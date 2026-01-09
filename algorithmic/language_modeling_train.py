from transformers import GPT2LMHeadModel, GPT2Config, TrainingArguments, Trainer, TrainerCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import random
from copy import deepcopy
import string
import argparse
import itertools
import os
from collections import Counter
from typing import Optional
import yaml

class NoPE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return torch.tensor(0.0, device=x.device) 

class NoPEGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer.wpe = NoPE()

class RegGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, coef):
        super().__init__(config)
        self.coef = coef

    def forward(self, *args, labels: Optional[torch.LongTensor] = None, **kwargs):
        outputs = super().forward(*args, labels=labels, **kwargs)

        if labels is not None:
            loss2 = self.compute_regularizer()

            if isinstance(outputs, tuple):
                outputs = (outputs[0] + loss2 * self.coef,) + outputs[1:]
            else:
                outputs.loss = outputs.loss + loss2 * self.coef
        return outputs
    
    def compute_regularizer(self):
        pe = self.transformer.wpe.weight # (num_embeddings, embedding_dim)

        square_sum = 0
        for block in self.transformer.h:
            w_matrix = block.attn.c_attn.weight # W_qkv for this layer (including all heads), 
            # it can first be split (by columns) into 3 equal part, correspond to q, k, v. Each part then be spit into many parts for each head
            k_offset = block.attn.embed_dim
            head_dim = block.attn.head_dim
            for i in range(block.attn.num_heads):
                w_query = w_matrix[:, i*head_dim : (i+1)*head_dim]  # W_q for head i
                w_key = w_matrix[:, k_offset+i*head_dim : k_offset+(i+1)*head_dim]  # W_k for head i

                product = (pe @ w_query) @ ((pe @ w_key).T)
                product = (torch.tril(product)**2).sum(dim=0).mean()
                square_sum = square_sum + product

        return square_sum

class customTokenizer():
    def __init__(self, vocab: list[str]):
        normal_tkn_num = len(vocab) # each element is a token

        self.bos_token = "<bos>"
        self.sep_token = "<sep>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.bos_token_id = normal_tkn_num
        self.sep_token_id = normal_tkn_num + 1
        self.eos_token_id = normal_tkn_num + 2
        self.pad_token_id = normal_tkn_num + 3
        self.special_token_ids = [self.bos_token_id, self.sep_token_id, self.eos_token_id, self.pad_token_id]
        self.special_tokens = [self.bos_token, self.sep_token, self.eos_token, self.pad_token]
        assert all(t not in vocab for t in self.special_tokens)
        
        # self.vocab = {"0": 0, "1": 1}
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.vocab[self.bos_token] = self.bos_token_id
        self.vocab[self.sep_token] = self.sep_token_id
        self.vocab[self.eos_token] = self.eos_token_id
        self.vocab[self.pad_token] = self.pad_token_id

        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.padding_side = "right"

    def __call__(self, strings: list[str] | str, **kwargs):
        # this func is not used, since the data generator does not generate str
        # string is tokenized by white space
        if type(strings) == str:
            strings = [strings]
        ids = []
        strings = [s.split(" ") for s in strings]
        max_len = max(map(lambda x: len(x), strings))
        for s in strings:
            ids.append( list(map(lambda x: self.vocab[x], s)) + [self.pad_token_id] * (max_len-len(s)) )

        return {"input_ids": torch.LongTensor(ids)}

    def convert_ids_to_tokens(self, ids: list[int], rm_special=False):
        if rm_special:
            return [self.vocab_inv[i] for i in ids if i not in self.special_token_ids]
        else:
            return list(map(lambda x: self.vocab_inv[x], ids))

    def __len__(self):
        return len(self.vocab)
    
class BinaryMajorityDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        assert len(tokenizer) == 6
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)   # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)
            while True:
                num_zero = random.randint(0, length)
                if num_zero != length-num_zero:
                    break
            instance = [0, ] * num_zero + [1, ] * (length - num_zero)
            random.shuffle(instance)
            ans = 0 if num_zero > length-num_zero else 1

            instance.insert(0, self.tokenizer.bos_token_id)
            instance.append(self.tokenizer.sep_token_id)
            instance.append(ans)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + bits.. + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, self.max_test_length - length)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class MajorityDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)   # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)
            while True:
                instance = random.choices(range(len(self.tokenizer)-4), k=length)
                most_common = Counter(instance).most_common(2)
                if len(most_common) < 2 or most_common[0][1] > most_common[1][1]:
                    break
            ans = most_common[0][0]

            instance.insert(0, self.tokenizer.bos_token_id)
            instance.append(self.tokenizer.sep_token_id)
            instance.append(ans)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + bits.. + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, self.max_test_length - length)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label

class BinaryMajorityInterleaveDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int, period: int):
        super().__init__()
        self.tokenizer = tokenizer 
        assert len(tokenizer) == 6
        self.range_min, self.range_max = length_range
        self.range_min = max(3, self.range_min)
        self.max_test_length = max_test_length
        self.period = period
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            total_length = random.randint(self.range_min, self.range_max)
            length = round(total_length / self.period)
            if length * self.period > self.range_max:
                length -= 1
            if length * self.period < self.range_min:
                length += 1
            
            instances = []
            answers = []
            for i in range(self.period):
                while True:
                    num_zero = random.randint(0, length)
                    if num_zero != length-num_zero:
                        break
                instance = [0, ] * num_zero + [1, ] * (length - num_zero)
                random.shuffle(instance)
                instances.append(instance)

                ans = 0 if num_zero > length-num_zero else 1
                answers.append(ans)

            whole_instance = [val for tup in zip(*instances) for val in tup]

            whole_instance.insert(0, self.tokenizer.bos_token_id)
            whole_instance.append(self.tokenizer.sep_token_id)
            whole_instance.extend(answers)
            whole_instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(whole_instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length*self.period+2] = [self.tokenizer.pad_token_id,] * (length*self.period+2)   # bos + bits.. + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, self.max_test_length - length*self.period)
            else:
                offset = 0
            pos_ids = list(range(offset, len(whole_instance)+offset))

            yield whole_instance, pos_ids, label


class UniqueCopyDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert len(tokenizer) - 4 >= max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied
            
            temp = random.sample(range(len(self.tokenizer)-4), length)
            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.extend(temp)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + ... + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, (self.max_test_length - length) * 2)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class RepeatCopyDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied
            
            temp = random.choices(range(len(self.tokenizer)-4), k=length)
            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.extend(temp)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + ... + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, (self.max_test_length - length) * 2)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class SortDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert len(tokenizer) - 4 >= max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied

            temp = random.sample(range(len(self.tokenizer)-4), length)
            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.extend(sorted(temp))
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + bits.. + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, (self.max_test_length - length) * 2)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class ParityDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max) 
            num_ones = random.randint(0, length)
            temp = [self.tokenizer.vocab["1"]] * num_ones + [self.tokenizer.vocab["0"]] * (length - num_ones)
            random.shuffle(temp)
            ans = self.tokenizer.vocab["e"] if num_ones % 2 == 0 else self.tokenizer.vocab["o"]

            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.append(ans)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + bits.. + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, self.max_test_length - length)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label

class AdditionDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.range_min = max(4, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied

            len_operand1 = random.randint(1, length-3)
            len_operand2 = length - 2 - len_operand1
            
            if len_operand1 > 1:
                operand1 = ["1"] + random.choices(["0", "1"], k=len_operand1-1)
            else:
                operand1 = random.choices(["0", "1"], k=1)
            if len_operand2 > 1:
                operand2 = ["1"] + random.choices(["0", "1"], k=len_operand2-1)
            else:
                operand2 = random.choices(["0", "1"], k=1)

            ans = int("0b" + "".join(operand1), 2) + int("0b" + "".join(operand2), 2)
            ans = list(bin(ans)[2:])

            instance = [self.tokenizer.bos_token]
            instance.extend(operand1)
            instance.append("+")
            instance.extend(operand2)
            instance.append("=")
            instance.extend(ans)
            instance.append(self.tokenizer.eos_token)

            instance = list(map(lambda x: self.tokenizer.vocab[x], instance))

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+1] = [self.tokenizer.pad_token_id,] * (length+1)   # bos + bits.. + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, self.max_test_length*2 - len(instance))
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class EvalDataset(Dataset):
    def __init__(self, d: IterableDataset, num_data: int) -> None:
        super().__init__()
        self.data = []
        for i, item in enumerate(d):
            if i >= num_data:
                break
            self.data.append(item)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class FixedDataset(Dataset):
    def __init__(self, examples):
        super().__init__()
        self.data = examples
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def generate_deduplicated_datasets(dataset_class, tokenizer, train_range, test_range,
                                 max_length, train_size=50000, test_size=2000,
                                 train_seed=42, test_seed=1337, **kwargs):
    """
    Generate deduplicated train and test datasets ensuring no overlap.

    Args:
        dataset_class: The dataset class to instantiate (e.g., BinaryMajorityDataset)
        tokenizer: Custom tokenizer for the task
        train_range: Tuple (min, max) for training lengths
        test_range: Tuple (min, max) for test lengths
        max_length: Maximum length for positional embeddings
        train_size: Number of training examples to generate
        test_size: Number of test examples to generate
        train_seed: Random seed for training data generation (default: 42)
        test_seed: Random seed for test data generation (default: 1337)
        **kwargs: Additional arguments for dataset class (e.g., period=3)

    Returns:
        tuple: (train_dataset, test_dataset) as FixedDataset instances
    """
    # Generate test set first with different seed
    original_torch_state = torch.get_rng_state()
    original_random_state = random.getstate()

    torch.manual_seed(test_seed)
    random.seed(test_seed)
    test_examples = []
    test_gen = dataset_class(tokenizer, test_range, max_length, **kwargs)
    for i, example in enumerate(test_gen):
        if i >= test_size:
            break
        test_examples.append(example)

    # Generate train set with original seed, skip duplicates
    torch.set_rng_state(original_torch_state)
    random.setstate(original_random_state)
    torch.manual_seed(train_seed)
    random.seed(train_seed)

    # Create set of test examples for duplicate checking
    test_set = set(tuple(ex[0]) for ex in test_examples)

    train_examples = []
    train_gen = dataset_class(tokenizer, train_range, max_length, **kwargs)

    for example in train_gen:
        if tuple(example[0]) not in test_set:
            train_examples.append(example)
        if len(train_examples) >= train_size:
            break

    # Restore original random states
    torch.set_rng_state(original_torch_state)
    random.setstate(original_random_state)

    return FixedDataset(train_examples), FixedDataset(test_examples)

class customCollator():
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        input_ids, pos_ids, labels = tuple(zip(*examples))
        max_len = max(len(item) for item in input_ids)

        [item.extend([self.pad_id,] * (max_len - len(item))) for item in input_ids]
        input_ids = torch.LongTensor(input_ids)
        [item.extend([self.pad_id,] * (max_len - len(item))) for item in labels]
        labels = torch.LongTensor(labels)
        labels[labels == self.pad_id] = -100
        [item.extend([item[-1],] * (max_len - len(item))) for item in pos_ids]
        pos_ids = torch.LongTensor(pos_ids)
        
        batch = {"input_ids": input_ids, "position_ids": pos_ids, "labels": labels}
        return batch
    
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    predictions = np.argmax(shift_logits, axis=-1)
    correct = np.all((predictions == shift_labels) | (shift_labels == -100), axis=1)
    return {"acc": correct.sum() / len(correct)}

def compute_run_statistics(results_per_run, test_length_ranges, train_length_range):
    """
    Compute mean, std, best, and worst statistics across multiple runs.

    Args:
        results_per_run: List of dicts, each containing eval metrics for one run
        test_length_ranges: List of test length ranges
        train_length_range: Training length range

    Returns:
        dict: Statistics including mean, std, best_run, worst_run
    """
    if len(results_per_run) == 0:
        return {}

    # Collect all metric names from first run
    metric_names = list(results_per_run[0].keys())

    # Compute mean and std for each metric
    stats = {}
    for metric_name in metric_names:
        values = [run[metric_name] for run in results_per_run]
        stats[f"{metric_name}_mean"] = np.mean(values)
        stats[f"{metric_name}_std"] = np.std(values)

    # Find best and worst runs based on train accuracy
    # The key format is: eval_len{X}-{Y}_train_acc
    train_acc_key = f"eval_len{train_length_range[0]}-{train_length_range[1]}_train_acc"

    # Debug: check if key exists, if not find what keys we have
    if train_acc_key not in results_per_run[0]:
        # Fallback: find any key with 'train' in it
        train_keys = [k for k in results_per_run[0].keys() if 'train' in k.lower()]
        if train_keys:
            train_acc_key = train_keys[0]
        else:
            # Use first available metric
            train_acc_key = list(results_per_run[0].keys())[0]

    best_idx = np.argmax([run[train_acc_key] for run in results_per_run])
    worst_idx = np.argmin([run[train_acc_key] for run in results_per_run])

    stats['best_run'] = {
        'run_idx': best_idx,
        'metrics': results_per_run[best_idx]
    }
    stats['worst_run'] = {
        'run_idx': worst_idx,
        'metrics': results_per_run[worst_idx]
    }

    return stats

def setup_task_datasets(task, tokenizer, train_length_range, test_length_ranges, max_test_length,
                       train_size, test_size, test_num, train_seed=42, test_seed=1337):
    """
    Helper function to generate train and test datasets for a given task.

    Args:
        task: Task name (e.g., "bin_majority")
        tokenizer: Custom tokenizer for the task
        train_length_range: Tuple (min, max) for training lengths
        test_length_ranges: List of tuples for test length ranges
        max_test_length: Maximum test length
        train_size: Number of training examples
        test_size: Number of test examples
        test_num: Number of test examples for additional test ranges
        train_seed: Random seed for training data (default: 42)
        test_seed: Random seed for test data (default: 1337)

    Returns:
        tuple: (train_dataset, test_dataset_dict)
    """
    dataset_config = {
        "bin_majority": (BinaryMajorityDataset, {}),
        "majority": (MajorityDataset, {}),
        "bin_majority_interleave": (BinaryMajorityInterleaveDataset, {"period": 3}),
        "unique_copy": (UniqueCopyDataset, {}),
        "repeat_copy": (RepeatCopyDataset, {}),
        "sort": (SortDataset, {}),
        "parity": (ParityDataset, {}),
        "addition": (AdditionDataset, {}),
    }

    dataset_class, extra_kwargs = dataset_config[task]

    # Generate deduplicated train and same-length test datasets with specified seeds
    train_dataset, test_same = generate_deduplicated_datasets(
        dataset_class, tokenizer, train_length_range, train_length_range,
        max_test_length, train_size, test_size, train_seed, test_seed, **extra_kwargs)

    # Create test dataset dictionary
    test_dataset = {
        f"len{train_length_range[0]}-{train_length_range[1]}_train": train_dataset,
        f"len{train_length_range[0]}-{train_length_range[1]}_test": test_same,
        # Keep longer-length tests as before
        **{f"len{test_range[0]}-{test_range[1]}": EvalDataset(dataset_class(tokenizer, test_range, -1, **extra_kwargs), test_num)
            for test_range in test_length_ranges[1:]}  # Skip first range (same length)
    }

    return train_dataset, test_dataset

class myCallback(TrainerCallback):
    def __init__(self, log_to_summary=True):
        """
        Args:
            log_to_summary: If True, write results to summary file. If False, only do early stopping.
        """
        super().__init__()
        self.log_to_summary = log_to_summary

    def on_evaluate(self, state, args, control, metrics=None, logs=None, eval_dataloader=None, **kwargs):
        assert metrics["epoch"] >= getattr(self, "current_epoch", 0)
        if metrics["epoch"] > getattr(self, "current_epoch", 0):
            self.latest_acc = {}
            self.current_epoch = metrics["epoch"]
        for key in metrics.keys():
            if key.endswith("acc"):
                self.latest_acc[key] = metrics[key]
        # Now we have train + test for same length, plus additional test ranges
        if len(self.latest_acc) == len(test_length_ranges) + 1:
            if (self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_train_acc"] == 1.0) or (self.current_epoch == 1.0):
                if self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_train_acc"] == 1.0:
                    control.should_training_stop = True
                    global fit_train_data
                    fit_train_data = True
                    msg = f"early stop {self.current_epoch}\t\t"
                else:
                    msg = "reach max step\t\t"
                if self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_train_acc"] >= 0.99:
                    msg = ">> " + msg

                # Only log to summary file if log_to_summary is True (for n_runs == 1)
                if self.log_to_summary:
                    print(f"{n_layer}l{n_head}h{d_model}d_n{train_size}\t\t", msg, "\t\t".join([f"{k}: {v}" for k, v in self.latest_acc.items()]), f"\t\tlr: {lr}", file=summary_f)
                    summary_f.flush()

                if (self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_train_acc"] == 1.0) and (self.latest_acc[f"eval_len{test_length_ranges[1][0]}-{test_length_ranges[1][1]}_acc"] == 1.0):
                    global should_stop
                    should_stop = True
                


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["bin_majority", "majority", "bin_majority_interleave", "unique_copy", "repeat_copy", "sort", "parity", "addition"])
    parser.add_argument("--nope", action="store_true")
    parser.add_argument("--regularize", type=float, default=0.0)
    parser.add_argument("--train_size", type=int, default=50000, help="Number of training examples")
    parser.add_argument("--test_size", type=int, default=2000, help="Number of test examples")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(0)
    random.seed(0)

    # Load configuration from YAML file if provided, otherwise use defaults
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Extract data configuration
        train_length_range = tuple(config['data']['train_length_range'])
        test_length_ranges = [train_length_range] + [tuple(r) for r in config['data']['test_length_ranges']]
        max_test_length = test_length_ranges[-1][1]

        # Extract training configuration
        batch_size = config['training']['batch_size']
        test_num = config['training']['test_num']
        n_runs = config['training'].get('n_runs', 1)  # Default to 1 run for backward compatibility
        base_seed = config['training'].get('base_seed', 0)  # Default to seed 0

        # Handle train_size and test_size from config
        # Convert to list if single value for consistency with sweeping
        if 'train_size' in config['data']:
            train_sizes = config['data']['train_size']
            if not isinstance(train_sizes, list):
                train_sizes = [train_sizes]
        else:
            train_sizes = [args.train_size]

        if 'test_size' in config['data']:
            args.test_size = config['data']['test_size']

        # Generate all combinations of model configs including train_size
        configs = list(itertools.product(
            config['model']['n_layers'],
            config['model']['n_heads'],
            config['model']['d_model'],
            config['training']['learning_rate'],
            train_sizes
        ))
    else:
        # Default configuration (backwards compatible)
        train_length_range = (0, 50)
        test_length_ranges = [train_length_range] + [(51, 100)]
        max_test_length = test_length_ranges[-1][1]
        batch_size = 64
        test_num = 2_000
        n_runs = 1  # Default to single run
        base_seed = 0  # Default base seed

        # Default configs - now includes train_size
        configs = [(l, 1, 16, 1e-3, args.train_size) for l in [1, 2, 4, 8, 16]]

    per_device_bz = batch_size // torch.cuda.device_count() if torch.cuda.is_available() else batch_size

    # Tokenizer and n_positions setup (done once, outside the config loop)
    # Dataset generation is now done inside the loop to support train_size sweeping
    if args.task == "bin_majority":
        tokenizer = customTokenizer(["0", "1"])
        n_positions = max_test_length + 4      # bos, sep, ans, eos

    elif args.task == "majority":
        tokenizer = customTokenizer(list(string.ascii_lowercase))
        n_positions = max_test_length + 4      # bos, sep, ans, eos

    elif args.task == "bin_majority_interleave":
        tokenizer = customTokenizer(["0", "1"])
        n_positions = max_test_length + 6    # ans

    elif args.task == "unique_copy":
        tokenizer = customTokenizer([str(i) for i in range(max_test_length)])
        n_positions = max_test_length*2 + 3  # bos, sep, eos

    elif args.task == "repeat_copy":
        tokenizer = customTokenizer(["a", "b"])
        n_positions = max_test_length*2 + 3  # bos, sep, eos

    elif args.task == "sort":
        tokenizer = customTokenizer([str(i) for i in range(max_test_length)])
        n_positions = max_test_length*2 + 3  # bos, sep, eos

    elif args.task == "parity":
        tokenizer = customTokenizer(["0", "1", "e", "o"])       # even, odd
        n_positions = max_test_length + 4  # bos, sep, ans, eos

    elif args.task == "addition":
        tokenizer = customTokenizer(["0", "1", "+", "="])
        n_positions = max_test_length*2  # bos, ans, eos

    
    task_path = f"./lm-out-new-{args.task}"
    if not os.path.exists(task_path):
        os.mkdir(task_path)
    if args.nope:
        suffix = "-nope"
    elif args.regularize != 0:
        suffix = f"-reg{args.regularize}"
    else:
        suffix = ""
    summary_f = open(os.path.join(task_path, f"summary{suffix}.txt"), "w")

    should_stop = False
    fit_train_data = False
    for n_layer, n_head, d_model, lr, train_size in configs:

        if n_layer > 4:
            max_steps = 60_000
            warmup_steps = 3000
        else:
            max_steps = 30_000
            warmup_steps = 0

        # Collect results across multiple runs
        results_per_run = []

        for run_idx in range(n_runs):
            print(f"\n{'='*80}")
            print(f"Config: {n_layer}l{n_head}h{d_model}d_n{train_size} | Run {run_idx + 1}/{n_runs}")
            print(f"{'='*80}\n")

            # Set seeds for this run
            model_seed = base_seed + run_idx
            data_seed_train = 2 * model_seed
            data_seed_test = 2 * model_seed + 1

            # Generate fresh datasets for this run with run-specific seeds
            train_dataset, test_dataset = setup_task_datasets(
                args.task, tokenizer, train_length_range, test_length_ranges,
                max_test_length, train_size, args.test_size, test_num,
                data_seed_train, data_seed_test
            )

            # Print examples only for the first run of the first configuration
            if (n_layer, n_head, d_model, lr, train_size) == configs[0] and run_idx == 0:
                for i in range(3):
                    print("\ninput example:")
                    print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}_test"][i][0])))
                    print("label example:")
                    print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}_test"][i][2])))

            # Set model initialization seed
            torch.manual_seed(model_seed)
            random.seed(model_seed)

            output_dir = f"{n_layer}l{n_head}h{d_model}d_n{train_size}_run{run_idx}{'smalllr' if lr == 1e-4 else ''}{suffix}"
            output_dir = os.path.join(task_path, output_dir)

            cfg = GPT2Config(vocab_size=len(tokenizer),
                        n_positions=n_positions,
                        n_embd=d_model,
                        n_layer=n_layer,
                        n_head=n_head,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        attn_pdrop=0,
                        resid_pdrop=0,
                        embd_pdrop=0,
                        )

            if args.nope:
                model = NoPEGPT2LMHeadModel(cfg)
            elif args.regularize != 0:
                model = RegGPT2LMHeadModel(cfg, args.regularize)
            else:
                model = GPT2LMHeadModel(cfg)

            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                per_device_train_batch_size=per_device_bz,
                per_device_eval_batch_size=per_device_bz,
                max_steps=max_steps,
                eval_strategy="steps",
                eval_steps=3_000,
                save_strategy="no",
                logging_strategy="steps",
                logging_steps=3_000,
                learning_rate=lr,
                weight_decay=0.01,
                optim='adamw_torch',
                lr_scheduler_type='linear',
                warmup_steps=warmup_steps,
                report_to="none",
            )

            data_collator = customCollator(tokenizer.pad_token_id)

            # Only log to summary file during training if n_runs == 1
            # For multiple runs, we'll log aggregated stats after all runs complete
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[myCallback(log_to_summary=(n_runs == 1))],
            )

            trainer.train()

            # Evaluate on all test sets and collect results
            run_results = {}
            for test_name, test_data in test_dataset.items():
                # Use test_name as metric prefix for trainer.evaluate
                metrics = trainer.evaluate(test_data, metric_key_prefix=f"eval_{test_name}")

                # Debug: print what keys we got (only for first run)
                if run_idx == 0 and (n_layer, n_head, d_model, lr, train_size) == configs[0]:
                    print(f"DEBUG: test_name='{test_name}', metric keys: {list(metrics.keys())}")

                # Extract accuracy - trainer adds 'eval_' prefix, so key is 'eval_{test_name}_acc'
                acc_key = f"eval_{test_name}_acc"
                # Store with consistent naming: eval_len{X}-{Y}_train_acc or eval_len{X}-{Y}_acc
                result_key = f"eval_{test_name}_acc"
                run_results[result_key] = metrics.get(acc_key, 0.0)

            # Debug: print final run_results keys (only for first run)
            if run_idx == 0 and (n_layer, n_head, d_model, lr, train_size) == configs[0]:
                print(f"DEBUG: run_results keys: {list(run_results.keys())}")

            results_per_run.append(run_results)

            # Clean up to free memory
            del train_dataset, test_dataset, model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Compute statistics across runs (only if n_runs > 1)
        if n_runs > 1:
            stats = compute_run_statistics(results_per_run, test_length_ranges, train_length_range)

            # Log aggregated results to summary file
            print(f"\n{n_layer}l{n_head}h{d_model}d_n{train_size}\tmean\t", file=summary_f, end="")
            for metric_name in sorted([k for k in stats.keys() if k.endswith('_mean')]):
                base_name = metric_name.replace('_mean', '')
                mean_val = stats[metric_name]
                std_val = stats[f"{base_name}_std"]
                print(f"{base_name}: {mean_val:.4f}±{std_val:.4f}\t", file=summary_f, end="")
            print(f"lr: {lr}", file=summary_f)

            # Log best run
            best_run = stats['best_run']
            print(f"{n_layer}l{n_head}h{d_model}d_n{train_size}\tbest\t", file=summary_f, end="")
            for metric_name, value in sorted(best_run['metrics'].items()):
                print(f"{metric_name}: {value:.4f}\t", file=summary_f, end="")
            print(f"(run {best_run['run_idx']})", file=summary_f)

            # Log worst run
            worst_run = stats['worst_run']
            print(f"{n_layer}l{n_head}h{d_model}d_n{train_size}\tworst\t", file=summary_f, end="")
            for metric_name, value in sorted(worst_run['metrics'].items()):
                print(f"{metric_name}: {value:.4f}\t", file=summary_f, end="")
            print(f"(run {worst_run['run_idx']})", file=summary_f)

            summary_f.flush()
        else:
            # Single run - log normally (backward compatible)
            run_results = results_per_run[0]
            print(f"{n_layer}l{n_head}h{d_model}d_n{train_size}\t", file=summary_f, end="")
            for metric_name, value in sorted(run_results.items()):
                print(f"{metric_name}: {value:.4f}\t", file=summary_f, end="")
            print(f"lr: {lr}", file=summary_f)
            summary_f.flush()

    
    summary_f.close()