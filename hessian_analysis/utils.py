"""
Utility functions for Hessian analysis.

Provides functions to load models, create dataloaders, and compute Hessian metrics.
"""

import torch
import numpy as np
import sys
import os
import string
import re
from torch.utils.data import DataLoader, Subset
from transformers import GPT2LMHeadModel, GPT2Config

# Add patched PyHessian to path BEFORE importing
_current_dir = os.path.dirname(__file__)
_algorithmic_dir = os.path.join(os.path.dirname(_current_dir), 'algorithmic')
sys.path.insert(0, os.path.join(_algorithmic_dir, 'pyhessian_patched'))
from hessian import hessian

# Import from training script
sys.path.insert(0, _algorithmic_dir)
from language_modeling_train import (
    NoPEGPT2LMHeadModel,
    RegGPT2LMHeadModel,
    customTokenizer,
    setup_task_datasets,
    customCollator,
)


def load_model_and_config(model_path, device='cuda'):
    """
    Load saved model from HuggingFace Trainer checkpoint.

    Automatically detects model type (standard GPT2, NoPE, or Regularized)
    from the checkpoint directory name.

    Args:
        model_path: Path to checkpoint directory (contains pytorch_model.bin and config.json)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: Loaded PyTorch model
        config: GPT2Config object
        model_type: String indicating model type ('standard', 'nope', or 'reg')
    """
    # Handle multiple checkpoint directory structures:
    # 1. Direct path to model files (pytorch_model.bin or model.safetensors in model_path)
    # 2. Path to checkpoint-N subdirectory
    # 3. Path to directory containing checkpoint-N subdirectories

    # Check if model files exist directly in the provided path
    has_model_files = (
        os.path.isfile(os.path.join(model_path, "pytorch_model.bin")) or
        os.path.isfile(os.path.join(model_path, "model.safetensors"))
    )

    if has_model_files:
        # Case 1 or 2: Direct path to model files
        checkpoint_dir = model_path
    else:
        # Case 3: Look for checkpoint subdirectories
        try:
            entries = os.listdir(model_path)
            checkpoints = [d for d in entries if d.startswith("checkpoint-") and
                          os.path.isdir(os.path.join(model_path, d))]
            if checkpoints:
                # Use the highest numbered checkpoint
                checkpoint_nums = [int(d.split("-")[1]) for d in checkpoints]
                latest_checkpoint = f"checkpoint-{max(checkpoint_nums)}"
                checkpoint_dir = os.path.join(model_path, latest_checkpoint)
            else:
                # No checkpoint subdirectories, assume model_path is correct
                checkpoint_dir = model_path
        except (FileNotFoundError, PermissionError):
            checkpoint_dir = model_path

    config_path = os.path.join(checkpoint_dir, "config.json")

    # Check for both pytorch_model.bin and model.safetensors
    weights_path_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")
    weights_path_safetensors = os.path.join(checkpoint_dir, "model.safetensors")

    if os.path.exists(weights_path_safetensors):
        weights_path = weights_path_safetensors
        weights_format = "safetensors"
    elif os.path.exists(weights_path_bin):
        weights_path = weights_path_bin
        weights_format = "pytorch"
    else:
        raise FileNotFoundError(
            f"Model weights not found. Looked for:\n"
            f"  - {weights_path_safetensors}\n"
            f"  - {weights_path_bin}"
        )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    print(f"Loading model from: {checkpoint_dir}")

    # Load config
    config = GPT2Config.from_json_file(config_path)

    # Detect model type from directory name
    # Look at parent directories to find nope/reg markers
    path_parts = checkpoint_dir.split(os.sep)
    model_type = 'standard'
    reg_coef = None

    for part in path_parts:
        if 'nope' in part.lower():
            model_type = 'nope'
            break
        elif 'reg' in part.lower():
            model_type = 'reg'
            # Extract regularization coefficient from directory name
            match = re.search(r'reg([\d.]+)', part)
            if match:
                reg_coef = float(match.group(1))
            else:
                reg_coef = 0.1  # Default if not specified
            break

    # Instantiate appropriate model
    if model_type == 'nope':
        print("Loading NoPE model (no positional embeddings)")
        model = NoPEGPT2LMHeadModel(config)
    elif model_type == 'reg':
        print(f"Loading Regularized model (reg_coef={reg_coef})")
        model = RegGPT2LMHeadModel(config, reg_coef)
    else:
        print("Loading standard GPT2 model")
        model = GPT2LMHeadModel(config)

    # Load weights (handle both pytorch and safetensors formats)
    if weights_format == "safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location=device)

    # Load state dict with strict=False to handle tied weights
    # (lm_head.weight is tied to transformer.wte.weight in GPT2)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Check that only expected tied weights are missing
    expected_missing = ['lm_head.weight']  # This is tied to wte.weight
    unexpected_missing = [k for k in missing_keys if k not in expected_missing]

    if unexpected_missing:
        raise RuntimeError(
            f"Unexpected missing keys in state_dict: {unexpected_missing}\n"
            f"Expected missing (tied weights): {[k for k in missing_keys if k in expected_missing]}"
        )

    if unexpected_keys:
        print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")

    model.to(device)
    model.eval()

    print(f"Model loaded successfully: {model_type}")
    print(f"  Format: {weights_format}")
    print(f"  Layers: {config.n_layer}, Heads: {config.n_head}, d_model: {config.n_embd}")

    return model, config, model_type


def get_tokenizer_and_n_positions(task, max_test_length):
    """
    Get tokenizer and n_positions for a given task.

    Args:
        task: Task name (e.g., 'bin_majority')
        max_test_length: Maximum test sequence length

    Returns:
        tokenizer: customTokenizer instance
        n_positions: Number of positions for the model
    """
    if task == "bin_majority":
        tokenizer = customTokenizer(["0", "1"])
        n_positions = max_test_length + 4  # bos, sep, ans, eos
    elif task == "majority":
        tokenizer = customTokenizer(list(string.ascii_lowercase))
        n_positions = max_test_length + 4
    elif task == "bin_majority_interleave":
        tokenizer = customTokenizer(["0", "1"])
        n_positions = max_test_length + 6
    elif task == "unique_copy":
        tokenizer = customTokenizer([str(i) for i in range(max_test_length)])
        n_positions = max_test_length * 2 + 3
    elif task == "repeat_copy":
        tokenizer = customTokenizer(["a", "b"])
        n_positions = max_test_length * 2 + 3
    elif task == "sort":
        tokenizer = customTokenizer([str(i) for i in range(max_test_length)])
        n_positions = max_test_length * 2 + 3
    elif task == "parity":
        tokenizer = customTokenizer(["0", "1", "e", "o"])
        n_positions = max_test_length + 4
    elif task == "addition":
        tokenizer = customTokenizer(["0", "1", "+", "="])
        n_positions = max_test_length * 2
    else:
        raise ValueError(f"Unknown task: {task}")

    return tokenizer, n_positions


def create_hessian_dataloaders(task, train_range, test_ranges,
                                hessian_batch_size=200, max_samples=500,
                                train_seed=0, test_seed=1):
    """
    Create dataloaders for Hessian computation.

    Creates three separate dataloaders:
    - Train distribution (e.g., length 0-50)
    - Same-length test distribution (e.g., length 0-50, different seed)
    - Longer-length test distribution (e.g., length 51-100)

    Args:
        task: Task name (e.g., 'bin_majority')
        train_range: Training length range (e.g., (0, 50))
        test_ranges: List of test ranges (e.g., [(0, 50), (51, 100)])
        hessian_batch_size: Batch size for Hessian computation (~200)
        max_samples: Number of samples to use per dataset (for efficiency)
        train_seed: Seed for training data generation
        test_seed: Seed for test data generation

    Returns:
        dict: {dataset_name: dataloader}
        tokenizer: The tokenizer used
    """
    max_test_length = test_ranges[-1][1]

    # Get tokenizer
    tokenizer, _ = get_tokenizer_and_n_positions(task, max_test_length)

    print(f"\nCreating datasets for task: {task}")
    print(f"  Train range: {train_range}")
    print(f"  Test ranges: {test_ranges}")
    print(f"  Max samples per dataset: {max_samples}")

    # Use setup_task_datasets to create datasets with proper deduplication
    train_dataset, test_dataset_dict = setup_task_datasets(
        task, tokenizer, train_range, test_ranges,
        max_test_length,
        train_size=max_samples,
        test_size=max_samples,
        test_num=max_samples,
        train_seed=train_seed,
        test_seed=test_seed
    )

    # Create dataloaders
    collator = customCollator(tokenizer.pad_token_id)
    dataloaders = {}

    for name, dataset in test_dataset_dict.items():
        # Limit dataset size for efficiency
        dataset_size = len(dataset)
        if dataset_size > max_samples:
            dataset = Subset(dataset, range(max_samples))
            print(f"  {name}: using {max_samples} samples (truncated from {dataset_size})")
        else:
            print(f"  {name}: using {dataset_size} samples")

        dataloaders[name] = DataLoader(
            dataset,
            batch_size=hessian_batch_size,
            shuffle=False,
            collate_fn=collator
        )

    return dataloaders, tokenizer


class BatchWrapper:
    """
    Wrapper for batch dict that provides a .size() method for PyHessian compatibility.

    PyHessian expects inputs.size(0) to get batch size, but our collator returns dicts.
    This wrapper makes the dict behave like a tensor for PyHessian's needs.
    """
    def __init__(self, batch_dict):
        self.batch_dict = batch_dict
        self._batch_size = batch_dict['input_ids'].size(0)

    def size(self, dim=None):
        """Return batch size when called with dim=0, otherwise delegate to input_ids."""
        if dim == 0 or dim is None:
            return self._batch_size
        return self.batch_dict['input_ids'].size(dim)

    def to(self, device):
        """Move all tensors in the batch to the specified device."""
        moved_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in self.batch_dict.items()}
        return BatchWrapper(moved_dict)

    def __getitem__(self, key):
        """Allow dict-like access."""
        return self.batch_dict[key]

    def keys(self):
        return self.batch_dict.keys()


class ModelWrapper(torch.nn.Module):
    """
    Wrapper for models to handle custom batch format for PyHessian.

    PyHessian expects a simple forward pass, but our models need
    input_ids, position_ids, and labels in a specific format.
    """
    def __init__(self, model, ignore_regularization=True):
        """
        Args:
            model: The GPT2 model (standard, NoPE, or Reg)
            ignore_regularization: If True, don't add regularization loss
                                   (for RegGPT2LMHeadModel) during Hessian computation
        """
        super().__init__()
        self.model = model
        self.ignore_regularization = ignore_regularization
        self.is_reg_model = isinstance(model, RegGPT2LMHeadModel)

    def forward(self, batch_wrapper):
        """
        Forward pass that accepts BatchWrapper from dataloader.

        Args:
            batch_wrapper: BatchWrapper containing batch dict

        Returns:
            model outputs (with logits and loss)
        """
        # Extract the actual batch dict
        if isinstance(batch_wrapper, BatchWrapper):
            batch_dict = batch_wrapper.batch_dict
        else:
            # Fallback: assume it's already a dict
            batch_dict = batch_wrapper

        # For regularized models, temporarily disable regularization for Hessian
        if self.is_reg_model and self.ignore_regularization:
            original_coef = self.model.coef
            self.model.coef = 0.0

        outputs = self.model(
            input_ids=batch_dict['input_ids'],
            position_ids=batch_dict['position_ids'],
            labels=batch_dict['labels']
        )

        # Restore regularization coefficient
        if self.is_reg_model and self.ignore_regularization:
            self.model.coef = original_coef

        return outputs


def hessian_criterion(outputs, target):
    """
    Criterion for PyHessian that extracts loss from model outputs.

    Args:
        outputs: Model outputs (CausalLMOutputWithCrossAttentions)
        target: Target labels (not used, loss already computed in forward pass)

    Returns:
        scalar loss
    """
    return outputs.loss


class HessianDataLoaderWrapper:
    """
    Wrapper for DataLoader to work with PyHessian's expected format.

    PyHessian expects: for data, target in dataloader
    Our collator returns: {input_ids, position_ids, labels}

    This wrapper converts the dict batch to a BatchWrapper that has .size() method.
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            # Wrap the batch dict so PyHessian can call .size(0) on it
            batch_wrapped = BatchWrapper(batch)
            # PyHessian will pass batch_wrapped to model, and labels to criterion
            yield batch_wrapped, batch['labels']

    def __len__(self):
        return len(self.dataloader)


def compute_hessian_metrics(model, dataloader, device='cuda', verbose=True, seed=42,
                           top_k=100):
    """
    Compute Hessian top eigenvalues and spectral density using PyHessian.

    Args:
        model: PyTorch model
        dataloader: DataLoader with batches of data
        device: Device to run on ('cuda' or 'cpu')
        verbose: Whether to print progress
        seed: Random seed for reproducibility (default: 42)
        top_k: Number of top eigenvalues to compute (default: 100)

    Returns:
        dict with keys: top_eigenvalues, density_eigen, density_weight
    """
    # Set random seeds for reproducibility
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Disable flash attention for Hessian computation (it doesn't support double backward)
    # This is necessary for PyTorch 2.0+ which uses flash attention by default
    import torch.nn.functional as F
    old_sdpa_enabled = torch.backends.cuda.sdp_kernel

    # Disable all optimized attention kernels
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)  # Use standard math implementation

    try:
        # Wrap model and dataloader for PyHessian compatibility
        wrapped_model = ModelWrapper(model, ignore_regularization=True)
        wrapped_dataloader = HessianDataLoaderWrapper(dataloader)

        # Initialize PyHessian
        if verbose:
            print("  Initializing PyHessian...")

        hessian_comp = hessian(
            wrapped_model,
            hessian_criterion,
            dataloader=wrapped_dataloader,
            cuda=(device.type == 'cuda')
        )

        # Compute top-k eigenvalues
        if verbose:
            print(f"  Computing top {top_k} eigenvalues...")
        top_eigenvalues, _ = hessian_comp.eigenvalues(maxIter=150, tol=1e-5, top_n=top_k)

        # Compute eigenvalue density
        if verbose:
            print("  Computing eigenvalue spectral density...")

        density_eigen, density_weight = hessian_comp.density(iter=100, n_v=5)

        result = {
            'top_eigenvalues': top_eigenvalues,
            'density_eigen': density_eigen,
            'density_weight': density_weight
        }

        if verbose:
            print(f"    Computed {len(top_eigenvalues)} eigenvalues")
            print(f"    Range: [{top_eigenvalues[-1]:.4f}, {top_eigenvalues[0]:.4f}]")

        return result
    finally:
        # Restore attention settings
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
