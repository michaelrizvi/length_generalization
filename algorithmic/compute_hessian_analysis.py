#!/usr/bin/env python3
"""
Compute Hessian eigenvalue analysis for trained models.

This script computes the Hessian eigenvalue spectral density (ESD), top eigenvalue,
and trace for trained models on algorithmic tasks. It generates separate analyses
for train, same-length test, and longer-length test distributions.

Usage:
    python compute_hessian_analysis.py \
        --model_path ./lm-out-new-bin_majority/1l1h16d_n50000_run0/checkpoint-30000 \
        --task bin_majority \
        --output_dir ./hessian_results \
        --batch_size 200 \
        --max_samples 500

Requirements:
    - pyhessian (pip install pyhessian)
    - Model checkpoint saved by language_modeling_train.py
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add patched PyHessian to path BEFORE importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyhessian_patched'))
from hessian import hessian

from transformers import GPT2LMHeadModel, GPT2Config
import json
import string
import re
from torch.utils.data import DataLoader, Subset

# Import from training script
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
                           compute_top_k=False, top_k=1000):
    """
    Compute Hessian eigenvalue, trace, and density using PyHessian.

    Args:
        model: PyTorch model
        dataloader: DataLoader with batches of data
        device: Device to run on ('cuda' or 'cpu')
        verbose: Whether to print progress
        seed: Random seed for reproducibility (default: 42)
        compute_top_k: If True, compute top-k eigenvalues using Lanczos (default: False)
        top_k: Number of top eigenvalues to compute (default: 1000)

    Returns:
        dict with keys: top_eigenvalue, trace, density_eigen, density_weight
                       (and optionally top_k_eigenvalues if compute_top_k=True)
    """
    # Set random seeds for reproducibility
    import random
    import numpy as np
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

        # Compute top eigenvalue
        if verbose:
            print("  Computing top eigenvalue...")
        top_eigenvalues, _ = hessian_comp.eigenvalues(maxIter=300, tol=1e-5, top_n=1)

        # Compute trace
        if verbose:
            print("  Computing trace...")
        trace = hessian_comp.trace(maxIter=300, tol=1e-5)
        # Trace returns a list, take the mean as the final estimate
        if isinstance(trace, (list, tuple)):
            trace = sum(trace) / len(trace)

        # Compute eigenvalue density
        if verbose:
            print("  Computing eigenvalue spectral density...")

        density_eigen, density_weight = hessian_comp.density(iter=300, n_v=10)

        result = {
            'top_eigenvalue': float(top_eigenvalues[0]),
            'trace': float(trace),
            'density_eigen': density_eigen,
            'density_weight': density_weight
        }

        # Optionally compute top-k eigenvalues using Lanczos
        if compute_top_k:
            if verbose:
                print(f"  Computing top {top_k} eigenvalues using Lanczos method...")
                print(f"    (This may take several hours for large k)")

            # Use Lanczos with more iterations to get top-k eigenvalues
            # We run multiple trials (n_v) and take the trial with highest top eigenvalue
            lanczos_eigen, lanczos_weight = hessian_comp.density(iter=top_k, n_v=5)

            # lanczos_eigen is list of lists (one per n_v trial)
            # Each trial gives us ~top_k eigenvalues
            # Take the trial with the highest max eigenvalue (most reliable)
            max_eigenvalues = [max(trial) for trial in lanczos_eigen]
            best_trial_idx = np.argmax(max_eigenvalues)

            # Extract eigenvalues from best trial and sort descending
            top_k_eigenvalues = np.array(lanczos_eigen[best_trial_idx])
            top_k_eigenvalues = np.sort(top_k_eigenvalues)[::-1]  # Sort descending

            result['top_k_eigenvalues'] = top_k_eigenvalues.tolist()
            result['top_k_count'] = len(top_k_eigenvalues)

            if verbose:
                print(f"    Computed {len(top_k_eigenvalues)} eigenvalues")
                print(f"    Range: [{top_k_eigenvalues[-1]:.4f}, {top_k_eigenvalues[0]:.4f}]")

        return result
    finally:
        # Restore attention settings
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def plot_top_k_eigenvalues(eigenvalues, output_path, title="", log_scale=True):
    """
    Plot sorted top-k eigenvalues.

    Args:
        eigenvalues: Array of eigenvalues (sorted descending)
        output_path: Path to save PDF
        title: Title for the plot
        log_scale: If True, use log scale for y-axis (default: True)
    """
    eigenvalues = np.array(eigenvalues)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: All eigenvalues
    x = np.arange(1, len(eigenvalues) + 1)
    ax1.plot(x, eigenvalues, linewidth=2, color='#2E86AB', marker='o', markersize=2, alpha=0.7)
    ax1.set_xlabel('Eigenvalue Rank', fontsize=14)
    ax1.set_ylabel('Eigenvalue', fontsize=14)
    if log_scale:
        ax1.set_yscale('log')
        ax1.set_ylabel('Eigenvalue (log scale)', fontsize=14)
    ax1.set_title(f'Top-{len(eigenvalues)} Eigenvalues\n{title}', fontsize=16)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add annotations for key statistics
    textstr = '\n'.join((
        f'λ₁ (max): {eigenvalues[0]:.2f}',
        f'λ₁₀₀: {eigenvalues[min(99, len(eigenvalues)-1)]:.2f}',
        f'λ₁₀₀₀: {eigenvalues[min(999, len(eigenvalues)-1)]:.2f}' if len(eigenvalues) >= 1000 else '',
        f'Sum: {np.sum(eigenvalues):.2f}',
    ))
    ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right plot: Cumulative explained variance
    cumsum = np.cumsum(np.abs(eigenvalues))
    total = np.sum(np.abs(eigenvalues))
    cumulative_variance = cumsum / total * 100

    ax2.plot(x, cumulative_variance, linewidth=2, color='#A23B72')
    ax2.set_xlabel('Number of Eigenvalues', fontsize=14)
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=14)
    ax2.set_title('Cumulative Variance Explained', fontsize=16)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 105])

    # Add horizontal lines at key percentages
    for pct in [50, 90, 95, 99]:
        idx = np.argmax(cumulative_variance >= pct)
        if cumulative_variance[idx] >= pct:
            ax2.axhline(y=pct, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax2.text(len(eigenvalues) * 0.02, pct + 1, f'{pct}% at λ{idx+1}',
                    fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Top-k eigenvalue plot saved to: {output_path}")


def plot_esd(density_eigen, density_weight, output_path, title=""):
    """
    Plot and save eigenvalue spectral density as PDF.

    Args:
        density_eigen: List of lists of eigenvalue estimates (one per n_v run)
        density_weight: List of lists of density weights (one per n_v run)
        output_path: Path to save PDF
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))

    # PyHessian's density() returns list of lists (one per n_v run)
    # We need to flatten or plot each run
    if isinstance(density_eigen, list) and len(density_eigen) > 0 and isinstance(density_eigen[0], list):
        # Multiple runs - plot each one
        for i, (eigen_vals, weights) in enumerate(zip(density_eigen, density_weight)):
            # Sort by eigenvalue magnitude for better visualization
            sorted_pairs = sorted(zip(eigen_vals, weights), key=lambda x: x[0])
            eigen_vals_sorted = [x[0] for x in sorted_pairs]
            weights_sorted = [x[1] for x in sorted_pairs]

            plt.semilogy(eigen_vals_sorted, weights_sorted, linewidth=2, alpha=0.7,
                        label=f'Run {i+1}' if len(density_eigen) > 1 else None)
        if len(density_eigen) > 1:
            plt.legend()
    else:
        # Single flat array - sort as well
        sorted_pairs = sorted(zip(density_eigen, density_weight), key=lambda x: x[0])
        eigen_vals_sorted = [x[0] for x in sorted_pairs]
        weights_sorted = [x[1] for x in sorted_pairs]
        plt.semilogy(eigen_vals_sorted, weights_sorted, linewidth=2, color='#2E86AB')

    plt.xlabel('Eigenvalue', fontsize=14)
    plt.ylabel('Density (log scale)', fontsize=14)
    plt.title(f'Hessian Eigenvalue Spectral Density\n{title}', fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ESD plot saved to: {output_path}")


def parse_test_ranges(args_test_ranges):
    """
    Parse test ranges from command line arguments.

    Args:
        args_test_ranges: List of integers like [51, 100, 101, 150]

    Returns:
        List of tuples like [(51, 100), (101, 150)]
    """
    if len(args_test_ranges) % 2 != 0:
        raise ValueError("test_ranges must have an even number of values (pairs of min,max)")

    ranges = []
    for i in range(0, len(args_test_ranges), 2):
        ranges.append((args_test_ranges[i], args_test_ranges[i+1]))
    return ranges


def main():
    parser = argparse.ArgumentParser(
        description="Compute Hessian analysis for trained models on algorithmic tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a trained model
  python compute_hessian_analysis.py \\
      --model_path ./lm-out-new-bin_majority/1l1h16d_n50000_run0/checkpoint-30000 \\
      --task bin_majority \\
      --output_dir ./hessian_results

  # With custom batch size and sample count
  python compute_hessian_analysis.py \\
      --model_path ./lm-out-new-parity/4l2h256d_n50000_run0/checkpoint-30000 \\
      --task parity \\
      --output_dir ./hessian_results/parity \\
      --batch_size 100 \\
      --max_samples 300
        """
    )

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint directory")
    parser.add_argument("--task", type=str, required=True,
                       choices=["bin_majority", "majority", "bin_majority_interleave",
                               "unique_copy", "repeat_copy", "sort", "parity", "addition"],
                       help="Task name")
    parser.add_argument("--output_dir", type=str, default="./hessian_results",
                       help="Directory to save results (default: ./hessian_results)")
    parser.add_argument("--batch_size", type=int, default=200,
                       help="Batch size for Hessian computation (default: 200)")
    parser.add_argument("--max_samples", type=int, default=500,
                       help="Maximum samples per dataset (default: 500)")
    parser.add_argument("--train_range", type=int, nargs=2, default=[0, 50],
                       help="Training length range (default: 0 50)")
    parser.add_argument("--test_ranges", type=int, nargs='+', default=[51, 100],
                       help="Test length ranges as pairs: min1 max1 min2 max2 ... (default: 51 100)")
    parser.add_argument("--train_seed", type=int, default=0,
                       help="Random seed for training data generation (default: 0)")
    parser.add_argument("--test_seed", type=int, default=1,
                       help="Random seed for test data generation (default: 1)")
    parser.add_argument("--hessian_seed", type=int, default=42,
                       help="Random seed for Hessian computation (default: 42)")
    parser.add_argument("--compute_top_k", action="store_true",
                       help="Compute top-k eigenvalues using Lanczos method")
    parser.add_argument("--top_k", type=int, default=1000,
                       help="Number of top eigenvalues to compute (default: 1000)")

    args = parser.parse_args()

    # Parse test ranges
    train_range = tuple(args.train_range)
    test_ranges = [train_range] + parse_test_ranges(args.test_ranges)

    print("="*80)
    print("Hessian Eigenvalue Analysis for Algorithmic Tasks")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Task: {args.task}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max samples per dataset: {args.max_samples}")
    print(f"  Train range: {train_range}")
    print(f"  Test ranges: {test_ranges[1:]}")
    print(f"  Data seeds: train={args.train_seed}, test={args.test_seed}")
    print(f"  Hessian seed: {args.hessian_seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Load model
    print("\n" + "="*80)
    print("Step 1: Loading Model")
    print("="*80)
    model, config, model_type = load_model_and_config(args.model_path, device)

    # Create dataloaders
    print("\n" + "="*80)
    print("Step 2: Creating Datasets")
    print("="*80)
    dataloaders, tokenizer = create_hessian_dataloaders(
        args.task, train_range, test_ranges,
        args.batch_size, args.max_samples,
        args.train_seed, args.test_seed
    )

    # Extract model config for directory naming
    model_config_str = f"{config.n_layer}l{config.n_head}h{config.n_embd}d"

    # Create output directory structure: output_dir/task/model_config/
    task_output_dir = os.path.join(args.output_dir, args.task, model_config_str)
    os.makedirs(task_output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {task_output_dir}")

    # Compute Hessian for each dataset
    print("\n" + "="*80)
    print("Step 3: Computing Hessian Metrics")
    print("="*80)

    all_results = {}
    for dataset_name, dataloader in dataloaders.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        results = compute_hessian_metrics(
            model, dataloader, device, verbose=True, seed=args.hessian_seed,
            compute_top_k=args.compute_top_k, top_k=args.top_k
        )
        all_results[dataset_name] = results

        # Plot ESD
        output_path = os.path.join(task_output_dir, f"esd_{dataset_name}.pdf")
        plot_esd(
            results['density_eigen'],
            results['density_weight'],
            output_path,
            title=f"{dataset_name} ({model_config_str}, {model_type})"
        )

        # Plot top-k eigenvalues if computed
        if 'top_k_eigenvalues' in results:
            output_path = os.path.join(task_output_dir, f"top_k_eigenvalues_{dataset_name}.pdf")
            plot_top_k_eigenvalues(
                results['top_k_eigenvalues'],
                output_path,
                title=f"{dataset_name} ({model_config_str}, {model_type})"
            )

        # Print summary
        print(f"\n  Results:")
        print(f"    Top Eigenvalue: {results['top_eigenvalue']:.6f}")
        print(f"    Trace: {results['trace']:.6f}")
        print(f"    ESD: Computed successfully")
        if 'top_k_eigenvalues' in results:
            print(f"    Top-K Eigenvalues: {results['top_k_count']} computed")

    # Save all results as JSON
    print("\n" + "="*80)
    print("Step 4: Saving Results")
    print("="*80)

    json_results = {
        'model_path': args.model_path,
        'task': args.task,
        'model_config': {
            'n_layers': config.n_layer,
            'n_heads': config.n_head,
            'd_model': config.n_embd,
            'model_type': model_type
        },
        'hessian_config': {
            'batch_size': args.batch_size,
            'max_samples': args.max_samples,
            'train_seed': args.train_seed,
            'test_seed': args.test_seed,
            'hessian_seed': args.hessian_seed
        },
        'datasets': {}
    }

    for dataset_name, results in all_results.items():
        json_results['datasets'][dataset_name] = {
            'top_eigenvalue': results['top_eigenvalue'],
            'trace': results['trace']
        }

    json_path = os.path.join(task_output_dir, "hessian_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nAll metrics saved to: {json_path}")

    # Print final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nModel: {model_config_str} ({model_type})")
    print(f"Task: {args.task}\n")

    for dataset_name, results in all_results.items():
        print(f"{dataset_name}:")
        print(f"  Top Eigenvalue: {results['top_eigenvalue']:.6f}")
        print(f"  Trace:          {results['trace']:.6f}")
        print(f"  ESD Plot:       esd_{dataset_name}.pdf")
        print()

    print(f"All results saved to: {task_output_dir}")
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
