#!/usr/bin/env python3
"""
Plot Hessian eigenvalue analysis for a single model across multiple dataset ranges.

This script takes a single trained model and evaluates its Hessian on datasets
of increasing length. Produces overlay plots showing how the model's spectrum
changes as the evaluation dataset grows.

Usage:
    python plot_hessian_many_datasets.py \
        --model_path ./lm-out-new-unique_copy/2l1h64d_n10000_len1-10_run0 \
        --task unique_copy \
        --dataset_ranges 1-10 1-20 1-50 1-100 \
        --output_dir ./hessian_results/many_datasets

Output:
  - hessian_results/many_datasets/{model_name}/esd_comparison.pdf
  - hessian_results/many_datasets/{model_name}/top_eigenvalues_comparison.pdf
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import gc
from pathlib import Path

from utils import (
    load_model_and_config,
    create_hessian_dataloaders,
    compute_hessian_metrics,
)


def parse_range(range_str):
    """
    Parse a range string like '1-10' into (1, 10).

    Args:
        range_str: String in format 'min-max'

    Returns:
        tuple: (min, max)

    Raises:
        ValueError: If format is invalid
    """
    parts = range_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid range format: {range_str}. Expected format: 'min-max'")
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        raise ValueError(f"Invalid range format: {range_str}. Expected format: 'min-max'")


def extract_train_length_range(model_path):
    """
    Extract training length range from model path.

    Expected pattern: len{min}-{max}_run{N}
    E.g., "2l1h64d_n10000_len1-10_run0" -> (1, 10)

    Args:
        model_path: Path to model checkpoint

    Returns:
        tuple: (min_length, max_length)

    Raises:
        ValueError: If pattern not found
    """
    path_parts = str(model_path).rstrip('/').split('/')
    
    for part in reversed(path_parts):
        match = re.search(r'len(\d+)-(\d+)', part)
        if match:
            min_len = int(match.group(1))
            max_len = int(match.group(2))
            return (min_len, max_len)
    
    raise ValueError(
        f"Could not extract train length range from model path: {model_path}\n"
        f"Expected pattern like 'len1-10' in one of the directory names."
    )


def plot_esd_overlay(all_results, dataset_ranges, output_path, model_name):
    """
    Plot eigenvalue spectral density curves overlaid for different dataset ranges.

    Args:
        all_results: Dict mapping dataset_range -> ESD data (density_eigen, density_weight)
        dataset_ranges: List of (min, max) tuples for labels
        output_path: Path to save PDF
        model_name: Name of model for title
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color palette for different dataset ranges
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_ranges)))

    for idx, dataset_range in enumerate(sorted(dataset_ranges)):
        if dataset_range not in all_results or all_results[dataset_range] is None:
            continue

        density_eigen, density_weight = all_results[dataset_range]

        # Average across runs if multiple
        if isinstance(density_eigen, list) and len(density_eigen) > 0 and isinstance(density_eigen[0], list):
            # Multiple runs - compute mean
            all_eigen_vals = []
            all_weights = []
            
            for eigen_vals, weights in zip(density_eigen, density_weight):
                sorted_pairs = sorted(zip(eigen_vals, weights), key=lambda x: x[0])
                all_eigen_vals.append([x[0] for x in sorted_pairs])
                all_weights.append([x[1] for x in sorted_pairs])
            
            # Convert to numpy arrays for averaging
            all_eigen_vals_np = np.array(all_eigen_vals)
            all_weights_np = np.array(all_weights)
            
            # Compute mean eigenvalues and weights across runs
            mean_eigen = np.mean(all_eigen_vals_np, axis=0)
            mean_weight = np.mean(all_weights_np, axis=0)
            
            eigen_vals_sorted = mean_eigen
            weights_sorted = mean_weight
        else:
            # Single flat array - sort as well
            sorted_pairs = sorted(zip(density_eigen, density_weight), key=lambda x: x[0])
            eigen_vals_sorted = [x[0] for x in sorted_pairs]
            weights_sorted = [x[1] for x in sorted_pairs]

        label = f"len {dataset_range[0]}-{dataset_range[1]}"
        ax.semilogy(eigen_vals_sorted, weights_sorted, linewidth=2.5, alpha=0.8,
                   label=label, color=colors[idx])

    ax.set_xlabel('Eigenvalue', fontsize=14)
    ax.set_ylabel('Density (log scale)', fontsize=14)
    ax.set_title(f'Hessian Eigenvalue Spectral Density\n{model_name} across Dataset Ranges', fontsize=16)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ESD overlay plot saved to: {output_path}")


def plot_top_eigenvalues_overlay(all_results, dataset_ranges, output_path, model_name, top_k=100):
    """
    Plot top eigenvalues overlaid for different dataset ranges.

    Args:
        all_results: Dict mapping dataset_range -> top_eigenvalues
        dataset_ranges: List of (min, max) tuples for labels
        output_path: Path to save PDF
        model_name: Name of model for title
        top_k: Number of top eigenvalues to plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color palette for different dataset ranges
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_ranges)))

    for idx, dataset_range in enumerate(sorted(dataset_ranges)):
        if dataset_range not in all_results or all_results[dataset_range] is None:
            continue

        eigenvalues = all_results[dataset_range]

        # Average across runs if multiple
        if isinstance(eigenvalues, list) and len(eigenvalues) > 0 and isinstance(eigenvalues[0], list):
            # Multiple runs - compute mean
            eigenvalues_np = np.array(eigenvalues)
            eigenvalues_mean = np.mean(eigenvalues_np, axis=0)
        else:
            eigenvalues_mean = np.array(eigenvalues)

        # Take top k
        top_eigs = eigenvalues_mean[:min(top_k, len(eigenvalues_mean))]

        label = f"len {dataset_range[0]}-{dataset_range[1]}"
        x = np.arange(1, len(top_eigs) + 1)
        ax.plot(x, top_eigs, linewidth=2.5, marker='o', markersize=4, alpha=0.8,
               label=label, color=colors[idx])

    ax.set_xlabel('Eigenvalue Rank', fontsize=14)
    ax.set_ylabel('Eigenvalue', fontsize=14)
    ax.set_title(f'Top {top_k} Eigenvalues\n{model_name} across Dataset Ranges', fontsize=16)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Top eigenvalue overlay plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Hessian analysis for a single model across multiple dataset ranges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default ranges
  python plot_hessian_many_datasets.py \\
      --model_path ./lm-out-new-unique_copy/2l1h64d_n10000_len1-10_run0 \\
      --task unique_copy

  # Custom dataset ranges
  python plot_hessian_many_datasets.py \\
      --model_path ./lm-out-new-unique_copy/2l1h64d_n10000_len1-10_run0 \\
      --task unique_copy \\
      --dataset_ranges 1-10 1-20 1-30 1-50 1-100 \\
      --batch_size 200 \\
      --max_samples 500
        """
    )

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint directory")
    parser.add_argument("--task", type=str, required=True,
                       choices=["bin_majority", "majority", "bin_majority_interleave",
                               "unique_copy", "repeat_copy", "sort", "parity", "addition"],
                       help="Task name")
    parser.add_argument("--dataset_ranges", type=str, nargs='+',
                       default=['1-10', '1-20', '1-50', '1-100'],
                       help="Dataset length ranges to analyze (e.g., 1-10 1-20 1-50 1-100)")
    parser.add_argument("--output_dir", type=str, default="./hessian_results/many_datasets",
                       help="Output directory for results (default: ./hessian_results/many_datasets)")
    parser.add_argument("--batch_size", type=int, default=400,
                       help="Batch size for Hessian computation (default: 200)")
    parser.add_argument("--max_samples", type=int, default=500,
                       help="Maximum samples per dataset (default: 500)")
    parser.add_argument("--hessian_seed", type=int, default=42,
                       help="Random seed for Hessian computation (default: 42)")
    parser.add_argument("--data_seed", type=int, default=0,
                       help="Random seed for data generation (default: 0)")
    parser.add_argument("--top_k", type=int, default=15,
                       help="Number of top eigenvalues to plot (default: 100)")

    args = parser.parse_args()

    # Validate inputs
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Parse dataset ranges
    try:
        dataset_ranges = [parse_range(r) for r in args.dataset_ranges]
    except ValueError as e:
        raise ValueError(f"Error parsing dataset ranges: {e}")

    # Extract model training length
    try:
        train_length_range = extract_train_length_range(model_path)
    except ValueError as e:
        print(f"Warning: {e}")
        train_length_range = None

    print("="*80)
    print("Hessian Analysis: Single Model, Multiple Dataset Ranges")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model path: {args.model_path}")
    if train_length_range:
        print(f"  Model trained on lengths: {train_length_range[0]}-{train_length_range[1]}")
    print(f"  Task: {args.task}")
    print(f"  Dataset ranges: {[f'{r[0]}-{r[1]}' for r in dataset_ranges]}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max samples per dataset: {args.max_samples}")
    print(f"  Top k eigenvalues: {args.top_k}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Load model once
    print("\n" + "="*80)
    print("Step 1: Loading Model")
    print("="*80)
    try:
        model, config, model_type = load_model_and_config(args.model_path, device)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        raise

    # Create output directory
    model_name = model_path.name
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Store results for each dataset range
    all_esd_results = {}
    all_eigenvalue_results = {}

    # Analyze each dataset range
    print("\n" + "="*80)
    print("Step 2: Computing Hessian Metrics for Each Dataset Range")
    print("="*80)

    for range_idx, (min_len, max_len) in enumerate(dataset_ranges):
        print(f"\n{'='*60}")
        print(f"Dataset {range_idx + 1}/{len(dataset_ranges)}: len {min_len}-{max_len}")
        print(f"{'='*60}")

        try:
            # Create dataloaders for this dataset range (independent from model's test ranges)
            print(f"  Creating dataloaders...")
            dataloaders, tokenizer = create_hessian_dataloaders(
                args.task, 
                train_range=(min_len, max_len),
                test_ranges=[(min_len, max_len)],  # Only evaluate on this range
                hessian_batch_size=args.batch_size,
                max_samples=args.max_samples,
                train_seed=args.data_seed,
                test_seed=args.data_seed + 1
            )

            # Get the first dataloader (should be the train set)
            train_dataloader = list(dataloaders.values())[0]

            print(f"  Computing Hessian metrics...")
            results = compute_hessian_metrics(
                model, train_dataloader, device, verbose=True, seed=args.hessian_seed,
                top_k=args.top_k
            )

            # Store results
            dataset_range = (min_len, max_len)
            all_esd_results[dataset_range] = (results['density_eigen'], results['density_weight'])
            all_eigenvalue_results[dataset_range] = results['top_eigenvalues']

            # Print summary
            print(f"\n  Results:")
            print(f"    Top Eigenvalue: {results['top_eigenvalues'][0]:.6f}")
            print(f"    Num Eigenvalues: {len(results['top_eigenvalues'])}")

            # Cleanup: Delete dataloaders and tokenizer to free memory
            print(f"  Cleaning up memory...")
            del dataloaders
            del tokenizer
            try:
                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"  Warning: CUDA memory cleanup raised error (safe to ignore): {e}")
            gc.collect()
            print(f"  Memory cleared for next dataset")

        except Exception as e:
            print(f"  ERROR computing Hessian: {e}")
            dataset_range = (min_len, max_len)
            all_esd_results[dataset_range] = None
            all_eigenvalue_results[dataset_range] = None
            continue

    # Generate plots
    print("\n" + "="*80)
    print("Step 3: Generating Comparison Plots")
    print("="*80)

    # Plot ESD overlay
    esd_path = os.path.join(output_dir, "esd_comparison.pdf")
    print(f"\nGenerating ESD comparison plot...")
    plot_esd_overlay(
        all_esd_results,
        dataset_ranges,
        esd_path,
        model_name
    )

    # Plot top eigenvalues overlay
    eigenval_path = os.path.join(output_dir, "top_eigenvalues_comparison.pdf")
    print(f"Generating top eigenvalues comparison plot...")
    plot_top_eigenvalues_overlay(
        all_eigenvalue_results,
        dataset_ranges,
        eigenval_path,
        model_name,
        args.top_k
    )

    # Cleanup model
    print(f"\nCleaning up model from memory...")
    del model
    try:
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"Warning: CUDA memory cleanup raised error (safe to ignore): {e}")
    gc.collect()

    # Print final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nModel: {model_name} ({model_type})")
    print(f"Task: {args.task}\n")

    for dataset_range in sorted(dataset_ranges):
        if dataset_range in all_eigenvalue_results and all_eigenvalue_results[dataset_range] is not None:
            eigenvalues = all_eigenvalue_results[dataset_range]
            print(f"Dataset len {dataset_range[0]}-{dataset_range[1]}:")
            print(f"  Top Eigenvalue: {eigenvalues[0]:.6f}")
            print(f"  Num Eigenvalues: {len(eigenvalues)}")

    print(f"\nOutput plots:")
    print(f"  - esd_comparison.pdf")
    print(f"  - top_eigenvalues_comparison.pdf")
    print(f"\nAll results saved to: {output_dir}")
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
