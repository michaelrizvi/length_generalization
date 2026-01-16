#!/usr/bin/env python3
"""
Plot Hessian eigenvalue analysis for a single model checkpoint.

This script takes a model checkpoint path and produces plots of the top k eigenvalues
and eigenvalue spectral density (ESD) for train, same-length test, and generalization
test distributions.

Usage:
    python plot_hessian_simple.py \
        --model_path ./lm-out-new-unique_copy/2l1h64d_n10000_len1-10_run0 \
        --task unique_copy \
        --output_dir ./hessian_results/simple \
        --batch_size 200 \
        --max_samples 500 \
        --top_k 100

The script extracts train length range from the model folder name pattern.
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
from pathlib import Path

from utils import (
    load_model_and_config,
    create_hessian_dataloaders,
    compute_hessian_metrics,
)


def extract_train_length_range(model_path):
    """
    Extract train length range from model checkpoint directory name.

    Expected pattern: len{min}-{max} or similar
    E.g., "2l1h64d_n10000_len1-10_run0" -> (1, 10)

    Args:
        model_path: Path to model checkpoint (or parent directory)

    Returns:
        tuple: (min_length, max_length)

    Raises:
        ValueError: If pattern not found
    """
    # Get the last meaningful directory name (could be checkpoint dir or parent)
    path_parts = model_path.rstrip('/').split('/')
    
    # Search through path parts for the pattern
    for part in reversed(path_parts):
        match = re.search(r'len(\d+)-(\d+)', part)
        if match:
            min_len = int(match.group(1))
            max_len = int(match.group(2))
            return (min_len, max_len)
    
    raise ValueError(
        f"Could not extract train length range from model path: {model_path}\n"
        f"Expected pattern like 'len1-10' in one of the directory names.\n"
        f"Path parts: {path_parts}"
    )


def plot_top_k_eigenvalues(eigenvalues, output_path, title="", log_scale=False):
    """
    Plot sorted top-k eigenvalues.

    Args:
        eigenvalues: Array of eigenvalues (sorted descending)
        output_path: Path to save PDF
        title: Title for the plot
        log_scale: If True, use log scale for y-axis (default: False)
    """
    eigenvalues = np.array(eigenvalues)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(1, len(eigenvalues) + 1)
    ax.plot(x, eigenvalues, linewidth=2, color='#2E86AB', marker='o', markersize=3, alpha=0.7)
    ax.set_xlabel('Eigenvalue Rank', fontsize=14)
    ax.set_ylabel('Eigenvalue', fontsize=14)
    if log_scale:
        ax.set_yscale('log')
    ax.set_title(f'Top {len(eigenvalues)} Eigenvalues\n{title}', fontsize=16)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add annotations for key statistics
    textstr = '\n'.join((
        f'λ₁ (max): {eigenvalues[0]:.4f}',
        f'λ₁₀: {eigenvalues[min(9, len(eigenvalues)-1)]:.4f}',
        f'λ₁₀₀: {eigenvalues[min(99, len(eigenvalues)-1)]:.4f}' if len(eigenvalues) >= 100 else '',
    ))
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Top eigenvalue plot saved to: {output_path}")


def export_eigenvalues_to_csv(all_results, output_path):
    """
    Export top eigenvalues from all datasets to CSV.

    Creates a DataFrame with eigenvalue rank and values from each dataset.

    Args:
        all_results: Dict mapping dataset names to results dicts
        output_path: Path to save CSV
    """
    # Get max number of eigenvalues
    max_eigenvalues = max(len(results['top_eigenvalues']) for results in all_results.values())
    
    # Initialize data dictionary
    data = {
        'rank': list(range(1, max_eigenvalues + 1))
    }
    
    # Add eigenvalues from each dataset
    for dataset_name in sorted(all_results.keys()):
        eigenvalues = all_results[dataset_name]['top_eigenvalues']
        # Pad with NaN if this dataset has fewer eigenvalues
        padded_eigenvalues = list(eigenvalues) + [np.nan] * (max_eigenvalues - len(eigenvalues))
        data[dataset_name] = padded_eigenvalues
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"    Eigenvalues CSV saved to: {output_path}")


def export_esd_to_csv(all_results, output_path):
    """
    Export eigenvalue spectral density from all datasets to CSV.

    Creates a DataFrame with eigenvalues and their corresponding density weights
    from each dataset.

    Args:
        all_results: Dict mapping dataset names to results dicts
        output_path: Path to save CSV
    """
    # Collect all eigenvalues across datasets to create a unified eigenvalue axis
    all_eigenvalues = set()
    processed_data = {}
    
    for dataset_name in all_results.keys():
        density_eigen = all_results[dataset_name]['density_eigen']
        density_weight = all_results[dataset_name]['density_weight']
        
        # Process the density data (handle list of lists or flat arrays)
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
            eigen_vals_sorted = np.array([x[0] for x in sorted_pairs])
            weights_sorted = np.array([x[1] for x in sorted_pairs])
        
        processed_data[dataset_name] = {
            'eigenvalues': eigen_vals_sorted,
            'weights': weights_sorted
        }
        all_eigenvalues.update(eigen_vals_sorted)
    
    # Create unified eigenvalue axis (sorted)
    unified_eigenvalues = sorted(list(all_eigenvalues))
    
    # Create DataFrame with interpolation for alignment
    data = {'eigenvalue': unified_eigenvalues}
    
    for dataset_name in sorted(processed_data.keys()):
        # Create a mapping from eigenvalue to weight for this dataset
        eigen_weight_dict = dict(zip(processed_data[dataset_name]['eigenvalues'], 
                                     processed_data[dataset_name]['weights']))
        # Map to unified eigenvalue axis, using NaN for missing values
        weights = [eigen_weight_dict.get(ev, np.nan) for ev in unified_eigenvalues]
        data[f'density_weight_{dataset_name}'] = weights
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"    ESD CSV saved to: {output_path}")


def plot_esd(density_eigen, density_weight, output_path, title=""):
    """
    Plot and save eigenvalue spectral density as PDF.

    Averages across multiple n_v runs for statistical validity.

    Args:
        density_eigen: List of lists of eigenvalue estimates (one per n_v run)
        density_weight: List of lists of density weights (one per n_v run)
        output_path: Path to save PDF
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))

    # PyHessian's density() returns list of lists (one per n_v run)
    # Average across runs instead of plotting separately
    if isinstance(density_eigen, list) and len(density_eigen) > 0 and isinstance(density_eigen[0], list):
        # Multiple runs - compute mean
        all_eigen_vals = []
        all_weights = []
        
        for eigen_vals, weights in zip(density_eigen, density_weight):
            # Sort by eigenvalue magnitude for better visualization
            sorted_pairs = sorted(zip(eigen_vals, weights), key=lambda x: x[0])
            all_eigen_vals.append([x[0] for x in sorted_pairs])
            all_weights.append([x[1] for x in sorted_pairs])
        
        # Convert to numpy arrays for averaging
        # Need to handle potentially different lengths, so we'll compute mean at each eigenvalue
        all_eigen_vals_np = np.array(all_eigen_vals)
        all_weights_np = np.array(all_weights)
        
        # Compute mean eigenvalues and weights across runs
        mean_eigen = np.mean(all_eigen_vals_np, axis=0)
        mean_weight = np.mean(all_weights_np, axis=0)
        
        plt.semilogy(mean_eigen, mean_weight, linewidth=2.5, color='#2E86AB', label='Mean (n_v runs)')
        plt.legend()
        
        # Add annotation showing number of runs
        textstr = f'Averaged over {len(density_eigen)} runs'
        plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Single flat array - sort as well
        sorted_pairs = sorted(zip(density_eigen, density_weight), key=lambda x: x[0])
        eigen_vals_sorted = [x[0] for x in sorted_pairs]
        weights_sorted = [x[1] for x in sorted_pairs]
        plt.semilogy(eigen_vals_sorted, weights_sorted, linewidth=2.5, color='#2E86AB')

    plt.xlabel('Eigenvalue', fontsize=14)
    plt.ylabel('Density (log scale)', fontsize=14)
    plt.title(f'Hessian Eigenvalue Spectral Density\n{title}', fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ESD plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Hessian analysis for a single model checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - extracts train length from model path
  python plot_hessian_simple.py \\
      --model_path ./lm-out-new-unique_copy/2l1h64d_n10000_len1-10_run0 \\
      --task unique_copy

  # With custom output directory and parameters
  python plot_hessian_simple.py \\
      --model_path ./lm-out-new-unique_copy/2l1h64d_n10000_len1-10_run0 \\
      --task unique_copy \\
      --output_dir ./hessian_results/simple \\
      --batch_size 100 \\
      --max_samples 300 \\
      --top_k 100
        """
    )

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint directory")
    parser.add_argument("--task", type=str, required=True,
                       choices=["bin_majority", "majority", "bin_majority_interleave",
                               "unique_copy", "repeat_copy", "sort", "parity", "addition"],
                       help="Task name")
    parser.add_argument("--output_dir", type=str, default="./hessian_results/simple",
                       help="Base output directory (default: ./hessian_results/simple)")
    parser.add_argument("--batch_size", type=int, default=200,
                       help="Batch size for Hessian computation (default: 200)")
    parser.add_argument("--max_samples", type=int, default=500,
                       help="Maximum samples per dataset (default: 500)")
    parser.add_argument("--train_seed", type=int, default=0,
                       help="Random seed for training data generation (default: 0)")
    parser.add_argument("--test_seed", type=int, default=1,
                       help="Random seed for test data generation (default: 1)")
    parser.add_argument("--hessian_seed", type=int, default=42,
                       help="Random seed for Hessian computation (default: 42)")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Number of top eigenvalues to compute (default: 100)")

    args = parser.parse_args()

    print("="*80)
    print("Hessian Plot for Single Model")
    print("="*80)

    # Extract train length range from model path
    print(f"\nExtracting train length range from model path...")
    train_range = extract_train_length_range(args.model_path)
    print(f"  Train range: {train_range}")

    # Create test ranges: same-length test and generalization test
    # Assuming generalization test is twice the training length
    test_range_same = train_range
    test_range_gen = (train_range[1] + 1, train_range[1] * 2)
    test_ranges = [test_range_same, test_range_gen]

    print(f"\nConfiguration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Task: {args.task}")
    print(f"  Train range: {train_range}")
    print(f"  Test ranges: {test_ranges}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max samples per dataset: {args.max_samples}")
    print(f"  Top eigenvalues to compute: {args.top_k}")

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

    # Create output directory structure
    # output_dir/task/model_path (relative or basename)
    model_basename = os.path.basename(args.model_path.rstrip('/'))
    task_output_dir = os.path.join(args.output_dir, args.task, model_basename)
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
            top_k=args.top_k
        )
        all_results[dataset_name] = results

        # Plot ESD
        output_path = os.path.join(task_output_dir, f"esd_{dataset_name}.pdf")
        plot_esd(
            results['density_eigen'],
            results['density_weight'],
            output_path,
            title=f"{dataset_name} ({model_type})"
        )

        # Plot top eigenvalues
        output_path = os.path.join(task_output_dir, f"top_eigenvalues_{dataset_name}.pdf")
        plot_top_k_eigenvalues(
            results['top_eigenvalues'],
            output_path,
            title=f"{dataset_name} ({model_type})"
        )

        # Print summary
        print(f"\n  Results:")
        print(f"    Top Eigenvalue: {results['top_eigenvalues'][0]:.6f}")
        print(f"    ESD: Computed successfully")
        print(f"    Top Eigenvalues: {len(results['top_eigenvalues'])} computed")

    # Print final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nModel: {model_basename} ({model_type})")
    print(f"Task: {args.task}\n")

    for dataset_name, results in all_results.items():
        print(f"{dataset_name}:")
        print(f"  Top Eigenvalue: {results['top_eigenvalues'][0]:.6f}")
        print(f"  Num Eigenvalues: {len(results['top_eigenvalues'])}")
        print(f"  ESD Plot:       esd_{dataset_name}.pdf")
        print(f"  Eigenvalue Plot: top_eigenvalues_{dataset_name}.pdf")
        print()

    print(f"All results saved to: {task_output_dir}")
    
    # Export data to CSV
    print("\n" + "="*80)
    print("Exporting Data to CSV")
    print("="*80)
    
    eigenvalues_csv_path = os.path.join(task_output_dir, f"{model_basename}_eigenvalues.csv")
    export_eigenvalues_to_csv(all_results, eigenvalues_csv_path)
    
    esd_csv_path = os.path.join(task_output_dir, f"{model_basename}_esd.csv")
    export_esd_to_csv(all_results, esd_csv_path)
    
    print(f"\nCSV files saved to: {task_output_dir}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
