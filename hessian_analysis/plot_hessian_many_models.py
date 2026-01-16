#!/usr/bin/env python3
"""
Plot Hessian eigenvalue analysis for multiple model checkpoints on the training set only.

This script takes a list of model checkpoint paths and produces overlay plots comparing
the top k eigenvalues and eigenvalue spectral density (ESD) across models on their
respective training distributions.

The training length range is automatically extracted from each model folder name.
Results are plotted on single comparison figures with different colors for each model.

Usage:
    python plot_hessian_many_models.py \
        --model_paths ./lm-out-new-unique_copy/2l1h64d_n10000_len1-10_run0 \
                      ./lm-out-new-unique_copy/2l1h64d_n10000_len1-50_run0 \
        --task unique_copy \
        --output_dir ./hessian_results/many_models \
        --batch_size 200 \
        --max_samples 500 \
        --top_k 100

Output:
  - esd_comparison.pdf: ESD curves for all models overlaid on one plot
  - top_eigenvalues_comparison.pdf: Top eigenvalues for all models overlaid on one plot
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import gc
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


def extract_model_config(model_path):
    """
    Extract model configuration string from path.

    Expected pattern: {layers}l{heads}h{d_model}d or similar
    E.g., "2l1h64d_n10000_len1-10_run0" -> "2l1h64d"

    Args:
        model_path: Path to model checkpoint (or parent directory)

    Returns:
        str: Model config string

    Raises:
        ValueError: If pattern not found
    """
    path_parts = model_path.rstrip('/').split('/')
    
    # Search through path parts for the pattern
    for part in reversed(path_parts):
        match = re.search(r'(\d+l\d+h\d+d)', part)
        if match:
            return match.group(1)
    
    raise ValueError(
        f"Could not extract model config from model path: {model_path}\n"
        f"Expected pattern like '2l1h64d' in one of the directory names.\n"
        f"Path parts: {path_parts}"
    )


def plot_top_k_eigenvalues_overlay(all_results, output_path, title=""):
    """
    Plot sorted top-k eigenvalues for multiple models on same figure.

    Args:
        all_results: Dict mapping model names to results dicts
        output_path: Path to save PDF
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color palette for different models
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_results)))

    for (model_name, data), color in zip(sorted(all_results.items(), key=lambda x: x[1]['train_range']), colors):
        train_range = data['train_range']
        eigenvalues = np.array(data['results']['top_eigenvalues'])

        x = np.arange(1, len(eigenvalues) + 1)
        label = f"len {train_range[0]}-{train_range[1]}"
        ax.plot(x, eigenvalues, linewidth=2.5, marker='o', markersize=4, 
                alpha=0.8, label=label, color=color)

    ax.set_xlabel('Eigenvalue Rank', fontsize=14)
    ax.set_ylabel('Eigenvalue', fontsize=14)
    ax.set_title(f'Top Eigenvalues Comparison (Train Set)\n{title}', fontsize=16)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Top eigenvalue overlay plot saved to: {output_path}")


def plot_esd_overlay(all_results, output_path, title=""):
    """
    Plot eigenvalue spectral density for multiple models on same figure.

    Averages across multiple n_v runs for statistical validity.

    Args:
        all_results: Dict mapping model names to results dicts
        output_path: Path to save PDF
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color palette for different models
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_results)))

    for (model_name, data), color in zip(sorted(all_results.items(), key=lambda x: x[1]['train_range']), colors):
        train_range = data['train_range']
        density_eigen = data['results']['density_eigen']
        density_weight = data['results']['density_weight']

        # PyHessian's density() returns list of lists (one per n_v run)
        # Average across runs for better statistical validity
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

        label = f"len {train_range[0]}-{train_range[1]}"
        ax.semilogy(eigen_vals_sorted, weights_sorted, linewidth=2.5, alpha=0.8,
                   label=label, color=color)

    ax.set_xlabel('Eigenvalue', fontsize=14)
    ax.set_ylabel('Density (log scale)', fontsize=14)
    ax.set_title(f'Hessian Eigenvalue Spectral Density Comparison (Train Set)\n{title}', fontsize=16)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ESD overlay plot saved to: {output_path}")


def export_comparison_eigenvalues_to_csv(all_model_results, output_path):
    """
    Export top eigenvalues from all models to CSV for comparison.

    Creates a DataFrame with eigenvalue rank and values from each model.

    Args:
        all_model_results: Dict mapping model paths to results dicts
        output_path: Path to save CSV
    """
    # Get max number of eigenvalues
    max_eigenvalues = max(len(data['results']['top_eigenvalues']) 
                         for data in all_model_results.values())
    
    # Initialize data dictionary with rank
    data = {
        'rank': list(range(1, max_eigenvalues + 1))
    }
    
    # Add eigenvalues from each model, sorted by train range
    sorted_models = sorted(all_model_results.items(), key=lambda x: x[1]['train_range'])
    for model_path, model_data in sorted_models:
        train_range = model_data['train_range']
        eigenvalues = model_data['results']['top_eigenvalues']
        
        # Create column name with train range
        col_name = f"len_{train_range[0]}-{train_range[1]}"
        
        # Pad with NaN if this model has fewer eigenvalues
        padded_eigenvalues = list(eigenvalues) + [np.nan] * (max_eigenvalues - len(eigenvalues))
        data[col_name] = padded_eigenvalues
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"  Comparison eigenvalues CSV saved to: {output_path}")


def export_comparison_esd_to_csv(all_model_results, output_path):
    """
    Export eigenvalue spectral density from all models to CSV for comparison.

    Creates a DataFrame with eigenvalues and their corresponding density weights
    from each model.

    Args:
        all_model_results: Dict mapping model paths to results dicts
        output_path: Path to save CSV
    """
    # Collect all eigenvalues across all models to create a unified eigenvalue axis
    all_eigenvalues = set()
    processed_data = {}
    
    sorted_models = sorted(all_model_results.items(), key=lambda x: x[1]['train_range'])
    
    for model_path, model_data in sorted_models:
        train_range = model_data['train_range']
        density_eigen = model_data['results']['density_eigen']
        density_weight = model_data['results']['density_weight']
        
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
        
        col_name = f"len_{train_range[0]}-{train_range[1]}"
        processed_data[col_name] = {
            'eigenvalues': eigen_vals_sorted,
            'weights': weights_sorted
        }
        all_eigenvalues.update(eigen_vals_sorted)
    
    # Create unified eigenvalue axis (sorted)
    unified_eigenvalues = sorted(list(all_eigenvalues))
    
    # Create DataFrame with interpolation for alignment
    data = {'eigenvalue': unified_eigenvalues}
    
    for col_name in sorted(processed_data.keys()):
        # Create a mapping from eigenvalue to weight for this model
        eigen_weight_dict = dict(zip(processed_data[col_name]['eigenvalues'], 
                                     processed_data[col_name]['weights']))
        # Map to unified eigenvalue axis, using NaN for missing values
        weights = [eigen_weight_dict.get(ev, np.nan) for ev in unified_eigenvalues]
        data[f'density_weight_{col_name}'] = weights
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"  Comparison ESD CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Hessian analysis for multiple model checkpoints on training set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare models with different training lengths
  python plot_hessian_many_models.py \\
      --model_paths ./lm-out-new-unique_copy/2l1h64d_n10000_len1-10_run0 \\
                    ./lm-out-new-unique_copy/2l1h64d_n10000_len1-50_run0 \\
                    ./lm-out-new-unique_copy/2l1h64d_n10000_len1-100_run0 \\
      --task unique_copy

  # With custom output directory
  python plot_hessian_many_models.py \\
      --model_paths ./lm-out-new-unique_copy/2l1h64d_n10000_len1-10_run0 \\
                    ./lm-out-new-unique_copy/2l1h64d_n10000_len1-50_run0 \\
      --task unique_copy \\
      --output_dir ./hessian_results/many_models \\
      --batch_size 100 \\
      --max_samples 300
        """
    )

    parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                       help="Paths to model checkpoints (space-separated)")
    parser.add_argument("--task", type=str, required=True,
                       choices=["bin_majority", "majority", "bin_majority_interleave",
                               "unique_copy", "repeat_copy", "sort", "parity", "addition"],
                       help="Task name (same for all models)")
    parser.add_argument("--output_dir", type=str, default="./hessian_results/many_models",
                       help="Output directory for all results (default: ./hessian_results/many_models)")
    parser.add_argument("--batch_size", type=int, default=400,
                       help="Batch size for Hessian computation (default: 200)")
    parser.add_argument("--max_samples", type=int, default=500,
                       help="Maximum samples per dataset (default: 500)")
    parser.add_argument("--train_seed", type=int, default=0,
                       help="Random seed for training data generation (default: 0)")
    parser.add_argument("--test_seed", type=int, default=1,
                       help="Random seed for test data generation (default: 1)")
    parser.add_argument("--hessian_seed", type=int, default=42,
                       help="Random seed for Hessian computation (default: 42)")
    parser.add_argument("--top_k", type=int, default=15,
                       help="Number of top eigenvalues to compute (default: 100)")

    args = parser.parse_args()

    print("="*80)
    print("Hessian Plot for Multiple Models (Train Set Only)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Number of models to analyze: {len(args.model_paths)}")
    print(f"Task: {args.task}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory with task subdirectory
    task_output_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(task_output_dir, exist_ok=True)
    print(f"Results will be saved to: {task_output_dir}")

    # Collect results for all models
    all_model_results = {}
    model_configs = {}

    for idx, model_path in enumerate(args.model_paths):
        print(f"\n{'='*80}")
        print(f"Model {idx + 1}/{len(args.model_paths)}")
        print(f"{'='*80}")

        # Extract train length range from model path
        print(f"\nProcessing: {model_path}")
        try:
            train_range = extract_train_length_range(model_path)
            model_config = extract_model_config(model_path)
            print(f"  Model config: {model_config}")
            print(f"  Train length range: {train_range}")
        except ValueError as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping this model...")
            continue

        model_configs[model_path] = {
            'config': model_config,
            'train_range': train_range
        }

        # Load model
        print(f"\n  Step 1: Loading Model")
        try:
            model, config, model_type = load_model_and_config(model_path, device)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            print(f"  Skipping this model...")
            continue

        # Create dataloaders for train set only
        print(f"\n  Step 2: Creating Train Dataset")
        try:
            # For train-only analysis, we only need the train range (no test ranges)
            dataloaders, tokenizer = create_hessian_dataloaders(
                args.task, train_range, [train_range],  # Only train range for test_ranges
                args.batch_size, args.max_samples,
                args.train_seed, args.test_seed
            )
        except Exception as e:
            print(f"  ERROR creating dataloaders: {e}")
            print(f"  Skipping this model...")
            continue

        # Compute Hessian for train dataset only
        print(f"\n  Step 3: Computing Hessian Metrics (Train Set)")
        try:
            # Get the train dataloader (it should be the only one or the first one)
            train_dataloader = list(dataloaders.values())[0]

            results = compute_hessian_metrics(
                model, train_dataloader, device, verbose=True, seed=args.hessian_seed,
                top_k=args.top_k
            )

            all_model_results[model_path] = {
                'config': model_config,
                'train_range': train_range,
                'model_type': model_type,
                'results': results
            }

            # Print summary for this model
            print(f"\n  Results:")
            print(f"    Top Eigenvalue: {results['top_eigenvalues'][0]:.6f}")
            print(f"    Num Eigenvalues: {len(results['top_eigenvalues'])}")

        except Exception as e:
            print(f"  ERROR computing Hessian: {e}")
            print(f"  Skipping this model...")
            continue

        # Cleanup: Delete model, dataloaders, and tokenizer to free GPU/CPU memory
        print(f"\n  Cleaning up memory...")
        del model
        del dataloaders
        del tokenizer
        torch.cuda.empty_cache()  # Force GPU memory cleanup
        gc.collect()  # Force Python garbage collection
        print(f"  Memory cleared for next model")

    # Generate overlay plots
    print(f"\n{'='*80}")
    print("Generating Overlay Plots")
    print(f"{'='*80}")

    if all_model_results:
        # Extract model config from first model
        first_config = list(all_model_results.values())[0]['config']
        
        # Plot ESD overlay
        esd_path = os.path.join(task_output_dir, f"esd_comparison.pdf")
        print(f"\nGenerating ESD comparison plot...")
        plot_esd_overlay(
            all_model_results,
            esd_path,
            title=f"{first_config} - Multiple Training Lengths"
        )

        # Plot top eigenvalues overlay
        eigenval_path = os.path.join(task_output_dir, f"top_eigenvalues_comparison.pdf")
        print(f"Generating top eigenvalues comparison plot...")
        plot_top_k_eigenvalues_overlay(
            all_model_results,
            eigenval_path,
            title=f"{first_config} - Multiple Training Lengths"
        )

    # Export data to CSV
    print(f"\n{'='*80}")
    print("Exporting Comparison Data to CSV")
    print(f"{'='*80}\n")
    
    if all_model_results:
        eigenvalues_csv_path = os.path.join(task_output_dir, "comparison_eigenvalues.csv")
        export_comparison_eigenvalues_to_csv(all_model_results, eigenvalues_csv_path)
        
        esd_csv_path = os.path.join(task_output_dir, "comparison_esd.csv")
        export_comparison_esd_to_csv(all_model_results, esd_csv_path)

    # Print final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nTask: {args.task}")
    print(f"Successfully analyzed {len(all_model_results)} models\n")

    for model_path, data in sorted(all_model_results.items(), key=lambda x: x[1]['train_range']):
        config = data['config']
        train_range = data['train_range']
        results = data['results']

        print(f"len {train_range[0]}-{train_range[1]}:")
        print(f"  Model path: {model_path}")
        print(f"  Top Eigenvalue: {results['top_eigenvalues'][0]:.6f}")
        print(f"  Num Eigenvalues: {len(results['top_eigenvalues'])}")
        print()

    print(f"Output plots:")
    print(f"  - esd_comparison.pdf")
    print(f"  - top_eigenvalues_comparison.pdf")
    print(f"\nOutput CSV files:")
    print(f"  - comparison_eigenvalues.csv")
    print(f"  - comparison_esd.csv")
    print(f"\nAll results saved to: {task_output_dir}")
    print(f"Total models analyzed: {len(all_model_results)}/{len(args.model_paths)}")
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
