#!/usr/bin/env python3
"""
Script to generate heatmaps from language modeling training results.

This script parses summary.txt files from the language modeling experiments
and generates three heatmaps showing:
1. Training accuracy (same length as training data)
2. Test accuracy on same-length strings (statistical generalization)
3. Test accuracy on longer strings (length generalization)

Usage:
    python plot_heatmaps.py <path_to_summary.txt>
"""

import argparse
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_config(config_str):
    """
    Parse configuration string like '1l1h16d_n50000' or '1l1h16d'.

    Returns:
        dict: Configuration parameters (n_layers, n_heads, d_model, train_size)
    """
    # Pattern: {n_layers}l{n_heads}h{d_model}d_n{train_size}
    # Note: train_size (_n{train_size}) may not always be present in older formats
    pattern = r'(\d+)l(\d+)h(\d+)d(?:_n(\d+))?'
    match = re.match(pattern, config_str)

    if not match:
        return None

    config = {
        'n_layers': int(match.group(1)),
        'n_heads': int(match.group(2)),
        'd_model': int(match.group(3)),
        'train_size': int(match.group(4)) if match.group(4) else None
    }

    return config


def parse_metric(metric_str):
    """
    Parse metric string, handling both single-run and multi-run formats.

    Examples:
        'eval_len0-50_train_acc: 0.9500'  -> ('eval_len0-50_train_acc', 0.9500)
        'eval_len0-50_train_acc: 0.9500±0.0200'  -> ('eval_len0-50_train_acc', 0.9500)

    Returns:
        tuple: (metric_name, value) or None if parsing fails
    """
    # Pattern: metric_name: value or metric_name: mean±std
    pattern = r'(eval_[\w\-]+):\s*([\d\.]+)(?:±[\d\.]+)?'
    match = re.search(pattern, metric_str)

    if not match:
        return None

    return match.group(1), float(match.group(2))


def parse_learning_rate(lr_str):
    """
    Parse learning rate from string like 'lr: 0.001'.

    Returns:
        float: Learning rate value or None if parsing fails
    """
    pattern = r'lr:\s*([\d\.e\-]+)'
    match = re.search(pattern, lr_str)

    if not match:
        return None

    return float(match.group(1))


def detect_metric_format(result):
    """
    Detect which metric format is being used in the results.

    Returns:
        dict with keys 'train_metric', 'test_same_metric', 'test_longer_metric'
        or None if format cannot be detected
    """
    metrics = [k for k in result.keys() if k.startswith('eval_')]

    # Try to detect format patterns
    format_info = {}

    # Pattern 1: New format with _train_acc, _test_acc suffixes
    # e.g., eval_len0-50_train_acc, eval_len0-50_test_acc, eval_len51-100_acc
    train_metrics = [m for m in metrics if '_train_acc' in m]
    test_same_metrics = [m for m in metrics if '_test_acc' in m]

    if train_metrics and test_same_metrics:
        # New format detected
        format_info['train_metric'] = train_metrics[0]
        format_info['test_same_metric'] = test_same_metrics[0]

        # Find the longer range metric (any metric that's not train or test_acc)
        longer_metrics = [m for m in metrics if '_train_acc' not in m and '_test_acc' not in m]
        if longer_metrics:
            format_info['test_longer_metric'] = longer_metrics[0]
        else:
            return None

        format_info['format_type'] = 'new'
        format_info['has_train_size'] = result.get('train_size') is not None
        return format_info

    # Pattern 2: Old format without _train_acc/_test_acc suffixes
    # e.g., eval_len0-50_acc, eval_len51-100_acc, eval_len101-150_acc
    # The first metric is training accuracy, second is same-length test, third is longer
    if len(metrics) >= 2:
        # Sort metrics by the length range to ensure consistent ordering
        # Extract the length range and sort
        def extract_range(metric):
            match = re.search(r'len(\d+)-(\d+)', metric)
            if match:
                return int(match.group(1)), int(match.group(2))
            return (999999, 999999)  # Put unmatched at end

        sorted_metrics = sorted(metrics, key=extract_range)

        if len(sorted_metrics) >= 2:
            format_info['train_metric'] = sorted_metrics[0]  # First range is training
            format_info['test_same_metric'] = sorted_metrics[0]  # Same as train for old format
            format_info['test_longer_metric'] = sorted_metrics[1]  # Second range is longer
            format_info['format_type'] = 'old'
            format_info['has_train_size'] = result.get('train_size') is not None
            return format_info

    return None


def infer_train_size_from_config(config_str, all_results):
    """
    When train_size is not in the config string, try to infer a default or use a placeholder.
    For now, we'll use a counter to assign sequential train sizes.
    """
    # This is a fallback - assign a default train_size if missing
    return 50000  # Default value used in the training script


def parse_summary_file(filepath):
    """
    Parse summary.txt file and extract all experiment results.

    Args:
        filepath: Path to summary.txt file

    Returns:
        tuple: (list of results, metric_format_info dict)
            - results: List of dictionaries with experiment configuration and metrics
            - metric_format_info: Dict describing which metrics were found and their mapping
    """
    results = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split line by tabs or multiple spaces
            parts = re.split(r'\t+|\s{2,}', line)

            if len(parts) < 3:
                continue

            # Parse configuration (first part)
            config = parse_config(parts[0])
            if config is None:
                continue

            # Check if this is a multi-run mean line (second part contains 'mean')
            # For multi-run, we only want the 'mean' lines
            is_mean_line = len(parts) > 1 and 'mean' in parts[1].lower()
            is_single_run = len(parts) > 1 and 'mean' not in parts[1].lower() and 'best' not in parts[1].lower() and 'worst' not in parts[1].lower()

            # Skip best/worst lines for multi-run experiments
            if 'best' in parts[1].lower() or 'worst' in parts[1].lower():
                continue

            # Only process mean lines (multi-run) or regular lines (single-run)
            if not (is_mean_line or is_single_run):
                continue

            # Parse metrics from remaining parts
            metrics = {}
            lr = None

            for part in parts[1:]:
                # Try to parse as metric
                metric_result = parse_metric(part)
                if metric_result:
                    metric_name, metric_value = metric_result
                    metrics[metric_name] = metric_value

                # Try to parse as learning rate
                lr_result = parse_learning_rate(part)
                if lr_result is not None:
                    lr = lr_result

            # Skip if no metrics found
            if not metrics:
                continue

            # Handle missing train_size - assign default if not present
            if config['train_size'] is None:
                config['train_size'] = 50000  # Default train size

            # Combine config, metrics, and lr
            result = {**config, **metrics, 'lr': lr}
            results.append(result)

    if not results:
        return results, None

    # Detect metric format from the first result
    metric_format_info = detect_metric_format(results[0])

    if metric_format_info is None:
        return results, None

    return results, metric_format_info


def select_best_hyperparameters(results, train_metric_name):
    """
    For each (train_size, n_layers) combination, select the configuration
    with the best training accuracy across different (n_heads, d_model, lr).

    Args:
        results: List of experiment results
        train_metric_name: Name of the training accuracy metric to use for selection

    Returns:
        list: Filtered list containing only the best configuration for each (train_size, n_layers)
    """
    # Group by (train_size, n_layers)
    grouped = {}

    for result in results:
        key = (result['train_size'], result['n_layers'])

        if key not in grouped:
            grouped[key] = []

        grouped[key].append(result)

    # Select best for each group
    best_results = []

    for key, group in grouped.items():
        # Find the result with highest train accuracy
        best = max(group, key=lambda x: x.get(train_metric_name, 0.0))
        best_results.append(best)

    return best_results


def create_heatmap_data(results, metric_name):
    """
    Create a 2D array for heatmap visualization.

    Args:
        results: List of experiment results (already filtered to best hyperparameters)
        metric_name: Name of the metric to visualize

    Returns:
        tuple: (dataframe, train_sizes, n_layers_list) for plotting
    """
    # Extract unique train_sizes and n_layers
    train_sizes = sorted(set(r['train_size'] for r in results))
    n_layers_list = sorted(set(r['n_layers'] for r in results))

    # Create a DataFrame with n_layers as rows and train_sizes as columns
    data = pd.DataFrame(index=n_layers_list, columns=train_sizes, dtype=float)

    # Fill in the data
    for result in results:
        train_size = result['train_size']
        n_layers = result['n_layers']
        value = result[metric_name]

        data.loc[n_layers, train_size] = value

    return data, train_sizes, n_layers_list


def plot_heatmap(data, metric_name, title, output_path):
    """
    Create and save a heatmap visualization.

    Args:
        data: DataFrame with heatmap data
        metric_name: Name of the metric being visualized
        title: Title for the plot
        output_path: Path to save the PDF file
    """
    # Set up the matplotlib figure with publication-quality settings
    plt.figure(figsize=(10, 8))

    # Set style for academic papers
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12

    # Create heatmap
    ax = sns.heatmap(
        data,
        annot=True,  # Show values in cells
        fmt='.3f',   # Format as 3 decimal places
        cmap='RdYlGn',  # Red-Yellow-Green colormap (red=low, green=high)
        vmin=0.0,    # Minimum value for color scale
        vmax=1.0,    # Maximum value for color scale
        cbar_kws={'label': 'Accuracy'},
        linewidths=0.5,  # Add gridlines between cells
        linecolor='gray',
        square=False,  # Don't force square cells
        mask=data.isna()  # Mask NaN values (leave blank)
    )

    # Set labels and title
    ax.set_xlabel('Training Sample Count', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Layers', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save to PDF
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Close figure to free memory
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate heatmaps from language modeling training results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_heatmaps.py algorithmic/lm-out-new-addition/summary.txt
  python plot_heatmaps.py /path/to/summary.txt
        """
    )

    parser.add_argument(
        'summary_file',
        type=str,
        help='Path to the summary.txt file containing experiment results'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory for PDF files (default: same as summary file)'
    )

    args = parser.parse_args()

    # Check if file exists
    summary_path = Path(args.summary_file)
    if not summary_path.exists():
        print(f"Error: File not found: {args.summary_file}", file=sys.stderr)
        sys.exit(1)

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = summary_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing summary file: {summary_path}")

    # Parse the summary file
    results, metric_format = parse_summary_file(summary_path)

    if not results:
        print("Error: No valid results found in the summary file.", file=sys.stderr)
        sys.exit(1)

    if metric_format is None:
        print("Error: Could not detect metric format in the summary file.", file=sys.stderr)
        print("Expected metrics like 'eval_len0-50_train_acc' or 'eval_len0-50_acc'", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(results)} experiment results")
    print(f"Detected format: {metric_format['format_type']}")
    print(f"  Training metric: {metric_format['train_metric']}")
    print(f"  Same-length test metric: {metric_format['test_same_metric']}")
    print(f"  Longer-length test metric: {metric_format['test_longer_metric']}")

    # Select best hyperparameters for each (train_size, n_layers) combination
    best_results = select_best_hyperparameters(results, metric_format['train_metric'])

    print(f"Selected {len(best_results)} best configurations")

    # Define the three metrics to visualize based on detected format
    metrics = [
        (metric_format['train_metric'], 'Training Accuracy (Same Length)', 'heatmap_train_acc.pdf'),
        (metric_format['test_same_metric'], 'Test Accuracy (Same Length)', 'heatmap_test_same_acc.pdf'),
        (metric_format['test_longer_metric'], 'Test Accuracy (Longer Strings)', 'heatmap_test_longer_acc.pdf')
    ]

    # Generate each heatmap
    for metric_name, title, filename in metrics:
        print(f"\nGenerating heatmap for: {metric_name}")

        # Create heatmap data
        data, train_sizes, n_layers_list = create_heatmap_data(best_results, metric_name)

        print(f"  Data shape: {len(n_layers_list)} layers × {len(train_sizes)} sample counts")
        print(f"  Sample counts: {train_sizes}")
        print(f"  Layers: {n_layers_list}")

        # Plot and save
        output_path = output_dir / filename
        plot_heatmap(data, metric_name, title, output_path)

    print("\n✓ All heatmaps generated successfully!")


if __name__ == '__main__':
    main()
