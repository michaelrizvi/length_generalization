#!/usr/bin/env python
"""
Plot accuracy metrics from summary.txt files.

Usage:
    python plot_summary.py <path_to_summary.txt> [--output <output_path>]

Examples:
    python plot_summary.py lm-out-new-bin_majority/summary.txt --output results.png
    python plot_summary.py lm-out-new-bin_majority/summary.txt --output results.pdf
"""

import argparse
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def parse_summary_file(filepath):
    """
    Parse summary.txt file and extract configuration and accuracy metrics.

    Returns:
        dict: Dictionary mapping (n_layers, n_heads, d_model, lr) to accuracy metrics
    """
    data = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Skip lines that reached max step (didn't converge)
            if 'reach max step' in line:
                continue

            # Parse config: e.g., "1l1h16d"
            config_match = re.match(r'(\d+)l(\d+)h(\d+)d', line)
            if not config_match:
                continue

            n_layers = int(config_match.group(1))
            n_heads = int(config_match.group(2))
            d_model = int(config_match.group(3))

            # Parse learning rate
            lr_match = re.search(r'lr:\s*([\d.e-]+)', line)
            if not lr_match:
                continue
            lr = float(lr_match.group(1))

            # Parse all accuracy metrics
            acc_metrics = {}
            acc_pattern = r'(eval_[\w-]+_acc):\s*([\d.]+)'
            for match in re.finditer(acc_pattern, line):
                metric_name = match.group(1)
                acc_value = float(match.group(2))
                acc_metrics[metric_name] = acc_value

            if acc_metrics:  # Only add if we found accuracy metrics
                data[(n_layers, n_heads, d_model, lr)].append(acc_metrics)

    return data


def clean_metric_name(metric_name):
    """
    Clean up metric names for display.

    Examples:
        eval_len10-50_train_acc -> Train (10-50)
        eval_len10-50_test_acc -> Test (10-50)
        eval_len60-70_acc -> Test (60-70)
    """
    # Extract length range
    range_match = re.search(r'len(\d+-\d+)', metric_name)
    if not range_match:
        return metric_name

    length_range = range_match.group(1)

    # Determine if it's train or test
    if 'train' in metric_name:
        return f'Train ({length_range})'
    else:
        return f'Test ({length_range})'


def plot_accuracies(data, output_path=None, show_plot=True):
    """
    Create line plot of accuracies vs number of layers.

    Args:
        data: Dictionary from parse_summary_file
        output_path: Optional path to save the plot
        show_plot: Whether to display the plot interactively
    """
    # Average metrics for each configuration
    averaged_data = {}
    for config, metrics_list in data.items():
        n_layers, n_heads, d_model, lr = config

        # Average all metrics across runs
        avg_metrics = {}
        all_metric_names = set()
        for metrics in metrics_list:
            all_metric_names.update(metrics.keys())

        for metric_name in all_metric_names:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            if values:
                avg_metrics[metric_name] = np.mean(values)

        averaged_data[config] = avg_metrics

    # Sort by number of layers
    sorted_configs = sorted(averaged_data.keys(), key=lambda x: x[0])

    # Extract number of layers for x-axis
    n_layers_list = [config[0] for config in sorted_configs]

    # Collect all unique metric names and sort them
    all_metrics = set()
    for metrics in averaged_data.values():
        all_metrics.update(metrics.keys())

    # Sort metrics: train first, then test ranges in order
    def metric_sort_key(metric):
        if 'train' in metric:
            return (0, metric)
        else:
            # Extract the range for sorting test metrics
            range_match = re.search(r'len(\d+)-(\d+)', metric)
            if range_match:
                return (1, int(range_match.group(1)))
            return (2, metric)

    sorted_metrics = sorted(all_metrics, key=metric_sort_key)

    # Create plot with style similar to the reference
    plt.figure(figsize=(10, 6))

    # Use a color palette similar to the reference
    colors = ['#FF6B9D', '#8FBC4A', '#4ECDC4', '#9B59B6', '#E67E22', '#3498DB']

    # Plot each metric as a line
    for idx, metric_name in enumerate(sorted_metrics):
        y_values = []
        for config in sorted_configs:
            metrics = averaged_data[config]
            y_values.append(metrics.get(metric_name, np.nan))

        color = colors[idx % len(colors)]
        clean_name = clean_metric_name(metric_name)

        plt.plot(n_layers_list, y_values,
                marker='o',
                markersize=8,
                linewidth=2.5,
                color=color,
                label=clean_name)

    # Styling to match reference plot
    plt.xlabel('Number of Layers', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Accuracy vs Number of Layers', fontsize=14, fontweight='bold')

    # Set y-axis range from 0 to 1
    plt.ylim(0, 1.0)

    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Legend
    plt.legend(loc='best', fontsize=10, framealpha=0.9)

    # Set background color similar to reference
    ax = plt.gca()
    ax.set_facecolor('#E8E8F0')
    plt.gcf().patch.set_facecolor('white')

    # Adjust layout
    plt.tight_layout()

    # Save and/or show
    if output_path:
        # Matplotlib automatically detects format from extension (.png, .pdf, .svg, etc.)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format=None)
        print(f"Plot saved to {output_path}")

    # Only show interactively if requested
    if show_plot:
        try:
            plt.show()
        except:
            pass  # Silently fail if no display available


def main():
    parser = argparse.ArgumentParser(
        description='Plot accuracy metrics from summary.txt files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('summary_file', type=str,
                       help='Path to summary.txt file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for the plot. Supported formats: .png, .pdf, .svg, .eps (optional, will show plot regardless)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the plot interactively')

    args = parser.parse_args()

    # Parse the summary file
    print(f"Parsing {args.summary_file}...")
    data = parse_summary_file(args.summary_file)

    if not data:
        print("No data found in summary file. Make sure the file contains valid entries.")
        return

    print(f"Found {len(data)} unique configurations")

    # Create plot
    plot_accuracies(data, args.output, show_plot=not args.no_show)


if __name__ == '__main__':
    main()
