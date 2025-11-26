#!/usr/bin/env python3
"""
Trotterization Error Plotting - Visualization from Pre-computed Data

Generates plots from data computed by trotterization_error_compute.py.
Can be run multiple times with different options without recomputing.
Supports multiple data directories for comparison or merging.

Usage: python trotterization_error_plot.py <data_dir> [data_dir2 ...] [options]
Examples:
  # Single dataset
  python trotterization_error_plot.py outputs/trotterization_error/BM_8v/20241125_143022

  # Merge datasets (e.g., different time values) into one plot
  python trotterization_error_plot.py dir_t01 dir_t001 --merge

  # Compare datasets (show as separate lines)
  python trotterization_error_plot.py dir1 dir2 dir3 --compare

  # With options
  python trotterization_error_plot.py <data_dir> --error-type minmax
  python trotterization_error_plot.py <data_dir> --no-log --format png

Options:
  --error-type: std (default), minmax, iqr (interquartile range), none
  --no-log: Use linear scale instead of log scale
  --format: pdf (default), png, svg
  --title: Custom plot title
  --merge: Merge data from multiple directories into one plot (different time values)
  --compare: Generate comparison plot across multiple datasets (side-by-side)
  --output: Output directory for plots (default: first data_dir)
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def load_data(data_dir):
    """Load statistics and metadata from data directory."""
    data_dir = Path(data_dir)

    stats_file = data_dir / 'statistics.csv'
    meta_file = data_dir / 'metadata.json'

    if not stats_file.exists():
        raise FileNotFoundError(f"Statistics file not found: {stats_file}")

    stats_df = pd.read_csv(stats_file)

    metadata = {}
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            metadata = json.load(f)

    return stats_df, metadata


def load_multiple_data(data_dirs):
    """Load data from multiple directories."""
    all_data = []
    for data_dir in data_dirs:
        try:
            stats_df, metadata = load_data(data_dir)
            all_data.append({
                'dir': data_dir,
                'stats': stats_df,
                'metadata': metadata,
                'label': metadata.get('graph_type', Path(data_dir).parent.name)
            })
            print(f"Loaded: {data_dir}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    return all_data


def merge_data(data_dirs):
    """Merge statistics from multiple directories into one combined dataset.

    Useful when you have separate runs with different time values (e.g., t=0.1 and t=0.01)
    and want to combine them into a single convergence plot.
    """
    merged_stats = []
    merged_metadata = {}
    all_time_values = []
    all_trotter_steps = set()

    for data_dir in data_dirs:
        try:
            stats_df, metadata = load_data(data_dir)
            merged_stats.append(stats_df)

            # Collect time values
            time_vals = stats_df['time'].unique().tolist()
            all_time_values.extend(time_vals)

            # Collect trotter steps
            all_trotter_steps.update(stats_df['trotter_steps'].unique().tolist())

            # Merge metadata (take first one as base, update with others)
            if not merged_metadata:
                merged_metadata = metadata.copy()
            else:
                # Update time values list
                if 'time_values' in merged_metadata and 'time_values' in metadata:
                    merged_metadata['time_values'] = list(set(
                        merged_metadata['time_values'] + metadata['time_values']
                    ))

            print(f"Loaded: {data_dir} (time values: {time_vals})")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if not merged_stats:
        return None, None

    # Concatenate all statistics
    combined_df = pd.concat(merged_stats, ignore_index=True)

    # Check for duplicate (time, trotter_steps) combinations
    duplicates = combined_df.duplicated(subset=['time', 'trotter_steps'], keep=False)
    if duplicates.any():
        print(f"Warning: Found duplicate (time, trotter_steps) combinations. Using first occurrence.")
        combined_df = combined_df.drop_duplicates(subset=['time', 'trotter_steps'], keep='first')

    # Update metadata
    merged_metadata['time_values'] = sorted(list(set(all_time_values)))
    merged_metadata['trotter_steps'] = sorted(list(all_trotter_steps))
    merged_metadata['merged_from'] = [str(d) for d in data_dirs]

    print(f"\nMerged data:")
    print(f"  Time values: {merged_metadata['time_values']}")
    print(f"  Trotter steps: {sorted(combined_df['trotter_steps'].unique())}")

    return combined_df, merged_metadata


def create_convergence_plot(stats_df, metadata, args, output_dir):
    """Create convergence plot with customizable error bars."""
    time_values = stats_df['time'].unique()

    plt.figure(figsize=(12, 8))

    matching_color = '#1f77b4'  # Blue
    pauli_color = '#ff7f0e'     # Orange
    line_styles = ['-', '--', '-.', ':']

    for time_idx, time_val in enumerate(sorted(time_values)):
        line_style = line_styles[time_idx % len(line_styles)]
        time_data = stats_df[stats_df['time'] == time_val].sort_values('trotter_steps')

        steps = time_data['trotter_steps'].values
        matching_means = time_data['matching_mean'].values
        pauli_means = time_data['pauli_mean'].values

        # Determine error bars based on type
        if args.error_type == 'std':
            matching_yerr = time_data['matching_std'].values
            pauli_yerr = time_data['pauli_std'].values
        elif args.error_type == 'minmax':
            matching_yerr = [
                matching_means - time_data['matching_min'].values,
                time_data['matching_max'].values - matching_means
            ]
            pauli_yerr = [
                pauli_means - time_data['pauli_min'].values,
                time_data['pauli_max'].values - pauli_means
            ]
        elif args.error_type == 'iqr':
            matching_yerr = [
                matching_means - time_data['matching_q25'].values,
                time_data['matching_q75'].values - matching_means
            ]
            pauli_yerr = [
                pauli_means - time_data['pauli_q25'].values,
                time_data['pauli_q75'].values - pauli_means
            ]
        else:  # none
            matching_yerr = None
            pauli_yerr = None

        # Filter valid data
        valid_mask = ~np.isnan(matching_means) & ~np.isnan(pauli_means)
        valid_steps = steps[valid_mask]
        valid_matching = matching_means[valid_mask]
        valid_pauli = pauli_means[valid_mask]

        if matching_yerr is not None:
            if isinstance(matching_yerr, list):
                valid_matching_yerr = [matching_yerr[0][valid_mask], matching_yerr[1][valid_mask]]
                valid_pauli_yerr = [pauli_yerr[0][valid_mask], pauli_yerr[1][valid_mask]]
            else:
                valid_matching_yerr = matching_yerr[valid_mask]
                valid_pauli_yerr = pauli_yerr[valid_mask]
        else:
            valid_matching_yerr = None
            valid_pauli_yerr = None

        if len(valid_steps) > 0:
            plt.errorbar(valid_steps, valid_matching,
                        yerr=valid_matching_yerr,
                        marker='o', linewidth=2, capsize=5, label=f'Matching t={time_val}',
                        color=matching_color, linestyle=line_style)

            plt.errorbar(valid_steps, valid_pauli,
                        yerr=valid_pauli_yerr,
                        marker='s', linewidth=2, capsize=5, label=f'Pauli t={time_val}',
                        color=pauli_color, linestyle=line_style)

    plt.xlabel('Trotter Steps', fontsize=12)
    plt.ylabel('2-norm Difference from Exact CTQW', fontsize=12)

    if args.title:
        plt.title(args.title)
    else:
        graph_type = metadata.get('graph_type', 'Unknown')
        n_vertices = metadata.get('n_vertices', '?')
        n_graphs = metadata.get('n_graphs_processed', '?')
        plt.title(f'Trotterization Error: Matching vs Pauli\n{graph_type} graphs, {n_vertices} vertices, {n_graphs} graphs')

    if not args.no_log:
        plt.yscale('log')

    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    error_suffix = f"_{args.error_type}" if args.error_type != 'std' else ""
    scale_suffix = "_linear" if args.no_log else ""
    plot_file = output_dir / f'convergence{error_suffix}{scale_suffix}.{args.format}'
    plt.savefig(plot_file, format=args.format, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Plot saved to {plot_file}")


def create_multi_dataset_plot(all_data, args, output_dir):
    """Create comparison plot across multiple datasets."""
    # Use a colormap for different datasets
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Matching comparison
    ax_matching = axes[0]
    # Right plot: Pauli comparison
    ax_pauli = axes[1]

    for idx, data in enumerate(all_data):
        stats_df = data['stats']
        label = data['label']
        color = colors[idx]
        marker = markers[idx % len(markers)]

        # Get first time value for simplicity (or could loop over all)
        time_values = sorted(stats_df['time'].unique())
        time_val = time_values[0]  # Use first time value

        time_data = stats_df[stats_df['time'] == time_val].sort_values('trotter_steps')

        steps = time_data['trotter_steps'].values
        matching_means = time_data['matching_mean'].values
        pauli_means = time_data['pauli_mean'].values
        matching_stds = time_data['matching_std'].values
        pauli_stds = time_data['pauli_std'].values

        valid_mask = ~np.isnan(matching_means) & ~np.isnan(pauli_means)

        if args.error_type == 'none':
            ax_matching.plot(steps[valid_mask], matching_means[valid_mask],
                            marker=marker, linewidth=2, color=color, label=label)
            ax_pauli.plot(steps[valid_mask], pauli_means[valid_mask],
                         marker=marker, linewidth=2, color=color, label=label)
        else:
            ax_matching.errorbar(steps[valid_mask], matching_means[valid_mask],
                                yerr=matching_stds[valid_mask],
                                marker=marker, linewidth=2, capsize=3, color=color, label=label)
            ax_pauli.errorbar(steps[valid_mask], pauli_means[valid_mask],
                             yerr=pauli_stds[valid_mask],
                             marker=marker, linewidth=2, capsize=3, color=color, label=label)

    for ax, title in [(ax_matching, 'Matching Decomposition'), (ax_pauli, 'Pauli Decomposition')]:
        ax.set_xlabel('Trotter Steps', fontsize=12)
        ax.set_ylabel('2-norm Error', fontsize=12)
        ax.set_title(title)
        if not args.no_log:
            ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(args.title if args.title else 'Trotterization Error Comparison Across Datasets', fontsize=14)
    plt.tight_layout()

    scale_suffix = "_linear" if args.no_log else ""
    plot_file = output_dir / f'multi_comparison{scale_suffix}.{args.format}'
    plt.savefig(plot_file, format=args.format, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Multi-dataset comparison saved to {plot_file}")


def create_multi_ratio_plot(all_data, args, output_dir):
    """Create ratio comparison plot across multiple datasets."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']

    plt.figure(figsize=(10, 6))

    for idx, data in enumerate(all_data):
        stats_df = data['stats']
        label = data['label']
        color = colors[idx]
        marker = markers[idx % len(markers)]

        time_values = sorted(stats_df['time'].unique())
        time_val = time_values[0]

        time_data = stats_df[stats_df['time'] == time_val].sort_values('trotter_steps')

        steps = time_data['trotter_steps'].values
        matching_means = time_data['matching_mean'].values
        pauli_means = time_data['pauli_mean'].values

        valid_mask = ~np.isnan(matching_means) & ~np.isnan(pauli_means) & (pauli_means > 0)
        ratio = matching_means[valid_mask] / pauli_means[valid_mask]

        plt.plot(steps[valid_mask], ratio, marker=marker, linewidth=2,
                 color=color, label=label)

    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal error')
    plt.xlabel('Trotter Steps', fontsize=12)
    plt.ylabel('Matching Error / Pauli Error', fontsize=12)
    plt.title(args.title if args.title else 'Error Ratio Comparison\n(< 1 means Matching is better)')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = output_dir / f'multi_ratio.{args.format}'
    plt.savefig(plot_file, format=args.format, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Multi-dataset ratio plot saved to {plot_file}")


def create_comparison_plot(stats_df, metadata, args, output_dir):
    """Create a plot comparing matching vs pauli directly."""
    time_values = stats_df['time'].unique()

    fig, axes = plt.subplots(1, len(time_values), figsize=(6*len(time_values), 5), squeeze=False)

    for idx, time_val in enumerate(sorted(time_values)):
        ax = axes[0, idx]
        time_data = stats_df[stats_df['time'] == time_val].sort_values('trotter_steps')

        steps = time_data['trotter_steps'].values
        matching_means = time_data['matching_mean'].values
        pauli_means = time_data['pauli_mean'].values

        valid_mask = ~np.isnan(matching_means) & ~np.isnan(pauli_means)

        # Scatter plot: matching vs pauli
        ax.scatter(matching_means[valid_mask], pauli_means[valid_mask],
                   c=steps[valid_mask], cmap='viridis', s=100, edgecolors='black')

        # Add diagonal line (equal error)
        all_vals = np.concatenate([matching_means[valid_mask], pauli_means[valid_mask]])
        min_val, max_val = np.min(all_vals), np.max(all_vals)
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal error')

        ax.set_xlabel('Matching Error', fontsize=11)
        ax.set_ylabel('Pauli Error', fontsize=11)
        ax.set_title(f't = {time_val}')

        if not args.no_log:
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                    norm=plt.Normalize(vmin=steps[valid_mask].min(),
                                                       vmax=steps[valid_mask].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Trotter Steps')

    plt.suptitle('Matching vs Pauli Error Comparison', fontsize=14)
    plt.tight_layout()

    scale_suffix = "_linear" if args.no_log else ""
    plot_file = output_dir / f'comparison{scale_suffix}.{args.format}'
    plt.savefig(plot_file, format=args.format, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Comparison plot saved to {plot_file}")


def create_ratio_plot(stats_df, metadata, args, output_dir):
    """Create plot showing the ratio of matching/pauli errors."""
    time_values = stats_df['time'].unique()

    plt.figure(figsize=(10, 6))
    line_styles = ['-', '--', '-.', ':']

    for time_idx, time_val in enumerate(sorted(time_values)):
        line_style = line_styles[time_idx % len(line_styles)]
        time_data = stats_df[stats_df['time'] == time_val].sort_values('trotter_steps')

        steps = time_data['trotter_steps'].values
        matching_means = time_data['matching_mean'].values
        pauli_means = time_data['pauli_mean'].values

        valid_mask = ~np.isnan(matching_means) & ~np.isnan(pauli_means) & (pauli_means > 0)
        ratio = matching_means[valid_mask] / pauli_means[valid_mask]

        plt.plot(steps[valid_mask], ratio, marker='o', linewidth=2,
                 linestyle=line_style, label=f't={time_val}')

    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal error')
    plt.xlabel('Trotter Steps', fontsize=12)
    plt.ylabel('Matching Error / Pauli Error', fontsize=12)
    plt.title('Error Ratio: Matching vs Pauli\n(< 1 means Matching is better)')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = output_dir / f'ratio.{args.format}'
    plt.savefig(plot_file, format=args.format, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Ratio plot saved to {plot_file}")


def main():
    parser = argparse.ArgumentParser(description='Plot trotterization error results')
    parser.add_argument('data_dirs', nargs='+', help='Path(s) to data directory (output from compute script)')
    parser.add_argument('--error-type', choices=['std', 'minmax', 'iqr', 'none'],
                        default='std', help='Type of error bars')
    parser.add_argument('--no-log', action='store_true', help='Use linear scale instead of log')
    parser.add_argument('--format', choices=['pdf', 'png', 'svg'], default='pdf',
                        help='Output format')
    parser.add_argument('--title', type=str, default=None, help='Custom plot title')
    parser.add_argument('--all', action='store_true', help='Generate all plot types')
    parser.add_argument('--merge', action='store_true',
                        help='Merge data from multiple directories into one plot (different time values)')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison plots across datasets (side-by-side)')
    parser.add_argument('--output', type=str, default=None, help='Output directory for plots')

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.data_dirs[0])

    # Merge mode: combine data from multiple directories into one plot
    if args.merge and len(args.data_dirs) > 1:
        stats_df, metadata = merge_data(args.data_dirs)

        if stats_df is None:
            print("Error: No valid data to merge")
            return

        # Generate plots with merged data (same as single dataset)
        create_convergence_plot(stats_df, metadata, args, output_dir)

        if args.all:
            create_comparison_plot(stats_df, metadata, args, output_dir)
            create_ratio_plot(stats_df, metadata, args, output_dir)

    # Single dataset mode
    elif len(args.data_dirs) == 1 and not args.compare:
        try:
            stats_df, metadata = load_data(args.data_dirs[0])
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        print(f"Loaded data from: {args.data_dirs[0]}")
        print(f"  Time values: {sorted(stats_df['time'].unique())}")
        print(f"  Trotter steps: {sorted(stats_df['trotter_steps'].unique())}")

        # Generate plots
        create_convergence_plot(stats_df, metadata, args, output_dir)

        if args.all:
            create_comparison_plot(stats_df, metadata, args, output_dir)
            create_ratio_plot(stats_df, metadata, args, output_dir)

    # Multi-dataset comparison mode (side-by-side)
    else:
        all_data = load_multiple_data(args.data_dirs)

        if not all_data:
            print("Error: No valid data directories found")
            return

        print(f"\nLoaded {len(all_data)} datasets")

        # Generate multi-dataset comparison plots
        create_multi_dataset_plot(all_data, args, output_dir)
        create_multi_ratio_plot(all_data, args, output_dir)

        # Also generate individual plots if requested
        if args.all:
            for data in all_data:
                data_output_dir = Path(data['dir'])
                create_convergence_plot(data['stats'], data['metadata'], args, data_output_dir)
                create_comparison_plot(data['stats'], data['metadata'], args, data_output_dir)
                create_ratio_plot(data['stats'], data['metadata'], args, data_output_dir)

    print("\nPlotting complete!")


if __name__ == "__main__":
    main()
