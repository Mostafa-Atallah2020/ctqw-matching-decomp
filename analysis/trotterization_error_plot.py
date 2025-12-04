#!/usr/bin/env python3
"""
Trotterization Error Plotting - Visualization from Pre-computed Data

Generates plots from data computed by trotterization_error_compute.py.
Automatically detects and aggregates data from multiple timestamp folders.

Usage: python trotterization_error_plot.py <data_dir> [options]
Examples:
  # Parent folder with timestamp subfolders (aggregates all runs)
  python trotterization_error_plot.py outputs/trotterization_error/disconnected_8v

  # Single timestamp folder
  python trotterization_error_plot.py outputs/trotterization_error/disconnected_8v/20241204_131918

  # With options
  python trotterization_error_plot.py <data_dir> --error-type minmax
  python trotterization_error_plot.py <data_dir> --no-log --format png

Options:
  --error-type: std (default), minmax, iqr (interquartile range), none
  --no-log: Use linear scale instead of log scale for y-axis
  --no-log-x: Use linear scale instead of log scale for x-axis
  --format: pdf (default), png, svg
  --title: Custom plot title
  --all: Generate all plot types
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import pandas as pd

# Publication quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'text.usetex': False,  # Set True if LaTeX is available
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})


def find_timestamp_folders(data_dir):
    """Find all timestamp folders containing raw_results.csv."""
    data_dir = Path(data_dir)
    timestamp_folders = []

    # Check if this is a timestamp folder itself
    if (data_dir / "raw_results.csv").exists():
        return [data_dir]

    # Look for timestamp subfolders
    for subfolder in sorted(data_dir.iterdir()):
        if subfolder.is_dir() and (subfolder / "raw_results.csv").exists():
            timestamp_folders.append(subfolder)

    return timestamp_folders


def load_and_aggregate_data(data_dir):
    """Load raw_results.csv from all timestamp folders and aggregate.

    Returns:
        stats_df: DataFrame with computed statistics (mean, std, min, max, etc.)
        metadata: Combined metadata from first folder
        raw_df: Combined raw data from all folders
    """
    data_dir = Path(data_dir)
    timestamp_folders = find_timestamp_folders(data_dir)

    if not timestamp_folders:
        raise FileNotFoundError(f"No raw_results.csv found in {data_dir} or its subfolders")

    print(f"Found {len(timestamp_folders)} data folder(s):")
    for folder in timestamp_folders:
        print(f"  - {folder.name}")

    # Load all raw results
    all_raw_dfs = []
    metadata = {}

    for folder in timestamp_folders:
        raw_file = folder / "raw_results.csv"
        meta_file = folder / "metadata.json"

        df = pd.read_csv(raw_file)
        df["run_folder"] = folder.name  # Track which run it came from
        all_raw_dfs.append(df)

        # Load metadata from first folder
        if not metadata and meta_file.exists():
            with open(meta_file, "r") as f:
                metadata = json.load(f)

    # Combine all raw data
    raw_df = pd.concat(all_raw_dfs, ignore_index=True)
    print(f"\nTotal data points: {len(raw_df)}")
    print(f"  Unique graphs: {raw_df['graph_index'].nunique()}")
    print(f"  Runs per configuration: {len(timestamp_folders)}")

    # Compute statistics grouped by (time, trotter_steps)
    stats_rows = []

    for time_val in sorted(raw_df["time"].unique()):
        for n_steps in sorted(raw_df["trotter_steps"].unique()):
            mask = (raw_df["time"] == time_val) & (raw_df["trotter_steps"] == n_steps)
            subset = raw_df[mask]

            if len(subset) == 0:
                continue

            matching_vals = subset["matching_diff"].dropna().values
            pauli_vals = subset["pauli_diff"].dropna().values

            stats_rows.append({
                "time": time_val,
                "trotter_steps": n_steps,
                "n_samples": len(matching_vals),
                # Matching statistics
                "matching_mean": np.mean(matching_vals) if len(matching_vals) > 0 else np.nan,
                "matching_std": np.std(matching_vals) if len(matching_vals) > 0 else np.nan,
                "matching_min": np.min(matching_vals) if len(matching_vals) > 0 else np.nan,
                "matching_max": np.max(matching_vals) if len(matching_vals) > 0 else np.nan,
                "matching_median": np.median(matching_vals) if len(matching_vals) > 0 else np.nan,
                "matching_q25": np.percentile(matching_vals, 25) if len(matching_vals) > 0 else np.nan,
                "matching_q75": np.percentile(matching_vals, 75) if len(matching_vals) > 0 else np.nan,
                # Pauli statistics
                "pauli_mean": np.mean(pauli_vals) if len(pauli_vals) > 0 else np.nan,
                "pauli_std": np.std(pauli_vals) if len(pauli_vals) > 0 else np.nan,
                "pauli_min": np.min(pauli_vals) if len(pauli_vals) > 0 else np.nan,
                "pauli_max": np.max(pauli_vals) if len(pauli_vals) > 0 else np.nan,
                "pauli_median": np.median(pauli_vals) if len(pauli_vals) > 0 else np.nan,
                "pauli_q25": np.percentile(pauli_vals, 25) if len(pauli_vals) > 0 else np.nan,
                "pauli_q75": np.percentile(pauli_vals, 75) if len(pauli_vals) > 0 else np.nan,
            })

    stats_df = pd.DataFrame(stats_rows)

    # Update metadata
    metadata["n_runs"] = len(timestamp_folders)
    metadata["n_total_samples"] = len(raw_df)

    return stats_df, metadata, raw_df


def create_convergence_plot(stats_df, metadata, args, output_dir):
    """Create convergence plot with customizable error bars."""
    time_values = stats_df["time"].unique()

    plt.figure(figsize=(12, 8))

    matching_color = "#1f77b4"  # Blue
    pauli_color = "#ff7f0e"  # Orange
    line_styles = ["-", "--", "-.", ":"]

    for time_idx, time_val in enumerate(sorted(time_values)):
        line_style = line_styles[time_idx % len(line_styles)]
        time_data = stats_df[stats_df["time"] == time_val].sort_values("trotter_steps")

        steps = time_data["trotter_steps"].values
        matching_means = time_data["matching_mean"].values
        pauli_means = time_data["pauli_mean"].values

        # Determine error bars based on type
        if args.error_type == "std":
            matching_yerr = time_data["matching_std"].values
            pauli_yerr = time_data["pauli_std"].values
        elif args.error_type == "minmax":
            matching_yerr = [
                matching_means - time_data["matching_min"].values,
                time_data["matching_max"].values - matching_means,
            ]
            pauli_yerr = [
                pauli_means - time_data["pauli_min"].values,
                time_data["pauli_max"].values - pauli_means,
            ]
        elif args.error_type == "iqr":
            matching_yerr = [
                matching_means - time_data["matching_q25"].values,
                time_data["matching_q75"].values - matching_means,
            ]
            pauli_yerr = [
                pauli_means - time_data["pauli_q25"].values,
                time_data["pauli_q75"].values - pauli_means,
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
            plt.errorbar(
                valid_steps,
                valid_matching,
                yerr=valid_matching_yerr,
                marker="o",
                linewidth=2,
                capsize=5,
                label=f"Matching t={time_val}",
                color=matching_color,
                linestyle=line_style,
            )

            plt.errorbar(
                valid_steps,
                valid_pauli,
                yerr=valid_pauli_yerr,
                marker="s",
                linewidth=2,
                capsize=5,
                label=f"Pauli t={time_val}",
                color=pauli_color,
                linestyle=line_style,
            )

    plt.xlabel("Trotter Steps", fontsize=12)
    plt.ylabel("2-norm Difference from Exact CTQW", fontsize=12)

    if args.title:
        plt.title(args.title)
    else:
        graph_type = metadata.get("graph_type", "Unknown")
        n_vertices = metadata.get("n_vertices", "?")
        n_graphs = metadata.get("n_graphs_processed", "?")
        n_runs = metadata.get("n_runs", 1)
        plt.title(
            f"Trotterization Error: Matching vs Pauli\n{graph_type} graphs, {n_vertices} vertices, {n_graphs} graphs, {n_runs} run(s)"
        )

    if not args.no_log:
        plt.yscale("log")

    if not args.no_log_x:
        plt.xscale("log")

    # Create two separate legends side by side: Matching on left, Pauli on right
    handles, labels = plt.gca().get_legend_handles_labels()
    matching_handles = [h for h, l in zip(handles, labels) if "Matching" in l]
    matching_labels = [l.replace("Matching ", "") for l in labels if "Matching" in l]
    pauli_handles = [h for h, l in zip(handles, labels) if "Pauli" in l]
    pauli_labels = [l.replace("Pauli ", "") for l in labels if "Pauli" in l]

    # First legend for Matching (left side of pair)
    leg1 = plt.legend(matching_handles, matching_labels,
                      loc="upper right", title="Matching", fontsize=12, title_fontsize=13,
                      bbox_to_anchor=(0.78, 1.0), framealpha=0.9, borderpad=0.3)
    plt.gca().add_artist(leg1)

    # Second legend for Pauli (right side, adjacent to Matching)
    leg2 = plt.legend(pauli_handles, pauli_labels,
                      loc="upper right", title="Pauli", fontsize=12, title_fontsize=13,
                      bbox_to_anchor=(1.0, 1.0), framealpha=0.9, borderpad=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    error_suffix = f"_{args.error_type}" if args.error_type != "std" else ""
    scale_suffix = "_linear" if args.no_log else ""
    xscale_suffix = "_linearx" if args.no_log_x else ""
    plot_file = output_dir / f"convergence{error_suffix}{scale_suffix}{xscale_suffix}.{args.format}"
    plt.savefig(plot_file, format=args.format, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Plot saved to {plot_file}")


def create_comparison_plot(stats_df, metadata, args, output_dir):
    """Create a plot comparing matching vs pauli directly."""
    time_values = stats_df["time"].unique()

    fig, axes = plt.subplots(1, len(time_values), figsize=(6 * len(time_values), 5), squeeze=False)

    for idx, time_val in enumerate(sorted(time_values)):
        ax = axes[0, idx]
        time_data = stats_df[stats_df["time"] == time_val].sort_values("trotter_steps")

        steps = time_data["trotter_steps"].values
        matching_means = time_data["matching_mean"].values
        pauli_means = time_data["pauli_mean"].values

        valid_mask = ~np.isnan(matching_means) & ~np.isnan(pauli_means)

        # Scatter plot: matching vs pauli
        ax.scatter(
            matching_means[valid_mask],
            pauli_means[valid_mask],
            c=steps[valid_mask],
            cmap="viridis",
            s=100,
            edgecolors="black",
        )

        # Add diagonal line (equal error)
        all_vals = np.concatenate([matching_means[valid_mask], pauli_means[valid_mask]])
        min_val, max_val = np.min(all_vals), np.max(all_vals)
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Equal error")

        ax.set_xlabel("Matching Error", fontsize=11)
        ax.set_ylabel("Pauli Error", fontsize=11)
        ax.set_title(f"t = {time_val}")

        if not args.no_log:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=steps[valid_mask].min(), vmax=steps[valid_mask].max()),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Trotter Steps")

    plt.suptitle("Matching vs Pauli Error Comparison", fontsize=14)
    plt.tight_layout()

    scale_suffix = "_linear" if args.no_log else ""
    plot_file = output_dir / f"comparison{scale_suffix}.{args.format}"
    plt.savefig(plot_file, format=args.format, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Comparison plot saved to {plot_file}")


def create_ratio_plot(stats_df, metadata, args, output_dir):
    """Create plot showing the ratio of matching/pauli errors."""
    time_values = stats_df["time"].unique()

    plt.figure(figsize=(10, 6))
    line_styles = ["-", "--", "-.", ":"]

    for time_idx, time_val in enumerate(sorted(time_values)):
        line_style = line_styles[time_idx % len(line_styles)]
        time_data = stats_df[stats_df["time"] == time_val].sort_values("trotter_steps")

        steps = time_data["trotter_steps"].values
        matching_means = time_data["matching_mean"].values
        pauli_means = time_data["pauli_mean"].values

        valid_mask = ~np.isnan(matching_means) & ~np.isnan(pauli_means) & (pauli_means > 0)
        ratio = matching_means[valid_mask] / pauli_means[valid_mask]

        plt.plot(
            steps[valid_mask],
            ratio,
            marker="o",
            linewidth=2,
            linestyle=line_style,
            label=f"t={time_val}",
        )

    plt.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Equal error")
    plt.xlabel("Trotter Steps", fontsize=12)
    plt.ylabel("Matching Error / Pauli Error", fontsize=12)
    plt.title("Error Ratio: Matching vs Pauli\n(< 1 means Matching is better)")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = output_dir / f"ratio.{args.format}"
    plt.savefig(plot_file, format=args.format, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Ratio plot saved to {plot_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot trotterization error results")
    parser.add_argument(
        "data_dir", help="Path to data directory (can be parent with timestamp subfolders)"
    )
    parser.add_argument(
        "--error-type",
        choices=["std", "minmax", "iqr", "none"],
        default="std",
        help="Type of error bars",
    )
    parser.add_argument("--no-log", action="store_true", help="Use linear scale instead of log for y-axis")
    parser.add_argument("--no-log-x", action="store_true", help="Use linear scale instead of log for x-axis")
    parser.add_argument(
        "--format", choices=["pdf", "png", "svg"], default="pdf", help="Output format"
    )
    parser.add_argument("--title", type=str, default=None, help="Custom plot title")
    parser.add_argument("--all", action="store_true", help="Generate all plot types")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = data_dir

    try:
        stats_df, metadata, raw_df = load_and_aggregate_data(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"\nStatistics computed:")
    print(f"  Time values: {sorted(stats_df['time'].unique())}")
    print(f"  Trotter steps: {sorted(stats_df['trotter_steps'].unique())}")

    # Save aggregated data to CSV files
    stats_file = output_dir / "aggregated_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"Aggregated statistics saved to {stats_file}")

    raw_file = output_dir / "aggregated_raw_data.csv"
    raw_df.to_csv(raw_file, index=False)
    print(f"Aggregated raw data saved to {raw_file}")

    # Save metadata
    meta_file = output_dir / "aggregated_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_file}")

    # Generate plots
    create_convergence_plot(stats_df, metadata, args, output_dir)

    if args.all:
        create_comparison_plot(stats_df, metadata, args, output_dir)
        create_ratio_plot(stats_df, metadata, args, output_dir)

    print("\nPlotting complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
