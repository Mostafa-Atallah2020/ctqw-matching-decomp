#!/usr/bin/env python3
"""
CX Count Scaling Analysis - Plotting Script

Generates plots showing how CX gate count scales with number of vertices
for Matching vs Pauli decomposition. Works with any graph type.

Usage:
  python cx_scaling_plot.py <output_dir>
  python cx_scaling_plot.py analysis/outputs/cx_scaling/erdos_renyi

Input:
  - raw_results.csv: Per-graph CX counts (from cx_scaling_compute.py)
  - summary.csv: Aggregated statistics by vertex count
  - metadata.json: Configuration info
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# PRX Quantum style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Shading alpha for error regions
SHADE_ALPHA = 0.25

# Colors (matching trotterization_error_plot.py)
MATCHING_COLOR = "#1f77b4"  # Blue
PAULI_COLOR = "#ff7f0e"  # Orange


def load_data(output_dir):
    """Load raw results, summary, and metadata from output directory."""
    output_dir = Path(output_dir)

    raw_file = output_dir / "raw_results.csv"
    summary_file = output_dir / "summary.csv"
    meta_file = output_dir / "metadata.json"

    if not raw_file.exists():
        raise FileNotFoundError(f"raw_results.csv not found in {output_dir}")

    raw_df = pd.read_csv(raw_file)
    summary_df = pd.read_csv(summary_file) if summary_file.exists() else None

    metadata = {}
    if meta_file.exists():
        with open(meta_file) as f:
            metadata = json.load(f)

    return raw_df, summary_df, metadata


def get_filename_prefix(raw_df, metadata):
    """Generate informative filename prefix from data."""
    # Get graph types
    if "graph_type" in raw_df.columns:
        graph_types = raw_df["graph_type"].unique()
        if len(graph_types) == 1:
            graph_type = graph_types[0]
        else:
            graph_type = "mixed"
    else:
        graph_type = "unknown"

    # Get vertex range
    vertices = sorted(raw_df["n_vertices"].unique())
    if len(vertices) == 1:
        vertex_str = f"{vertices[0]}v"
    else:
        vertex_str = f"{min(vertices)}-{max(vertices)}v"

    # Get graph count
    n_graphs = len(raw_df)

    return f"cx_scaling_{graph_type}_{vertex_str}_{n_graphs}graphs"


def compute_summary_for_subset(df):
    """Compute summary statistics for a subset of the raw data."""
    summary_rows = []
    for n_vertices in sorted(df["n_vertices"].unique()):
        subset = df[df["n_vertices"] == n_vertices]
        summary_rows.append({
            "n_vertices": n_vertices,
            "n_qubits": int(np.log2(n_vertices)),
            "n_graphs": len(subset),
            "avg_edges": subset["n_edges"].mean(),
            "matching_cx_mean": subset["matching_cx"].mean(),
            "matching_cx_std": subset["matching_cx"].std(),
            "pauli_cx_mean": subset["pauli_cx"].mean(),
            "pauli_cx_std": subset["pauli_cx"].std(),
            "cx_ratio_mean": subset["cx_ratio"].mean(),
            "cx_ratio_std": subset["cx_ratio"].std(),
            "matching_wins": (subset["cx_diff"] < 0).sum(),
            "pauli_wins": (subset["cx_diff"] > 0).sum(),
            "draws": (subset["cx_diff"] == 0).sum(),
        })
    return pd.DataFrame(summary_rows)


def plot_cx_vs_vertices(raw_df, summary_df, output_dir, prefix, title_suffix="", show_stars=False):
    """Plot CX count vs number of vertices with shaded error regions."""
    fig, ax = plt.subplots(figsize=(5, 4))

    vertices = summary_df["n_vertices"].values
    matching_mean = summary_df["matching_cx_mean"].values
    matching_std = summary_df["matching_cx_std"].values
    pauli_mean = summary_df["pauli_cx_mean"].values
    pauli_std = summary_df["pauli_cx_std"].values

    # Shaded error regions (clamp lower bound to avoid log scale issues)
    matching_lower = np.maximum(matching_mean - matching_std, 1)
    matching_upper = matching_mean + matching_std
    pauli_lower = np.maximum(pauli_mean - pauli_std, 1)
    pauli_upper = pauli_mean + pauli_std

    ax.fill_between(vertices, matching_lower, matching_upper,
                    color=MATCHING_COLOR, alpha=SHADE_ALPHA)
    ax.fill_between(vertices, pauli_lower, pauli_upper,
                    color=PAULI_COLOR, alpha=SHADE_ALPHA)

    # Mean lines with markers
    ax.plot(vertices, matching_mean, 'o-', color=MATCHING_COLOR,
            label="Matching", markersize=5, linewidth=1.5)
    ax.plot(vertices, pauli_mean, 's-', color=PAULI_COLOR,
            label="Pauli", markersize=5, linewidth=1.5)

    # Mark Matching wins and draws with stars (optional)
    if show_stars and raw_df is not None and "cx_diff" in raw_df.columns:
        np.random.seed(42)  # Reproducible jitter

        # Matching wins (green stars)
        matching_wins = raw_df[raw_df["cx_diff"] < 0]
        if len(matching_wins) > 0:
            x_jitter = np.random.uniform(-0.15, 0.15, len(matching_wins))
            y_jitter = np.random.uniform(-0.1, 0.1, len(matching_wins))
            x_pos = matching_wins["n_vertices"].values * (1 + x_jitter)
            y_pos = matching_wins["matching_cx"].values * (1 + y_jitter)
            ax.scatter(x_pos, y_pos,
                      marker='*', color='green', s=100, zorder=5,
                      label=f"Matching wins ({len(matching_wins)})", alpha=0.9,
                      edgecolors='darkgreen', linewidth=0.5)

        # Draws (gray stars)
        draws = raw_df[raw_df["cx_diff"] == 0]
        if len(draws) > 0:
            x_jitter = np.random.uniform(-0.15, 0.15, len(draws))
            y_jitter = np.random.uniform(-0.1, 0.1, len(draws))
            x_pos = draws["n_vertices"].values * (1 + x_jitter)
            y_pos = draws["matching_cx"].values * (1 + y_jitter)
            ax.scatter(x_pos, y_pos,
                      marker='*', color='gray', s=100, zorder=5,
                      label=f"Draws ({len(draws)})", alpha=0.9,
                      edgecolors='black', linewidth=0.5)

    ax.set_xlabel("Number of Vertices $N$")
    ax.set_ylabel("CX Gate Count")
    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='black', framealpha=1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    plt.tight_layout()

    # Informative filename: cx_count_erdos_renyi_p0_10_4-128v.pdf
    v_min, v_max = int(vertices.min()), int(vertices.max())
    graph_type = title_suffix.replace("_", "") if title_suffix else "combined"
    filename = f"cx_count_erdos_renyi_{graph_type}_{v_min}-{v_max}v.pdf"
    plot_file = output_dir / filename
    fig.savefig(plot_file, format="pdf", bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {plot_file}")
    plt.close()


def plot_cx_ratio(summary_df, output_dir, prefix, title_suffix=""):
    """Plot the ratio of Matching/Pauli CX counts with shaded error region."""
    fig, ax = plt.subplots(figsize=(5, 4))

    vertices = summary_df["n_vertices"].values
    ratio_mean = summary_df["cx_ratio_mean"].values
    ratio_std = summary_df["cx_ratio_std"].values if "cx_ratio_std" in summary_df.columns else np.zeros_like(ratio_mean)

    # Shaded error region
    ax.fill_between(vertices, ratio_mean - ratio_std, ratio_mean + ratio_std,
                    color=MATCHING_COLOR, alpha=SHADE_ALPHA)

    # Mean line with markers
    ax.plot(vertices, ratio_mean, 'o-', color=MATCHING_COLOR,
            markersize=5, linewidth=1.5)

    # Reference line at 1.0
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1,
               label='Equal CX count')

    ax.set_xlabel("Number of Vertices $N$")
    ax.set_ylabel("CX Ratio (Matching / Pauli)")
    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='black', framealpha=1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xscale('log', base=2)

    plt.tight_layout()

    # Informative filename: cx_ratio_erdos_renyi_p0_10_4-128v.pdf
    v_min, v_max = int(vertices.min()), int(vertices.max())
    graph_type = title_suffix.replace("_", "") if title_suffix else "combined"
    filename = f"cx_ratio_erdos_renyi_{graph_type}_{v_min}-{v_max}v.pdf"
    plot_file = output_dir / filename
    fig.savefig(plot_file, format="pdf", bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {plot_file}")
    plt.close()


def plot_win_rate(summary_df, output_dir, prefix):
    """Plot win rate for Matching vs Pauli across vertex counts."""
    fig, ax = plt.subplots(figsize=(5, 4))

    vertices = summary_df["n_vertices"].values
    n_graphs = summary_df["n_graphs"].values
    matching_wins = summary_df["matching_wins"].values
    pauli_wins = summary_df["pauli_wins"].values
    draws = summary_df["draws"].values

    matching_pct = 100 * matching_wins / n_graphs
    pauli_pct = 100 * pauli_wins / n_graphs
    draw_pct = 100 * draws / n_graphs

    width = 0.25
    x = np.arange(len(vertices))

    ax.bar(x - width, matching_pct, width, label='Matching wins', color=MATCHING_COLOR)
    ax.bar(x, draw_pct, width, label='Draw', color='#7f7f7f')
    ax.bar(x + width, pauli_pct, width, label='Pauli wins', color=PAULI_COLOR)

    ax.set_xlabel("Number of Vertices $N$")
    ax.set_ylabel("Win Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in vertices])
    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='black', framealpha=1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax.set_ylim(0, 100)

    plt.tight_layout()

    plot_file = output_dir / "cx_winrate.pdf"
    fig.savefig(plot_file, format="pdf", bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {plot_file}")
    plt.close()


def plot_scatter_by_edges(raw_df, output_dir, prefix):
    """Scatter plot of CX count vs number of edges."""
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(raw_df["n_edges"], raw_df["matching_cx"],
               alpha=0.5, label="Matching", color=MATCHING_COLOR, s=20,
               edgecolors='black', linewidth=0.3)
    ax.scatter(raw_df["n_edges"], raw_df["pauli_cx"],
               alpha=0.5, label="Pauli", color=PAULI_COLOR, s=20,
               edgecolors='black', linewidth=0.3)

    ax.set_xlabel("Number of Edges $m$")
    ax.set_ylabel("CX Gate Count")
    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='black', framealpha=1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()

    plot_file = output_dir / "cx_vs_edges.pdf"
    fig.savefig(plot_file, format="pdf", bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {plot_file}")
    plt.close()


def plot_cx_diff_histogram(raw_df, output_dir, prefix):
    """Histogram of CX difference (Matching - Pauli) by vertex count."""
    vertex_list = sorted(raw_df["n_vertices"].unique())
    n_plots = len(vertex_list)

    fig, axes = plt.subplots(1, n_plots, figsize=(3.5 * n_plots, 3))
    if n_plots == 1:
        axes = [axes]

    for ax, n_v in zip(axes, vertex_list):
        subset = raw_df[raw_df["n_vertices"] == n_v]
        diff = subset["cx_diff"].values

        ax.hist(diff, bins=30, color=MATCHING_COLOR, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

        # Mean line
        mean_diff = diff.mean()
        ax.axvline(x=mean_diff, color=PAULI_COLOR, linestyle='-', linewidth=1.5)

        ax.set_xlabel("CX Diff (M$-$P)")
        ax.set_ylabel("Count")
        ax.set_title(f"$N$={n_v}")

        ax.text(0.95, 0.95, f"$\\mu$={mean_diff:.1f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')

    plt.tight_layout()

    plot_file = output_dir / "cx_diff_hist.pdf"
    fig.savefig(plot_file, format="pdf", bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {plot_file}")
    plt.close()


def plot_boxplot(raw_df, output_dir, prefix):
    """Box plot showing CX distribution by vertex count."""
    fig, ax = plt.subplots(figsize=(5, 4))

    vertex_list = sorted(raw_df["n_vertices"].unique())
    matching_data = [raw_df[raw_df["n_vertices"] == v]["matching_cx"].values for v in vertex_list]
    pauli_data = [raw_df[raw_df["n_vertices"] == v]["pauli_cx"].values for v in vertex_list]

    positions = np.arange(len(vertex_list))
    width = 0.35

    bp1 = ax.boxplot(matching_data, positions=positions - width/2, widths=width,
                     patch_artist=True,
                     boxprops=dict(facecolor=MATCHING_COLOR, alpha=0.7),
                     medianprops=dict(color='black', linewidth=1),
                     flierprops=dict(marker='o', markersize=3, alpha=0.5))
    bp2 = ax.boxplot(pauli_data, positions=positions + width/2, widths=width,
                     patch_artist=True,
                     boxprops=dict(facecolor=PAULI_COLOR, alpha=0.7),
                     medianprops=dict(color='black', linewidth=1),
                     flierprops=dict(marker='o', markersize=3, alpha=0.5))

    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in vertex_list])
    ax.set_xlabel("Number of Vertices $N$")
    ax.set_ylabel("CX Gate Count")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Matching", "Pauli"],
              loc='upper left', frameon=True, fancybox=False,
              edgecolor='black', framealpha=1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')

    plt.tight_layout()

    plot_file = output_dir / "cx_boxplot.pdf"
    fig.savefig(plot_file, format="pdf", bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {plot_file}")
    plt.close()


def print_summary(summary_df):
    """Print summary statistics to console."""
    print("\n" + "=" * 70)
    print("CX SCALING SUMMARY")
    print("=" * 70)
    print(f"{'Vertices':<10} {'Graphs':<8} {'Match CX':<15} {'Pauli CX':<15} {'Ratio':<8} {'Match Win%':<10}")
    print("-" * 70)

    for _, row in summary_df.iterrows():
        match_cx = f"{row['matching_cx_mean']:.1f} ± {row['matching_cx_std']:.1f}"
        pauli_cx = f"{row['pauli_cx_mean']:.1f} ± {row['pauli_cx_std']:.1f}"
        win_pct = 100 * row["matching_wins"] / row["n_graphs"] if row["n_graphs"] > 0 else 0

        print(f"{int(row['n_vertices']):<10} {int(row['n_graphs']):<8} "
              f"{match_cx:<15} {pauli_cx:<15} "
              f"{row['cx_ratio_mean']:<8.2f} {win_pct:<10.1f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Plot CX scaling results for Matching vs Pauli decomposition"
    )
    parser.add_argument(
        "output_dir", help="Directory containing cx_scaling compute results"
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate all plot types"
    )
    parser.add_argument(
        "--stars", action="store_true", help="Show stars for Matching wins and draws"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        return 1

    # Find subfolders with data (each probability gets its own folder)
    subfolders = [d for d in output_dir.iterdir() if d.is_dir() and (d / "summary.csv").exists()]

    if not subfolders:
        # Fallback: check if data is in main folder
        if (output_dir / "summary.csv").exists():
            subfolders = [output_dir]
        else:
            print(f"Error: No data found in {output_dir} or its subfolders")
            return 1

    print(f"Found {len(subfolders)} data folders to process")

    # Process each subfolder
    for subfolder in sorted(subfolders):
        print(f"\n{'='*60}")
        print(f"Processing: {subfolder.name}")
        print(f"{'='*60}")

        try:
            raw_df, summary_df, metadata = load_data(subfolder)

            if summary_df is None or summary_df.empty:
                print(f"  Warning: No summary data found in {subfolder.name}")
                continue

            # Detect graph type
            if "graph_type" in raw_df.columns:
                graph_types = raw_df["graph_type"].unique()
                graph_type_str = ", ".join(sorted(set(graph_types)))
            else:
                graph_type_str = subfolder.name

            print(f"Graph type: {graph_type_str}")
            print(f"Loaded {len(raw_df)} graph results across {len(summary_df)} vertex counts")

            # Get filename prefix
            prefix = get_filename_prefix(raw_df, metadata)

            # Print summary
            print_summary(summary_df)

            # Generate plots
            print("\nGenerating plots...")
            plot_cx_vs_vertices(raw_df, summary_df, subfolder, prefix, title_suffix="", show_stars=args.stars)
            plot_cx_ratio(summary_df, subfolder, prefix, title_suffix="")

            if args.all:
                plot_win_rate(summary_df, subfolder, prefix)
                plot_scatter_by_edges(raw_df, subfolder, prefix)
                plot_cx_diff_histogram(raw_df, subfolder, prefix)
                plot_boxplot(raw_df, subfolder, prefix)

        except Exception as e:
            print(f"  Error processing {subfolder.name}: {e}")

    print(f"\nAll plots saved to: {output_dir}")
    print("Plotting complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
