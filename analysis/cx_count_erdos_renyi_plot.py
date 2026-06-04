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

# npj Quantum Information style (11 pt body, matching document)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 1.2,
    'lines.markersize': 3.5,
    'legend.frameon': False,
})

# Shading alpha for error regions
SHADE_ALPHA = 0.15

# Colors (consistent across all plots in the report)
MATCHING_COLOR = "#1f77b4"   # blue - greedy / matching
PAULI_COLOR = "#d62728"      # red - Pauli
COMP_AWARE_COLOR = "#2ca02c" # green - comp-aware


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
        row = {
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
        }
        # Add depth stats if columns exist
        if "matching_depth" in subset.columns and "pauli_depth" in subset.columns:
            row["matching_depth_mean"] = subset["matching_depth"].mean()
            row["matching_depth_std"] = subset["matching_depth"].std()
            row["pauli_depth_mean"] = subset["pauli_depth"].mean()
            row["pauli_depth_std"] = subset["pauli_depth"].std()
        summary_rows.append(row)
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


def plot_depth_vs_vertices(raw_df, summary_df, output_dir, prefix, title_suffix=""):
    """Plot circuit depth vs number of vertices with shaded error regions."""
    if "matching_depth_mean" not in summary_df.columns:
        print("  Skipping depth plot: no depth data in summary")
        return

    fig, ax = plt.subplots(figsize=(5, 4))

    vertices = summary_df["n_vertices"].values
    matching_mean = summary_df["matching_depth_mean"].values
    matching_std = summary_df["matching_depth_std"].values
    pauli_mean = summary_df["pauli_depth_mean"].values
    pauli_std = summary_df["pauli_depth_std"].values

    # Shaded error regions
    matching_lower = np.maximum(matching_mean - matching_std, 1)
    matching_upper = matching_mean + matching_std
    pauli_lower = np.maximum(pauli_mean - pauli_std, 1)
    pauli_upper = pauli_mean + pauli_std

    ax.fill_between(vertices, matching_lower, matching_upper,
                    color=MATCHING_COLOR, alpha=SHADE_ALPHA)
    ax.fill_between(vertices, pauli_lower, pauli_upper,
                    color=PAULI_COLOR, alpha=SHADE_ALPHA)

    ax.plot(vertices, matching_mean, 'o-', color=MATCHING_COLOR,
            label="Matching", markersize=5, linewidth=1.5)
    ax.plot(vertices, pauli_mean, 's-', color=PAULI_COLOR,
            label="Pauli", markersize=5, linewidth=1.5)

    ax.set_xlabel("Number of Vertices $N$")
    ax.set_ylabel("Circuit Depth")
    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='black', framealpha=1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    plt.tight_layout()

    v_min, v_max = int(vertices.min()), int(vertices.max())
    graph_type = title_suffix.replace("_", "") if title_suffix else "combined"
    filename = f"depth_erdos_renyi_{graph_type}_{v_min}-{v_max}v.pdf"
    plot_file = output_dir / filename
    fig.savefig(plot_file, format="pdf", bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {plot_file}")
    plt.close()


COMP_AWARE_COLOR = "#2ca02c"  # Green


def plot_three_way(summary1, summary2, output_dir, metric="cx", label1="Greedy", label2="Comp-Aware", prob_label=""):
    """Three-way plot: Greedy Matching vs Comp-Aware Matching vs Pauli."""
    fig, ax = plt.subplots(figsize=(5, 4))

    if metric == "cx":
        m_key, m_std_key = "matching_cx_mean", "matching_cx_std"
        p_key, p_std_key = "pauli_cx_mean", "pauli_cx_std"
        ylabel = "CX Gate Count"
    else:
        m_key, m_std_key = "matching_depth_mean", "matching_depth_std"
        p_key, p_std_key = "pauli_depth_mean", "pauli_depth_std"
        ylabel = "Circuit Depth"

    if m_key not in summary1.columns or m_key not in summary2.columns:
        print(f"  Skipping three-way {metric} plot: missing data")
        return

    v1 = summary1["n_vertices"].values
    v2 = summary2["n_vertices"].values

    # Pauli (use from summary1 — same graphs)
    p_mean = summary1[p_key].values
    p_std = summary1[p_std_key].values
    p_lo = np.maximum(p_mean - p_std, 1)
    ax.fill_between(v1, p_lo, p_mean + p_std, color=PAULI_COLOR, alpha=SHADE_ALPHA)
    ax.plot(v1, p_mean, 's-', color=PAULI_COLOR, label="Pauli", markersize=5, linewidth=1.5)

    # Greedy matching
    m1_mean = summary1[m_key].values
    m1_std = summary1[m_std_key].values
    m1_lo = np.maximum(m1_mean - m1_std, 1)
    ax.fill_between(v1, m1_lo, m1_mean + m1_std, color=MATCHING_COLOR, alpha=SHADE_ALPHA)
    ax.plot(v1, m1_mean, 'o-', color=MATCHING_COLOR, label=f"Matching ({label1})", markersize=5, linewidth=1.5)

    # Comp-aware matching
    m2_mean = summary2[m_key].values
    m2_std = summary2[m_std_key].values
    m2_lo = np.maximum(m2_mean - m2_std, 1)
    ax.fill_between(v2, m2_lo, m2_mean + m2_std, color=COMP_AWARE_COLOR, alpha=SHADE_ALPHA)
    ax.plot(v2, m2_mean, '^-', color=COMP_AWARE_COLOR, label=f"Matching ({label2})", markersize=5, linewidth=1.5)

    ax.set_xlabel("Number of Vertices $N$")
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='black', framealpha=1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    plt.tight_layout()

    v_min = int(min(v1.min(), v2.min()))
    v_max = int(max(v1.max(), v2.max()))
    p_str = f"_{prob_label}" if prob_label else ""
    filename = f"{metric}_three_way{p_str}_{v_min}-{v_max}v.pdf"
    plot_file = Path(output_dir) / filename
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
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Second output dir for three-way comparison (e.g., greedy vs comp-aware)"
    )
    parser.add_argument(
        "--compare-output", type=str, default=None,
        help="Output directory for comparison plots (default: output_dir)"
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

            if raw_df is None or raw_df.empty:
                print(f"  Warning: No data found in {subfolder.name}")
                continue

            # Always recompute summary from raw data to include depth
            summary_df = compute_summary_for_subset(raw_df)

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
            plot_depth_vs_vertices(raw_df, summary_df, subfolder, prefix, title_suffix="")

            if args.all:
                plot_win_rate(summary_df, subfolder, prefix)
                plot_scatter_by_edges(raw_df, subfolder, prefix)
                plot_cx_diff_histogram(raw_df, subfolder, prefix)
                plot_boxplot(raw_df, subfolder, prefix)

        except Exception as e:
            print(f"  Error processing {subfolder.name}: {e}")

    # Three-way comparison if --compare is provided
    if args.compare:
        compare_dir = Path(args.compare)
        compare_output = Path(args.compare_output) if args.compare_output else output_dir

        if not compare_dir.exists():
            print(f"Error: Compare directory not found: {compare_dir}")
        else:
            compare_output.mkdir(parents=True, exist_ok=True)

            # Find matching subfolders between the two dirs
            compare_subs = [d for d in compare_dir.iterdir()
                            if d.is_dir() and (d / "raw_results.csv").exists()]

            print(f"\n{'='*60}")
            print("THREE-WAY COMPARISON PLOTS")
            print(f"{'='*60}")
            print(f"  Dir 1 (Greedy):      {output_dir}")
            print(f"  Dir 2 (Comp-Aware):  {compare_dir}")
            print(f"  Output:              {compare_output}")

            for sub1 in sorted(subfolders):
                # Find matching subfolder in compare dir
                sub2 = compare_dir / sub1.name
                if not sub2.exists() or not (sub2 / "raw_results.csv").exists():
                    print(f"  Skipping {sub1.name}: no matching data in compare dir")
                    continue

                try:
                    raw1, _, _ = load_data(sub1)
                    raw2, _, _ = load_data(sub2)

                    if raw1 is None or raw2 is None:
                        continue

                    summary1 = compute_summary_for_subset(raw1)
                    summary2 = compute_summary_for_subset(raw2)

                    prob_label = sub1.name  # e.g., "p0_01"
                    sub_output = compare_output / sub1.name
                    sub_output.mkdir(parents=True, exist_ok=True)

                    print(f"\n  Generating three-way plots for {prob_label}...")
                    plot_three_way(summary1, summary2, sub_output, metric="cx", prob_label=prob_label)
                    plot_three_way(summary1, summary2, sub_output, metric="depth", prob_label=prob_label)

                except Exception as e:
                    print(f"  Error on {sub1.name}: {e}")

    print(f"\nAll plots saved to: {output_dir}")
    print("Plotting complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
