#!/usr/bin/env python3
"""
Plot CX count and depth vs number of vertices for connected graph results.

Reads aggregated_stats_raw.csv from each connected_{N}v/counting_{N}v/ folder
and produces scaling plots with error bars (std across runs).

Usage:
  python plot_connected_scaling.py <base_dir>
  python plot_connected_scaling.py analysis/outputs_comp_aware/matching_vs_pauli
  python plot_connected_scaling.py analysis/outputs/connected

  # Compare two heuristics side by side:
  python plot_connected_scaling.py analysis/outputs/connected --compare analysis/outputs_comp_aware/matching_vs_pauli
"""

import argparse
import sys
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

SHADE_ALPHA = 0.15
MATCHING_COLOR = "#1f77b4"   # blue - greedy
PAULI_COLOR = "#d62728"      # red - Pauli
COMP_AWARE_COLOR = "#2ca02c" # green - comp-aware


def load_connected_data(base_dir):
    """Load aggregated raw data from all connected_{N}v folders.

    Supports two directory layouts:
      Layout A (greedy):     base_dir/connected_{N}v/counting_{N}v/aggregated_stats_raw.csv
      Layout B (comp_aware): base_dir/connected_{N}v/counting_{N}v/aggregated_stats_raw.csv
    """
    base_dir = Path(base_dir)
    records = []

    # Find all connected_*v folders
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("connected_"):
            continue

        # Extract vertex count from folder name
        try:
            n_vertices = int(d.name.replace("connected_", "").replace("v", ""))
        except ValueError:
            continue

        # Find counting subfolder
        counting_dirs = list(d.glob("counting_*/"))
        if not counting_dirs:
            continue

        counting_dir = counting_dirs[0]
        raw_csv = counting_dir / "aggregated_stats_raw.csv"

        if not raw_csv.exists():
            # Try to read individual run folders instead
            run_dirs = sorted([rd for rd in counting_dir.iterdir()
                               if rd.is_dir() and rd.name.startswith("20")])
            if not run_dirs:
                print(f"  Skipping {d.name}: no aggregated data or run folders")
                continue

            # Parse summary + averages files from individual runs
            for run_dir in run_dirs:
                avg_files = list((run_dir / "results").glob("averages_*.txt"))
                summary_files = list((run_dir / "results").glob("summary_*.txt"))
                if not avg_files:
                    continue

                run_data = {}
                # Read summary for win/lose/draw counts
                if summary_files:
                    with open(summary_files[0]) as f:
                        for line in f:
                            if ":" in line:
                                k, v = line.strip().split(":", 1)
                                try:
                                    run_data[k.strip()] = float(v.strip())
                                except ValueError:
                                    pass

                # Read averages for CX/depth metrics
                with open(avg_files[0]) as f:
                    for line in f:
                        if ":" in line:
                            k, v = line.strip().split(":", 1)
                            try:
                                run_data[k.strip()] = float(v.strip())
                            except ValueError:
                                pass

                record = {"n_vertices": n_vertices}
                for key in ["win", "lose", "draw",
                            "win_matching_cx", "win_matching_depth",
                            "win_pauli_cx", "win_pauli_depth",
                            "lose_matching_cx", "lose_matching_depth",
                            "lose_pauli_cx", "lose_pauli_depth",
                            "draw_matching_cx", "draw_matching_depth",
                            "draw_pauli_cx", "draw_pauli_depth"]:
                    record[key] = run_data.get(key, np.nan)
                records.append(record)
            continue

        # Read aggregated raw CSV - skip comment lines
        lines = raw_csv.read_text().splitlines()
        header_idx = 0
        data_lines = []
        for i, line in enumerate(lines):
            if line.startswith("#") or line.strip() == "":
                continue
            if line.startswith("run_id,"):
                header_idx = i
                data_lines.append(line)
                continue
            if data_lines and not line.startswith(",") and not line.startswith("#"):
                # Only include data rows (not summary rows)
                parts = line.split(",")
                if parts[0].strip().isdigit():
                    data_lines.append(line)

        if len(data_lines) < 2:
            print(f"  Skipping {d.name}: no data rows in CSV")
            continue

        # Parse CSV from filtered lines
        from io import StringIO
        df = pd.read_csv(StringIO("\n".join(data_lines)))

        # Clean percentage columns
        for col in df.columns:
            if "_pct" in col:
                df[col] = df[col].astype(str).str.replace("%", "").apply(
                    lambda x: float(x) if x.strip() else np.nan
                )

        df["n_vertices"] = n_vertices

        for _, row in df.iterrows():
            records.append(row.to_dict())

    if not records:
        return None

    return pd.DataFrame(records)


def compute_overall_matching_cx(row):
    """Compute weighted average matching CX across win/lose/draw categories."""
    total = 0
    weighted_cx = 0
    for cat in ["win", "lose", "draw"]:
        count = row.get(cat, 0)
        cx = row.get(f"{cat}_matching_cx", np.nan)
        if pd.notna(count) and pd.notna(cx) and count > 0:
            # Clean percentage strings
            if isinstance(count, str):
                count = float(count.replace("%", ""))
            count = float(count)
            total += count
            weighted_cx += count * float(cx)
    return weighted_cx / total if total > 0 else np.nan


def compute_overall_pauli_cx(row):
    """Compute weighted average Pauli CX across win/lose/draw categories."""
    total = 0
    weighted_cx = 0
    for cat in ["win", "lose", "draw"]:
        count = row.get(cat, 0)
        cx = row.get(f"{cat}_pauli_cx", np.nan)
        if pd.notna(count) and pd.notna(cx) and count > 0:
            if isinstance(count, str):
                count = float(count.replace("%", ""))
            count = float(count)
            total += count
            weighted_cx += count * float(cx)
    return weighted_cx / total if total > 0 else np.nan


def compute_overall_depth(row, method):
    """Compute weighted average depth across categories."""
    total = 0
    weighted_depth = 0
    for cat in ["win", "lose", "draw"]:
        count = row.get(cat, 0)
        depth = row.get(f"{cat}_{method}_depth", np.nan)
        if pd.notna(count) and pd.notna(depth) and count > 0:
            if isinstance(count, str):
                count = float(count.replace("%", ""))
            count = float(count)
            total += count
            weighted_depth += count * float(depth)
    return weighted_depth / total if total > 0 else np.nan


def prepare_scaling_data(df):
    """Compute per-run overall CX and depth, then aggregate by vertex count."""
    df = df.copy()
    df["overall_matching_cx"] = df.apply(compute_overall_matching_cx, axis=1)
    df["overall_pauli_cx"] = df.apply(compute_overall_pauli_cx, axis=1)
    df["overall_matching_depth"] = df.apply(
        lambda r: compute_overall_depth(r, "matching"), axis=1)
    df["overall_pauli_depth"] = df.apply(
        lambda r: compute_overall_depth(r, "pauli"), axis=1)

    # Aggregate by vertex count
    summary = df.groupby("n_vertices").agg(
        matching_cx_mean=("overall_matching_cx", "mean"),
        matching_cx_std=("overall_matching_cx", "std"),
        pauli_cx_mean=("overall_pauli_cx", "mean"),
        pauli_cx_std=("overall_pauli_cx", "std"),
        matching_depth_mean=("overall_matching_depth", "mean"),
        matching_depth_std=("overall_matching_depth", "std"),
        pauli_depth_mean=("overall_pauli_depth", "mean"),
        pauli_depth_std=("overall_pauli_depth", "std"),
        n_runs=("overall_matching_cx", "count"),
    ).reset_index()

    # Fill NaN std with 0 (single run)
    summary = summary.fillna(0)

    return summary


def plot_scaling(summary, output_dir, metric="cx", label_suffix="",
                 summary2=None, label1="Greedy", label2="Comp-Aware"):
    """Plot CX or depth vs vertices with shaded error regions."""
    fig, ax = plt.subplots(figsize=(5, 4))

    vertices = summary["n_vertices"].values
    matching_key = f"matching_{metric}_mean"
    matching_std_key = f"matching_{metric}_std"
    pauli_key = f"pauli_{metric}_mean"
    pauli_std_key = f"pauli_{metric}_std"

    ylabel = "CX Gate Count" if metric == "cx" else "Circuit Depth"

    if summary2 is not None:
        # Comparison mode: matching from both heuristics
        vertices2 = summary2["n_vertices"].values

        # Pauli (same for both)
        p_mean = summary["pauli_cx_mean" if metric == "cx" else "pauli_depth_mean"].values
        p_std = summary["pauli_cx_std" if metric == "cx" else "pauli_depth_std"].values
        p_lo = np.maximum(p_mean - p_std, 1)
        ax.fill_between(vertices, p_lo, p_mean + p_std,
                        color=PAULI_COLOR, alpha=SHADE_ALPHA)
        ax.plot(vertices, p_mean, 's-', color=PAULI_COLOR,
                label="Pauli", markersize=5, linewidth=1.5)

        # Heuristic 1 (greedy)
        m1_mean = summary[matching_key].values
        m1_std = summary[matching_std_key].values
        m1_lo = np.maximum(m1_mean - m1_std, 1)
        ax.fill_between(vertices, m1_lo, m1_mean + m1_std,
                        color=MATCHING_COLOR, alpha=SHADE_ALPHA)
        ax.plot(vertices, m1_mean, 'o-', color=MATCHING_COLOR,
                label=f"Matching ({label1})", markersize=5, linewidth=1.5)

        # Heuristic 2 (comp-aware)
        m2_mean = summary2[matching_key].values
        m2_std = summary2[matching_std_key].values
        m2_lo = np.maximum(m2_mean - m2_std, 1)
        ax.fill_between(vertices2, m2_lo, m2_mean + m2_std,
                        color=COMP_AWARE_COLOR, alpha=SHADE_ALPHA)
        ax.plot(vertices2, m2_mean, '^-', color=COMP_AWARE_COLOR,
                label=f"Matching ({label2})", markersize=5, linewidth=1.5)
    else:
        # Single heuristic mode
        m_mean = summary[matching_key].values
        m_std = summary[matching_std_key].values
        p_mean = summary[pauli_key].values
        p_std = summary[pauli_std_key].values

        m_lo = np.maximum(m_mean - m_std, 1)
        p_lo = np.maximum(p_mean - p_std, 1)

        ax.fill_between(vertices, m_lo, m_mean + m_std,
                        color=MATCHING_COLOR, alpha=SHADE_ALPHA)
        ax.fill_between(vertices, p_lo, p_mean + p_std,
                        color=PAULI_COLOR, alpha=SHADE_ALPHA)

        ax.plot(vertices, m_mean, 'o-', color=MATCHING_COLOR,
                label="Matching", markersize=5, linewidth=1.5)
        ax.plot(vertices, p_mean, 's-', color=PAULI_COLOR,
                label="Pauli", markersize=5, linewidth=1.5)

    ax.set_xlabel("Number of Vertices $N$")
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='black', framealpha=1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    plt.tight_layout()

    v_min, v_max = int(vertices.min()), int(vertices.max())
    suffix = f"_{label_suffix}" if label_suffix else ""
    filename = f"connected_{metric}_vs_vertices_{v_min}-{v_max}v{suffix}.pdf"
    plot_file = Path(output_dir) / filename
    fig.savefig(plot_file, format="pdf", bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {plot_file}")
    plt.close()


def plot_win_rate(summary_data, output_dir, label_suffix=""):
    """Plot win/lose/draw percentages from raw data."""
    # This needs the raw df, not the summary
    pass


def print_scaling_summary(summary):
    """Print scaling summary table."""
    print(f"\n{'='*75}")
    print("CONNECTED GRAPHS - SCALING SUMMARY")
    print(f"{'='*75}")
    print(f"{'Vertices':<10} {'Runs':<6} {'Match CX':<20} {'Pauli CX':<20} {'Match Depth':<20} {'Pauli Depth':<20}")
    print("-" * 96)

    for _, row in summary.iterrows():
        m_cx = f"{row['matching_cx_mean']:.1f} +/- {row['matching_cx_std']:.1f}"
        p_cx = f"{row['pauli_cx_mean']:.1f} +/- {row['pauli_cx_std']:.1f}"
        m_d = f"{row['matching_depth_mean']:.1f} +/- {row['matching_depth_std']:.1f}"
        p_d = f"{row['pauli_depth_mean']:.1f} +/- {row['pauli_depth_std']:.1f}"
        print(f"{int(row['n_vertices']):<10} {int(row['n_runs']):<6} {m_cx:<20} {p_cx:<20} {m_d:<20} {p_d:<20}")

    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot CX/depth scaling for connected graph results"
    )
    parser.add_argument(
        "base_dir",
        help="Base directory containing connected_{N}v folders"
    )
    parser.add_argument(
        "--compare",
        type=str, default=None,
        help="Second base directory for comparison (e.g., greedy vs comp-aware)"
    )
    parser.add_argument(
        "--label1", type=str, default="Greedy",
        help="Label for first dataset (default: Greedy)"
    )
    parser.add_argument(
        "--label2", type=str, default="Comp-Aware",
        help="Label for second dataset (default: Comp-Aware)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for plots (default: base_dir)"
    )

    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output) if args.output else base_dir

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return 1

    print(f"Loading data from: {base_dir}")
    df = load_connected_data(base_dir)

    if df is None or df.empty:
        print("Error: No data found")
        return 1

    summary = prepare_scaling_data(df)
    print_scaling_summary(summary)

    # Comparison mode
    summary2 = None
    if args.compare:
        compare_dir = Path(args.compare)
        if compare_dir.exists():
            print(f"\nLoading comparison data from: {compare_dir}")
            df2 = load_connected_data(compare_dir)
            if df2 is not None and not df2.empty:
                summary2 = prepare_scaling_data(df2)
                print(f"\nComparison ({args.label2}):")
                print_scaling_summary(summary2)

    # Generate plots
    print("\nGenerating plots...")
    label_suffix = "compare" if summary2 is not None else ""

    plot_scaling(summary, output_dir, metric="cx", label_suffix=label_suffix,
                 summary2=summary2, label1=args.label1, label2=args.label2)
    plot_scaling(summary, output_dir, metric="depth", label_suffix=label_suffix,
                 summary2=summary2, label1=args.label1, label2=args.label2)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
