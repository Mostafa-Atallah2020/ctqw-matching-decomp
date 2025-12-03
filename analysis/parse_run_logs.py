#!/usr/bin/env python3
"""
Parse multiple run logs and compute statistics (mean +/- std) for all properties.
Includes detailed graph properties, gate counts, and timing from log files.
Saves results to CSV format with informative organization.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np


def parse_summary_file(summary_path: Path) -> Dict[str, Any]:
    """Parse a summary file and return all properties."""
    results = {}

    with open(summary_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Try to parse as number
            try:
                if "." in value:
                    results[key] = float(value)
                else:
                    results[key] = int(value)
            except ValueError:
                results[key] = value

    return results


def parse_log_file(log_path: Path) -> Dict[str, Any]:
    """Parse a log file and extract detailed metrics."""
    results = {}

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the "Average Gate Counts by Category:" section
    gate_counts_match = re.search(
        r"Average Gate Counts by Category:(.*?)={50,}", content, re.DOTALL
    )
    if gate_counts_match:
        section = gate_counts_match.group(1)

        # Parse key: value pairs (format: [timestamp] key: value)
        for line in section.split("\n"):
            # Match pattern like "[2025-11-26 16:38:03] win_matching_cx: 17.33"
            match = re.search(r"\]\s*([a-zA-Z_]+):\s*(.+)$", line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()

                # Try to parse as number
                try:
                    if "." in value:
                        results[key] = float(value)
                    else:
                        results[key] = int(value)
                except ValueError:
                    results[key] = value

    return results


def analyze_folder(folder_path: Path) -> Tuple[Dict[str, List], int, List[str]]:
    """
    Analyze all runs in a folder.

    Returns:
        Tuple of (results dict with lists of values, total graphs per run, list of run timestamps)
    """
    results = {}
    total_per_run = 0
    run_timestamps = []

    # Find all timestamp subfolders
    run_dirs = sorted([d for d in folder_path.iterdir() if d.is_dir()])

    for run_dir in run_dirs:
        # Look for summary file in results subfolder
        results_dir = run_dir / "results"
        logs_dir = run_dir / "logs"

        if not results_dir.exists():
            continue

        # Find summary file (pattern: summary_*.txt)
        summary_files = list(results_dir.glob("summary_*.txt"))
        if not summary_files:
            continue

        summary_path = summary_files[0]
        run_results = parse_summary_file(summary_path)

        # Also parse log file for detailed metrics
        if logs_dir.exists():
            log_files = list(logs_dir.glob("analysis_*.log"))
            if log_files:
                log_results = parse_log_file(log_files[0])
                run_results.update(log_results)

        # Initialize keys on first run
        if not results:
            for key in run_results:
                results[key] = []

        # Append values
        for key, value in run_results.items():
            if key not in results:
                results[key] = [None] * len(run_timestamps)  # Backfill with None
            results[key].append(value)

        # Track timestamp
        run_timestamps.append(run_dir.name)

        # Calculate total for this run (win + lose + draw)
        if "win" in run_results and "lose" in run_results and "draw" in run_results:
            run_total = run_results["win"] + run_results["lose"] + run_results["draw"]
            if total_per_run == 0:
                total_per_run = run_total

    return results, total_per_run, run_timestamps


def compute_statistics(values: List) -> Dict[str, float]:
    """Compute statistics for a list of numeric values."""
    # Filter out None values
    numeric_values = [v for v in values if v is not None and isinstance(v, (int, float))]

    if not numeric_values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    arr = np.array(numeric_values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_all_statistics(results: Dict[str, List], total: int) -> Dict[str, Dict[str, float]]:
    """Compute statistics for all properties."""
    stats = {}

    for key, values in results.items():
        stats[key] = compute_statistics(values)

        # Add percentage stats for win/lose/draw
        if key in ["win", "lose", "draw"] and total > 0:
            numeric_values = [v for v in values if v is not None and isinstance(v, (int, float))]
            if numeric_values:
                percentages = np.array(numeric_values) / total * 100
                stats[f"{key}_pct"] = {
                    "mean": float(np.mean(percentages)),
                    "std": float(np.std(percentages)),
                    "min": float(np.min(percentages)),
                    "max": float(np.max(percentages)),
                }

    return stats


def format_stats(stats: Dict[str, Dict[str, float]], n_runs: int, total: int) -> str:
    """Format statistics for display."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Statistics from {n_runs} runs ({total} graphs per run)")
    lines.append("=" * 80)
    lines.append("")

    # Group properties
    result_props = ["win", "lose", "draw"]
    timing_props = [k for k in stats if "timing" in k.lower()]
    gate_props = [k for k in stats if any(x in k.lower() for x in ["_cx", "_u3", "_depth"])]
    graph_props = [k for k in stats if "graph_" in k.lower()]

    # Display results
    lines.append("--- Results ---")
    lines.append(f"{'Property':<20} {'Mean +/- Std':<25} {'Range':<20}")
    lines.append("-" * 65)

    for prop in result_props:
        if prop in stats:
            s = stats[prop]
            if s["mean"] is not None:
                mean_std = f"{s['mean']:.1f} +/- {s['std']:.1f}"
                range_str = f"[{s['min']:.0f}, {s['max']:.0f}]"

                # Add percentage
                pct_key = f"{prop}_pct"
                if pct_key in stats and stats[pct_key]["mean"] is not None:
                    pct_s = stats[pct_key]
                    pct_str = f"({pct_s['mean']:.1f}% +/- {pct_s['std']:.1f}%)"
                    mean_std = f"{mean_std} {pct_str}"

                label = {"win": "[WIN]", "lose": "[LOSE]", "draw": "[DRAW]"}.get(prop, prop)
                lines.append(f"{label:<20} {mean_std:<40} {range_str:<20}")

    # Display gate counts
    if gate_props:
        lines.append("")
        lines.append("--- Gate Counts (by category) ---")
        lines.append(f"{'Property':<35} {'Mean +/- Std':<25} {'Range':<20}")
        lines.append("-" * 80)

        for prop in sorted(gate_props):
            s = stats[prop]
            if s["mean"] is not None:
                mean_std = f"{s['mean']:.2f} +/- {s['std']:.2f}"
                range_str = f"[{s['min']:.2f}, {s['max']:.2f}]"
                lines.append(f"{prop:<35} {mean_std:<25} {range_str:<20}")

    # Display graph properties
    if graph_props:
        lines.append("")
        lines.append("--- Graph Properties (by category) ---")
        lines.append(f"{'Property':<40} {'Mean +/- Std':<25} {'Range':<20}")
        lines.append("-" * 85)

        for prop in sorted(graph_props):
            s = stats[prop]
            if s["mean"] is not None:
                mean_std = f"{s['mean']:.3f} +/- {s['std']:.3f}"
                range_str = f"[{s['min']:.3f}, {s['max']:.3f}]"
                lines.append(f"{prop:<40} {mean_std:<25} {range_str:<20}")

    # Display timing
    if timing_props:
        lines.append("")
        lines.append("--- Timing ---")
        lines.append(f"{'Property':<35} {'Mean +/- Std':<25} {'Range':<20}")
        lines.append("-" * 80)

        for prop in sorted(timing_props):
            s = stats[prop]
            if s["mean"] is not None:
                mean_std = f"{s['mean']:.4f} +/- {s['std']:.4f}"
                range_str = f"[{s['min']:.4f}, {s['max']:.4f}]"
                lines.append(f"{prop:<35} {mean_std:<25} {range_str:<20}")

    lines.append("")
    return "\n".join(lines)


def save_stats_json(stats: Dict[str, Dict[str, float]], output_path: Path, n_runs: int, total: int):
    """Save statistics to a JSON file."""
    output_data = {
        "n_runs": n_runs,
        "total_graphs_per_run": total,
        "statistics": stats,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def save_stats_csv(stats: Dict[str, Dict[str, float]], output_path: Path, n_runs: int, total: int):
    """Save statistics to an informative CSV file with grouped sections."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Metadata header
        writer.writerow(["# Aggregated Statistics"])
        writer.writerow(["# n_runs", n_runs])
        writer.writerow(["# total_graphs_per_run", total])
        writer.writerow([])

        # Results section
        writer.writerow(["## Results (Matching vs Pauli)"])
        writer.writerow(
            ["category", "count_mean", "count_std", "count_min", "count_max", "pct_mean", "pct_std"]
        )

        for prop in ["win", "lose", "draw"]:
            if prop in stats:
                s = stats[prop]
                pct_key = f"{prop}_pct"
                pct_s = stats.get(pct_key, {})

                writer.writerow(
                    [
                        prop.upper(),
                        f"{s['mean']:.2f}" if s["mean"] is not None else "",
                        f"{s['std']:.2f}" if s["std"] is not None else "",
                        f"{s['min']:.0f}" if s["min"] is not None else "",
                        f"{s['max']:.0f}" if s["max"] is not None else "",
                        f"{pct_s.get('mean', 0):.2f}%" if pct_s.get("mean") is not None else "",
                        f"{pct_s.get('std', 0):.2f}%" if pct_s.get("std") is not None else "",
                    ]
                )

        writer.writerow([])

        # Gate counts section
        gate_props = sorted(
            [
                k
                for k in stats
                if any(x in k.lower() for x in ["_cx", "_u3", "_depth"])
                and not k.startswith("config")
            ]
        )
        if gate_props:
            writer.writerow(["## Gate Counts by Category"])
            writer.writerow(["metric", "mean", "std", "min", "max"])

            for prop in gate_props:
                s = stats[prop]
                writer.writerow(
                    [
                        prop,
                        f"{s['mean']:.2f}" if s["mean"] is not None else "",
                        f"{s['std']:.2f}" if s["std"] is not None else "",
                        f"{s['min']:.2f}" if s["min"] is not None else "",
                        f"{s['max']:.2f}" if s["max"] is not None else "",
                    ]
                )

            writer.writerow([])

        # Graph properties section
        graph_props = sorted([k for k in stats if "graph_" in k.lower()])
        if graph_props:
            writer.writerow(["## Graph Properties by Category"])
            writer.writerow(["metric", "mean", "std", "min", "max"])

            for prop in graph_props:
                s = stats[prop]
                writer.writerow(
                    [
                        prop,
                        f"{s['mean']:.4f}" if s["mean"] is not None else "",
                        f"{s['std']:.4f}" if s["std"] is not None else "",
                        f"{s['min']:.4f}" if s["min"] is not None else "",
                        f"{s['max']:.4f}" if s["max"] is not None else "",
                    ]
                )

            writer.writerow([])

        # Timing section
        timing_props = sorted([k for k in stats if "timing" in k.lower()])
        if timing_props:
            writer.writerow(["## Timing Statistics (seconds)"])
            writer.writerow(["metric", "mean", "std", "min", "max"])

            for prop in timing_props:
                s = stats[prop]
                readable_name = prop.replace("timing_", "").replace("_", " ").title()
                writer.writerow(
                    [
                        readable_name,
                        f"{s['mean']:.6f}" if s["mean"] is not None else "",
                        f"{s['std']:.6f}" if s["std"] is not None else "",
                        f"{s['min']:.6f}" if s["min"] is not None else "",
                        f"{s['max']:.6f}" if s["max"] is not None else "",
                    ]
                )

            writer.writerow([])

        # Summary row for quick reference
        writer.writerow(["## Summary"])
        win_pct = stats.get("win_pct", {})
        lose_pct = stats.get("lose_pct", {})

        if win_pct.get("mean") is not None:
            writer.writerow(
                ["win_rate", f"{win_pct.get('mean', 0):.1f}% +/- {win_pct.get('std', 0):.1f}%"]
            )
            writer.writerow(
                ["lose_rate", f"{lose_pct.get('mean', 0):.1f}% +/- {lose_pct.get('std', 0):.1f}%"]
            )


def save_raw_data_csv(
    results: Dict[str, List], run_timestamps: List[str], output_path: Path, total: int
):
    """Save raw data from all runs to CSV with percentages."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Organize columns by category
        result_cols = ["win", "lose", "draw"]
        gate_cols = sorted(
            [
                k
                for k in results
                if any(x in k.lower() for x in ["_cx", "_u3", "_depth"])
                and not k.startswith("config")
            ]
        )
        graph_cols = sorted([k for k in results if "graph_" in k.lower()])
        timing_cols = sorted([k for k in results if "timing" in k.lower()])
        config_cols = sorted([k for k in results if k.startswith("config_")])
        other_cols = [
            k
            for k in results
            if k not in result_cols
            and k not in gate_cols
            and k not in graph_cols
            and k not in timing_cols
            and k not in config_cols
        ]

        # Build header with percentages for results
        header = ["run_id", "timestamp"]
        for col in result_cols:
            if col in results:
                header.extend([col, f"{col}_pct"])
        header.extend(gate_cols)
        header.extend(graph_cols)
        header.extend(timing_cols)
        header.extend(config_cols)
        header.extend(other_cols)

        writer.writerow(header)

        # Data rows
        for i, timestamp in enumerate(run_timestamps):
            row = [i + 1, timestamp]

            # Results with percentages
            for col in result_cols:
                if col in results and i < len(results[col]):
                    val = results[col][i]
                    row.append(val)
                    if val is not None and total > 0:
                        row.append(f"{val / total * 100:.2f}%")
                    else:
                        row.append("")
                elif col in results:
                    row.extend(["", ""])

            # Gate columns
            for col in gate_cols:
                if col in results and i < len(results[col]):
                    val = results[col][i]
                    row.append(f"{val:.2f}" if isinstance(val, float) else val)
                else:
                    row.append("")

            # Graph columns
            for col in graph_cols:
                if col in results and i < len(results[col]):
                    val = results[col][i]
                    row.append(f"{val:.4f}" if isinstance(val, float) else val)
                else:
                    row.append("")

            # Timing columns
            for col in timing_cols:
                if col in results and i < len(results[col]):
                    val = results[col][i]
                    row.append(f"{val:.6f}" if isinstance(val, float) else val)
                else:
                    row.append("")

            # Config columns
            for col in config_cols:
                if col in results and i < len(results[col]):
                    row.append(results[col][i])
                else:
                    row.append("")

            # Other columns
            for col in other_cols:
                if col in results and i < len(results[col]):
                    row.append(results[col][i])
                else:
                    row.append("")

            writer.writerow(row)

        # Add summary rows
        writer.writerow([])
        writer.writerow(["# Summary Statistics"])

        # MEAN row
        mean_row = ["", "MEAN"]
        for col in result_cols:
            if col in results:
                vals = [v for v in results[col] if v is not None]
                if vals:
                    mean_val = np.mean(vals)
                    mean_row.append(f"{mean_val:.2f}")
                    mean_row.append(f"{mean_val / total * 100:.2f}%")
                else:
                    mean_row.extend(["", ""])

        for col in gate_cols:
            if col in results:
                vals = [v for v in results[col] if v is not None and isinstance(v, (int, float))]
                mean_row.append(f"{np.mean(vals):.2f}" if vals else "")
            else:
                mean_row.append("")

        for col in graph_cols:
            if col in results:
                vals = [v for v in results[col] if v is not None and isinstance(v, (int, float))]
                mean_row.append(f"{np.mean(vals):.4f}" if vals else "")
            else:
                mean_row.append("")

        for col in timing_cols:
            if col in results:
                vals = [v for v in results[col] if v is not None and isinstance(v, (int, float))]
                mean_row.append(f"{np.mean(vals):.6f}" if vals else "")
            else:
                mean_row.append("")

        writer.writerow(mean_row)

        # STD row
        std_row = ["", "STD"]
        for col in result_cols:
            if col in results:
                vals = [v for v in results[col] if v is not None]
                if vals:
                    std_val = np.std(vals)
                    std_row.append(f"{std_val:.2f}")
                    std_row.append(f"{std_val / total * 100:.2f}%")
                else:
                    std_row.extend(["", ""])

        for col in gate_cols:
            if col in results:
                vals = [v for v in results[col] if v is not None and isinstance(v, (int, float))]
                std_row.append(f"{np.std(vals):.2f}" if vals else "")
            else:
                std_row.append("")

        for col in graph_cols:
            if col in results:
                vals = [v for v in results[col] if v is not None and isinstance(v, (int, float))]
                std_row.append(f"{np.std(vals):.4f}" if vals else "")
            else:
                std_row.append("")

        for col in timing_cols:
            if col in results:
                vals = [v for v in results[col] if v is not None and isinstance(v, (int, float))]
                std_row.append(f"{np.std(vals):.6f}" if vals else "")
            else:
                std_row.append("")

        writer.writerow(std_row)


def main():
    parser = argparse.ArgumentParser(
        description="Parse run logs and compute statistics for all properties",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to folder containing run subfolders (e.g., outputs/matching_vs_pauli/odd_8v)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output base name (without extension). Default: <folder>/aggregated_stats",
    )

    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"[ERROR] Folder not found: {folder_path}")
        return 1

    print(f"[INFO] Analyzing folder: {folder_path}")

    results, total, run_timestamps = analyze_folder(folder_path)
    n_runs = len(run_timestamps)

    if n_runs == 0:
        print("[ERROR] No valid runs found in folder")
        return 1

    print(f"[INFO] Found {n_runs} runs")
    print(f"[INFO] Properties found: {len(results)} metrics")

    # Categorize properties for summary
    gate_props = [k for k in results if any(x in k.lower() for x in ["_cx", "_u3", "_depth"])]
    graph_props = [k for k in results if "graph_" in k.lower()]
    timing_props = [k for k in results if "timing" in k.lower()]

    print(f"[INFO]   - Gate count metrics: {len(gate_props)}")
    print(f"[INFO]   - Graph property metrics: {len(graph_props)}")
    print(f"[INFO]   - Timing metrics: {len(timing_props)}")

    stats = compute_all_statistics(results, total)

    # Display statistics
    print(format_stats(stats, n_runs, total))

    # Determine output paths
    output_base = Path(args.output) if args.output else folder_path / "aggregated_stats"

    # Save to JSON
    json_path = output_base.with_suffix(".json")
    save_stats_json(stats, json_path, n_runs, total)
    print(f"[OK] JSON saved to: {json_path}")

    # Save statistics to CSV
    csv_path = output_base.with_suffix(".csv")
    save_stats_csv(stats, csv_path, n_runs, total)
    print(f"[OK] Statistics CSV saved to: {csv_path}")

    # Save raw data to CSV
    raw_csv_path = output_base.parent / f"{output_base.stem}_raw.csv"
    save_raw_data_csv(results, run_timestamps, raw_csv_path, total)
    print(f"[OK] Raw data CSV saved to: {raw_csv_path}")

    return 0


if __name__ == "__main__":
    exit(main())
