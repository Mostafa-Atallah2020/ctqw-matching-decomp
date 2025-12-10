#!/usr/bin/env python3
"""
CX Count Scaling Analysis - Compute Script

Computes CX gate counts for Matching vs Pauli decomposition across different
graph sizes to analyze how gate count scales with number of vertices.
Works with any graph type (Erdős-Rényi, path, bipartite, etc.).

Usage:
  python cx_scaling_compute.py <g6_file1> <g6_file2> ... [options]
  python cx_scaling_compute.py analysis/graphs/erdos_renyi/*.g6 -o analysis/outputs/cx_scaling

Output:
  - raw_results.csv: Per-graph CX counts
  - summary.csv: Aggregated statistics by vertex count
  - metadata.json: Configuration info
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import MultiEdgeGraph, MatchingDecomposition, PauliDecomposition
from src.utils.graph import load_graphs_from_g6

from qiskit import transpile


def get_circuit_metrics(qc, seed=42):
    """Transpile circuit and return CX count, U3 count, and depth."""
    transpiled = transpile(
        qc,
        basis_gates=["cx", "u3"],
        optimization_level=3,
        seed_transpiler=seed
    )
    counts = transpiled.count_ops()
    return {
        "cx_count": counts.get("cx", 0),
        "u3_count": counts.get("u3", 0),
        "depth": transpiled.depth()
    }


def analyze_graph(edges, n_steps=1, delta_t=0.1):
    """Analyze a single graph and return metrics for both decompositions."""
    G = MultiEdgeGraph(edges)

    # Matching decomposition
    matching_decomp = MatchingDecomposition(G)
    matching_qc = matching_decomp.build_circuit(n_steps=n_steps, delta_t=delta_t)
    matching_metrics = get_circuit_metrics(matching_qc)

    # Pauli decomposition
    pauli_decomp = PauliDecomposition(G)
    pauli_qc = pauli_decomp.build_circuit(n_steps=n_steps, delta_t=delta_t)
    pauli_metrics = get_circuit_metrics(pauli_qc)

    return {
        "n_qubits": G.n_qubits,
        "n_vertices": 2 ** G.n_qubits,
        "n_edges": len(edges),
        "n_matchings": matching_decomp.num_matchings(),
        "matching_cx": matching_metrics["cx_count"],
        "matching_u3": matching_metrics["u3_count"],
        "matching_depth": matching_metrics["depth"],
        "pauli_cx": pauli_metrics["cx_count"],
        "pauli_u3": pauli_metrics["u3_count"],
        "pauli_depth": pauli_metrics["depth"],
        "cx_diff": matching_metrics["cx_count"] - pauli_metrics["cx_count"],
        "cx_ratio": matching_metrics["cx_count"] / pauli_metrics["cx_count"] if pauli_metrics["cx_count"] > 0 else np.nan
    }


def process_g6_file(g6_file, n_steps=1, delta_t=0.1, verbose=True):
    """Process all graphs in a G6 file."""
    graphs, metadata = load_graphs_from_g6(g6_file)

    if not graphs:
        print(f"  No graphs loaded from {g6_file}")
        return [], metadata

    n_vertices = metadata.get("vertices", "?")
    graph_type = metadata.get("type", "unknown")

    if verbose:
        print(f"  Processing {len(graphs)} graphs ({n_vertices}v, type={graph_type})")

    results = []
    for i, edges in enumerate(graphs):
        if not edges:
            continue

        try:
            result = analyze_graph(edges, n_steps, delta_t)
            result["graph_index"] = i
            result["graph_type"] = graph_type
            result["source_file"] = Path(g6_file).name
            results.append(result)

            if verbose:
                m_cx = result["matching_cx"]
                p_cx = result["pauli_cx"]
                diff = result["cx_diff"]
                winner = "M" if diff < 0 else ("P" if diff > 0 else "=")
                print(f"    [{i+1:3d}/{len(graphs)}] edges={len(edges):3d}  M={m_cx:4d}  P={p_cx:4d}  diff={diff:+4d}  [{winner}]")

        except Exception as e:
            if verbose:
                print(f"    [{i+1:3d}/{len(graphs)}] Error: {e}")
            continue

    return results, metadata


def save_results(all_results, output_dir):
    """Save raw results and compute summary statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    df = pd.DataFrame(all_results)
    raw_file = output_dir / "raw_results.csv"
    df.to_csv(raw_file, index=False)
    print(f"Raw results saved to {raw_file}")

    # Compute summary by n_vertices
    summary_rows = []
    for n_vertices in sorted(df["n_vertices"].unique()):
        subset = df[df["n_vertices"] == n_vertices]

        summary_rows.append({
            "n_vertices": n_vertices,
            "n_qubits": int(np.log2(n_vertices)),
            "n_graphs": len(subset),
            "avg_edges": subset["n_edges"].mean(),
            "avg_matchings": subset["n_matchings"].mean(),
            # Matching stats
            "matching_cx_mean": subset["matching_cx"].mean(),
            "matching_cx_std": subset["matching_cx"].std(),
            "matching_cx_min": subset["matching_cx"].min(),
            "matching_cx_max": subset["matching_cx"].max(),
            "matching_depth_mean": subset["matching_depth"].mean(),
            # Pauli stats
            "pauli_cx_mean": subset["pauli_cx"].mean(),
            "pauli_cx_std": subset["pauli_cx"].std(),
            "pauli_cx_min": subset["pauli_cx"].min(),
            "pauli_cx_max": subset["pauli_cx"].max(),
            "pauli_depth_mean": subset["pauli_depth"].mean(),
            # Comparison stats
            "cx_diff_mean": subset["cx_diff"].mean(),
            "cx_diff_std": subset["cx_diff"].std(),
            "cx_ratio_mean": subset["cx_ratio"].mean(),
            "cx_ratio_std": subset["cx_ratio"].std(),
            "matching_wins": (subset["cx_diff"] < 0).sum(),
            "pauli_wins": (subset["cx_diff"] > 0).sum(),
            "draws": (subset["cx_diff"] == 0).sum(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / "summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")

    return df, summary_df


def save_metadata(args, file_list, output_dir):
    """Save metadata about the computation."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "n_steps": args.n_steps,
        "delta_t": args.delta_t,
        "input_files": [str(f) for f in file_list],
        "n_files": len(file_list)
    }

    meta_file = Path(output_dir) / "metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute CX scaling for Matching vs Pauli decomposition"
    )
    parser.add_argument(
        "g6_files", nargs="+", help="G6 files to process (supports glob patterns)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--n-steps", type=int, default=1,
        help="Number of Trotter steps (default: 1)"
    )
    parser.add_argument(
        "--delta-t", type=float, default=0.1,
        help="Time step (default: 0.1)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=True,
        help="Verbose output"
    )

    args = parser.parse_args()

    # Expand glob patterns
    from glob import glob
    file_list = []
    for pattern in args.g6_files:
        matches = glob(pattern)
        if matches:
            file_list.extend(matches)
        elif Path(pattern).exists():
            file_list.append(pattern)

    if not file_list:
        print("No input files found")
        return 1

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        script_dir = Path(__file__).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = script_dir / "outputs" / "cx_scaling" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect graph types from filenames
    from src.utils.graph.g6_utils import parse_g6_filename
    graph_types = set()
    for f in file_list:
        meta = parse_g6_filename(f)
        if "type" in meta:
            graph_types.add(meta["type"])
    graph_type_str = ", ".join(sorted(graph_types)) if graph_types else "unknown"

    print("=" * 60)
    print("CX SCALING ANALYSIS - Matching vs Pauli")
    print("=" * 60)
    print(f"Graph type(s): {graph_type_str}")
    print(f"Input files: {len(file_list)}")
    print(f"Trotter steps: {args.n_steps}")
    print(f"Delta t: {args.delta_t}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Sort files by vertex count (extract number before 'v' in filename)
    def get_vertex_count(filepath):
        import re
        match = re.search(r'_(\d+)v\.g6$', filepath)
        return int(match.group(1)) if match else 0

    sorted_files = sorted(file_list, key=get_vertex_count)

    # Process all files
    all_results = []
    for g6_file in sorted_files:
        print(f"\nProcessing: {Path(g6_file).name}")
        results, _ = process_g6_file(
            g6_file,
            n_steps=args.n_steps,
            delta_t=args.delta_t,
            verbose=args.verbose
        )
        all_results.extend(results)

    if not all_results:
        print("\nNo results to save")
        return 1

    # Save results
    print("\n" + "-" * 60)
    df, summary_df = save_results(all_results, output_dir)
    save_metadata(args, file_list, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY BY VERTEX COUNT")
    print("=" * 60)
    print(f"{'Vertices':<10} {'Graphs':<8} {'Match CX':<12} {'Pauli CX':<12} {'Diff':<10} {'Win%':<8}")
    print("-" * 60)
    for _, row in summary_df.iterrows():
        win_pct = 100 * row["matching_wins"] / row["n_graphs"] if row["n_graphs"] > 0 else 0
        print(f"{int(row['n_vertices']):<10} {int(row['n_graphs']):<8} "
              f"{row['matching_cx_mean']:<12.1f} {row['pauli_cx_mean']:<12.1f} "
              f"{row['cx_diff_mean']:<10.1f} {win_pct:<8.1f}")

    print("\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"Run: python cx_count_erdos_renyi_plot.py {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
