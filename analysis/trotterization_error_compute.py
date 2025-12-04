#!/usr/bin/env python3
"""
Trotterization Error Computation - Data Generation

Computes operator norm differences between Matching/Pauli decompositions and exact CTQW.
Saves all raw data and pre-computed statistics (means, stds) for later plotting.

Usage: python trotterization_error_compute.py <g6_file> [options]
Examples:
  python trotterization_error_compute.py graphs.g6
  python trotterization_error_compute.py graphs.g6 -m 1 -M 20 -s 2 -t 0.1 0.5 1.0

Output files:
  - raw_results.csv: All individual graph results
  - statistics.csv: Pre-computed means and stds for plotting
  - metadata.json: Configuration and metadata
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
sys.path.append(str(Path(__file__).parent.parent))

np.random.seed(123456789)
np.set_printoptions(floatmode="maxprec")

# Use new refactored classes
from src.core import MultiEdgeGraph, MatchingDecomposition, PauliDecomposition
from src.utils import get_exact_evolution_operator
from src.utils.graph import load_graphs_from_g6

# Qiskit imports
from qiskit import transpile
from qiskit.quantum_info import Operator


def analyze_graph_properties(edges, n_vertices=None):
    """Analyze basic properties of the graph.

    Args:
        edges: Set of edge tuples (bitstring pairs)
        n_vertices: Expected number of vertices (from metadata). If None, inferred from edges.
    """
    if not edges:
        return {"n_vertices": 0, "n_edges": 0, "is_empty": True}

    n_edges = len(edges)

    # Use expected n_vertices from metadata if provided, otherwise infer from bitstring length
    if n_vertices is None:
        # Infer from bitstring length (edges are tuples of bitstrings)
        first_edge = next(iter(edges))
        n_qubits = len(first_edge[0])
        n_vertices = 2 ** n_qubits

    max_edges = n_vertices * (n_vertices - 1) // 2

    degrees = defaultdict(int)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1

    return {
        "n_vertices": n_vertices,
        "n_edges": n_edges,
        "density": n_edges / max_edges if max_edges > 0 else 0,
        "min_degree": min(degrees.values()) if degrees else 0,
        "max_degree": max(degrees.values()) if degrees else 0,
        "avg_degree": np.mean(list(degrees.values())) if degrees else 0,
    }


def create_exact_ctqw_operator(edges, time):
    """Create exact CTQW operator using matrix exponentiation."""
    G = MultiEdgeGraph(edges)
    exact_op = get_exact_evolution_operator(time, G.hamiltonian)
    return exact_op


def create_matching_circuit_operator(edges, n_steps, total_time):
    """Create quantum operator using MatchingDecomposition class."""
    G = MultiEdgeGraph(edges)
    decomp = MatchingDecomposition(G)
    qc = decomp.build_circuit(n_steps=n_steps, delta_t=total_time)
    qc_transpiled = transpile(qc, basis_gates=["cx", "u3"], optimization_level=3)
    return Operator(qc_transpiled)


def create_pauli_circuit_operator(edges, n_steps, total_time):
    """Create quantum operator using PauliDecomposition class."""
    G = MultiEdgeGraph(edges)
    decomp = PauliDecomposition(G)
    qc = decomp.build_circuit(n_steps=n_steps, delta_t=total_time)
    return Operator(qc)


def compute_operator_differences(edges, trotter_steps_list, time_values):
    """Compute 2-norm differences between matching/Pauli and exact CTQW."""
    results = {
        "trotter_steps": trotter_steps_list,
        "time_values": time_values,
        "matching_differences": {},
        "pauli_differences": {},
        "properties": analyze_graph_properties(edges),
    }

    for time_val in time_values:
        print(f"    Processing time value: {time_val}")

        try:
            exact_op = create_exact_ctqw_operator(edges, time_val)
        except Exception as e:
            print(f"      Error creating exact CTQW for time {time_val}: {e}")
            results["matching_differences"][time_val] = [np.nan] * len(trotter_steps_list)
            results["pauli_differences"][time_val] = [np.nan] * len(trotter_steps_list)
            continue

        matching_diffs = []
        pauli_diffs = []

        for n_steps in trotter_steps_list:
            print(f"      Trotter steps: {n_steps}", end=" ")

            try:
                matching_op = create_matching_circuit_operator(edges, n_steps, time_val)
                diff = matching_op - exact_op
                two_norm = np.linalg.norm(diff.data, ord=2)
                matching_diffs.append(two_norm)
                print(f"M={two_norm:.6f}", end=", ")
            except Exception as e:
                print(f"M=Error", end=", ")
                matching_diffs.append(np.nan)

            try:
                pauli_op = create_pauli_circuit_operator(edges, n_steps, time_val)
                diff = pauli_op - exact_op
                two_norm = np.linalg.norm(diff.data, ord=2)
                pauli_diffs.append(two_norm)
                print(f"P={two_norm:.6f}")
            except Exception as e:
                print(f"P=Error")
                pauli_diffs.append(np.nan)

        results["matching_differences"][time_val] = matching_diffs
        results["pauli_differences"][time_val] = pauli_diffs

    return results


def process_all_graphs(graphs, trotter_steps_list, time_values, expected_vertices=None):
    """Process all graphs and compute operator differences.

    Args:
        graphs: List of edge sets (each edge is a tuple of bitstrings)
        trotter_steps_list: List of Trotter step counts to test
        time_values: List of time values to test
        expected_vertices: Expected number of vertices from metadata (used for reporting)
    """
    all_results = []
    valid_graphs = 0
    skipped_no_edges = 0

    # Determine n_vertices and n_qubits from metadata or first graph's bitstring length
    if expected_vertices is not None:
        n_vertices = expected_vertices
        n_qubits = int(np.log2(n_vertices))
    else:
        # Infer from first graph's bitstring length
        first_edge = next(iter(graphs[0]))
        n_qubits = len(first_edge[0])
        n_vertices = 2 ** n_qubits

    print(f"  Using {n_vertices} vertices ({n_qubits} qubits) from metadata")

    for i, edges in enumerate(graphs):
        print(f"Processing graph {i+1}/{len(graphs)}")

        if not edges:
            print(f"  Skipping: no edges")
            skipped_no_edges += 1
            continue

        props = analyze_graph_properties(edges, n_vertices)
        print(f"  Graph: {n_vertices} vertices ({n_qubits} qubits), {props['n_edges']} edges")

        if n_qubits > 8:
            print(f"    WARNING: {n_qubits} qubits will create large operators")

        try:
            results = compute_operator_differences(edges, trotter_steps_list, time_values)
            results["graph_index"] = i
            results["n_qubits"] = n_qubits
            results["properties"] = props  # Update with correct n_vertices
            all_results.append(results)
            valid_graphs += 1
        except Exception as e:
            print(f"  Error processing graph {i}: {e}")
            continue

    print(f"\nProcessing summary:")
    print(f"  Successfully processed: {valid_graphs}/{len(graphs)} graphs")
    if skipped_no_edges > 0:
        print(f"  Skipped (no edges): {skipped_no_edges}")

    return all_results


def save_raw_results(all_results, output_dir):
    """Save all raw results to CSV."""
    if not all_results:
        return None

    rows = []
    time_values = all_results[0]["time_values"]
    trotter_steps = all_results[0]["trotter_steps"]

    for result in all_results:
        for time_val in time_values:
            for step_idx, n_steps in enumerate(trotter_steps):
                matching_diff = result["matching_differences"].get(
                    time_val, [np.nan] * len(trotter_steps)
                )[step_idx]
                pauli_diff = result["pauli_differences"].get(
                    time_val, [np.nan] * len(trotter_steps)
                )[step_idx]

                rows.append(
                    {
                        "graph_index": result["graph_index"],
                        "n_qubits": result["n_qubits"],
                        "n_vertices": result["properties"]["n_vertices"],
                        "n_edges": result["properties"]["n_edges"],
                        "density": result["properties"]["density"],
                        "time": time_val,
                        "trotter_steps": n_steps,
                        "matching_diff": matching_diff,
                        "pauli_diff": pauli_diff,
                    }
                )

    df = pd.DataFrame(rows)
    csv_file = output_dir / "raw_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"Raw results saved to {csv_file}")
    return df


def save_metadata(metadata, args, output_dir, n_graphs_processed):
    """Save metadata and configuration."""
    meta = {
        "graph_type": metadata.get("type", "unknown"),
        "n_vertices": metadata.get("vertices", "unknown"),
        "n_graphs_in_file": metadata.get("n_graphs", "unknown"),
        "n_graphs_processed": n_graphs_processed,
        "trotter_steps": list(range(args.min_steps, args.max_steps + 1, args.step_inc)),
        "time_values": args.time_values,
        "timestamp": datetime.now().isoformat(),
        "g6_file": str(args.g6_file),
    }

    json_file = output_dir / "metadata.json"
    with open(json_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Compute trotterization errors for CTQW")
    parser.add_argument("g6_file", help="Path to G6 file")
    parser.add_argument("-m", "--min_steps", type=int, default=1, help="Minimum Trotter steps")
    parser.add_argument("-M", "--max_steps", type=int, default=20, help="Maximum Trotter steps")
    parser.add_argument("-s", "--step_inc", type=int, default=2, help="Step increment")
    parser.add_argument(
        "-t", "--time_values", nargs="+", type=float, default=[0.1], help="Time values"
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    # Load graphs
    graphs, metadata = load_graphs_from_g6(args.g6_file)
    if not graphs:
        print("No graphs loaded")
        return

    # Setup output directory with timestamp
    script_dir = Path(__file__).parent
    graph_type = metadata.get("type", "graph")
    n_vertices = metadata.get("vertices", "N")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output:
        output_dir = Path(args.output) / timestamp
    else:
        output_dir = (
            script_dir
            / "outputs"
            / "trotterization_error"
            / f"{graph_type}_{n_vertices}v"
            / timestamp
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate Trotter steps list
    trotter_steps = list(range(args.min_steps, args.max_steps + 1, args.step_inc))

    print(f"\nComputation Configuration:")
    print(f"  Trotter steps: {trotter_steps}")
    print(f"  Time values: {args.time_values}")
    print(f"  Output: {output_dir}")

    # Process graphs
    expected_vertices = metadata.get("vertices")
    all_results = process_all_graphs(graphs, trotter_steps, args.time_values, expected_vertices)

    # Save all data
    if all_results:
        save_raw_results(all_results, output_dir)
        save_metadata(metadata, args, output_dir, len(all_results))

    print(f"\nComputation complete! Data saved to: {output_dir}")
    print(f"Run 'python trotterization_error_plot.py {output_dir}' to generate plots")


if __name__ == "__main__":
    main()
