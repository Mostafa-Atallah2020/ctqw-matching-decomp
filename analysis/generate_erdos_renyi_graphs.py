#!/usr/bin/env python3
"""
Erdős-Rényi Random Graph Generator.

Generates random graphs using the G(n,p) model where each edge
is included independently with probability p.

Usage:
  python generate_erdos_renyi_graphs.py --vertices 8 16 32 -n 100 -p 0.1 0.2
  python generate_erdos_renyi_graphs.py --vertices 16 -n 100 -p 0.1 --seed 42
"""

import argparse
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np

# Add parent directory to path to import src utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

from ctqw_matching_decomp.utils.graph.properties import is_connected


def get_hamming_distance(u, v):
    """Calculate Hamming distance between two integers."""
    return bin(u ^ v).count("1")


def generate_erdos_renyi_graphs(n_vertices, n_graphs, p, seed=None, verbose=True):
    """
    Generate Erdős-Rényi random graphs G(n, p).

    Args:
        n_vertices: Number of vertices (must be power of 2)
        n_graphs: Number of graphs to generate
        p: Edge probability (0 to 1)
        seed: Random seed for reproducibility
        verbose: Print progress info

    Returns:
        List of NetworkX graphs
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    graphs = []
    seen_hashes = set()
    attempts = 0
    max_attempts = n_graphs * 100

    n_bits = int(np.log2(n_vertices))

    if verbose:
        print(f"Generating {n_graphs} Erdős-Rényi G({n_vertices}, {p}) graphs")

    while len(graphs) < n_graphs and attempts < max_attempts:
        attempts += 1

        # Generate random graph
        G = nx.erdos_renyi_graph(n_vertices, p, seed=random.randint(0, 2**31))

        # Skip empty graphs
        if G.number_of_edges() == 0:
            continue

        # Check uniqueness using graph hash
        try:
            h = nx.weisfeiler_lehman_graph_hash(G)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
        except:
            pass

        graphs.append(G)

        if verbose and len(graphs) % 20 == 0:
            print(f"  Generated {len(graphs)}/{n_graphs} graphs...")

    if len(graphs) < n_graphs and verbose:
        print(f"Note: Only generated {len(graphs)} unique graphs after {attempts} attempts")

    return graphs[:n_graphs]


def print_statistics(graphs, n_vertices, p, verbose=True):
    """Print statistics about generated graphs."""
    if not verbose or not graphs:
        return

    n_bits = int(np.log2(n_vertices))

    # Basic stats
    connected = sum(1 for G in graphs if is_connected(G))
    bipartite = sum(1 for G in graphs if nx.is_bipartite(G))

    edge_counts = [G.number_of_edges() for G in graphs]
    avg_edges = np.mean(edge_counts)
    min_edges = min(edge_counts)
    max_edges = max(edge_counts)

    # Expected edges for G(n, p)
    max_possible = n_vertices * (n_vertices - 1) // 2
    expected_edges = p * max_possible

    # H distribution totals
    h_totals = {}
    for G in graphs:
        for u, v in G.edges():
            hd = get_hamming_distance(u, v)
            h_totals[hd] = h_totals.get(hd, 0) + 1

    total_edges = sum(h_totals.values())

    print(f"\nStatistics (n={len(graphs)} graphs, p={p}):")
    print(f"  Connected: {connected}/{len(graphs)} ({100*connected/len(graphs):.1f}%)")
    print(f"  Bipartite: {bipartite}/{len(graphs)}")
    print(f"  Edges: min={min_edges}, avg={avg_edges:.1f}, max={max_edges} (expected: {expected_edges:.1f})")
    if total_edges > 0:
        h_dist = ", ".join(
            [f"H{h}: {count/total_edges*100:.1f}%" for h, count in sorted(h_totals.items())]
        )
        print(f"  H distribution: {h_dist}")


def save_graphs(graphs, output_path, verbose=True):
    """Save graphs in graph6 format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for G in graphs:
            f.write(nx.to_graph6_bytes(G, header=False).decode("ascii"))

    if verbose:
        print(f"Saved {len(graphs)} graphs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Erdős-Rényi random graphs G(n, p)"
    )
    parser.add_argument(
        "-n", "--n-graphs", type=int, default=100, help="Number of graphs to generate"
    )
    parser.add_argument(
        "--vertices", type=int, nargs="+", default=[8, 16, 32, 64, 128], help="Vertex count(s) to generate"
    )
    parser.add_argument(
        "-p", "--probability", type=float, nargs="+", default=[0.1],
        help="Edge probability (can specify multiple values)"
    )
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true", default=True, help="Verbose output")

    args = parser.parse_args()

    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "graphs" / "erdos_renyi"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ERDŐS-RÉNYI RANDOM GRAPH GENERATOR")
    print("=" * 60)
    print(f"Model: G(n, p) - each edge included with probability p")
    print(f"Probabilities: {args.probability}")
    print(f"Vertices: {args.vertices}")
    print("=" * 60)

    for n_vertices in args.vertices:
        if n_vertices <= 0 or (n_vertices & (n_vertices - 1)) != 0:
            print(f"Error: {n_vertices} is not a power of 2")
            continue

        for p in args.probability:
            print(f"\n" + "-" * 60)
            print(f"Generating G({n_vertices}, {p})")

            # Use different seed for each (n, p) combination
            seed = args.seed + int(p * 1000) + n_vertices if args.seed else None

            graphs = generate_erdos_renyi_graphs(
                n_vertices, args.n_graphs, p, seed=seed, verbose=args.verbose
            )

            if not graphs:
                print(f"No graphs generated for n={n_vertices}, p={p}")
                continue

            print_statistics(graphs, n_vertices, p, args.verbose)

            # Create filename with probability (replace . with _)
            p_str = f"{p:.2f}".replace(".", "_")
            output_file = output_dir / f"{len(graphs)}graph_erdos_renyi_p{p_str}_{n_vertices}v.g6"
            save_graphs(graphs, output_file, args.verbose)

    print("\n" + "=" * 60)
    print("Generation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
