#!/usr/bin/env python3
"""
Graph Generation Script

This script generates random connected graphs for quantum walk operator difference norm analysis.
Generated graphs are saved in graph6 format to ../data/graphs directory.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np


def generate_test_graphs(n_vertices, n_graphs=100, seed=None):
    """Generate test graphs ensuring they are connected and non-isomorphic."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    graphs = []
    attempts = 0
    max_attempts = n_graphs * 10  # Prevent infinite loops

    print(f"Generating {n_graphs} connected, non-isomorphic graphs with {n_vertices} vertices...")

    while len(graphs) < n_graphs and attempts < max_attempts:
        attempts += 1

        # Generate random graph with random number of edges
        min_edges = n_vertices - 1  # Minimum for connectivity
        max_edges = n_vertices * (n_vertices - 1) // 2  # Complete graph
        n_edges = random.randint(min_edges, max_edges)

        g = nx.gnm_random_graph(n_vertices, n_edges)

        # Check if connected
        if not nx.is_connected(g):
            continue

        # Check if isomorphic to any existing graph (expensive for large sets)
        is_duplicate = False
        if len(graphs) < 50:  # Only check isomorphism for smaller sets
            for existing_graph in graphs:
                if nx.is_isomorphic(g, existing_graph):
                    is_duplicate = True
                    break

        if not is_duplicate:
            graphs.append(g)
            if len(graphs) % 10 == 0:
                print(f"  Generated {len(graphs)}/{n_graphs} graphs...")

    if len(graphs) < n_graphs:
        print(
            f"Warning: Only generated {len(graphs)} graphs out of {n_graphs} requested after {attempts} attempts"
        )

    return graphs


def save_graphs(graphs, n_vertices, n_graphs, output_dir, verbose=True):
    """Save graphs in graph6 format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{len(graphs)}graph_random_{n_vertices}c.g6"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        for graph in graphs:
            f.write(nx.to_graph6_bytes(graph, header=False).decode("ascii"))

    if verbose:
        print(f"Saved {len(graphs)} graphs to {filepath}")
    return filepath


def main():
    # Set default directory based on script location in ./analysis
    script_dir = os.path.dirname(__file__)
    default_output_dir = os.path.join(script_dir, "..", "data", "graphs")

    parser = argparse.ArgumentParser(
        description="Generate random connected graphs for quantum walk operator difference norm analysis"
    )

    # Graph generation parameters
    parser.add_argument(
        "--vertex-sizes",
        nargs="+",
        type=int,
        default=[8, 16, 32],
        help="List of vertex sizes for graphs (default: 8 16 32)",
    )
    parser.add_argument(
        "--n-graphs",
        type=int,
        default=100,
        help="Number of graphs to generate per vertex size (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # Output directory
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help=f"Directory to save generated graphs (default: {default_output_dir})",
    )

    # Control options
    parser.add_argument("--force", action="store_true", help="Overwrite existing graph files")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        print(f"Graph Generation Configuration:")
        print(f"  Vertex sizes: {args.vertex_sizes}")
        print(f"  Number of graphs per size: {args.n_graphs}")
        print(f"  Seed: {args.seed}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Force overwrite: {args.force}")

    for n_vertices in args.vertex_sizes:
        print(f"\n{'='*60}")
        print(f"Generating graphs with {n_vertices} vertices")
        print(f"{'='*60}")

        # Check if file already exists
        output_dir = Path(args.output_dir)
        expected_filename = f"{args.n_graphs}graph_random_{n_vertices}c.g6"
        expected_filepath = output_dir / expected_filename

        if expected_filepath.exists() and not args.force:
            print(f"Graph file {expected_filepath} already exists. Use --force to overwrite.")
            continue

        # Generate graphs
        graphs = generate_test_graphs(n_vertices, args.n_graphs, args.seed)

        # Save graphs
        saved_filepath = save_graphs(
            graphs, n_vertices, args.n_graphs, args.output_dir, args.verbose
        )

        if args.verbose:
            print(f"Successfully generated and saved {len(graphs)} graphs")

            # Print some basic statistics
            total_edges = sum(len(g.edges()) for g in graphs)
            avg_edges = total_edges / len(graphs) if graphs else 0
            min_edges = min(len(g.edges()) for g in graphs) if graphs else 0
            max_edges = max(len(g.edges()) for g in graphs) if graphs else 0

            print(f"Graph statistics:")
            print(f"  Average edges: {avg_edges:.1f}")
            print(f"  Edge range: {min_edges} - {max_edges}")

    if args.verbose:
        print(f"\nGraph generation complete! All graphs saved to {args.output_dir}")
        print("Ready for operator difference norm analysis!")


if __name__ == "__main__":
    main()
