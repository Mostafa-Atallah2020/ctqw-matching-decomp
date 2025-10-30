#!/usr/bin/env python3
"""
Graph Generation Script

This script generates random connected graphs for quantum walk operator difference norm analysis.
Generated graphs are saved in graph6 format to ../data/graphs directory.
Supports regular graphs and sparse bipartite graphs.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np


def generate_sparse_bipartite_graph(n_vertices, sparsity_factor=0.3, seed=None):
    """
    Generate a sparse bipartite graph.

    Args:
        n_vertices: Total number of vertices
        sparsity_factor: Controls edge density (0 < sparsity_factor < 1)
                        Lower values = sparser graphs
        seed: Random seed for reproducibility

    Returns:
        NetworkX graph object
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Split vertices into two sets for bipartite structure
    n1 = n_vertices // 2
    n2 = n_vertices - n1

    # Create bipartite graph with specified sparsity
    # Maximum possible edges in bipartite graph is n1 * n2
    max_edges = n1 * n2

    # Calculate number of edges based on sparsity factor
    # Ensure minimum connectivity (at least n_vertices - 1 edges)
    min_edges = max(n_vertices - 1, int(max_edges * 0.1))  # At least 10% or connectivity minimum
    target_edges = max(min_edges, int(max_edges * sparsity_factor))

    # Generate random bipartite graph
    g = nx.bipartite.random_graph(n1, n2, target_edges / max_edges, seed=seed)

    # Ensure connectivity by adding edges if necessary
    if not nx.is_connected(g):
        # Get the two bipartite sets
        top_nodes = {n for n, d in g.nodes(data=True) if d["bipartite"] == 0}
        bottom_nodes = set(g) - top_nodes

        # Find connected components
        components = list(nx.connected_components(g))

        # Connect components by adding minimal edges
        for i in range(len(components) - 1):
            # Find a node from each bipartite set in consecutive components
            comp1 = components[i]
            comp2 = components[i + 1]

            # Find nodes from different bipartite sets to connect
            comp1_top = comp1.intersection(top_nodes)
            comp1_bottom = comp1.intersection(bottom_nodes)
            comp2_top = comp2.intersection(top_nodes)
            comp2_bottom = comp2.intersection(bottom_nodes)

            # Add edge between different bipartite sets
            if comp1_top and comp2_bottom:
                g.add_edge(random.choice(list(comp1_top)), random.choice(list(comp2_bottom)))
            elif comp1_bottom and comp2_top:
                g.add_edge(random.choice(list(comp1_bottom)), random.choice(list(comp2_top)))

    return g


def is_bipartite_isomorphic(g1, g2):
    """Check if two bipartite graphs are isomorphic, respecting bipartite structure."""
    if not (nx.is_bipartite(g1) and nx.is_bipartite(g2)):
        return False

    # Get bipartite sets for both graphs
    try:
        top1 = {n for n, d in g1.nodes(data=True) if d.get("bipartite") == 0}
        bottom1 = set(g1.nodes()) - top1
        top2 = {n for n, d in g2.nodes(data=True) if d.get("bipartite") == 0}
        bottom2 = set(g2.nodes()) - top2

        # Check if set sizes match
        if len(top1) != len(top2) or len(bottom1) != len(bottom2):
            return False

        # Use NetworkX's bipartite isomorphism matcher
        from networkx.algorithms import isomorphism

        matcher = isomorphism.GraphMatcher(g1, g2)
        return matcher.is_isomorphic()

    except:
        # Fallback to regular isomorphism check
        return nx.is_isomorphic(g1, g2)


def generate_canonical_hash(graph, graph_type="random"):
    """Generate a canonical hash for faster duplicate detection."""
    try:
        if graph_type == "bipartite" and nx.is_bipartite(graph):
            # For bipartite graphs, create a more specific hash
            top_nodes = {n for n, d in graph.nodes(data=True) if d.get("bipartite") == 0}
            bottom_nodes = set(graph.nodes()) - top_nodes

            # Create degree sequences for each bipartite set
            top_degrees = sorted([graph.degree(n) for n in top_nodes])
            bottom_degrees = sorted([graph.degree(n) for n in bottom_nodes])

            # Include bipartite set sizes and degree sequences
            return (
                len(top_nodes),
                len(bottom_nodes),
                tuple(top_degrees),
                tuple(bottom_degrees),
                graph.number_of_edges(),
            )
        else:
            # For regular graphs, use degree sequence and basic properties
            degree_sequence = sorted([d for n, d in graph.degree()])
            return (graph.number_of_nodes(), graph.number_of_edges(), tuple(degree_sequence))
    except:
        # Fallback hash
        return (graph.number_of_nodes(), graph.number_of_edges())


def generate_test_graphs(
    n_vertices, n_graphs=100, seed=None, graph_type="random", sparsity_factor=0.3
):
    """Generate test graphs ensuring they are connected and non-isomorphic."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    graphs = []
    graph_hashes = set()  # For fast preliminary duplicate detection
    attempts = 0
    max_attempts = n_graphs * 20  # Increased attempts for better success rate
    isomorphism_check_limit = 100  # Check isomorphism for more graphs

    if graph_type == "bipartite":
        print(
            f"Generating {n_graphs} connected, non-isomorphic sparse bipartite graphs with {n_vertices} vertices (sparsity={sparsity_factor})..."
        )
    else:
        print(
            f"Generating {n_graphs} connected, non-isomorphic graphs with {n_vertices} vertices..."
        )

    while len(graphs) < n_graphs and attempts < max_attempts:
        attempts += 1

        if graph_type == "bipartite":
            g = generate_sparse_bipartite_graph(n_vertices, sparsity_factor, seed)
        else:
            # Generate random graph with random number of edges
            min_edges = n_vertices - 1  # Minimum for connectivity
            max_edges = n_vertices * (n_vertices - 1) // 2  # Complete graph
            n_edges = random.randint(min_edges, max_edges)
            g = nx.gnm_random_graph(n_vertices, n_edges)

        # Check if connected
        if not nx.is_connected(g):
            continue

        # For bipartite graphs, verify bipartite property
        if graph_type == "bipartite" and not nx.is_bipartite(g):
            continue

        # Fast hash-based duplicate detection
        graph_hash = generate_canonical_hash(g, graph_type)
        if graph_hash in graph_hashes:
            continue

        # Detailed isomorphism checking for smaller collections
        is_duplicate = False
        if len(graphs) < isomorphism_check_limit:
            for existing_graph in graphs:
                if graph_type == "bipartite":
                    if is_bipartite_isomorphic(g, existing_graph):
                        is_duplicate = True
                        break
                else:
                    if nx.is_isomorphic(g, existing_graph):
                        is_duplicate = True
                        break

        if not is_duplicate:
            graphs.append(g)
            graph_hashes.add(graph_hash)
            if len(graphs) % 10 == 0:
                print(f"  Generated {len(graphs)}/{n_graphs} graphs (attempts: {attempts})...")

    if len(graphs) < n_graphs:
        print(
            f"Warning: Only generated {len(graphs)} graphs out of {n_graphs} requested after {attempts} attempts"
        )
        print(f"Consider:")
        print(f"  - Reducing the number of requested graphs")
        print(f"  - Increasing vertex count (more possible unique graphs)")
        if graph_type == "bipartite":
            print(f"  - Adjusting sparsity factor (current: {sparsity_factor})")

    return graphs


def save_graphs(
    graphs,
    n_vertices,
    n_graphs,
    output_dir,
    graph_type="random",
    sparsity_factor=None,
    verbose=True,
):
    """Save graphs in graph6 format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{len(graphs)}graph_{graph_type}_{n_vertices}c.g6"

    filepath = output_dir / filename

    with open(filepath, "w") as f:
        for graph in graphs:
            f.write(nx.to_graph6_bytes(graph, header=False).decode("ascii"))

    if verbose:
        print(f"Saved {len(graphs)} graphs to {filepath}")
    return filepath


def print_graph_statistics(graphs, graph_type="random", verbose=True):
    """Print statistics about generated graphs."""
    if not graphs or not verbose:
        return

    total_edges = sum(len(g.edges()) for g in graphs)
    avg_edges = total_edges / len(graphs) if graphs else 0
    min_edges = min(len(g.edges()) for g in graphs) if graphs else 0
    max_edges = max(len(g.edges()) for g in graphs) if graphs else 0

    print(f"Graph statistics:")
    print(f"  Average edges: {avg_edges:.1f}")
    print(f"  Edge range: {min_edges} - {max_edges}")

    if graph_type == "bipartite":
        # Additional bipartite-specific statistics
        bipartite_sets = []
        for g in graphs:
            if nx.is_bipartite(g):
                top_nodes = {n for n, d in g.nodes(data=True) if d.get("bipartite") == 0}
                bottom_nodes = set(g) - top_nodes
                bipartite_sets.append((len(top_nodes), len(bottom_nodes)))

        if bipartite_sets:
            avg_set1 = sum(s[0] for s in bipartite_sets) / len(bipartite_sets)
            avg_set2 = sum(s[1] for s in bipartite_sets) / len(bipartite_sets)
            print(f"  Average bipartite set sizes: {avg_set1:.1f} | {avg_set2:.1f}")


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

    # Graph type parameters
    parser.add_argument(
        "--graph-type",
        type=str,
        choices=["random", "bipartite"],
        default="random",
        help="Type of graphs to generate (default: random)",
    )
    parser.add_argument(
        "--sparsity-factor",
        type=float,
        default=0.3,
        help="Sparsity factor for bipartite graphs (0 < factor < 1, lower = sparser) (default: 0.3)",
    )

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

    # Validate sparsity factor
    if args.graph_type == "bipartite" and not (0 < args.sparsity_factor < 1):
        print("Error: sparsity-factor must be between 0 and 1 for bipartite graphs")
        sys.exit(1)

    if args.verbose:
        print(f"Graph Generation Configuration:")
        print(f"  Graph type: {args.graph_type}")
        if args.graph_type == "bipartite":
            print(f"  Sparsity factor: {args.sparsity_factor}")
        print(f"  Vertex sizes: {args.vertex_sizes}")
        print(f"  Number of graphs per size: {args.n_graphs}")
        print(f"  Seed: {args.seed}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Force overwrite: {args.force}")

    for n_vertices in args.vertex_sizes:
        print(f"\n{'='*60}")
        if args.graph_type == "bipartite":
            print(f"Generating sparse bipartite graphs with {n_vertices} vertices")
        else:
            print(f"Generating graphs with {n_vertices} vertices")
        print(f"{'='*60}")

        # Check if file already exists
        output_dir = Path(args.output_dir)
        if args.graph_type == "bipartite":
            sparsity_str = f"_s{args.sparsity_factor:.1f}".replace(".", "")
            expected_filename = (
                f"{args.n_graphs}graph_{args.graph_type}{sparsity_str}_{n_vertices}c.g6"
            )
        else:
            expected_filename = f"{args.n_graphs}graph_{args.graph_type}_{n_vertices}c.g6"
        expected_filepath = output_dir / expected_filename

        if expected_filepath.exists() and not args.force:
            print(f"Graph file {expected_filepath} already exists. Use --force to overwrite.")
            continue

        # Generate graphs
        graphs = generate_test_graphs(
            n_vertices, args.n_graphs, args.seed, args.graph_type, args.sparsity_factor
        )

        # Save graphs
        saved_filepath = save_graphs(
            graphs,
            n_vertices,
            args.n_graphs,
            args.output_dir,
            args.graph_type,
            args.sparsity_factor,
            args.verbose,
        )

        if args.verbose:
            print(f"Successfully generated and saved {len(graphs)} graphs")
            print_graph_statistics(graphs, args.graph_type, args.verbose)

    if args.verbose:
        print(f"\nGraph generation complete! All graphs saved to {args.output_dir}")
        print("Ready for operator difference norm analysis!")


if __name__ == "__main__":
    main()
