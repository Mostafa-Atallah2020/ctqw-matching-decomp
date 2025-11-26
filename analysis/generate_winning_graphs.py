#!/usr/bin/env python3
"""
Odd vs Even Qubit Graph Generator for Matching vs Pauli Win.

Based on analysis of verified winners:

EVEN QUBITS (4, 6, 8 → 16v, 64v, 256v):
  - Very sparse: 3-5 edges
  - Many components (highly disconnected)
  - LOW H1 ratio: 0-35% (H2/H3 dominate!)
  - Max degree: 1-2
  - NO subspace clustering - edges across full hypercube

ODD QUBITS (3, 5, 7 → 8v, 32v, 128v):
  - Sparse: 3 to n_bits edges
  - Disconnected: 3+ components
  - Mixed H1 ratio: 30-60%
  - Max degree: 2
  - NO subspace clustering - edges across full hypercube

Usage:
  python generate_odd_even.py --vertices 8 16 32 64 128 -n 200
"""

import argparse
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np


def get_hamming_distance(u, v):
    """Calculate Hamming distance between two integers."""
    return bin(u ^ v).count('1')


def get_all_edges_by_hamming(n_vertices):
    """Get all possible edges categorized by Hamming distance."""
    n_bits = int(np.log2(n_vertices))
    edges_by_h = {h: [] for h in range(1, n_bits + 1)}

    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            hd = get_hamming_distance(i, j)
            edges_by_h[hd].append((i, j))

    return edges_by_h


def generate_even_qubit_graph(n_vertices, seed=None):
    """
    Generate graph for EVEN qubit count (4, 6, 8... qubits).

    Pattern from verified 16v winners:
    - 3-5 edges ONLY
    - H1 ratio: 0-35% (H2/H3 dominate)
    - Many components (12-13 for 16v, scales with vertices)
    - Max degree: 1-2
    - NO subspace clustering
    """
    if seed is not None:
        random.seed(seed)

    n_bits = int(np.log2(n_vertices))

    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))

    edges_by_h = get_all_edges_by_hamming(n_vertices)

    # Shuffle all edge lists
    for h in edges_by_h:
        random.shuffle(edges_by_h[h])

    # Target: 3 to 2*n_bits edges (more variety)
    target_edges = random.randint(3, 2 * n_bits)

    # H weights: strongly prefer H2 and H3
    # H1: 10-20%, H2: 35-45%, H3: 30-40%, H4+: 10-15%
    h_weights = []
    for h in range(1, n_bits + 1):
        if h == 1:
            h_weights.append(0.15)
        elif h == 2:
            h_weights.append(0.40)
        elif h == 3:
            h_weights.append(0.35)
        else:
            h_weights.append(0.10 / max(1, n_bits - 3))  # Split remaining among higher H

    # Normalize weights
    total = sum(h_weights)
    h_weights = [w / total for w in h_weights]

    added_edges = []
    vertex_degrees = {v: 0 for v in range(n_vertices)}
    max_degree = random.choice([2, 2, 3])  # Allow higher degree for variety

    attempts = 0
    max_attempts = 1000

    while len(added_edges) < target_edges and attempts < max_attempts:
        attempts += 1

        # Choose Hamming distance based on weights
        h = random.choices(range(1, n_bits + 1), weights=h_weights)[0]

        if not edges_by_h[h]:
            continue

        # Pick random edge
        edge = edges_by_h[h].pop()
        u, v = edge

        if vertex_degrees[u] < max_degree and vertex_degrees[v] < max_degree:
            G.add_edge(u, v)
            added_edges.append(edge)
            vertex_degrees[u] += 1
            vertex_degrees[v] += 1

    return G


def generate_odd_qubit_graph(n_vertices, seed=None):
    """
    Generate graph for ODD qubit count (3, 5, 7... qubits).

    Pattern from verified 8v and 32v winners:
    - 3 to n_bits edges (sparse)
    - Mixed H1 ratio: 30-60%
    - Disconnected (3+ components)
    - Max degree: 2
    - NO subspace clustering
    """
    if seed is not None:
        random.seed(seed)

    n_bits = int(np.log2(n_vertices))

    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))

    edges_by_h = get_all_edges_by_hamming(n_vertices)

    # Shuffle all edge lists
    for h in edges_by_h:
        random.shuffle(edges_by_h[h])

    # Target: 3 to 2*n_bits edges (more variety)
    target_edges = random.randint(3, 2 * n_bits)

    # H weights: balanced mix, slightly favoring H1
    # H1: 35-45%, H2: 25-35%, H3+: 20-30%
    h_weights = []
    for h in range(1, n_bits + 1):
        if h == 1:
            h_weights.append(0.40)
        elif h == 2:
            h_weights.append(0.30)
        elif h == 3:
            h_weights.append(0.20)
        else:
            h_weights.append(0.10 / max(1, n_bits - 3))

    # Normalize weights
    total = sum(h_weights)
    h_weights = [w / total for w in h_weights]

    added_edges = []
    vertex_degrees = {v: 0 for v in range(n_vertices)}
    max_degree = random.choice([2, 2, 3])  # Allow higher degree for variety

    attempts = 0
    max_attempts = 1000

    while len(added_edges) < target_edges and attempts < max_attempts:
        attempts += 1

        # Choose Hamming distance based on weights
        h = random.choices(range(1, n_bits + 1), weights=h_weights)[0]

        if not edges_by_h[h]:
            continue

        # Pick random edge
        edge = edges_by_h[h].pop()
        u, v = edge

        if vertex_degrees[u] < max_degree and vertex_degrees[v] < max_degree:
            G.add_edge(u, v)
            added_edges.append(edge)
            vertex_degrees[u] += 1
            vertex_degrees[v] += 1

    return G


def generate_winning_graph(n_vertices, seed=None):
    """Generate a graph using odd/even qubit formula."""
    n_bits = int(np.log2(n_vertices))

    if n_bits % 2 == 0:  # Even qubits: 4, 6, 8...
        return generate_even_qubit_graph(n_vertices, seed)
    else:  # Odd qubits: 3, 5, 7...
        return generate_odd_qubit_graph(n_vertices, seed)


def compute_metrics(G, n_vertices):
    """Compute metrics and check if graph matches formula."""
    n_bits = int(np.log2(n_vertices))
    n_edges = G.number_of_edges()

    if n_edges == 0:
        return None, False

    # H1 ratio
    h1_count = sum(1 for u, v in G.edges() if get_hamming_distance(u, v) == 1)
    h1_ratio = h1_count / n_edges

    # Components
    n_components = nx.number_connected_components(G)
    component_ratio = n_components / n_vertices

    # Max degree
    max_deg = max(d for n, d in G.degree())

    metrics = {
        'n_edges': n_edges,
        'h1_ratio': h1_ratio,
        'n_components': n_components,
        'component_ratio': component_ratio,
        'max_degree': max_deg
    }

    # Validation based on odd/even - relaxed constraints for more variety
    if n_bits % 2 == 0:  # Even qubits
        valid = (
            3 <= n_edges <= 2 * n_bits and  # Allow more edges
            h1_ratio <= 0.6 and              # Relaxed H1 ratio
            component_ratio >= 0.4 and       # Relaxed component ratio
            max_deg <= 3                     # Allow degree 3
        )
    else:  # Odd qubits
        valid = (
            3 <= n_edges <= 2 * n_bits and  # Allow more edges
            component_ratio >= 0.2 and       # Relaxed component ratio
            max_deg <= 3                     # Allow degree 3
        )

    return metrics, valid


def generate_graphs(n_vertices, n_graphs, seed=None, verbose=True):
    """Generate multiple graphs using odd/even formula."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    graphs = []
    seen_hashes = set()
    attempts = 0
    max_attempts = n_graphs * 100

    n_bits = int(np.log2(n_vertices))
    is_even = n_bits % 2 == 0

    if verbose:
        print(f"Generating {n_graphs} graphs for {n_vertices} vertices ({n_bits} qubits)")
        if is_even:
            print(f"Using EVEN-qubit formula: 3-5 edges, H2/H3 dominated, highly disconnected")
        else:
            print(f"Using ODD-qubit formula: 3-{n_bits} edges, mixed H1/H2/H3, disconnected")

    while len(graphs) < n_graphs and attempts < max_attempts:
        attempts += 1

        G = generate_winning_graph(n_vertices)

        metrics, valid = compute_metrics(G, n_vertices)

        if not valid:
            continue

        # Check for duplicates
        try:
            graph_hash = nx.weisfeiler_lehman_graph_hash(G)
        except:
            graph_hash = (G.number_of_edges(), tuple(sorted(dict(G.degree()).values())))

        if graph_hash in seen_hashes:
            continue

        graphs.append(G)
        seen_hashes.add(graph_hash)

        if verbose and len(graphs) % 50 == 0:
            print(f"  Generated {len(graphs)}/{n_graphs}")

    if len(graphs) < n_graphs and verbose:
        print(f"Note: Generated {len(graphs)} graphs after {attempts} attempts")

    return graphs


def print_statistics(graphs, n_vertices, verbose=True):
    """Print statistics about generated graphs."""
    if not verbose or not graphs:
        return

    n_bits = int(np.log2(n_vertices))
    all_metrics = [compute_metrics(G, n_vertices)[0] for G in graphs]
    all_metrics = [m for m in all_metrics if m is not None]

    # Count H distribution
    h_totals = {}
    for G in graphs:
        for u, v in G.edges():
            hd = get_hamming_distance(u, v)
            h_totals[hd] = h_totals.get(hd, 0) + 1

    total_edges = sum(h_totals.values())

    print(f"\nStatistics (n={len(graphs)} graphs):")
    print(f"  Formula type: {'EVEN' if n_bits % 2 == 0 else 'ODD'}-qubit")
    print(f"  Edges: {np.mean([m['n_edges'] for m in all_metrics]):.1f}")
    print(f"  H1 ratio: {np.mean([m['h1_ratio'] for m in all_metrics]):.2f}")
    print(f"  Components: {np.mean([m['n_components'] for m in all_metrics]):.1f}")
    print(f"  Max degree: {np.mean([m['max_degree'] for m in all_metrics]):.1f}")
    if total_edges > 0:
        h_dist = ", ".join([f"H{h}: {count/total_edges*100:.0f}%" for h, count in sorted(h_totals.items())])
        print(f"  H distribution: {h_dist}")


def save_graphs(graphs, output_path, verbose=True):
    """Save graphs in graph6 format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for G in graphs:
            f.write(nx.to_graph6_bytes(G, header=False).decode('ascii'))

    if verbose:
        print(f"Saved {len(graphs)} graphs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate winning graphs using odd/even qubit formulae'
    )
    parser.add_argument('-n', '--n-graphs', type=int, default=200,
                        help='Number of graphs to generate')
    parser.add_argument('--vertices', type=int, nargs='+', default=[8, 16, 32],
                        help='Vertex count(s) to generate')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='Verbose output')

    args = parser.parse_args()

    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / 'graphs'
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ODD vs EVEN QUBIT WINNING GRAPH FORMULAE")
    print("=" * 60)
    print("EVEN qubits (4, 6, 8... -> 16v, 64v, 256v):")
    print("  - 3-5 edges, H2/H3 dominated, very disconnected")
    print("ODD qubits (3, 5, 7... -> 8v, 32v, 128v):")
    print("  - 3-n_bits edges, mixed H1/H2/H3, disconnected")
    print("NO subspace clustering for either formula")
    print("=" * 60)

    for n_vertices in args.vertices:
        if n_vertices <= 0 or (n_vertices & (n_vertices - 1)) != 0:
            print(f"Error: {n_vertices} is not a power of 2")
            continue

        print(f"\n" + "-" * 60)

        graphs = generate_graphs(
            n_vertices, args.n_graphs,
            seed=args.seed, verbose=args.verbose
        )

        if not graphs:
            print(f"No graphs generated for {n_vertices} vertices")
            continue

        print_statistics(graphs, n_vertices, args.verbose)

        n_bits = int(np.log2(n_vertices))
        formula_type = "even" if n_bits % 2 == 0 else "odd"
        output_file = output_dir / f"{len(graphs)}graph_{formula_type}_{n_vertices:d}v.g6"
        save_graphs(graphs, output_file, args.verbose)

    print("\n" + "=" * 60)
    print("Generation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
