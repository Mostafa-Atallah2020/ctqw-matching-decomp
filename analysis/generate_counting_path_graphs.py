#!/usr/bin/env python3
"""
Counting Path Graph Generator for Connected Graphs where Matching Wins.

The winning pattern is a Hamiltonian path in binary counting order:
  0 -> 1 -> 2 -> 3 -> ... -> n-1

This creates edges with XOR pattern:
  H dist: {1: 2^(n-1), 2: 2^(n-2), 3: 2^(n-3), ..., n: 1}

Generates variations by:
1. Permuting the counting order (using different starting points/orderings)
2. Adding small perturbations to the path
3. Using shifted counting patterns

Usage:
  python generate_counting_path_graphs.py --vertices 8 16 32 64 128 -n 200
"""

import argparse
import random
import sys
from pathlib import Path
from itertools import permutations

import networkx as nx
import numpy as np

# Add parent directory to path to import src utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.graph.properties import is_connected


def get_hamming_distance(u, v):
    """Calculate Hamming distance between two integers."""
    return bin(u ^ v).count('1')


def create_counting_path(n_vertices, offset=0, reverse=False):
    """
    Create a counting path: 0-1-2-3-...-n-1 (with optional offset and reverse).
    """
    n_bits = int(np.log2(n_vertices))
    order = list(range(n_vertices))

    # Apply offset (rotation)
    order = order[offset:] + order[:offset]

    # Apply reverse
    if reverse:
        order = order[::-1]

    edges = set()
    for i in range(len(order) - 1):
        u = format(order[i], f'0{n_bits}b')
        v = format(order[i + 1], f'0{n_bits}b')
        edges.add((min(u, v), max(u, v)))  # Canonical order

    return edges


def create_xor_permuted_path(n_vertices, xor_mask, seed=None):
    """
    Create a path where vertices are XOR'd with a mask.
    This permutes which vertices are connected while maintaining structure.
    """
    if seed is not None:
        random.seed(seed)

    n_bits = int(np.log2(n_vertices))

    # XOR all vertex numbers with the mask
    permuted = [(i ^ xor_mask) for i in range(n_vertices)]

    # Create path in this permuted order
    edges = set()
    for i in range(n_vertices - 1):
        u = format(permuted[i], f'0{n_bits}b')
        v = format(permuted[i + 1], f'0{n_bits}b')
        edges.add((min(u, v), max(u, v)))

    return edges


def create_perturbed_path(n_vertices, n_swaps=1, seed=None):
    """
    Create a counting path with random adjacent swaps.
    """
    if seed is not None:
        random.seed(seed)

    n_bits = int(np.log2(n_vertices))
    order = list(range(n_vertices))

    # Perform random adjacent swaps
    for _ in range(n_swaps):
        if len(order) >= 3:
            i = random.randint(1, len(order) - 2)
            order[i], order[i + 1] = order[i + 1], order[i]

    edges = set()
    for i in range(len(order) - 1):
        u = format(order[i], f'0{n_bits}b')
        v = format(order[i + 1], f'0{n_bits}b')
        edges.add((min(u, v), max(u, v)))

    return edges


def create_segment_reversed_path(n_vertices, segment_size=4, seed=None):
    """
    Create a path where segments are reversed.
    """
    if seed is not None:
        random.seed(seed)

    n_bits = int(np.log2(n_vertices))
    order = list(range(n_vertices))

    # Reverse some segments
    for start in range(0, n_vertices, segment_size):
        end = min(start + segment_size, n_vertices)
        if random.random() > 0.5:
            order[start:end] = reversed(order[start:end])

    edges = set()
    for i in range(len(order) - 1):
        u = format(order[i], f'0{n_bits}b')
        v = format(order[i + 1], f'0{n_bits}b')
        edges.add((min(u, v), max(u, v)))

    return edges


def edges_to_graph(edges, n_vertices):
    """Convert edge set to NetworkX graph."""
    n_bits = int(np.log2(n_vertices))
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))
    for u, v in edges:
        G.add_edge(int(u, 2), int(v, 2))
    return G


def compute_metrics(edges, n_vertices):
    """Compute metrics for a graph."""
    n_bits = int(np.log2(n_vertices))
    n_edges = len(edges)

    if n_edges == 0:
        return None, False

    G = edges_to_graph(edges, n_vertices)

    # Must be connected
    if not is_connected(G):
        return None, False

    # H distribution
    h_dist = {}
    for u, v in edges:
        hd = sum(a != b for a, b in zip(u, v))
        h_dist[hd] = h_dist.get(hd, 0) + 1

    h1_ratio = h_dist.get(1, 0) / n_edges if n_edges > 0 else 0

    metrics = {
        'n_edges': n_edges,
        'h1_ratio': h1_ratio,
        'h_dist': h_dist,
        'bipartite': nx.is_bipartite(G),
        'diameter': nx.diameter(G) if is_connected(G) else -1
    }

    # Valid if connected and Hamiltonian path size
    valid = is_connected(G) and n_edges == n_vertices - 1

    return metrics, valid


def create_gray_code_path(n_vertices, seed=None):
    """
    Create a Gray code path - each step flips exactly one bit.
    This is a Hamiltonian path on the hypercube with all H1 edges.
    """
    n_bits = int(np.log2(n_vertices))

    # Generate Gray code sequence
    gray_order = []
    for i in range(n_vertices):
        gray = i ^ (i >> 1)  # Convert to Gray code
        gray_order.append(gray)

    edges = set()
    for i in range(len(gray_order) - 1):
        u = format(gray_order[i], f'0{n_bits}b')
        v = format(gray_order[i + 1], f'0{n_bits}b')
        edges.add((min(u, v), max(u, v)))

    return edges


def create_random_hamiltonian_path(n_vertices, seed=None):
    """
    Create a random Hamiltonian path visiting all vertices.
    Uses random neighbor selection to build a path.
    """
    if seed is not None:
        random.seed(seed)

    n_bits = int(np.log2(n_vertices))

    # Start from random vertex
    start = random.randint(0, n_vertices - 1)
    visited = {start}
    path = [start]

    # Try to extend path
    while len(path) < n_vertices:
        current = path[-1]
        # Find all unvisited neighbors (at any Hamming distance)
        candidates = [v for v in range(n_vertices) if v not in visited]
        if not candidates:
            break
        # Pick random unvisited vertex
        next_v = random.choice(candidates)
        path.append(next_v)
        visited.add(next_v)

    if len(path) < n_vertices:
        return None

    edges = set()
    for i in range(len(path) - 1):
        u = format(path[i], f'0{n_bits}b')
        v = format(path[i + 1], f'0{n_bits}b')
        edges.add((min(u, v), max(u, v)))

    return edges


def create_path_with_extra_edges(n_vertices, base_path_edges, n_extra=1, seed=None):
    """
    Take a base path and add extra edges to create cycles.
    This maintains connectivity while adding variety.
    """
    if seed is not None:
        random.seed(seed)

    n_bits = int(np.log2(n_vertices))
    edges = set(base_path_edges)

    # Get all possible edges not in path
    all_edges = []
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            u = format(i, f'0{n_bits}b')
            v = format(j, f'0{n_bits}b')
            edge = (min(u, v), max(u, v))
            if edge not in edges:
                all_edges.append(edge)

    random.shuffle(all_edges)

    # Add extra edges
    for i in range(min(n_extra, len(all_edges))):
        edges.add(all_edges[i])

    return edges


def create_bit_reversed_path(n_vertices, seed=None):
    """
    Create path where vertices are visited in bit-reversed order.
    e.g., for 8 vertices: 0,4,2,6,1,5,3,7
    """
    n_bits = int(np.log2(n_vertices))

    def bit_reverse(x, bits):
        result = 0
        for _ in range(bits):
            result = (result << 1) | (x & 1)
            x >>= 1
        return result

    order = [bit_reverse(i, n_bits) for i in range(n_vertices)]

    edges = set()
    for i in range(len(order) - 1):
        u = format(order[i], f'0{n_bits}b')
        v = format(order[i + 1], f'0{n_bits}b')
        edges.add((min(u, v), max(u, v)))

    return edges


def create_partial_counting_path(n_vertices, path_length, start=0, seed=None):
    """
    Create a partial counting path of given length.
    Starts at 'start' and visits 'path_length' consecutive vertices.
    NOTE: This creates disconnected graphs if path_length < n_vertices.
    """
    if seed is not None:
        random.seed(seed)

    n_bits = int(np.log2(n_vertices))

    # Create path of specified length
    vertices = []
    current = start
    for _ in range(path_length):
        vertices.append(current)
        current = (current + 1) % n_vertices

    edges = set()
    for i in range(len(vertices) - 1):
        u = format(vertices[i], f'0{n_bits}b')
        v = format(vertices[i + 1], f'0{n_bits}b')
        edges.add((min(u, v), max(u, v)))

    return edges


def create_multi_component_counting_path(n_vertices, n_components, seed=None):
    """
    Create multiple disjoint counting paths.
    Each component is a consecutive counting segment.
    """
    if seed is not None:
        random.seed(seed)

    n_bits = int(np.log2(n_vertices))

    # Split vertices into n_components groups
    vertices_per_comp = n_vertices // n_components
    extra = n_vertices % n_components

    edges = set()
    start = 0
    for i in range(n_components):
        comp_size = vertices_per_comp + (1 if i < extra else 0)
        for j in range(comp_size - 1):
            u = format(start + j, f'0{n_bits}b')
            v = format(start + j + 1, f'0{n_bits}b')
            edges.add((min(u, v), max(u, v)))
        start += comp_size

    return edges


def generate_graphs(n_vertices, n_graphs, seed=None, verbose=True):
    """Generate multiple connected path-based graphs with varying structures."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    graphs = []
    seen_hashes = set()

    n_bits = int(np.log2(n_vertices))

    if verbose:
        print(f"Generating {n_graphs} CONNECTED PATH graphs for {n_vertices} vertices ({n_bits} qubits)")
        print(f"Strategies: counting paths, Gray code, random Hamiltonian, paths + extra edges")

    def add_graph(edges):
        """Helper to add a graph if unique and connected."""
        G = edges_to_graph(edges, n_vertices)
        if not is_connected(G):
            return False
        try:
            h = nx.weisfeiler_lehman_graph_hash(G)
            if h not in seen_hashes:
                graphs.append(G)
                seen_hashes.add(h)
                return True
        except:
            pass
        return False

    # 1. Base counting path (0->1->2->...->n-1)
    base_edges = create_counting_path(n_vertices)
    add_graph(base_edges)

    # 2. Gray code path (all H1 edges)
    gray_edges = create_gray_code_path(n_vertices)
    add_graph(gray_edges)

    # 3. Bit-reversed path
    bit_rev_edges = create_bit_reversed_path(n_vertices)
    add_graph(bit_rev_edges)

    # 4. XOR-permuted counting paths (Hamiltonian)
    for xor_mask in range(1, n_vertices):
        if len(graphs) >= n_graphs:
            break
        edges = create_xor_permuted_path(n_vertices, xor_mask)
        add_graph(edges)

    # 5. Base paths with extra edges (creates cycles)
    base_paths = [base_edges, gray_edges, bit_rev_edges]
    for base in base_paths:
        for n_extra in range(1, min(n_bits + 1, 6)):
            if len(graphs) >= n_graphs:
                break
            for _ in range(5):  # Try multiple random extra edge sets
                edges = create_path_with_extra_edges(n_vertices, base, n_extra)
                add_graph(edges)

    # 6. Random Hamiltonian paths
    attempts = 0
    while len(graphs) < n_graphs and attempts < n_graphs * 20:
        attempts += 1
        edges = create_random_hamiltonian_path(n_vertices)
        if edges:
            add_graph(edges)

    # 7. Random Hamiltonian paths with extra edges
    attempts = 0
    while len(graphs) < n_graphs and attempts < n_graphs * 30:
        attempts += 1
        base = create_random_hamiltonian_path(n_vertices)
        if base:
            n_extra = random.randint(1, min(n_bits, 5))
            edges = create_path_with_extra_edges(n_vertices, base, n_extra)
            add_graph(edges)

    if len(graphs) < n_graphs and verbose:
        print(f"Note: Generated {len(graphs)} unique graphs")

    return graphs[:n_graphs]


def print_statistics(graphs, n_vertices, verbose=True):
    """Print statistics about generated graphs."""
    if not verbose or not graphs:
        return

    n_bits = int(np.log2(n_vertices))

    # Check all connected
    connected = sum(1 for G in graphs if is_connected(G))
    bipartite = sum(1 for G in graphs if nx.is_bipartite(G))

    # H distribution totals
    h_totals = {}
    for G in graphs:
        for u, v in G.edges():
            hd = get_hamming_distance(u, v)
            h_totals[hd] = h_totals.get(hd, 0) + 1

    total_edges = sum(h_totals.values())

    print(f"\nStatistics (n={len(graphs)} graphs):")
    print(f"  Connected: {connected}/{len(graphs)}")
    print(f"  Bipartite: {bipartite}/{len(graphs)}")
    print(f"  Edges per graph: {total_edges / len(graphs):.1f}")
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
        description='Generate COUNTING PATH connected graphs for Matching win'
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
    print("COUNTING PATH GRAPH GENERATOR")
    print("=" * 60)
    print("Pattern: Hamiltonian path in binary counting order")
    print("  0 -> 1 -> 2 -> ... -> n-1")
    print("Properties:")
    print("  - CONNECTED (single component)")
    print("  - Exactly n-1 edges (spanning path)")
    print("  - H dist: H1:50%, H2:25%, H3:12.5%, ...")
    print("  - MATCHING WINS for this structure!")
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

        output_file = output_dir / f"{len(graphs)}graph_counting_{n_vertices:d}v.g6"
        save_graphs(graphs, output_file, args.verbose)

    print("\n" + "=" * 60)
    print("Generation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
