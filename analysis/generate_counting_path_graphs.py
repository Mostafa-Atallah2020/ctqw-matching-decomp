#!/usr/bin/env python3
"""
H1-Maximizing Graph Generator for Connected Graphs where Matching Wins.

Based on analysis of win/lose patterns, winning graphs have:
  - HIGH hamming_1_count: 70%+ of edges are H1 (single bit flip)
  - LOW avg_hamming: ~1.5-1.6 vs 2.5-2.8 for losing
  - LOW clustering: near zero (avoid triangles)
  - Higher diameter: stretched out paths

Key strategies:
1. Gray code paths - ALL edges are H1 (optimal)
2. Hypercube subgraphs - only H1 edges
3. Sparse trees using only H1 edges
4. Avoid high-Hamming edges that favor Pauli

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
    return bin(u ^ v).count("1")


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
        u = format(order[i], f"0{n_bits}b")
        v = format(order[i + 1], f"0{n_bits}b")
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
        u = format(permuted[i], f"0{n_bits}b")
        v = format(permuted[i + 1], f"0{n_bits}b")
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
        u = format(order[i], f"0{n_bits}b")
        v = format(order[i + 1], f"0{n_bits}b")
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
        u = format(order[i], f"0{n_bits}b")
        v = format(order[i + 1], f"0{n_bits}b")
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
        "n_edges": n_edges,
        "h1_ratio": h1_ratio,
        "h_dist": h_dist,
        "bipartite": nx.is_bipartite(G),
        "diameter": nx.diameter(G) if is_connected(G) else -1,
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
        u = format(gray_order[i], f"0{n_bits}b")
        v = format(gray_order[i + 1], f"0{n_bits}b")
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
        u = format(path[i], f"0{n_bits}b")
        v = format(path[i + 1], f"0{n_bits}b")
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
            u = format(i, f"0{n_bits}b")
            v = format(j, f"0{n_bits}b")
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
        u = format(order[i], f"0{n_bits}b")
        v = format(order[i + 1], f"0{n_bits}b")
        edges.add((min(u, v), max(u, v)))

    return edges


def get_h1_neighbors(v, n_bits):
    """Get all vertices at Hamming distance 1 from v."""
    neighbors = []
    for i in range(n_bits):
        neighbor = v ^ (1 << i)
        neighbors.append(neighbor)
    return neighbors


def create_h1_spanning_tree(n_vertices, seed=None):
    """
    Create a spanning tree using ONLY H1 edges (hypercube edges).
    Uses BFS/DFS from a random start to build connected tree.
    """
    if seed is not None:
        random.seed(seed)

    n_bits = int(np.log2(n_vertices))

    # Start from random vertex
    start = random.randint(0, n_vertices - 1)
    visited = {start}
    edges = set()
    frontier = [start]

    while len(visited) < n_vertices and frontier:
        # Pick random vertex from frontier
        current = random.choice(frontier)

        # Get unvisited H1 neighbors
        h1_neighbors = get_h1_neighbors(current, n_bits)
        unvisited = [n for n in h1_neighbors if n not in visited]

        if unvisited:
            # Add random unvisited H1 neighbor
            next_v = random.choice(unvisited)
            visited.add(next_v)
            u_str = format(current, f"0{n_bits}b")
            v_str = format(next_v, f"0{n_bits}b")
            edges.add((min(u_str, v_str), max(u_str, v_str)))
            frontier.append(next_v)
        else:
            # Remove from frontier if no unvisited H1 neighbors
            frontier.remove(current)

    return edges if len(visited) == n_vertices else None


def create_h1_path_dfs(n_vertices, seed=None):
    """
    Try to create a Hamiltonian path using only H1 edges.
    This is a path on the hypercube graph.
    """
    if seed is not None:
        random.seed(seed)

    n_bits = int(np.log2(n_vertices))

    # Try multiple starting points
    for _ in range(10):
        start = random.randint(0, n_vertices - 1)
        visited = {start}
        path = [start]

        while len(path) < n_vertices:
            current = path[-1]
            h1_neighbors = get_h1_neighbors(current, n_bits)
            unvisited = [n for n in h1_neighbors if n not in visited]

            if not unvisited:
                break

            # Prefer neighbors with more unvisited H1 neighbors (Warnsdorff's rule)
            def count_exits(v):
                return len([n for n in get_h1_neighbors(v, n_bits) if n not in visited and n != v])

            unvisited.sort(key=count_exits)
            next_v = unvisited[0]  # Choose vertex with fewest exits

            path.append(next_v)
            visited.add(next_v)

        if len(path) == n_vertices:
            edges = set()
            for i in range(len(path) - 1):
                u = format(path[i], f"0{n_bits}b")
                v = format(path[i + 1], f"0{n_bits}b")
                edges.add((min(u, v), max(u, v)))
            return edges

    return None


def create_h1_tree_with_extras(n_vertices, n_extra=1, seed=None):
    """
    Create H1 spanning tree plus extra H1 edges.
    All edges are H1, creating cycles but maintaining low Hamming.
    """
    if seed is not None:
        random.seed(seed)

    n_bits = int(np.log2(n_vertices))

    # Get base H1 tree
    base_edges = create_h1_spanning_tree(n_vertices, seed)
    if base_edges is None:
        return None

    edges = set(base_edges)

    # Find all H1 edges not in tree
    available_h1 = []
    for v in range(n_vertices):
        for neighbor in get_h1_neighbors(v, n_bits):
            if neighbor > v:  # Avoid duplicates
                u_str = format(v, f"0{n_bits}b")
                v_str = format(neighbor, f"0{n_bits}b")
                edge = (min(u_str, v_str), max(u_str, v_str))
                if edge not in edges:
                    available_h1.append(edge)

    random.shuffle(available_h1)

    # Add extra H1 edges
    for i in range(min(n_extra, len(available_h1))):
        edges.add(available_h1[i])

    return edges


def create_shifted_gray_code(n_vertices, shift=0, seed=None):
    """
    Create Gray code path with bit rotation/shift.
    Still all H1 edges but visits vertices in different order.
    """
    n_bits = int(np.log2(n_vertices))

    # Generate Gray code with shift
    gray_order = []
    for i in range(n_vertices):
        gray = i ^ (i >> 1)
        # Apply circular bit shift
        shifted = ((gray << shift) | (gray >> (n_bits - shift))) & (n_vertices - 1)
        gray_order.append(shifted)

    edges = set()
    for i in range(len(gray_order) - 1):
        u = format(gray_order[i], f"0{n_bits}b")
        v = format(gray_order[i + 1], f"0{n_bits}b")
        edges.add((min(u, v), max(u, v)))

    return edges


def create_xor_gray_code(n_vertices, xor_mask, seed=None):
    """
    Create Gray code path with XOR transformation.
    Permutes vertices while maintaining H1 edge structure.
    """
    n_bits = int(np.log2(n_vertices))

    gray_order = []
    for i in range(n_vertices):
        gray = i ^ (i >> 1)
        transformed = gray ^ xor_mask
        gray_order.append(transformed)

    edges = set()
    for i in range(len(gray_order) - 1):
        u = format(gray_order[i], f"0{n_bits}b")
        v = format(gray_order[i + 1], f"0{n_bits}b")
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
        u = format(vertices[i], f"0{n_bits}b")
        v = format(vertices[i + 1], f"0{n_bits}b")
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
            u = format(start + j, f"0{n_bits}b")
            v = format(start + j + 1, f"0{n_bits}b")
            edges.add((min(u, v), max(u, v)))
        start += comp_size

    return edges


def generate_graphs(n_vertices, n_graphs, seed=None, verbose=True):
    """
    Generate multiple connected graphs optimized for Matching wins.

    KEY INSIGHT: Winning graphs follow the COUNTING PATH Hamming distribution:
      H1: 50%, H2: 25%, H3: 12.5%, H4: 6.25%, H5: 3.125%

    This is NOT 100% H1 - the mixed Hamming distances from counting are important!
    Pure H1 graphs (Gray code, hypercube subgraphs) actually LOSE to Pauli.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    graphs = []
    seen_hashes = set()

    n_bits = int(np.log2(n_vertices))

    if verbose:
        print(
            f"Generating {n_graphs} COUNTING PATH graphs for {n_vertices} vertices ({n_bits} qubits)"
        )
        print(f"Strategies: counting paths, XOR-permuted counting, perturbed counting")

    def add_graph(edges):
        """Helper to add a graph if unique and connected."""
        if edges is None:
            return False
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

    # ===========================================
    # TIER 1: Pure counting paths (winning pattern!)
    # H distribution: H1:50%, H2:25%, H3:12.5%, etc.
    # ===========================================

    # 1. Base counting path (0->1->2->...->n-1)
    base_edges = create_counting_path(n_vertices)
    add_graph(base_edges)

    # 2. XOR-permuted counting paths (different vertex ordering)
    for xor_mask in range(1, n_vertices):
        if len(graphs) >= n_graphs:
            break
        edges = create_xor_permuted_path(n_vertices, xor_mask)
        add_graph(edges)

    # 3. Offset counting paths
    for offset in range(1, n_vertices):
        if len(graphs) >= n_graphs:
            break
        edges = create_counting_path(n_vertices, offset=offset)
        add_graph(edges)

    # 4. Reversed counting paths
    for offset in range(n_vertices):
        if len(graphs) >= n_graphs:
            break
        edges = create_counting_path(n_vertices, offset=offset, reverse=True)
        add_graph(edges)

    # 5. Perturbed counting paths (small variations)
    for n_swaps in range(1, min(n_bits * 2, 15)):
        if len(graphs) >= n_graphs:
            break
        for _ in range(100):
            edges = create_perturbed_path(n_vertices, n_swaps, seed=random.randint(0, 10000000))
            add_graph(edges)

    # 6. Segment-reversed counting paths
    for seg_size in [2, 4, 8, 16, 32, 64]:
        if seg_size >= n_vertices:
            continue
        if len(graphs) >= n_graphs:
            break
        for _ in range(100):
            edges = create_segment_reversed_path(
                n_vertices, seg_size, seed=random.randint(0, 10000000)
            )
            add_graph(edges)

    # ===========================================
    # TIER 2: Counting paths with extra edges
    # ===========================================

    # 7. Counting paths with extra edges (more variety)
    if len(graphs) < n_graphs:
        for n_extra in range(1, min(n_bits, 6)):
            if len(graphs) >= n_graphs:
                break
            for _ in range(200):
                edges = create_path_with_extra_edges(
                    n_vertices, base_edges, n_extra, seed=random.randint(0, 10000000)
                )
                add_graph(edges)

    # 8. XOR-permuted paths with extra edges
    if len(graphs) < n_graphs:
        for xor_mask in range(1, min(n_vertices, 32)):
            if len(graphs) >= n_graphs:
                break
            xor_base = create_xor_permuted_path(n_vertices, xor_mask)
            for n_extra in range(1, min(n_bits, 4)):
                if len(graphs) >= n_graphs:
                    break
                for _ in range(20):
                    edges = create_path_with_extra_edges(
                        n_vertices, xor_base, n_extra, seed=random.randint(0, 10000000)
                    )
                    add_graph(edges)

    # 9. Random Hamiltonian paths (might have counting-like structure)
    attempts = 0
    while len(graphs) < n_graphs and attempts < n_graphs * 100:
        attempts += 1
        edges = create_random_hamiltonian_path(n_vertices, seed=random.randint(0, 10000000))
        if edges:
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
        h_dist = ", ".join(
            [f"H{h}: {count/total_edges*100:.0f}%" for h, count in sorted(h_totals.items())]
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
        description="Generate COUNTING PATH connected graphs for Matching win"
    )
    parser.add_argument(
        "-n", "--n-graphs", type=int, default=200, help="Number of graphs to generate"
    )
    parser.add_argument(
        "--vertices", type=int, nargs="+", default=[8, 16, 32], help="Vertex count(s) to generate"
    )
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true", default=True, help="Verbose output")

    args = parser.parse_args()

    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "graphs"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COUNTING PATH GRAPH GENERATOR (Matching Win)")
    print("=" * 60)
    print("KEY INSIGHT: Winning graphs have COUNTING PATH H distribution:")
    print("  H1: 50%, H2: 25%, H3: 12.5%, H4: 6.25%, H5: 3.125%")
    print()
    print("NOTE: 100% H1 graphs (Gray code) LOSE to Pauli!")
    print("      The mixed Hamming distribution is what wins.")
    print("Strategies:")
    print("  1. Counting paths: 0->1->2->...->n-1")
    print("  2. XOR-permuted counting paths")
    print("  3. Perturbed counting paths")
    print("=" * 60)

    for n_vertices in args.vertices:
        if n_vertices <= 0 or (n_vertices & (n_vertices - 1)) != 0:
            print(f"Error: {n_vertices} is not a power of 2")
            continue

        print(f"\n" + "-" * 60)

        graphs = generate_graphs(n_vertices, args.n_graphs, seed=args.seed, verbose=args.verbose)

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
