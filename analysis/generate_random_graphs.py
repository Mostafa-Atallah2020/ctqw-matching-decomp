#!/usr/bin/env python3
"""
Pure Random Graph Generator for CTQW Analysis.

Generates PURELY RANDOM graphs at 3 sparsity levels for unbiased analysis.
NO Hamming-aware engineering, NO winning formulas - just random graphs.

This allows unbiased analysis of which graph properties naturally correlate
with Matching vs Pauli performance.

Sparsity Levels (3-tier):
  - sparse: density 0.02 - 0.15
  - medium: density 0.15 - 0.50
  - dense:  density 0.50 - 0.90

For each level, generates both connected and disconnected graphs.

Usage:
  python generate_random_graphs.py --vertices 32 64 -n 200 --seed 42
  python generate_random_graphs.py --vertices 32 -n 100 --density sparse --seed 42
"""

import argparse
import random
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np


# =============================================================================
# DENSITY LEVEL DEFINITIONS (3-tier, same as unified generator)
# =============================================================================

DENSITY_LEVELS = {
    "sparse": (0.02, 0.15),
    "medium": (0.15, 0.50),
    "dense": (0.50, 0.90),
}


# =============================================================================
# PURE RANDOM GRAPH GENERATION
# =============================================================================


def generate_random_graph(n_vertices, density, seed=None):
    """
    Generate a purely random graph using G(n,p) Erdős-Rényi model.

    No Hamming distance considerations - completely random edge selection.

    Args:
        n_vertices: Number of vertices
        density: Edge probability (0-1)
        seed: Random seed

    Returns:
        NetworkX graph
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Use G(n,p) model - each edge exists with probability p
    G = nx.gnp_random_graph(n_vertices, density, seed=seed)

    return G


def generate_random_connected_graph(
    n_vertices, min_density, max_density, seed=None, max_attempts=100
):
    """
    Generate a random connected graph at specified density range.

    Simply generates random graphs until one is connected.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for attempt in range(max_attempts):
        density = random.uniform(min_density, max_density)
        graph_seed = random.randint(0, 10000000)

        G = generate_random_graph(n_vertices, density, seed=graph_seed)

        if G.number_of_edges() > 0 and nx.is_connected(G):
            return G

    return None


def generate_random_disconnected_graph(
    n_vertices, min_density, max_density, seed=None, max_attempts=100
):
    """
    Generate a random disconnected graph at specified density range.

    For sparse densities: naturally disconnected graphs are common.
    For medium/dense densities: generate dense graph then remove minimum cut.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # For dense graphs, use the minimum cut approach directly
    if min_density >= 0.50:
        return generate_dense_disconnected_graph(n_vertices, min_density, max_density, seed)

    for attempt in range(max_attempts):
        density = random.uniform(min_density, max_density)
        graph_seed = random.randint(0, 10000000)

        G = generate_random_graph(n_vertices, density, seed=graph_seed)

        if G.number_of_edges() == 0:
            continue

        # If already disconnected, use it
        if not nx.is_connected(G):
            return G

        # For medium densities, try to disconnect by removing minimum cut
        if density > 0.15:
            G_disc = disconnect_by_minimum_cut(G, min_density, max_density)
            if G_disc is not None and not nx.is_connected(G_disc) and G_disc.number_of_edges() > 0:
                return G_disc

    return None


def generate_dense_disconnected_graph(n_vertices, min_density, max_density, seed=None):
    """
    Generate a disconnected graph at dense density levels.

    Strategy:
    1. Generate a dense connected graph at slightly higher density
    2. Find the minimum edge cut
    3. Remove those edges to disconnect
    4. Verify final density is in target range
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    max_edges = n_vertices * (n_vertices - 1) // 2

    # Generate at higher density to have room after cut removal
    target_density = random.uniform(min_density, max_density)
    # Add buffer for edges we'll remove
    generation_density = min(0.95, target_density + 0.05)

    graph_seed = random.randint(0, 10000000)
    G = generate_random_graph(n_vertices, generation_density, seed=graph_seed)

    if G.number_of_edges() == 0 or not nx.is_connected(G):
        # If already disconnected, check density
        if G.number_of_edges() > 0:
            actual_density = G.number_of_edges() / max_edges
            if min_density <= actual_density <= max_density:
                return G
        return None

    # Find and remove minimum edge cut
    try:
        cut_edges = nx.minimum_edge_cut(G)
        if cut_edges:
            G.remove_edges_from(cut_edges)

            # Verify it's disconnected and in density range
            if not nx.is_connected(G):
                actual_density = G.number_of_edges() / max_edges
                if min_density <= actual_density <= max_density:
                    return G
                # If density too high, remove more random edges
                elif actual_density > max_density:
                    target_edges = int(random.uniform(min_density, max_density) * max_edges)
                    while G.number_of_edges() > target_edges:
                        edges = list(G.edges())
                        if not edges:
                            break
                        G.remove_edge(*random.choice(edges))
                    if not nx.is_connected(G) and G.number_of_edges() > 0:
                        return G
    except nx.NetworkXError:
        pass

    return None


def disconnect_by_minimum_cut(G, min_density, max_density):
    """
    Disconnect a connected graph by removing the minimum edge cut.

    This is more reliable than bridge removal for dense graphs.
    """
    if not nx.is_connected(G):
        return G.copy()

    G_copy = G.copy()
    n_vertices = G_copy.number_of_nodes()
    max_edges = n_vertices * (n_vertices - 1) // 2

    # Try minimum edge cut
    try:
        cut_edges = nx.minimum_edge_cut(G_copy)
        if cut_edges:
            G_copy.remove_edges_from(cut_edges)

            if not nx.is_connected(G_copy) and G_copy.number_of_edges() > 0:
                actual_density = G_copy.number_of_edges() / max_edges
                if min_density <= actual_density <= max_density:
                    return G_copy
    except nx.NetworkXError:
        pass

    # Fallback: try finding bridges
    try:
        bridges = list(nx.bridges(G_copy))
        if bridges:
            bridge = random.choice(bridges)
            G_copy.remove_edge(*bridge)
            if not nx.is_connected(G_copy):
                return G_copy
    except:
        pass

    return None


# =============================================================================
# BATCH GENERATION
# =============================================================================


def generate_graphs_at_density(
    n_vertices, n_graphs, density_level, connected, seed=None, show_progress=True
):
    """
    Generate random graphs at a specific density level.

    Args:
        n_vertices: Number of vertices
        n_graphs: Number of graphs to generate
        density_level: 'sparse', 'medium', or 'dense'
        connected: True for connected, False for disconnected
        seed: Random seed
        show_progress: Show progress in terminal

    Returns:
        List of NetworkX graphs
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    min_density, max_density = DENSITY_LEVELS[density_level]

    graphs = []
    seen_hashes = set()
    attempts = 0
    max_attempts = n_graphs * 100

    # Progress tracking
    start_time = time.time()
    last_progress_time = start_time

    while len(graphs) < n_graphs and attempts < max_attempts:
        attempts += 1
        graph_seed = random.randint(0, 10000000)

        if connected:
            G = generate_random_connected_graph(
                n_vertices, min_density, max_density, seed=graph_seed
            )
        else:
            G = generate_random_disconnected_graph(
                n_vertices, min_density, max_density, seed=graph_seed
            )

        if G is None or G.number_of_edges() == 0:
            continue

        # Verify connectivity status
        is_connected = nx.is_connected(G)
        if connected and not is_connected:
            continue
        if not connected and is_connected:
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

        # Progress display
        current_time = time.time()
        if show_progress and (current_time - last_progress_time >= 0.5 or len(graphs) == n_graphs):
            elapsed = current_time - start_time
            progress = len(graphs) / n_graphs * 100
            speed = len(graphs) / elapsed if elapsed > 0 else 0
            eta = (n_graphs - len(graphs)) / speed if speed > 0 else 0

            print(
                f"\r  Progress: {progress:>5.1f}% ({len(graphs):>4}/{n_graphs}) | "
                f"Speed: {speed:>5.1f}/s | ETA: {eta:>5.1f}s | "
                f"Attempts: {attempts}",
                end="",
                flush=True,
            )
            last_progress_time = current_time

    if show_progress:
        print()  # New line after progress

    return graphs


# =============================================================================
# STATISTICS
# =============================================================================


def compute_graph_stats(G):
    """Compute basic statistics for a graph."""
    n_vertices = G.number_of_nodes()
    n_edges = G.number_of_edges()
    max_edges = n_vertices * (n_vertices - 1) // 2

    return {
        "n_vertices": n_vertices,
        "n_edges": n_edges,
        "density": n_edges / max_edges if max_edges > 0 else 0,
        "is_connected": nx.is_connected(G),
        "n_components": nx.number_connected_components(G),
        "max_degree": max(d for _, d in G.degree()) if n_edges > 0 else 0,
        "avg_degree": sum(d for _, d in G.degree()) / n_vertices if n_vertices > 0 else 0,
    }


def print_statistics(graphs, n_vertices, label=""):
    """Print statistics for a set of graphs."""
    if not graphs:
        print(f"  {label}: No graphs generated")
        return

    stats = [compute_graph_stats(G) for G in graphs]

    avg_edges = np.mean([s["n_edges"] for s in stats])
    avg_density = np.mean([s["density"] for s in stats])
    avg_components = np.mean([s["n_components"] for s in stats])
    avg_degree = np.mean([s["avg_degree"] for s in stats])

    print(f"\n  {label} Statistics (n={len(graphs)}):")
    print(f"    Avg edges:      {avg_edges:.1f}")
    print(f"    Avg density:    {avg_density:.3f}")
    print(f"    Avg components: {avg_components:.1f}")
    print(f"    Avg degree:     {avg_degree:.2f}")


def save_graphs(graphs, output_path, verbose=True):
    """Save graphs in graph6 format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for G in graphs:
            f.write(nx.to_graph6_bytes(G, header=False).decode("ascii"))

    if verbose:
        print(f"  Saved {len(graphs)} graphs to {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate PURELY RANDOM graphs at varying sparsity levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This generator creates completely random graphs using the G(n,p) model.
NO Hamming-aware engineering - just pure random edge selection.

Density Levels:
  sparse:  0.02 - 0.15 (few edges)
  medium:  0.15 - 0.50 (moderate edges)
  dense:   0.50 - 0.90 (many edges)

Examples:
  python generate_random_graphs.py --vertices 32 64 -n 200 --seed 42
  python generate_random_graphs.py --vertices 32 -n 100 --density sparse
        """,
    )

    parser.add_argument(
        "-n",
        "--n-graphs",
        type=int,
        default=200,
        help="Number of graphs per category (default: 200)",
    )
    parser.add_argument(
        "--vertices",
        type=int,
        nargs="+",
        default=[32, 64],
        help="Vertex count(s) to generate (default: 32 64)",
    )
    parser.add_argument(
        "--density",
        type=str,
        nargs="+",
        choices=["sparse", "medium", "dense", "all"],
        default=["all"],
        help="Density level(s) to generate (default: all)",
    )
    parser.add_argument(
        "--connectivity",
        type=str,
        choices=["connected", "disconnected", "both"],
        default="both",
        help="Connectivity type (default: both)",
    )
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "graphs" / "random"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    # Determine density levels
    if "all" in args.density:
        density_levels = list(DENSITY_LEVELS.keys())
    else:
        density_levels = args.density

    # Determine connectivity types
    if args.connectivity == "both":
        connectivity_types = [True, False]
    elif args.connectivity == "connected":
        connectivity_types = [True]
    else:
        connectivity_types = [False]

    if verbose:
        print("=" * 70)
        print("PURE RANDOM GRAPH GENERATOR")
        print("=" * 70)
        print()
        print("Method: G(n,p) Erdős-Rényi random graph model")
        print("NO Hamming-aware engineering - purely random edge selection")
        print()
        print("Density Levels:")
        for level, (min_d, max_d) in DENSITY_LEVELS.items():
            print(f"  {level:8s}: density {min_d:.2f} - {max_d:.2f}")
        print()
        print(f"Output: {output_dir}")
        print("=" * 70)

    total_start = time.time()

    for n_vertices in args.vertices:
        if n_vertices <= 0 or (n_vertices & (n_vertices - 1)) != 0:
            print(f"Warning: {n_vertices} is not a power of 2, skipping...")
            continue

        n_bits = int(np.log2(n_vertices))

        if verbose:
            print(f"\n{'='*70}")
            print(f"GENERATING FOR {n_vertices} VERTICES ({n_bits} QUBITS)")
            print(f"{'='*70}")

        for density_level in density_levels:
            min_d, max_d = DENSITY_LEVELS[density_level]

            if verbose:
                print(f"\n--- Density: {density_level.upper()} ({min_d:.2f} - {max_d:.2f}) ---")

            for connected in connectivity_types:
                conn_str = "connected" if connected else "disconnected"

                if verbose:
                    print(f"\nGenerating {args.n_graphs} {conn_str} graphs...")

                graphs = generate_graphs_at_density(
                    n_vertices,
                    args.n_graphs,
                    density_level,
                    connected,
                    seed=args.seed,
                    show_progress=verbose,
                )

                if graphs:
                    if verbose:
                        print_statistics(graphs, n_vertices, f"{density_level} {conn_str}")

                    output_file = (
                        output_dir
                        / f"{len(graphs)}graph_{density_level}_{conn_str}_{n_vertices}v.g6"
                    )
                    save_graphs(graphs, output_file, verbose)
                else:
                    print(
                        f"  WARNING: Could not generate {conn_str} graphs at {density_level} density"
                    )

    total_time = time.time() - total_start

    if verbose:
        print(f"\n{'='*70}")
        print(f"Generation complete! Total time: {total_time:.1f}s")
        print(f"Output directory: {output_dir}")
        print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
