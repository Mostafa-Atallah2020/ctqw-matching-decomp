"""
Compute graph properties from .g6 files and save as CSV.
Uses the properties module from src/utils/graph/properties.py
"""
import sys
import csv
import math
from pathlib import Path

# Add the project root to path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx

# Import directly from the properties module to avoid circular imports
from ctqw_matching_decomp.utils.graph.properties import (
    calculate_graph_properties,
    compute_hamming_statistics,
    hamming_distance
)


def mean(values):
    """Compute mean of a list."""
    if not values:
        return 0
    return sum(values) / len(values)


def std(values):
    """Compute standard deviation of a list."""
    if len(values) < 2:
        return 0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def get_bitstring_edges(graph: nx.Graph, n_qubits: int) -> set:
    """Convert graph edges to bitstring format for Hamming analysis."""
    edges = set()
    for u, v in graph.edges():
        u_str = format(u, f'0{n_qubits}b')
        v_str = format(v, f'0{n_qubits}b')
        edges.add((u_str, v_str))
    return edges


def compute_all_graph_properties(g6_file_path, n_qubits):
    """
    Read graphs from a .g6 file and compute properties for each graph.
    Uses the calculate_graph_properties function from src/utils/graph/properties.py

    Returns a list of dictionaries with properties for each graph.
    """
    graphs = list(nx.read_graph6(g6_file_path))

    all_properties = []
    for i, G in enumerate(graphs):
        # Use the properties module
        props = calculate_graph_properties(G)

        # Add additional properties not in the module
        n_vertices = G.number_of_nodes()
        n_components = nx.number_connected_components(G)

        # Isolated vertices (degree 0)
        degrees = [G.degree(node) for node in G.nodes()]
        isolated_vertices = sum(1 for d in degrees if d == 0)

        # Compute Hamming statistics for edges
        bitstring_edges = get_bitstring_edges(G, n_qubits)
        hamming_stats = compute_hamming_statistics(bitstring_edges)

        all_properties.append({
            'graph_index': i,
            'n_vertices': n_vertices,
            'n_edges': props['edge_count'],
            'n_components': n_components,
            'isolated_vertices': isolated_vertices,
            'avg_degree': props['avg_degree'],
            'max_degree': props['max_degree'],
            'density': props['edge_density'],
            'is_bipartite': 1 if props['is_bipartite'] else 0,
            'diameter': props['diameter'] if props['diameter'] is not None else -1,
            'clique_number': props['clique_number'],
            'avg_clustering': props['avg_clustering'],
            'estimated_group_size': props['estimated_group_size'],
            'estimated_orbit_count': props['estimated_orbit_count'],
            # Hamming statistics
            'avg_hamming': hamming_stats['avg_hamming'],
            'min_hamming': hamming_stats['min_hamming'],
            'max_hamming': hamming_stats['max_hamming'],
            'hamming_1_count': hamming_stats['hamming_1_count'],
            'hamming_gt1_count': hamming_stats['hamming_gt1_count'],
        })

    return all_properties


def compute_summary_stats(properties):
    """
    Compute mean and std for each property.
    """
    numeric_cols = ['n_edges', 'n_components', 'isolated_vertices', 'avg_degree',
                    'max_degree', 'density', 'is_bipartite', 'clique_number',
                    'avg_clustering', 'estimated_group_size', 'estimated_orbit_count',
                    'avg_hamming', 'hamming_1_count', 'hamming_gt1_count']

    summary = {}
    for col in numeric_cols:
        values = [p[col] for p in properties if p[col] is not None and p[col] != -1]
        summary[f'{col}_mean'] = mean(values)
        summary[f'{col}_std'] = std(values)

    # Diameter (only for connected graphs)
    diameter_values = [p['diameter'] for p in properties if p['diameter'] is not None and p['diameter'] != -1]
    summary['diameter_mean'] = mean(diameter_values) if diameter_values else None
    summary['diameter_std'] = std(diameter_values) if diameter_values else None

    # Bipartite percentage
    bipartite_values = [p['is_bipartite'] for p in properties]
    summary['bipartite_pct_mean'] = mean(bipartite_values) * 100
    summary['bipartite_pct_std'] = std(bipartite_values) * 100

    return summary


def save_csv(data, filepath, fieldnames):
    """Save list of dicts to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def process_graph_files(input_dir, output_dir, graph_type):
    """
    Process all .g6 files in input_dir and save results to output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    all_summaries = []

    g6_files = sorted(input_path.glob("*.g6"))
    for g6_file in g6_files:
        print(f"Processing {g6_file.name}...")

        # Extract vertex count from filename
        filename = g6_file.stem
        # Parse vertex count (e.g., "200graph_counting_128v" -> 128)
        parts = filename.split('_')
        n_vertices = None
        for part in parts:
            if part.endswith('v'):
                n_vertices = int(part[:-1])
                break

        if n_vertices is None:
            print(f"  Warning: Could not extract vertex count from {filename}")
            continue

        # Calculate number of qubits
        n_qubits = int(math.log2(n_vertices))

        # Compute properties using the properties module
        properties = compute_all_graph_properties(str(g6_file), n_qubits)

        # Determine output subdirectory
        if graph_type == "connected":
            output_subdir = output_path / f"connected_{n_vertices}v" / f"counting_{n_vertices}v"
        else:
            output_subdir = output_path / f"sparse_disconnected_{n_vertices}v"

        output_subdir.mkdir(parents=True, exist_ok=True)

        # Save per-graph properties
        per_graph_file = output_subdir / "graph_properties.csv"
        fieldnames = ['graph_index', 'n_vertices', 'n_edges', 'n_components', 'isolated_vertices',
                      'avg_degree', 'max_degree', 'density', 'is_bipartite', 'diameter',
                      'clique_number', 'avg_clustering', 'estimated_group_size', 'estimated_orbit_count',
                      'avg_hamming', 'min_hamming', 'max_hamming', 'hamming_1_count', 'hamming_gt1_count']
        save_csv(properties, per_graph_file, fieldnames)
        print(f"  Saved per-graph properties to {per_graph_file}")

        # Compute and save summary statistics
        summary = compute_summary_stats(properties)
        summary['n_vertices'] = n_vertices
        summary['n_graphs'] = len(properties)
        summary['graph_type'] = graph_type

        summary_file = output_subdir / "graph_properties_summary.csv"
        summary_fieldnames = ['graph_type', 'n_vertices', 'n_graphs',
                              'n_edges_mean', 'n_edges_std',
                              'n_components_mean', 'n_components_std',
                              'isolated_vertices_mean', 'isolated_vertices_std',
                              'avg_degree_mean', 'avg_degree_std',
                              'max_degree_mean', 'max_degree_std',
                              'density_mean', 'density_std',
                              'is_bipartite_mean', 'is_bipartite_std',
                              'bipartite_pct_mean', 'bipartite_pct_std',
                              'diameter_mean', 'diameter_std',
                              'clique_number_mean', 'clique_number_std',
                              'avg_clustering_mean', 'avg_clustering_std',
                              'estimated_group_size_mean', 'estimated_group_size_std',
                              'estimated_orbit_count_mean', 'estimated_orbit_count_std',
                              'avg_hamming_mean', 'avg_hamming_std',
                              'hamming_1_count_mean', 'hamming_1_count_std',
                              'hamming_gt1_count_mean', 'hamming_gt1_count_std']
        save_csv([summary], summary_file, summary_fieldnames)
        print(f"  Saved summary statistics to {summary_file}")

        all_summaries.append(summary)

    # Save combined summary for all vertex counts
    combined_file = output_path / f"{graph_type}_all_summary.csv"
    all_summaries_sorted = sorted(all_summaries, key=lambda x: x['n_vertices'])
    summary_fieldnames = ['graph_type', 'n_vertices', 'n_graphs',
                          'n_edges_mean', 'n_edges_std',
                          'n_components_mean', 'n_components_std',
                          'isolated_vertices_mean', 'isolated_vertices_std',
                          'avg_degree_mean', 'avg_degree_std',
                          'max_degree_mean', 'max_degree_std',
                          'density_mean', 'density_std',
                          'is_bipartite_mean', 'is_bipartite_std',
                          'bipartite_pct_mean', 'bipartite_pct_std',
                          'diameter_mean', 'diameter_std',
                          'clique_number_mean', 'clique_number_std',
                          'avg_clustering_mean', 'avg_clustering_std',
                          'estimated_group_size_mean', 'estimated_group_size_std',
                          'estimated_orbit_count_mean', 'estimated_orbit_count_std',
                          'avg_hamming_mean', 'avg_hamming_std',
                          'hamming_1_count_mean', 'hamming_1_count_std',
                          'hamming_gt1_count_mean', 'hamming_gt1_count_std']
    save_csv(all_summaries_sorted, combined_file, summary_fieldnames)
    print(f"\nSaved combined summary to {combined_file}")

    return all_summaries_sorted


def main():
    base_dir = Path(r"F:\UTK\Research\DOE_FOA_project\code\dyn-CTQW\analysis")

    # Process connected graphs
    print("=" * 60)
    print("Processing CONNECTED graphs")
    print("=" * 60)
    connected_summaries = process_graph_files(
        input_dir=base_dir / "graphs" / "connected",
        output_dir=base_dir / "outputs" / "matching_vs_pauli" / "connected",
        graph_type="connected"
    )

    print("\n" + "=" * 60)
    print("Processing DISCONNECTED graphs")
    print("=" * 60)
    disconnected_summaries = process_graph_files(
        input_dir=base_dir / "graphs" / "disconnected",
        output_dir=base_dir / "outputs" / "matching_vs_pauli" / "disconnected",
        graph_type="disconnected"
    )

    # Print summary table for LaTeX
    print("\n" + "=" * 60)
    print("SUMMARY FOR LATEX TABLE")
    print("=" * 60)

    print("\nConnected Graphs:")
    print("-" * 140)
    print(f"{'Vertices':<10} {'Edges':<15} {'Components':<12} {'Isolated':<12} {'Avg Degree':<15} {'Max Degree':<12} {'Density':<15} {'Bipartite %':<15}")
    print("-" * 140)
    for row in connected_summaries:
        edges = f"{row['n_edges_mean']:.1f} ± {row['n_edges_std']:.1f}"
        comps = f"{row['n_components_mean']:.0f}"
        isolated = f"{row['isolated_vertices_mean']:.0f}"
        avg_deg = f"{row['avg_degree_mean']:.2f} ± {row['avg_degree_std']:.2f}"
        max_deg = f"{row['max_degree_mean']:.1f} ± {row['max_degree_std']:.1f}"
        density = f"{row['density_mean']:.4f}"
        bipartite = f"{row['bipartite_pct_mean']:.0f}%"
        print(f"{int(row['n_vertices']):<10} {edges:<15} {comps:<12} {isolated:<12} {avg_deg:<15} {max_deg:<12} {density:<15} {bipartite:<15}")

    print("\nDisconnected Graphs:")
    print("-" * 140)
    print(f"{'Vertices':<10} {'Edges':<15} {'Components':<12} {'Isolated':<12} {'Avg Degree':<15} {'Max Degree':<12} {'Density':<15} {'Bipartite %':<15}")
    print("-" * 140)
    for row in disconnected_summaries:
        edges = f"{row['n_edges_mean']:.1f} ± {row['n_edges_std']:.1f}"
        comps = f"{row['n_components_mean']:.1f} ± {row['n_components_std']:.1f}"
        isolated = f"{row['isolated_vertices_mean']:.1f} ± {row['isolated_vertices_std']:.1f}"
        avg_deg = f"{row['avg_degree_mean']:.2f} ± {row['avg_degree_std']:.2f}"
        max_deg = f"{row['max_degree_mean']:.1f} ± {row['max_degree_std']:.1f}"
        density = f"{row['density_mean']:.4f}"
        bipartite = f"{row['bipartite_pct_mean']:.0f}%"
        print(f"{int(row['n_vertices']):<10} {edges:<15} {comps:<12} {isolated:<12} {avg_deg:<15} {max_deg:<12} {density:<15} {bipartite:<15}")


if __name__ == "__main__":
    main()
