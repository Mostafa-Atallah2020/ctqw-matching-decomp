#!/usr/bin/env python3
"""
Graph Properties CSV Generator

Reads win/lose/draw G6 files from a matching vs pauli analysis output folder
and generates a CSV with graph properties for each category.

Usage: python graph_properties_csv.py <run_folder>
Example:
  python graph_properties_csv.py outputs/matching_vs_pauli/BM_8v/20251125_193616

Output:
  - graph_properties.csv in the same run folder
"""

import sys
import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.graph import (
    calculate_graph_properties,
    compute_hamming_statistics,
    graph_to_bitstring_edges,
)


def load_graphs_from_g6_file(filepath):
    """Load graphs from a G6 file, returning list of NetworkX graphs."""
    graphs = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    G = nx.from_graph6_bytes(line.encode())
                    graphs.append(G)
                except Exception as e:
                    print(f"Warning: Failed to parse line: {e}")
    return graphs


def compute_all_properties(graph, graph_index, category):
    """Compute all properties for a single graph."""
    # Basic graph properties
    props = calculate_graph_properties(graph)

    # Add graph index and category
    row = {
        'graph_index': graph_index,
        'category': category,
        'n_vertices': len(graph.nodes()),
        'n_edges': props['edge_count'],
        'edge_density': props['edge_density'],
        'is_bipartite': props['is_bipartite'],
        'is_connected': nx.is_connected(graph),
        'diameter': props['diameter'],
        'clique_number': props['clique_number'],
        'max_degree': props['max_degree'],
        'avg_degree': props['avg_degree'],
        'avg_clustering': props['avg_clustering'],
        'estimated_group_size': props['estimated_group_size'],
        'estimated_orbit_count': props['estimated_orbit_count'],
    }

    # Hamming statistics (for bitstring representation)
    try:
        edges = graph_to_bitstring_edges(graph)
        hamming_stats = compute_hamming_statistics(edges)
        row['total_hamming'] = hamming_stats['total_hamming']
        row['avg_hamming'] = hamming_stats['avg_hamming']
        row['min_hamming'] = hamming_stats['min_hamming']
        row['max_hamming'] = hamming_stats['max_hamming']
        row['hamming_1_count'] = hamming_stats['hamming_1_count']
        row['hamming_gt1_count'] = hamming_stats['hamming_gt1_count']
    except Exception as e:
        print(f"Warning: Failed to compute Hamming stats for graph {graph_index}: {e}")
        row['total_hamming'] = None
        row['avg_hamming'] = None
        row['min_hamming'] = None
        row['max_hamming'] = None
        row['hamming_1_count'] = None
        row['hamming_gt1_count'] = None

    return row


def process_g6_file(filepath, category):
    """Process a G6 file and return list of property dictionaries."""
    if not filepath.exists():
        print(f"  {category}: file not found ({filepath.name})")
        return []

    graphs = load_graphs_from_g6_file(filepath)
    print(f"  {category}: {len(graphs)} graphs")

    rows = []
    for i, graph in enumerate(graphs):
        row = compute_all_properties(graph, i, category)
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Generate CSV with graph properties from win/lose/draw G6 files'
    )
    parser.add_argument('run_folder', help='Path to matching vs pauli analysis output folder')
    parser.add_argument('-o', '--output', help='Output CSV filename (default: graph_properties.csv)')

    args = parser.parse_args()

    run_folder = Path(args.run_folder)
    results_folder = run_folder / 'results'

    if not results_folder.exists():
        print(f"Error: Results folder not found: {results_folder}")
        return 1

    # Find G6 files
    g6_files = list(results_folder.glob('*.g6'))
    if not g6_files:
        print(f"Error: No G6 files found in {results_folder}")
        return 1

    print(f"Processing run folder: {run_folder}")
    print(f"Found {len(g6_files)} G6 files")

    # Categorize files
    categories = {'win': None, 'lose': None, 'draw': None}
    for g6_file in g6_files:
        name = g6_file.stem.lower()
        for cat in categories:
            if name.startswith(cat):
                categories[cat] = g6_file
                break

    # Process each category
    all_rows = []
    for category, filepath in categories.items():
        if filepath:
            rows = process_g6_file(filepath, category)
            all_rows.extend(rows)

    if not all_rows:
        print("Error: No graphs processed")
        return 1

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)

    # Reorder columns
    column_order = [
        'graph_index', 'category', 'n_vertices', 'n_edges', 'edge_density',
        'is_bipartite', 'is_connected', 'diameter', 'clique_number',
        'max_degree', 'avg_degree', 'avg_clustering',
        'total_hamming', 'avg_hamming', 'min_hamming', 'max_hamming',
        'hamming_1_count', 'hamming_gt1_count',
        'estimated_group_size', 'estimated_orbit_count'
    ]
    df = df[[col for col in column_order if col in df.columns]]

    # Save to CSV
    output_file = args.output if args.output else run_folder / 'graph_properties.csv'
    output_file = Path(output_file)
    df.to_csv(output_file, index=False)

    print(f"\nSaved to: {output_file}")
    print(f"Total graphs: {len(df)}")
    print(f"\nCategory summary:")
    print(df['category'].value_counts().to_string())

    # Print summary statistics
    print(f"\nProperty summary by category:")
    summary = df.groupby('category')[['n_edges', 'edge_density', 'avg_hamming', 'diameter']].mean()
    print(summary.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
