"""
Analyze graphs to find percentage where:
- Matching decomposition has all matchings pairwise commuting
- Pauli decomposition has non-commuting terms

This adds a column to Table 1 showing the % of graphs where matching
decomposition is exact but Pauli decomposition requires Trotterization.
"""

import sys
import os
import itertools
from pathlib import Path
from typing import Set, Tuple, List, Dict
from collections import defaultdict
import json
from datetime import datetime

import numpy as np
import networkx as nx
from qiskit.quantum_info import Pauli, Operator

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ctqw_matching_decomp.utils.graph.g6_utils import load_graphs_from_g6, g6_to_edge_set
from ctqw_matching_decomp.core.multi_edge_graph import MultiEdgeGraph
from ctqw_matching_decomp.core.decompositions.matching import MatchingDecomposition


def matchings_commute(m1: Set[Tuple[str, str]], m2: Set[Tuple[str, str]]) -> bool:
    """
    Check if two matchings commute.

    Two matchings commute iff no edge in m1 shares exactly one vertex with any edge in m2.
    (Sharing 0 or 2 vertices is fine; sharing 1 causes anticommutation.)
    """
    for e1 in m1:
        v1 = set(e1)
        for e2 in m2:
            v2 = set(e2)
            shared = len(v1 & v2)
            if shared == 1:  # Exactly one shared vertex = anticommute
                return False
    return True


def check_matchings_all_commute(matchings: List[Set[Tuple[str, str]]], n_qubits: int) -> bool:
    """
    Check if all matchings in the decomposition pairwise commute WITH EARLY EXIT.

    Args:
        matchings: List of matchings (each is a set of edges)
        n_qubits: Number of qubits (unused now, kept for API compatibility)

    Returns:
        True if all matchings pairwise commute
    """
    if len(matchings) <= 1:
        return True

    # Check pairs with early exit
    for i in range(len(matchings)):
        for j in range(i + 1, len(matchings)):
            if not matchings_commute(matchings[i], matchings[j]):
                return False  # Early exit!

    return True


def paulis_commute(p1: str, p2: str) -> bool:
    """Check if two Pauli strings commute."""
    anticommute_count = 0
    for c1, c2 in zip(p1, p2):
        # Count positions where both are non-identity and different
        if c1 != 'I' and c2 != 'I' and c1 != c2:
            anticommute_count += 1
    # Commute if even number of anticommuting positions
    return anticommute_count % 2 == 0


def check_pauli_commutes_with_early_exit(hamiltonian: np.ndarray, n_qubits: int) -> Tuple[bool, int]:
    """
    Compute Pauli decomposition and check commutativity WITH EARLY EXIT.

    Returns as soon as we find two non-commuting Pauli terms.

    Args:
        hamiltonian: The Hamiltonian matrix
        n_qubits: Number of qubits

    Returns:
        Tuple of (all_commute, n_pauli_terms_found)
    """
    pauli_terms = []  # List of non-zero Pauli strings found so far

    for pauli_string in itertools.product("IXYZ", repeat=n_qubits):
        pauli_str = "".join(pauli_string)
        P = Pauli(pauli_str)
        P_op = Operator(P).data
        coeff = np.trace(P_op.conj().T @ hamiltonian) / (2**n_qubits)

        real_coeff = float(np.real(coeff))
        if not np.isclose(real_coeff, 0, atol=1e-10):
            # Check if this new term commutes with all previous terms
            for prev_term in pauli_terms:
                if not paulis_commute(pauli_str, prev_term):
                    # Found non-commuting pair - early exit!
                    return False, len(pauli_terms) + 1

            pauli_terms.append(pauli_str)

    return True, len(pauli_terms)


def compute_greedy_matchings(edges: Set[Tuple[str, str]]) -> List[Set[Tuple[str, str]]]:
    """
    Compute matchings using greedy algorithm grouped by bit-flip position.

    This replicates the matching algorithm from MatchingDecomposition class.

    Args:
        edges: Set of graph edges

    Returns:
        List of matchings
    """
    # Group edges by bit-flip position
    bit_flip_groups = defaultdict(list)
    multi_bit_edges = []

    for edge in edges:
        u, v = edge
        u_int, v_int = int(u, 2), int(v, 2)
        diff = u_int ^ v_int

        if diff == 0:
            continue  # Self-loop
        elif bin(diff).count("1") == 1:
            # Single bit flip
            position = (diff & -diff).bit_length() - 1
            bit_flip_groups[position].append(edge)
        else:
            # Multiple bit flips
            multi_bit_edges.append(edge)

    matchings = []

    # Process each bit-flip group
    for position in sorted(bit_flip_groups.keys()):
        group_edges = bit_flip_groups[position]

        while group_edges:
            matching = set()
            used_vertices = set()
            remaining = []

            for edge in group_edges:
                u, v = edge
                if u not in used_vertices and v not in used_vertices:
                    matching.add(edge)
                    used_vertices.add(u)
                    used_vertices.add(v)
                else:
                    remaining.append(edge)

            if matching:
                matchings.append(matching)
            group_edges = remaining

    # Add multi-bit edges to existing matchings or create new ones
    for edge in multi_bit_edges:
        u, v = edge
        placed = False

        for matching in matchings:
            vertices_in_matching = set()
            for e in matching:
                vertices_in_matching.add(e[0])
                vertices_in_matching.add(e[1])

            if u not in vertices_in_matching and v not in vertices_in_matching:
                matching.add(edge)
                placed = True
                break

        if not placed:
            matchings.append({edge})

    return matchings


def analyze_graph(edges: Set[Tuple[str, str]], graph_idx: int,
                  heuristic: str = 'greedy') -> Dict:
    """
    Analyze a single graph for commuting properties.

    Args:
        edges: Set of graph edges
        graph_idx: Index of the graph (for reporting)
        heuristic: Matching heuristic ('greedy' or 'compression_aware')

    Returns:
        Dictionary with analysis results
    """
    if not edges:
        return {
            'graph_idx': graph_idx,
            'n_edges': 0,
            'matchings_commute': True,
            'pauli_commutes': True,
            'matching_exact_pauli_not': False,
            'n_matchings': 0,
            'n_pauli_terms': 0
        }

    # Get n_qubits from edge length
    first_edge = next(iter(edges))
    n_qubits = len(first_edge[0])
    n_vertices = 2 ** n_qubits

    # Compute matchings using MatchingDecomposition
    G = MultiEdgeGraph(edges)
    decomp = MatchingDecomposition(G, heuristic=heuristic)
    matchings = decomp.get_matchings()
    matchings_commute = check_matchings_all_commute(matchings, n_qubits)

    # Build Hamiltonian and check Pauli commutativity with early exit
    hamiltonian = np.zeros((n_vertices, n_vertices), dtype=float)
    for u, v in edges:
        i, j = int(u, 2), int(v, 2)
        hamiltonian[i, j] = 1.0
        hamiltonian[j, i] = 1.0

    pauli_commutes, n_pauli_terms = check_pauli_commutes_with_early_exit(hamiltonian, n_qubits)

    # Key metric: matching exact but Pauli not
    matching_exact_pauli_not = matchings_commute and not pauli_commutes

    return {
        'graph_idx': graph_idx,
        'n_edges': len(edges),
        'matchings_commute': matchings_commute,
        'pauli_commutes': pauli_commutes,
        'matching_exact_pauli_not': matching_exact_pauli_not,
        'n_matchings': len(matchings),
        'n_pauli_terms': n_pauli_terms
    }


def analyze_g6_file(filepath: str, heuristic: str = 'greedy') -> Dict:
    """
    Analyze all graphs in a G6 file.

    Args:
        filepath: Path to G6 file
        heuristic: Matching heuristic ('greedy' or 'compression_aware')

    Returns:
        Dictionary with summary statistics
    """
    print(f"\nAnalyzing: {filepath}")

    # Load graphs
    graphs, metadata = load_graphs_from_g6(filepath)
    n_graphs = len(graphs)

    print(f"Loaded {n_graphs} graphs")

    # Analyze each graph
    results = []
    for i, edges in enumerate(graphs):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing graph {i + 1}/{n_graphs}...")

        result = analyze_graph(edges, i, heuristic=heuristic)
        results.append(result)

    # Compute statistics
    n_matchings_commute = sum(1 for r in results if r['matchings_commute'])
    n_pauli_commutes = sum(1 for r in results if r['pauli_commutes'])
    n_matching_exact_pauli_not = sum(1 for r in results if r['matching_exact_pauli_not'])

    pct_matchings_commute = 100 * n_matchings_commute / n_graphs if n_graphs > 0 else 0
    pct_pauli_commutes = 100 * n_pauli_commutes / n_graphs if n_graphs > 0 else 0
    pct_matching_exact_pauli_not = 100 * n_matching_exact_pauli_not / n_graphs if n_graphs > 0 else 0

    summary = {
        'filepath': filepath,
        'metadata': metadata,
        'n_graphs': n_graphs,
        'n_matchings_commute': n_matchings_commute,
        'n_pauli_commutes': n_pauli_commutes,
        'n_matching_exact_pauli_not': n_matching_exact_pauli_not,
        'pct_matchings_commute': pct_matchings_commute,
        'pct_pauli_commutes': pct_pauli_commutes,
        'pct_matching_exact_pauli_not': pct_matching_exact_pauli_not,
        'detailed_results': results
    }

    print(f"\nResults for {metadata.get('vertices', 'unknown')} vertices:")
    print(f"  Matchings all commute: {n_matchings_commute}/{n_graphs} ({pct_matchings_commute:.1f}%)")
    print(f"  Pauli terms all commute: {n_pauli_commutes}/{n_graphs} ({pct_pauli_commutes:.1f}%)")
    print(f"  Matching exact but Pauli not: {n_matching_exact_pauli_not}/{n_graphs} ({pct_matching_exact_pauli_not:.1f}%)")

    return summary


def main():
    """Main function to analyze graph datasets.

    Usage:
        python analyze_commuting_decompositions.py [graph_dir]

    Examples:
        python analyze_commuting_decompositions.py                    # uses graphs/connected
        python analyze_commuting_decompositions.py graphs/erdos_renyi
        python analyze_commuting_decompositions.py F:\\path\\to\\graphs
    """
    import argparse
    parser = argparse.ArgumentParser(description='Analyze commuting decompositions')
    parser.add_argument('graph_dir', nargs='?', default=None,
                        help='Directory containing G6 files (default: graphs/connected)')
    parser.add_argument('--heuristic', type=str, default='greedy',
                        choices=['greedy', 'compression_aware'],
                        help='Matching heuristic (default: greedy)')
    args = parser.parse_args()

    # Directory containing G6 files
    if args.graph_dir:
        graphs_dir = Path(args.graph_dir)
        if not graphs_dir.is_absolute():
            graphs_dir = Path(__file__).parent / args.graph_dir
    else:
        graphs_dir = Path(__file__).parent / "graphs" / "connected"

    # Output directory named after input directory
    dir_name = graphs_dir.name
    base = "outputs_comp_aware" if args.heuristic == "compression_aware" else "outputs"
    output_dir = Path(__file__).parent / base / f"commuting_analysis_{dir_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all G6 files
    g6_files = sorted(graphs_dir.glob("*.g6"))

    if not g6_files:
        print(f"No G6 files found in {graphs_dir}")
        return

    print(f"Found {len(g6_files)} G6 files to analyze")

    # Analyze each file
    all_summaries = []
    for g6_file in g6_files:
        summary = analyze_g6_file(str(g6_file), heuristic=args.heuristic)
        all_summaries.append(summary)

    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE (for Table 1 in paper)")
    print("="*80)
    print(f"{'Vertices':<12} {'Graphs':<10} {'Match Commute':<15} {'Pauli Commute':<15} {'Match Exact, Pauli Not':<25}")
    print("-"*80)

    table_data = []
    for summary in sorted(all_summaries, key=lambda x: x['metadata'].get('vertices', 0)):
        vertices = summary['metadata'].get('vertices', 'N/A')
        n_graphs = summary['n_graphs']
        pct_match = summary['pct_matchings_commute']
        pct_pauli = summary['pct_pauli_commutes']
        pct_exact = summary['pct_matching_exact_pauli_not']

        print(f"{vertices:<12} {n_graphs:<10} {pct_match:<15.1f}% {pct_pauli:<15.1f}% {pct_exact:<25.1f}%")

        table_data.append({
            'vertices': vertices,
            'n_graphs': n_graphs,
            'pct_matchings_commute': pct_match,
            'pct_pauli_commutes': pct_pauli,
            'pct_matching_exact_pauli_not': pct_exact
        })

    print("="*80)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary JSON
    summary_file = output_dir / f"commuting_analysis_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        # Remove detailed results for summary file (too large)
        summaries_for_json = []
        for s in all_summaries:
            s_copy = {k: v for k, v in s.items() if k != 'detailed_results'}
            summaries_for_json.append(s_copy)
        json.dump({
            'timestamp': timestamp,
            'summaries': summaries_for_json,
            'table_data': table_data
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    # Save detailed results
    detailed_file = output_dir / f"commuting_analysis_detailed_{timestamp}.json"
    with open(detailed_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'all_summaries': all_summaries
        }, f, indent=2, default=str)
    print(f"Detailed results saved to: {detailed_file}")

    # Save LaTeX table snippet
    latex_file = output_dir / f"table1_column_{timestamp}.tex"
    with open(latex_file, 'w') as f:
        f.write("% Column for Table 1: Matching Exact but Pauli Not (%)\n")
        f.write("% Add this column to the existing table\n\n")
        for row in table_data:
            f.write(f"% {row['vertices']} vertices: {row['pct_matching_exact_pauli_not']:.1f}%\n")
    print(f"LaTeX snippet saved to: {latex_file}")

    return table_data


if __name__ == "__main__":
    main()