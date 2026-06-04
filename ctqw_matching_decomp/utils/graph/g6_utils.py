"""
Graph6 file utilities for loading and processing quantum walk graphs.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


def parse_g6_filename(filename: str) -> Dict:
    """
    Extract metadata from G6 filename.

    Args:
        filename: Path to G6 file

    Returns:
        Dictionary with metadata (n_graphs, type, vertices, suffix, filename)
    """
    path = Path(filename)
    stem = path.stem

    # Special pattern for Erdős-Rényi: 100graph_erdos_renyi_p0_10_128v
    er_match = re.match(r'(\d+)graph_(erdos_renyi_p\d+_\d+)_(\d+)([a-z]*)', stem)
    if er_match:
        groups = er_match.groups()
        return {
            'n_graphs': int(groups[0]),
            'type': groups[1],
            'vertices': int(groups[2]),
            'suffix': groups[3] if groups[3] else None,
            'filename': path.name
        }

    # Try multiple patterns for different filename formats
    patterns = [
        r'(\d+)graph_([A-Za-z_]+)_(\d+)([a-z]*)',  # 100graph_bipartite_16c
        r'(\d+)graph_([A-Za-z]+)_(\d+)([a-z]*)',   # Standard format
        r'(\d+)graph([A-Za-z_]+)(\d+)([a-z]*)',    # Without underscore after graph
        r'(\d+)graph_([A-Za-z]+)(\d+)([a-z]*)',    # Alternative format
    ]

    for pattern in patterns:
        match = re.match(pattern, stem)
        if match:
            groups = match.groups()
            return {
                'n_graphs': int(groups[0]),
                'type': groups[1],
                'vertices': int(groups[2]),
                'suffix': groups[3] if len(groups) >= 4 and groups[3] else None,
                'filename': path.name
            }

    # If no pattern matches, try to extract just the numbers and words
    numbers = re.findall(r'\d+', stem)
    words = re.findall(r'[A-Za-z_]+', stem)

    result = {'filename': path.name}

    # Extract numbers: first is usually graph count, last is usually vertex count
    if len(numbers) >= 2:
        result['n_graphs'] = int(numbers[0])
        result['vertices'] = int(numbers[-1])
    elif len(numbers) == 1:
        if 'graph' in stem.lower():
            result['n_graphs'] = int(numbers[0])
        else:
            result['vertices'] = int(numbers[0])

    # Extract graph type from words (skip 'graph' itself)
    if words:
        graph_words = [w for w in words if w.lower() != 'graph']
        if graph_words:
            result['type'] = '_'.join(graph_words)
        else:
            result['type'] = 'Unknown'
    else:
        result['type'] = 'Unknown'

    return result


def g6_to_edge_set(g6_string: str) -> Optional[Set[Tuple[str, str]]]:
    """
    Convert G6 string to edge set with binary vertex labels.

    Note: Returns single-direction edges only (not bidirectional).
    MultiEdgeGraph automatically symmetrizes the Hamiltonian.

    Args:
        g6_string: Graph6 format string

    Returns:
        Set of edge tuples with binary string vertices, or None on error
    """
    try:
        G = nx.from_graph6_bytes(g6_string.encode())
        n_vertices = len(G.nodes())

        if n_vertices == 0:
            return set()

        # Calculate bits needed for binary representation
        n_bits = max(1, int(np.ceil(np.log2(max(2, n_vertices)))))

        # Create vertex mapping to binary strings
        vertex_map = {vertex: format(i, f'0{n_bits}b')
                     for i, vertex in enumerate(sorted(G.nodes()))}

        # Convert edges to binary format (single direction only)
        edge_set = set()
        for u, v in G.edges():
            u_bin, v_bin = vertex_map[u], vertex_map[v]
            edge_set.add((u_bin, v_bin))

        return edge_set

    except Exception as e:
        print(f"Error parsing G6 string: {e}")
        return None


def load_graphs_from_g6(filename: str) -> Tuple[List[Set[Tuple[str, str]]], Dict]:
    """
    Load all graphs from a G6 file.

    Args:
        filename: Path to G6 file

    Returns:
        Tuple of (list of edge sets, metadata dict)
    """
    metadata = parse_g6_filename(filename)
    graphs = []

    print(f"Loading graphs from {filename}")
    print(f"Expected: {metadata}")

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                edges = g6_to_edge_set(line)
                if edges is not None:
                    graphs.append(edges)
                else:
                    print(f"Warning: Failed to parse line {line_num + 1}")

    print(f"Successfully loaded {len(graphs)} graphs")
    return graphs, metadata


def graph_to_bitstring_edges(graph: nx.Graph) -> Set[Tuple[str, str]]:
    """
    Convert a NetworkX graph to bitstring edge set.

    Note: Returns single-direction edges only (not bidirectional).
    MultiEdgeGraph automatically symmetrizes the Hamiltonian.

    Args:
        graph: NetworkX graph with integer node labels

    Returns:
        Set of edge tuples with binary string vertices
    """
    num_nodes = len(graph.nodes)
    num_bits = len(bin(num_nodes - 1)) - 2
    node_to_bitstring = {node: format(node, f"0{num_bits}b") for node in graph.nodes}
    return {(node_to_bitstring[u], node_to_bitstring[v]) for u, v in graph.edges}
