"""
Graph properties utilities for CTQW analysis.

Provides functions for calculating various graph properties useful for
analyzing quantum walk decomposition performance.
"""

import math
from typing import Dict, Optional

import networkx as nx
import numpy as np


def calculate_graph_properties(graph: nx.Graph) -> Dict:
    """
    Calculate various graph properties for analysis.

    Args:
        graph: NetworkX graph object

    Returns:
        Dictionary containing computed graph properties
    """
    properties = {}

    # Basic properties
    properties["edge_count"] = len(graph.edges())
    properties["edge_density"] = nx.density(graph)
    properties["is_bipartite"] = nx.is_bipartite(graph)

    # Connected graph properties
    if nx.is_connected(graph):
        properties["diameter"] = nx.diameter(graph)
    else:
        properties["diameter"] = None

    # Clique number (maximum clique size)
    try:
        properties["clique_number"] = len(max(nx.find_cliques(graph), key=len))
    except Exception:
        properties["clique_number"] = 1

    # Degree properties
    degrees = [graph.degree(node) for node in graph.nodes()]
    properties["max_degree"] = max(degrees) if degrees else 0
    properties["avg_degree"] = sum(degrees) / len(degrees) if degrees else 0

    # Clustering coefficient
    properties["avg_clustering"] = nx.average_clustering(graph)

    # Automorphism group size estimation (simplified)
    try:
        properties["estimated_group_size"] = estimate_group_size(graph)
    except Exception:
        properties["estimated_group_size"] = 1

    # Orbit count estimation (simplified)
    try:
        properties["estimated_orbit_count"] = estimate_orbit_count(graph)
    except Exception:
        properties["estimated_orbit_count"] = len(graph.nodes())

    return properties


def estimate_group_size(graph: nx.Graph) -> int:
    """
    Rough estimation of automorphism group size.

    Uses simple heuristics based on symmetry indicators to estimate
    the size of the graph's automorphism group.

    Args:
        graph: NetworkX graph object

    Returns:
        Estimated automorphism group size
    """
    n = len(graph.nodes())
    if n <= 1:
        return 1

    # Check for some common symmetric structures
    if nx.is_regular(graph):
        degree = list(graph.degree())[0][1]
        if degree == n - 1:  # Complete graph
            return math.factorial(n)
        elif degree == 0:  # Empty graph
            return math.factorial(n)
        elif degree == 1:  # Matching or path-like
            # Rough estimate for matching-like structures
            return 2 ** (n // 2)

    # Default conservative estimate
    return max(1, n // 4)


def estimate_orbit_count(graph: nx.Graph) -> int:
    """
    Rough estimation of number of orbits under automorphism group.

    Groups nodes by degree sequence and other simple invariants
    to estimate the number of orbits.

    Args:
        graph: NetworkX graph object

    Returns:
        Estimated number of orbits
    """
    degree_sequence = sorted([graph.degree(node) for node in graph.nodes()])
    unique_degrees = len(set(degree_sequence))

    # Very rough heuristic
    return min(len(graph.nodes()), max(1, unique_degrees))


def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of 2.

    Args:
        n: Integer to check

    Returns:
        True if n is a power of 2, False otherwise
    """
    return n > 0 and (n & (n - 1)) == 0


def compute_hamming_statistics(edges: set) -> Dict:
    """
    Compute Hamming distance statistics for a set of bitstring edges.

    Args:
        edges: Set of edge tuples with binary string vertices

    Returns:
        Dictionary with Hamming distance statistics
    """
    if not edges:
        return {"total_hamming": 0, "avg_hamming": 0, "hamming_1_count": 0, "hamming_gt1_count": 0}

    hamming_distances = []
    hamming_1_count = 0
    hamming_gt1_count = 0

    for v1, v2 in edges:
        dist = sum(c1 != c2 for c1, c2 in zip(v1, v2))
        hamming_distances.append(dist)
        if dist == 1:
            hamming_1_count += 1
        else:
            hamming_gt1_count += 1

    return {
        "total_hamming": sum(hamming_distances),
        "avg_hamming": np.mean(hamming_distances) if hamming_distances else 0,
        "min_hamming": min(hamming_distances) if hamming_distances else 0,
        "max_hamming": max(hamming_distances) if hamming_distances else 0,
        "hamming_1_count": hamming_1_count,
        "hamming_gt1_count": hamming_gt1_count,
    }


# Simple wrapper functions for NetworkX graph properties

def count_edges(graph: nx.Graph) -> int:
    """Return the number of edges in the graph."""
    return nx.number_of_edges(graph)


def calculate_edge_density(graph: nx.Graph) -> float:
    """Calculate the edge density of the graph."""
    return nx.density(graph)


def is_bipartite(graph: nx.Graph) -> bool:
    """Check if the graph is bipartite."""
    return nx.is_bipartite(graph)


def is_connected(graph: nx.Graph) -> bool:
    """Check if the graph is connected."""
    return nx.is_connected(graph)


def find_diameter(graph: nx.Graph) -> float:
    """
    Find the diameter of the graph.

    Returns infinity for disconnected graphs.
    """
    if not nx.is_connected(graph):
        return float("inf")
    return nx.diameter(graph)


def find_max_clique(graph: nx.Graph) -> int:
    """Find the size of the maximum clique in the graph."""
    return len(max(nx.find_cliques(graph), key=len, default=[]))


def average_clustering(graph: nx.Graph) -> float:
    """Calculate the average clustering coefficient of the graph."""
    return nx.average_clustering(graph)


def hamming_distance(s1: str, s2: str) -> int:
    """
    Calculate the Hamming distance between two binary strings.

    Args:
        s1: First binary string
        s2: Second binary string

    Returns:
        Number of positions where the strings differ
    """
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))
