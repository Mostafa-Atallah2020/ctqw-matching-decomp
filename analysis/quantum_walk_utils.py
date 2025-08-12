# quantum_walk_utils.py

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from scipy.linalg import expm

from src.graphs import IntersectingEdgesGraph, MultiEdgeGraph, StaticGraph


@dataclass
class CircuitMetrics:
    cx_count: int
    u3_count: int
    depth: int


@dataclass
class MatchingInfo:
    """Information about a matching and its bit pattern"""
    edges: Set[Tuple[int, int]]
    bit_positions: List[int]  # Which bit positions this matching flips
    gray_code_index: int  # Position in Gray code ordering


class Logger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message: str):
        """Log message with timestamp to file only."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")
            f.flush()

    def log_metrics(self, method: str, cx_count: int, u3_count: int, depth: int):
        """Log circuit metrics."""
        self.log(f"{method} - CX: {cx_count}, U3: {u3_count}, Depth: {depth}")

    def log_comparison(self, graph_index: int, category: str, exact_diff: Tuple[int, int], 
                      pauli_diff: Tuple[int, int], is_bipartite: bool):
        """Log comparison results for a graph."""
        self.log(f"Graph {graph_index} Results:")
        self.log(f"Category: {category}")
        self.log(f"Matching vs Exact: {'win' if exact_diff[0] < 0 else 'lose' if exact_diff[0] > 0 else 'draw'} "
                f"(CX diff: {exact_diff[0]}, U3 diff: {exact_diff[1]})")
        self.log(f"Matching vs Pauli: {'win' if pauli_diff[0] < 0 else 'lose' if pauli_diff[0] > 0 else 'draw'} "
                f"(CX diff: {pauli_diff[0]}, U3 diff: {pauli_diff[1]})")
        self.log(f"Bipartite: {is_bipartite}")

    def log_labeling_comparison(self, graph_index: int, original_cost: int, matching_based_cost: int,
                               original_metrics: Optional[CircuitMetrics], matching_based_metrics: Optional[CircuitMetrics]):
        """Log labeling method comparison results."""
        self.log(f"Graph {graph_index} Labeling Comparison:")
        improvement = original_cost - matching_based_cost
        self.log(f"Original Hamming cost: {original_cost}")
        self.log(f"Matching-based Hamming cost: {matching_based_cost}")
        self.log(f"Hamming improvement: {improvement}")
        
        if original_metrics and matching_based_metrics:
            cx_improvement = original_metrics.cx_count - matching_based_metrics.cx_count
            u3_improvement = original_metrics.u3_count - matching_based_metrics.u3_count
            depth_improvement = original_metrics.depth - matching_based_metrics.depth
            self.log(f"Circuit improvements - CX: {cx_improvement}, U3: {u3_improvement}, Depth: {depth_improvement}")

    def log_final_stats(self, categories: Dict[str, int], total_graphs: int):
        """Log final statistics."""
        if total_graphs == 0:
            self.log("No graphs were successfully processed")
            self.log("Categories summary:")
            for category, count in categories.items():
                self.log(f"{category}: 0 (0.0%)")
        else:
            for category, count in categories.items():
                percentage = (count / total_graphs) * 100
                self.log(f"{category}: {count} ({percentage:.1f}%)")

    def log_labeling_summary(self, total_graphs: int, total_hamming_improvement: int, 
                           total_cx_improvement: float, total_u3_improvement: float):
        """Log summary statistics for matching-based labeling."""
        self.log("Matching-Based Labeling Summary Statistics:")
        self.log("=" * 40)
        self.log(f"Total graphs processed: {total_graphs}")
        
        if total_graphs > 0:
            avg_hamming_improvement = total_hamming_improvement / total_graphs
            avg_cx_improvement = total_cx_improvement / total_graphs
            avg_u3_improvement = total_u3_improvement / total_graphs
            
            self.log(f"Average Hamming improvement: {avg_hamming_improvement:.2f}")
            self.log(f"Average CX improvement: {avg_cx_improvement:.2f}")
            self.log(f"Average U3 improvement: {avg_u3_improvement:.2f}")


class GrayCode:
    """Utility class for Gray code operations"""
    
    @staticmethod
    def generate_gray_code(n: int) -> List[str]:
        """Generate n-bit Gray code sequence"""
        if n <= 0:
            return ['']
        if n == 1:
            return ['0', '1']
        
        smaller = GrayCode.generate_gray_code(n - 1)
        result = []
        
        # First half: add '0' prefix to smaller Gray code
        for code in smaller:
            result.append('0' + code)
        
        # Second half: add '1' prefix to reversed smaller Gray code
        for code in reversed(smaller):
            result.append('1' + code)
        
        return result
    
    @staticmethod
    def gray_to_binary(gray_str: str) -> str:
        """Convert Gray code string to binary string"""
        if not gray_str:
            return ''
        
        binary = [gray_str[0]]
        for i in range(1, len(gray_str)):
            # XOR current Gray bit with previous binary bit
            binary.append(str(int(binary[i-1]) ^ int(gray_str[i])))
        
        return ''.join(binary)
    
    @staticmethod
    def binary_to_gray(binary_str: str) -> str:
        """Convert binary string to Gray code string"""
        if not binary_str:
            return ''
        
        gray = [binary_str[0]]
        for i in range(1, len(binary_str)):
            # XOR current binary bit with previous binary bit
            gray.append(str(int(binary_str[i-1]) ^ int(binary_str[i])))
        
        return ''.join(gray)


class MatchingBasedHypercubeLabeler:
    """
    Implementation of the matching-based hypercube labeling algorithm.
    
    Algorithm:
    1. Find the biggest matching M in the graph
    2. Label vertices so that for every edge in M, endpoints are 0abcd and 1abcd
    3. For remaining edges, find matchings with structured bit patterns
    4. Use Gray code ordering for matchings to reduce CX gate count
    """
    
    @staticmethod
    def hamming_distance(s1: str, s2: str) -> int:
        """Calculate Hamming distance between two binary strings."""
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    @staticmethod
    def find_maximum_matching(graph: nx.Graph) -> Set[Tuple[int, int]]:
        """Find maximum matching in the graph using NetworkX."""
        matching = nx.max_weight_matching(graph, maxcardinality=True)
        return set(matching)
    
    @staticmethod
    def find_edge_disjoint_matchings(graph: nx.Graph, max_matchings: int = None) -> List[Set[Tuple[int, int]]]:
        """Find multiple edge-disjoint matchings to cover as many edges as possible."""
        matchings = []
        remaining_graph = graph.copy()
        
        while remaining_graph.edges() and (max_matchings is None or len(matchings) < max_matchings):
            # Find maximum matching in remaining graph
            matching = nx.max_weight_matching(remaining_graph, maxcardinality=True)
            if not matching:
                break
            
            matchings.append(set(matching))
            
            # Remove matched edges from remaining graph
            for u, v in matching:
                if remaining_graph.has_edge(u, v):
                    remaining_graph.remove_edge(u, v)
        
        return matchings
    
    @staticmethod
    def get_vertex_cover_matchings(graph: nx.Graph) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """Find two matchings M1 and M2 such that their vertices cover all vertices."""
        # Start with maximum matching
        M1 = MatchingBasedHypercubeLabeler.find_maximum_matching(graph)
        covered_vertices = set()
        for u, v in M1:
            covered_vertices.add(u)
            covered_vertices.add(v)
        
        uncovered_vertices = set(graph.nodes()) - covered_vertices
        
        if not uncovered_vertices:
            # M1 covers all vertices
            return M1, set()
        
        # Create subgraph of edges involving uncovered vertices
        subgraph_edges = []
        for u, v in graph.edges():
            if u in uncovered_vertices or v in uncovered_vertices:
                subgraph_edges.append((u, v))
        
        subgraph = graph.edge_subgraph(subgraph_edges)
        M2 = MatchingBasedHypercubeLabeler.find_maximum_matching(subgraph)
        
        return M1, M2
    
    @staticmethod
    def assign_matching_bit_patterns(matchings: List[Set[Tuple[int, int]]], num_bits: int) -> List[MatchingInfo]:
        """Assign bit patterns to matchings using Gray code ordering."""
        matching_infos = []
        
        # For simplicity, assign each matching to flip a specific bit position
        # In a more sophisticated implementation, we could use multiple bits per matching
        
        gray_codes = GrayCode.generate_gray_code(min(num_bits, len(matchings)))
        
        for i, matching in enumerate(matchings):
            if i < len(gray_codes):
                # Determine which bit positions differ from previous Gray code
                if i == 0:
                    bit_positions = [j for j, bit in enumerate(gray_codes[i]) if bit == '1']
                else:
                    prev_gray = gray_codes[i-1]
                    curr_gray = gray_codes[i]
                    bit_positions = [j for j, (p, c) in enumerate(zip(prev_gray, curr_gray)) if p != c]
                
                matching_infos.append(MatchingInfo(
                    edges=matching,
                    bit_positions=bit_positions if bit_positions else [i % num_bits],
                    gray_code_index=i
                ))
            else:
                # Fallback for excess matchings
                matching_infos.append(MatchingInfo(
                    edges=matching,
                    bit_positions=[i % num_bits],
                    gray_code_index=i
                ))
        
        return matching_infos
    
    @staticmethod
    def calculate_total_hamming_cost(graph: nx.Graph, labeling: Dict[int, str]) -> int:
        """Calculate total Hamming distance across all edges for a given labeling."""
        total_cost = 0
        for u, v in graph.edges():
            total_cost += MatchingBasedHypercubeLabeler.hamming_distance(labeling[u], labeling[v])
        return total_cost
    
    @staticmethod
    def matching_based_hypercube_labeling(graph: nx.Graph) -> Dict[int, str]:
        """
        Main matching-based hypercube labeling algorithm.
        """
        nodes = list(graph.nodes())
        num_nodes = len(nodes)
        
        if num_nodes == 0:
            return {}
        if num_nodes == 1:
            return {nodes[0]: '0'}
        
        # Calculate number of bits needed
        num_bits = max(1, math.ceil(math.log2(num_nodes)))
        
        # Step 1: Find the biggest matching
        max_matching = MatchingBasedHypercubeLabeler.find_maximum_matching(graph)
        
        # Initialize labeling
        labeling = {}
        used_labels = set()
        
        # Step 2: Label vertices in the maximum matching
        # For each edge in matching, endpoints get labels differing in first bit
        available_base_patterns = []
        base_bits = num_bits - 1  # Reserve first bit for matching pair distinction
        
        # Generate all possible base patterns
        for i in range(2**base_bits):
            pattern = format(i, f'0{base_bits}b')
            available_base_patterns.append(pattern)
        
        pattern_idx = 0
        for u, v in max_matching:
            if pattern_idx < len(available_base_patterns):
                base_pattern = available_base_patterns[pattern_idx]
                label_u = '0' + base_pattern
                label_v = '1' + base_pattern
                
                labeling[u] = label_u
                labeling[v] = label_v
                used_labels.add(label_u)
                used_labels.add(label_v)
                pattern_idx += 1
        
        # Step 3: Handle vertices not covered by maximum matching
        covered_vertices = set()
        for u, v in max_matching:
            covered_vertices.add(u)
            covered_vertices.add(v)
        
        uncovered_vertices = set(nodes) - covered_vertices
        
        if uncovered_vertices:
            # If maximum matching doesn't cover all vertices, find additional structure
            if len(uncovered_vertices) > len(available_base_patterns) - pattern_idx:
                # Need to find two matchings that together cover all vertices
                M1, M2 = MatchingBasedHypercubeLabeler.get_vertex_cover_matchings(graph)
                
                # Relabel using the two-matching approach
                labeling = {}
                used_labels = set()
                
                # Label M1 with second bit = 0
                pattern_idx = 0
                for u, v in M1:
                    if pattern_idx < 2**(num_bits - 2):
                        base_pattern = format(pattern_idx, f'0{num_bits-2}b')
                        label_u = '00' + base_pattern
                        label_v = '10' + base_pattern
                        
                        labeling[u] = label_u
                        labeling[v] = label_v
                        used_labels.add(label_u)
                        used_labels.add(label_v)
                        pattern_idx += 1
                
                # Label M2 with second bit = 1
                pattern_idx = 0
                for u, v in M2:
                    if u not in labeling and v not in labeling:
                        if pattern_idx < 2**(num_bits - 2):
                            base_pattern = format(pattern_idx, f'0{num_bits-2}b')
                            label_u = '01' + base_pattern
                            label_v = '11' + base_pattern
                            
                            labeling[u] = label_u
                            labeling[v] = label_v
                            used_labels.add(label_u)
                            used_labels.add(label_v)
                            pattern_idx += 1
                
                # Handle any remaining uncovered vertices
                remaining_vertices = set(nodes) - set(labeling.keys())
                for vertex in remaining_vertices:
                    for candidate_i in range(2**num_bits):
                        candidate = format(candidate_i, f'0{num_bits}b')
                        if candidate not in used_labels:
                            labeling[vertex] = candidate
                            used_labels.add(candidate)
                            break
            else:
                # Assign remaining vertices to available patterns
                for vertex in uncovered_vertices:
                    if pattern_idx < len(available_base_patterns):
                        # Try to place in a way that minimizes cost to existing neighbors
                        base_pattern = available_base_patterns[pattern_idx]
                        best_label = None
                        min_cost = float('inf')
                        
                        for prefix in ['0', '1']:
                            candidate = prefix + base_pattern
                            if candidate not in used_labels:
                                cost = 0
                                for neighbor in graph.neighbors(vertex):
                                    if neighbor in labeling:
                                        cost += MatchingBasedHypercubeLabeler.hamming_distance(
                                            candidate, labeling[neighbor])
                                
                                if cost < min_cost:
                                    min_cost = cost
                                    best_label = candidate
                        
                        if best_label:
                            labeling[vertex] = best_label
                            used_labels.add(best_label)
                            pattern_idx += 1
                
                # Handle any remaining vertices with simple assignment
                remaining_vertices = set(nodes) - set(labeling.keys())
                for vertex in remaining_vertices:
                    for candidate_i in range(2**num_bits):
                        candidate = format(candidate_i, f'0{num_bits}b')
                        if candidate not in used_labels:
                            labeling[vertex] = candidate
                            used_labels.add(candidate)
                            break
        
        return labeling
    
    @staticmethod
    def analyze_labeling_quality(graph: nx.Graph, labeling: Dict[int, str]) -> Dict[str, float]:
        """Analyze the quality of a labeling by computing statistics about edge Hamming distances."""
        if not graph.edges():
            return {'total_cost': 0, 'avg_cost': 0, 'edges_with_dist_1': 0, 'edges_with_dist_1_ratio': 0}
        
        hamming_distances = []
        edges_with_dist_1 = 0
        hamming_dist_counts = defaultdict(int)
        
        for u, v in graph.edges():
            dist = MatchingBasedHypercubeLabeler.hamming_distance(labeling[u], labeling[v])
            hamming_distances.append(dist)
            hamming_dist_counts[dist] += 1
            if dist == 1:
                edges_with_dist_1 += 1
        
        total_cost = sum(hamming_distances)
        avg_cost = total_cost / len(hamming_distances)
        edges_with_dist_1_ratio = edges_with_dist_1 / len(hamming_distances)
        
        # Find most common Hamming distance
        most_common_dist = max(hamming_dist_counts.keys(), key=lambda k: hamming_dist_counts[k])
        most_common_count = hamming_dist_counts[most_common_dist]
        most_common_ratio = most_common_count / len(hamming_distances)
        
        return {
            'total_cost': total_cost,
            'avg_cost': avg_cost,
            'edges_with_dist_1': edges_with_dist_1,
            'edges_with_dist_1_ratio': edges_with_dist_1_ratio,
            'max_hamming_dist': max(hamming_distances) if hamming_distances else 0,
            'min_hamming_dist': min(hamming_distances) if hamming_distances else 0,
            'most_common_hamming_dist': most_common_dist,
            'most_common_hamming_ratio': most_common_ratio,
            'hamming_dist_distribution': dict(hamming_dist_counts)
        }


# Keep the old DegreeBasedHypercubeLabeler for comparison
class DegreeBasedHypercubeLabeler:
    """Original degree-based implementation for comparison"""
    
    @staticmethod
    def hamming_distance(s1: str, s2: str) -> int:
        """Calculate Hamming distance between two binary strings."""
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    @staticmethod
    def get_hamming_neighbors(binary_str: str, used_labels: Set[str]) -> List[str]:
        """Get all possible Hamming distance 1 neighbors that aren't already used."""
        neighbors = []
        for i in range(len(binary_str)):
            neighbor = list(binary_str)
            neighbor[i] = '1' if neighbor[i] == '0' else '0'
            neighbor_str = ''.join(neighbor)
            if neighbor_str not in used_labels:
                neighbors.append(neighbor_str)
        return neighbors
    
    @staticmethod
    def calculate_total_hamming_cost(graph: nx.Graph, labeling: Dict[int, str]) -> int:
        """Calculate total Hamming distance across all edges for a given labeling."""
        total_cost = 0
        for u, v in graph.edges():
            total_cost += DegreeBasedHypercubeLabeler.hamming_distance(labeling[u], labeling[v])
        return total_cost
    
    @staticmethod
    def degree_based_hypercube_labeling(graph: nx.Graph) -> Dict[int, str]:
        """Original degree-based labeling algorithm."""
        nodes = list(graph.nodes())
        num_nodes = len(nodes)
        
        if num_nodes == 0:
            return {}
        if num_nodes == 1:
            return {nodes[0]: '0'}
        
        num_bits = max(1, math.ceil(math.log2(num_nodes)))
        
        # Find vertex with maximum degree
        degrees = dict(graph.degree())
        M = max(degrees.keys(), key=lambda v: degrees[v])
        
        labeling = {}
        used_labels = set()
        bit_positions_used = set()
        
        # Assign string z to M (start with all zeros)
        z = '0' * num_bits
        labeling[M] = z
        used_labels.add(z)
        
        # Get neighbors of M sorted by degree
        neighbors_M = list(graph.neighbors(M))
        neighbors_M.sort(key=lambda v: degrees[v], reverse=True)
        
        available_bit_positions = list(range(num_bits))
        
        for u in neighbors_M:
            if u not in labeling:
                bit_to_flip = None
                for pos in available_bit_positions:
                    if pos not in bit_positions_used:
                        bit_to_flip = pos
                        bit_positions_used.add(pos)
                        break
                
                if bit_to_flip is not None:
                    u_label = list(z)
                    u_label[bit_to_flip] = '1' if u_label[bit_to_flip] == '0' else '0'
                    u_label = ''.join(u_label)
                    
                    labeling[u] = u_label
                    used_labels.add(u_label)
                else:
                    available_h1 = DegreeBasedHypercubeLabeler.get_hamming_neighbors(z, used_labels)
                    if available_h1:
                        labeling[u] = available_h1[0]
                        used_labels.add(available_h1[0])
        
        # Process remaining vertices using breadth-first expansion
        queue = deque([M])
        processed = {M}
        
        while queue:
            current = queue.popleft()
            current_label = labeling[current]
            
            unprocessed_neighbors = [n for n in graph.neighbors(current) if n not in processed]
            
            for neighbor in unprocessed_neighbors:
                if neighbor not in labeling:
                    available_h1 = DegreeBasedHypercubeLabeler.get_hamming_neighbors(current_label, used_labels)
                    
                    if available_h1:
                        best_label = None
                        min_cost = float('inf')
                        
                        for candidate in available_h1:
                            cost = 0
                            for existing_neighbor in graph.neighbors(neighbor):
                                if existing_neighbor in labeling:
                                    cost += DegreeBasedHypercubeLabeler.hamming_distance(candidate, labeling[existing_neighbor])
                            
                            if cost < min_cost:
                                min_cost = cost
                                best_label = candidate
                        
                        if best_label:
                            labeling[neighbor] = best_label
                            used_labels.add(best_label)
                            processed.add(neighbor)
                            queue.append(neighbor)
        
        # Handle remaining unlabeled vertices
        remaining_vertices = [v for v in nodes if v not in labeling]
        
        for vertex in remaining_vertices:
            best_label = None
            min_cost = float('inf')
            
            for candidate_i in range(2**num_bits):
                candidate = format(candidate_i, f'0{num_bits}b')
                if candidate not in used_labels:
                    total_cost = 0
                    for neighbor in graph.neighbors(vertex):
                        if neighbor in labeling:
                            total_cost += DegreeBasedHypercubeLabeler.hamming_distance(candidate, labeling[neighbor])
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_label = candidate
            
            if best_label:
                labeling[vertex] = best_label
                used_labels.add(best_label)
            else:
                for candidate_i in range(2**num_bits):
                    candidate = format(candidate_i, f'0{num_bits}b')
                    if candidate not in used_labels:
                        labeling[vertex] = candidate
                        used_labels.add(candidate)
                        break
        
        return labeling
    
    @staticmethod
    def analyze_labeling_quality(graph: nx.Graph, labeling: Dict[int, str]) -> Dict[str, float]:
        """Analyze the quality of a labeling."""
        if not graph.edges():
            return {'total_cost': 0, 'avg_cost': 0, 'edges_with_dist_1': 0, 'edges_with_dist_1_ratio': 0}
        
        hamming_distances = []
        edges_with_dist_1 = 0
        hamming_dist_counts = defaultdict(int)
        
        for u, v in graph.edges():
            dist = DegreeBasedHypercubeLabeler.hamming_distance(labeling[u], labeling[v])
            hamming_distances.append(dist)
            hamming_dist_counts[dist] += 1
            if dist == 1:
                edges_with_dist_1 += 1
        
        total_cost = sum(hamming_distances)
        avg_cost = total_cost / len(hamming_distances)
        edges_with_dist_1_ratio = edges_with_dist_1 / len(hamming_distances)
        
        most_common_dist = max(hamming_dist_counts.keys(), key=lambda k: hamming_dist_counts[k])
        most_common_count = hamming_dist_counts[most_common_dist]
        most_common_ratio = most_common_count / len(hamming_distances)
        
        return {
            'total_cost': total_cost,
            'avg_cost': avg_cost,
            'edges_with_dist_1': edges_with_dist_1,
            'edges_with_dist_1_ratio': edges_with_dist_1_ratio,
            'max_hamming_dist': max(hamming_distances) if hamming_distances else 0,
            'min_hamming_dist': min(hamming_distances) if hamming_distances else 0,
            'most_common_hamming_dist': most_common_dist,
            'most_common_hamming_ratio': most_common_ratio,
            'hamming_dist_distribution': dict(hamming_dist_counts)
        }


class HeuristicHypercubeLabeler:
    """
    Heuristic hypercube labeler that uses the new matching-based approach.
    """
    
    @staticmethod
    def hamming_distance(s1: str, s2: str) -> int:
        """Calculate Hamming distance between two binary strings."""
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    @staticmethod
    def calculate_total_hamming_cost(graph: nx.Graph, labeling: Dict[int, str]) -> int:
        """Calculate total Hamming distance across all edges for a given labeling."""
        total_cost = 0
        for u, v in graph.edges():
            total_cost += HeuristicHypercubeLabeler.hamming_distance(labeling[u], labeling[v])
        return total_cost
    
    @staticmethod
    def heuristic_hypercube_labeling(graph: nx.Graph) -> Dict[int, str]:
        """Find hypercube labeling using the matching-based heuristic algorithm."""
        nodes = list(graph.nodes())
        num_nodes = len(nodes)
        
        if num_nodes == 0:
            return {}
        
        if num_nodes == 1:
            return {nodes[0]: '0'}
        
        # Use the new matching-based heuristic algorithm
        try:
            matching_based_result = MatchingBasedHypercubeLabeler.matching_based_hypercube_labeling(graph)
            if matching_based_result and len(matching_based_result) == num_nodes:
                # Verify all nodes are labeled
                if set(matching_based_result.keys()) == set(nodes):
                    return matching_based_result
        except Exception as e:
            print(f"Warning: Matching-based heuristic failed: {e}")
        
        # Fallback to degree-based approach
        try:
            degree_based_result = DegreeBasedHypercubeLabeler.degree_based_hypercube_labeling(graph)
            if degree_based_result and len(degree_based_result) == num_nodes:
                if set(degree_based_result.keys()) == set(nodes):
                    return degree_based_result
        except Exception as e:
            print(f"Warning: Degree-based heuristic also failed: {e}")
        
        # Ultimate fallback: simple sequential labeling
        print(f"Using simple sequential fallback for graph with {num_nodes} nodes")
        num_bits = max(1, math.ceil(math.log2(num_nodes)))
        
        # Sort by degree (descending), then by node ID for consistency
        degree_sorted_nodes = sorted(nodes, key=lambda v: (-graph.degree(v), v))
        
        fallback_labeling = {}
        for i, vertex in enumerate(degree_sorted_nodes):
            fallback_labeling[vertex] = format(i, f'0{num_bits}b')
        
        return fallback_labeling


class GraphProcessor:
    @staticmethod
    def graph_to_bitstring(graph: nx.Graph) -> Set[Tuple[str, str]]:
        """Original simple labeling method."""
        num_nodes = len(graph.nodes)
        if num_nodes <= 1:
            return set()
        num_bits = len(bin(num_nodes - 1)) - 2
        node_to_bitstring = {node: format(node, f"0{num_bits}b") for node in graph.nodes}
        return {(node_to_bitstring[u], node_to_bitstring[v]) for u, v in graph.edges}
    
    @staticmethod
    def graph_to_bitstring_hypercube(graph: nx.Graph) -> Set[Tuple[str, str]]:
        """Convert graph to bitstring with matching-based hypercube labeling."""
        heuristic_labeling = HeuristicHypercubeLabeler.heuristic_hypercube_labeling(graph)
        return {(heuristic_labeling[u], heuristic_labeling[v]) for u, v in graph.edges}

    @staticmethod
    def compare_labeling_methods(graph: nx.Graph) -> Dict[str, Dict]:
        """Compare original vs matching-based hypercube labeling methods and their Hamming costs."""
        results = {}
        
        # Original method
        original_edges = GraphProcessor.graph_to_bitstring(graph)
        original_cost = sum(HeuristicHypercubeLabeler.hamming_distance(u, v) for u, v in original_edges)
        results['original'] = {'edges': original_edges, 'cost': original_cost}
        
        # Matching-based method
        try:
            matching_based_edges = GraphProcessor.graph_to_bitstring_hypercube(graph)
            matching_based_cost = sum(HeuristicHypercubeLabeler.hamming_distance(u, v) for u, v in matching_based_edges)
            
            # Get detailed analysis of the labeling
            matching_based_labeling = HeuristicHypercubeLabeler.heuristic_hypercube_labeling(graph)
            quality_stats = MatchingBasedHypercubeLabeler.analyze_labeling_quality(graph, matching_based_labeling)
            
            results['matching_based'] = {
                'edges': matching_based_edges, 
                'cost': matching_based_cost,
                'quality_stats': quality_stats
            }
        except Exception as e:
            results['matching_based'] = {'edges': None, 'cost': float('inf'), 'error': str(e)}
        
        return results

    @staticmethod
    def parse_graph_filename(filepath: str) -> Dict[str, str]:
        """Parse graph filename to extract metadata."""
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]

        try:
            import re
            num_match = re.match(r"(\d+)graph_", base_name)
            if not num_match:
                raise ValueError("Could not find number of graphs")
            num_graphs = int(num_match.group(1))

            parts = base_name.split("graph_")[1].split("_")
            size = parts[0]
            vertex_info = parts[1]
            vertices = vertex_info[:-1]
            graph_type = vertex_info[-1]

            return {
                "num_graphs": num_graphs,
                "size": size,
                "vertices": vertices,
                "type": graph_type,
            }
        except Exception as e:
            raise ValueError(f"Invalid filename format. Expected '<num>graph_<size>_<vertices>c.g6', got: {filename}")


class BaseAnalyzer:
    def __init__(self, n_qubits: int, delta_t: float, matchings: str = "greedy", seed: int = 0):
        self.n_qubits = n_qubits
        self.delta_t = delta_t
        self.matchings = matchings
        self.seed = seed

    def analyze_circuit(self, qc: QuantumCircuit, runs: int = 10) -> CircuitMetrics:
        """Analyze circuit with fixed transpilation settings over multiple runs and return the average counts."""
        total_metrics = CircuitMetrics(cx_count=0, u3_count=0, depth=0)

        for _ in range(runs):
            try:
                transpiled_qc = transpile(
                    qc,
                    basis_gates=["cx", "u3"],
                    optimization_level=3,
                    routing_method="basic",
                    layout_method="trivial",
                )

                counts = transpiled_qc.count_ops()
                current_metrics = CircuitMetrics(
                    cx_count=counts.get("cx", 0),
                    u3_count=counts.get("u3", 0),
                    depth=transpiled_qc.depth(),
                )

                total_metrics.cx_count += current_metrics.cx_count
                total_metrics.u3_count += current_metrics.u3_count
                total_metrics.depth += current_metrics.depth

            except Exception as e:
                print(f"Transpilation error: {str(e)}")

        # Calculate averages
        average_metrics = CircuitMetrics(
            cx_count=total_metrics.cx_count / runs,
            u3_count=total_metrics.u3_count / runs,
            depth=total_metrics.depth / runs,
        )

        return average_metrics

    def analyze_matching(self, edges: Set[Tuple[str, str]], n_steps: int = 1) -> Optional[CircuitMetrics]:
        """Analyze circuit using matching method."""
        try:
            static_G = StaticGraph(edges)
            intersecting_G = IntersectingEdgesGraph(edges, self.matchings)

            qc = QuantumCircuit(static_G.n_qubits)
            for _ in range(n_steps):
                for subgraph in intersecting_G.subgraphs:
                    G = MultiEdgeGraph(subgraph.edges)
                    sub_qc = G.get_qc(simplified=False)
                    qc = qc.compose(sub_qc)

            metrics = self.analyze_circuit(qc)
            if metrics is None:
                raise Exception("Failed to analyze circuit")
            return metrics

        except Exception as e:
            print(f"Matching analysis error: {str(e)}")
            return None

    def analyze_exact(self, edges: Set[Tuple[str, str]]) -> Optional[CircuitMetrics]:
        """Analyze circuit using exact method."""
        try:
            static_G = StaticGraph(edges)
            H = -1j * self.delta_t * static_G.get_adj_mat()
            U = Operator(expm(H))

            qc = QuantumCircuit(static_G.n_qubits)
            qc.unitary(U, range(static_G.n_qubits))

            metrics = self.analyze_circuit(qc)
            if metrics is None:
                raise Exception("Failed to analyze circuit")
            return metrics

        except Exception as e:
            print(f"Exact analysis error: {str(e)}")
            return None

    def analyze_pauli(self, edges: Set[Tuple[str, str]]) -> Optional[CircuitMetrics]:
        """Analyze circuit using Pauli decomposition method."""
        try:
            static_G = StaticGraph(edges)
            H = static_G.get_adj_mat()
            n = static_G.n_qubits

            # Decompose the Hamiltonian into Pauli basis
            pauli_strings = []
            real_coeffs = []
            imag_coeffs = []

            for pauli_string in ["".join(p) for p in itertools.product("IXYZ", repeat=n)]:
                P = Pauli(pauli_string)
                P_op = Operator(P).data
                coeff = np.trace(P_op.conj().T @ H) / (2**n)
                if not np.isclose(coeff, 0, atol=1e-10):
                    pauli_strings.append(pauli_string)
                    real_coeffs.append(float(np.real(coeff)))
                    imag_coeffs.append(float(np.imag(coeff)))

            # Create quantum circuit
            qc = QuantumCircuit(n)

            # Real part evolution
            if any(c != 0 for c in real_coeffs):
                real_pauli_op = SparsePauliOp(pauli_strings, real_coeffs)
                real_evo_gate = PauliEvolutionGate(real_pauli_op, time=-self.delta_t)
                qc.append(real_evo_gate, range(n))

            # Imaginary part evolution
            if any(c != 0 for c in imag_coeffs):
                imag_pauli_op = SparsePauliOp(pauli_strings, imag_coeffs)
                imag_evo_gate = PauliEvolutionGate(imag_pauli_op, time=-self.delta_t)
                qc.append(imag_evo_gate, range(n))

            metrics = self.analyze_circuit(qc)
            if metrics is None:
                raise Exception("Failed to analyze circuit")
            return metrics

        except Exception as e:
            print(f"Pauli analysis error: {str(e)}")
            return None


class HypercubeLabelingAnalyzer(BaseAnalyzer):
    """Analyzer that uses matching-based hypercube embedding labeling."""
    
    def __init__(self, n_qubits: int, delta_t: float, matchings: str = "greedy", seed: int = 0):
        super().__init__(n_qubits, delta_t, matchings, seed)
    
    def analyze_matching_with_hypercube(self, graph: nx.Graph, n_steps: int = 1) -> Optional[CircuitMetrics]:
        """Analyze circuit using matching method with hypercube labeling."""
        try:
            edges = GraphProcessor.graph_to_bitstring_hypercube(graph)
            return self.analyze_matching(edges, n_steps)
        except Exception as e:
            print(f"Matching analysis error: {str(e)}")
            return None

    def analyze_exact_with_hypercube(self, graph: nx.Graph) -> Optional[CircuitMetrics]:
        """Analyze circuit using exact method with hypercube labeling."""
        try:
            edges = GraphProcessor.graph_to_bitstring_hypercube(graph)
            return self.analyze_exact(edges)
        except Exception as e:
            print(f"Exact analysis error: {str(e)}")
            return None

    def analyze_pauli_with_hypercube(self, graph: nx.Graph) -> Optional[CircuitMetrics]:
        """Analyze circuit using Pauli decomposition with hypercube labeling."""
        try:
            edges = GraphProcessor.graph_to_bitstring_hypercube(graph)
            return self.analyze_pauli(edges)
        except Exception as e:
            print(f"Pauli analysis error: {str(e)}")
            return None
    
    def compare_labeling_impact(self, graph: nx.Graph) -> Dict[str, Dict]:
        """Compare original vs hypercube labeling methods and their circuit metrics."""
        results = {}
        
        # Original labeling
        try:
            original_edges = GraphProcessor.graph_to_bitstring(graph)
            original_cost = sum(sum(c1 != c2 for c1, c2 in zip(u, v)) for u, v in original_edges)
            
            original_matching = self.analyze_matching(original_edges)
            original_exact = self.analyze_exact(original_edges)
            original_pauli = self.analyze_pauli(original_edges)
            
            results['original'] = {
                'hamming_cost': original_cost,
                'edges': original_edges,
                'matching_metrics': original_matching,
                'exact_metrics': original_exact, 
                'pauli_metrics': original_pauli
            }
            
        except Exception as e:
            results['original'] = {
                'error': str(e),
                'hamming_cost': float('inf'),
                'edges': None,
                'matching_metrics': None,
                'exact_metrics': None,
                'pauli_metrics': None
            }
        
        # Hypercube labeling
        try:
            hypercube_edges = GraphProcessor.graph_to_bitstring_hypercube(graph)
            hypercube_cost = sum(sum(c1 != c2 for c1, c2 in zip(u, v)) for u, v in hypercube_edges)
            
            hypercube_matching = self.analyze_matching_with_hypercube(graph)
            hypercube_exact = self.analyze_exact_with_hypercube(graph)
            hypercube_pauli = self.analyze_pauli_with_hypercube(graph)
            
            # Get quality statistics for the labeling
            hypercube_labeling = HeuristicHypercubeLabeler.heuristic_hypercube_labeling(graph)
            quality_stats = MatchingBasedHypercubeLabeler.analyze_labeling_quality(graph, hypercube_labeling)
            
            results['hypercube'] = {
                'hamming_cost': hypercube_cost,
                'edges': hypercube_edges,
                'matching_metrics': hypercube_matching,
                'exact_metrics': hypercube_exact, 
                'pauli_metrics': hypercube_pauli,
                'quality_stats': quality_stats
            }
            
        except Exception as e:
            results['hypercube'] = {
                'error': str(e),
                'hamming_cost': float('inf'),
                'edges': None,
                'matching_metrics': None,
                'exact_metrics': None,
                'pauli_metrics': None,
                'quality_stats': None
            }
        
        return results


class ResultsManager:
    def __init__(self, base_dir: str, graph_info: Dict[str, str]):
        self.base_dir = base_dir
        self.graph_info = graph_info

    def get_output_path(self, category: str, ext: str) -> str:
        filename = f"{category}_{self.graph_info['size']}_{self.graph_info['vertices']}c.{ext}"
        return os.path.join(self.base_dir, filename)

    def save_results(self, results: Dict, category: str):
        output_path = self.get_output_path(category, "txt")
        with open(output_path, "w") as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

    def save_categorized_graphs(self, results: List[Dict], original_graphs: List[nx.Graph]):
        """Save graphs to separate g6 files based on their categories."""
        graphs_by_category = defaultdict(list)

        for result, graph in zip(results, original_graphs):
            category = result["category"]
            graphs_by_category[category].append(graph)

        for category, graphs in graphs_by_category.items():
            output_path = self.get_output_path(category, "g6")
            with open(output_path, "w") as f:
                for G in graphs:
                    g6_string = nx.to_graph6_bytes(G, header=False).decode().strip()
                    f.write(f"{g6_string}\n")


class PlotManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_plot(self, plt_figure: plt.Figure, filename: str):
        plt_figure.savefig(os.path.join(self.output_dir, filename))
        plt.close(plt_figure)

    def create_pie_chart(self, categories: Dict[str, int], title: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 8))
        labels = list(categories.keys())
        sizes = list(categories.values())

        ax.pie(sizes, labels=labels, autopct="%1.1f%%")
        ax.set_title(title)

        return fig


def setup_directories(base_dir: str) -> Dict[str, str]:
    directories = {
        "logs": os.path.join(base_dir, "logs"),
        "plots": os.path.join(base_dir, "plots"),
        "results": os.path.join(base_dir, "results"),
    }
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    return directories