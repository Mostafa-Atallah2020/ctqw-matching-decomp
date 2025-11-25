from __future__ import annotations
from typing import List, Set, Tuple, TYPE_CHECKING
from collections import defaultdict
import random

import numpy as np
from qiskit import QuantumCircuit
from src.utils.circuit.graph_simplifier import build_matching_circuit_iteratively

if TYPE_CHECKING:
    from src.core.multi_edge_graph import MultiEdgeGraph


class MatchingDecomposition:
    """
    Quantum circuit construction using matching-based decomposition for CTQW.

    This class implements the matching-based Trotterization approach for
    Continuous-Time Quantum Walks (CTQW). It decomposes the graph Hamiltonian
    into matchings and builds quantum circuits accordingly.

    Attributes:
        graph: MultiEdgeGraph instance
        n_qubits: Number of qubits in the system
        edges: Set of edges defining the graph
        matchings: List of matchings computed from the edges
    """

    def __init__(self, graph: MultiEdgeGraph):
        """
        Initialize MatchingDecomposition with a MultiEdgeGraph.

        Args:
            graph: MultiEdgeGraph instance containing the graph edges

        Raises:
            TypeError: If graph is not a MultiEdgeGraph instance
        """
        # Check by class name to avoid issues with module reloading in notebooks
        if type(graph).__name__ != 'MultiEdgeGraph':
            raise TypeError(
                f"Expected MultiEdgeGraph, got {type(graph).__name__}. "
                "Please create a MultiEdgeGraph from your edges first."
            )

        self.graph = graph
        self.edges = graph.edges
        self.n_qubits = graph.n_qubits
        self.matchings = self._compute_matchings(self.edges)

    def _get_bit_flip_position(self, edge: Tuple[str, str]) -> int:
        """
        Determine which bit position differs between edge endpoints.

        Args:
            edge: Tuple of two binary strings

        Returns:
            Bit position that differs, -1 for self-loop, -2 for multi-bit difference
        """
        u, v = edge

        if isinstance(u, str) and isinstance(v, str):
            try:
                u_int = int(u, 2)
                v_int = int(v, 2)
            except ValueError:
                return -2
        else:
            u_int, v_int = u, v

        diff = u_int ^ v_int
        if diff == 0:
            return -1

        if bin(diff).count("1") == 1:
            position = (diff & -diff).bit_length() - 1
            return position
        else:
            return -2

    def _is_complete_bipartite(self, edge_list: List[Tuple[str, str]]) -> Tuple[bool, List[str], List[str]]:
        """
        Check if edges form a complete bipartite graph.

        Args:
            edge_list: List of edges

        Returns:
            Tuple of (is_complete_bipartite, left_set, right_set)
        """
        vertices = set()
        for u, v in edge_list:
            vertices.add(u)
            vertices.add(v)

        set_0 = {v for v in vertices if v[0] == '0'}
        set_1 = {v for v in vertices if v[0] == '1'}

        if len(set_0) != len(set_1):
            return False, None, None

        expected_edges = set()
        for u in set_0:
            for v in set_1:
                edge = tuple(sorted([u, v]))
                expected_edges.add(edge)

        actual_edges = {tuple(sorted([u, v])) for u, v in edge_list}

        if expected_edges == actual_edges:
            return True, sorted(set_0), sorted(set_1)

        return False, None, None

    def _matchings_commute(self, matching1: Set, matching2: Set, left_set: List[str], right_set: List[str]) -> bool:
        """
        Check if two matchings have commuting Hamiltonians.

        Args:
            matching1: First matching
            matching2: Second matching
            left_set: Left partition vertices
            right_set: Right partition vertices

        Returns:
            True if matchings commute
        """
        all_vertices = sorted(set(left_set) | set(right_set))
        n = len(all_vertices)
        vertex_to_idx = {v: i for i, v in enumerate(all_vertices)}

        A = np.zeros((n, n), dtype=float)
        B = np.zeros((n, n), dtype=float)

        for u, v in matching1:
            i, j = vertex_to_idx[u], vertex_to_idx[v]
            A[i, j] = 1.0
            A[j, i] = 1.0

        for u, v in matching2:
            i, j = vertex_to_idx[u], vertex_to_idx[v]
            B[i, j] = 1.0
            B[j, i] = 1.0

        AB = A @ B
        BA = B @ A
        commutator_norm = np.linalg.norm(AB - BA)

        return commutator_norm < 1e-10

    def _compute_matchings(self, edges: Set[Tuple[str, str]]) -> List[Set[Tuple[str, str]]]:
        """
        Compute optimal matchings for the graph edges.

        Decomposes graph edges into matchings (sets of non-adjacent edges)
        that can be executed in parallel during Trotterization.

        Args:
            edges: Set of graph edges

        Returns:
            List of matchings, where each matching is a set of edges
        """
        edges_list = list(edges)
        is_complete, left_set, right_set = self._is_complete_bipartite(edges_list)

        if is_complete:
            return self._compute_bipartite_matchings(edges_list, left_set, right_set)

        return self._compute_greedy_matchings(edges_list)

    def _compute_bipartite_matchings(
        self,
        edges_list: List[Tuple[str, str]],
        left_set: List[str],
        right_set: List[str]
    ) -> List[Set[Tuple[str, str]]]:
        """
        Compute matchings for complete bipartite graphs using commuting strategy.

        Args:
            edges_list: List of edges
            left_set: Left partition vertices
            right_set: Right partition vertices

        Returns:
            List of matchings
        """
        import networkx as nx
        from networkx.algorithms import bipartite

        G = nx.Graph()
        G.add_edges_from(edges_list)

        left_nodes = set(left_set)
        right_nodes = set(right_set)

        def try_generate_commuting_matchings(edge_set, num_attempts=10):
            if not edge_set:
                return []

            H = G.edge_subgraph(edge_set)
            available_left = sorted([v for v in H.nodes() if v in left_nodes])
            available_right = sorted([v for v in H.nodes() if v in right_nodes])

            if len(available_left) != len(available_right):
                return []

            matchings = []

            # Strategy 1: Natural ordering
            matching = set()
            matched_right = set()
            for u in available_left:
                for v in available_right:
                    if v not in matched_right:
                        edge = (u, v) if (u, v) in edge_set else (v, u) if (v, u) in edge_set else None
                        if edge:
                            matching.add(edge)
                            matched_right.add(v)
                            break
            if len(matching) == len(available_left):
                matchings.append(matching)

            # Strategy 2-N: Random orderings
            for attempt in range(min(num_attempts - 1, 9)):
                random.seed(attempt)
                shuffled_right = available_right.copy()
                random.shuffle(shuffled_right)

                matching = set()
                matched_right = set()
                for u in available_left:
                    for v in shuffled_right:
                        if v not in matched_right:
                            edge = (u, v) if (u, v) in edge_set else (v, u) if (v, u) in edge_set else None
                            if edge:
                                matching.add(edge)
                                matched_right.add(v)
                                break

                if len(matching) == len(available_left):
                    if matching not in matchings:
                        matchings.append(matching)

            return matchings

        def find_commuting_matching(selected_matchings, remaining_edges):
            if not remaining_edges:
                return None

            candidate_matchings = try_generate_commuting_matchings(remaining_edges, num_attempts=10)

            if not candidate_matchings:
                H = G.edge_subgraph(remaining_edges)
                matching_dict = bipartite.maximum_matching(H, top_nodes=left_nodes)

                matching_edges = set()
                for u, v in matching_dict.items():
                    if u in left_nodes:
                        edge = (u, v) if (u, v) in remaining_edges else (v, u)
                        matching_edges.add(edge)

                if matching_edges:
                    commutes = all(
                        self._matchings_commute(matching_edges, m, left_set, right_set)
                        for m in selected_matchings
                    )
                    return matching_edges, commutes
                return None

            for matching in candidate_matchings:
                if all(self._matchings_commute(matching, m, left_set, right_set) for m in selected_matchings):
                    return matching, True

            return candidate_matchings[0], False

        matchings = []
        remaining_edges = set(G.edges())

        while remaining_edges:
            result = find_commuting_matching(matchings, remaining_edges)

            if result is None:
                break

            matching_edges, _ = result
            matchings.append(matching_edges)

            for edge in matching_edges:
                remaining_edges.discard(edge)
                u, v = edge
                remaining_edges.discard((v, u))

        return matchings

    def _compute_greedy_matchings(self, edges_list: List[Tuple[str, str]]) -> List[Set[Tuple[str, str]]]:
        """
        Compute matchings using greedy algorithm grouped by bit flip position.

        Args:
            edges_list: List of edges

        Returns:
            List of matchings
        """
        edge_groups = defaultdict(list)
        for edge in edges_list:
            bit_pos = self._get_bit_flip_position(edge)
            edge_groups[bit_pos].append(edge)

        matchings = []

        for bit_position, group_edges in edge_groups.items():
            if bit_position >= 0:
                while group_edges:
                    current_matching = set()
                    remaining_edges = []

                    for edge in group_edges:
                        if not any(set(edge) & set(existing_edge) for existing_edge in current_matching):
                            current_matching.add(edge)
                        else:
                            remaining_edges.append(edge)

                    if current_matching:
                        matchings.append(current_matching)
                    group_edges = remaining_edges
            else:
                for edge in group_edges:
                    placed = False
                    for matching in matchings:
                        if not any(set(edge) & set(e) for e in matching):
                            matching.add(edge)
                            placed = True
                            break
                    if not placed:
                        matchings.append({edge})

        return matchings

    def build_circuit(
        self,
        n_steps: int,
        delta_t: float
    ) -> QuantumCircuit:
        """
        Build quantum circuit using matching-based decomposition.

        Builds one Trotter step circuit, then repeats it n_steps times.

        Args:
            n_steps: Number of Trotter steps
            delta_t: Total evolution time

        Returns:
            QuantumCircuit: Complete matching-based circuit
        """
        return build_matching_circuit_iteratively(
            self.n_qubits,
            n_steps,
            delta_t,
            self.matchings
        )

    def get_matchings(self) -> List[Set[Tuple[str, str]]]:
        """
        Return the computed matchings.

        Returns:
            List of matchings, where each matching is a set of edges
        """
        return self.matchings

    def num_matchings(self) -> int:
        """
        Return the number of matchings.

        Returns:
            Number of matchings in the decomposition
        """
        return len(self.matchings)
