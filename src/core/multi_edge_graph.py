from typing import Set, Tuple, List, Union
import numpy as np
import networkx as nx
from src.utils.graph.drawer import GraphDrawer


class MultiEdgeGraph:
    """
    Multi-edge graph data type for Continuous-Time Quantum Walk (CTQW) simulations.

    This class represents an undirected graph where vertices are labeled
    as binary strings (quantum computational basis states) and edges are
    tuples of two bitstrings.

    Attributes:
        edges: Set of edges as tuples of two binary strings
        n_qubits: Number of qubits (determined from vertex label length)
        n_vertices: Total number of vertices (2^n_qubits)
        hamiltonian: Adjacency matrix (Hamiltonian for CTQW)
    """

    def __init__(self, edges: Union[Set[Tuple[str, str]], List[Tuple[str, str]]]):
        """
        Initialize MultiEdgeGraph with edges.

        Args:
            edges: Set or list of tuples representing graph edges, where each tuple
                   contains two binary strings (e.g., {('00', '01'), ('01', '11')})

        Raises:
            ValueError: If edges is empty or contains invalid edge format
        """
        self._edges = self._validate_and_normalize_edges(edges)

        if not self._edges:
            raise ValueError("Edge set cannot be empty")

        first_edge = next(iter(self._edges))
        self._n_qubits = len(first_edge[0])
        self._n_vertices = 2 ** self._n_qubits

        # Compute integer representation of edges
        self._int_edges = self._compute_int_edges()

        # Build networkx graph for utilities
        self._nx_graph = self._build_networkx_graph()

        # Lazily computed properties
        self._hamiltonian = None
        self._hamming_1_edges = None
        self._hamming_gt1_edges = None

    def _validate_and_normalize_edges(
        self, edges: Union[Set[Tuple[str, str]], List[Tuple[str, str]]]
    ) -> Set[Tuple[str, str]]:
        """Validate and normalize edges to a set."""
        if isinstance(edges, list):
            edges = set(edges)
        elif not isinstance(edges, set):
            raise ValueError("Edges must be a set or list")

        for edge in edges:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise ValueError(f"Each edge must be a tuple of length 2, got: {edge}")
            v1, v2 = edge
            if not (isinstance(v1, str) and isinstance(v2, str)):
                raise ValueError(f"Edge vertices must be binary strings, got: {edge}")
            if len(v1) != len(v2):
                raise ValueError(f"Edge vertices must have same length, got: {edge}")

        return edges

    def _compute_int_edges(self) -> Set[Tuple[int, int]]:
        """Convert binary string edges to integer tuples."""
        return {(int(v1, 2), int(v2, 2)) for v1, v2 in self._edges}

    def _build_networkx_graph(self) -> nx.Graph:
        """Build networkx graph representation."""
        G = nx.Graph()
        G.add_nodes_from(range(self._n_vertices))
        G.add_edges_from(self._int_edges)
        return G

    @property
    def edges(self) -> Set[Tuple[str, str]]:
        """Return the set of edges as binary string tuples."""
        return self._edges

    @property
    def n_qubits(self) -> int:
        """Return the number of qubits."""
        return self._n_qubits

    @property
    def n_vertices(self) -> int:
        """Return the number of vertices."""
        return self._n_vertices

    @property
    def n_edges(self) -> int:
        """Return the number of edges."""
        return len(self._edges)

    @property
    def hamiltonian(self) -> np.ndarray:
        """
        Return the adjacency matrix (Hamiltonian for CTQW).

        The Hamiltonian is computed lazily and cached.

        Returns:
            numpy.ndarray: Adjacency matrix of shape (n_vertices, n_vertices)
        """
        if self._hamiltonian is None:
            self._hamiltonian = self._compute_hamiltonian()
        return self._hamiltonian

    def _compute_hamiltonian(self) -> np.ndarray:
        """Compute the adjacency matrix."""
        adj_matrix = np.zeros((self._n_vertices, self._n_vertices), dtype=float)

        for v1, v2 in self._int_edges:
            adj_matrix[v1, v2] = 1.0
            adj_matrix[v2, v1] = 1.0

        return adj_matrix

    @property
    def hamming_1_edges(self) -> Set[Tuple[str, str]]:
        """Return edges with Hamming distance 1."""
        if self._hamming_1_edges is None:
            self._split_by_hamming_distance()
        return self._hamming_1_edges

    @property
    def hamming_gt1_edges(self) -> Set[Tuple[str, str]]:
        """Return edges with Hamming distance > 1."""
        if self._hamming_gt1_edges is None:
            self._split_by_hamming_distance()
        return self._hamming_gt1_edges

    def _split_by_hamming_distance(self) -> None:
        """Split edges into sets based on Hamming distance."""
        self._hamming_1_edges = set()
        self._hamming_gt1_edges = set()

        for edge in self._edges:
            v1, v2 = edge
            dist = sum(c1 != c2 for c1, c2 in zip(v1, v2))
            if dist == 1:
                self._hamming_1_edges.add(edge)
            else:
                self._hamming_gt1_edges.add(edge)

    def hamming_cost(self) -> int:
        """
        Calculate total Hamming distance across all edges.

        Returns:
            int: Sum of Hamming distances for all edges
        """
        total = 0
        for v1, v2 in self._edges:
            total += sum(c1 != c2 for c1, c2 in zip(v1, v2))
        return total

    def avg_hamming_cost(self) -> float:
        """
        Calculate average Hamming distance per edge.

        Returns:
            float: Average Hamming distance
        """
        return self.hamming_cost() / len(self._edges) if self._edges else 0.0

    def draw(self):
        """Display the graph visualization."""
        return GraphDrawer(self._n_qubits, self._int_edges).show()

    def __repr__(self) -> str:
        return f"MultiEdgeGraph(n_qubits={self._n_qubits}, n_edges={len(self._edges)})"

    def __str__(self) -> str:
        return f"MultiEdgeGraph with {self._n_qubits} qubits, {len(self._edges)} edges: {self._edges}"

    def __len__(self) -> int:
        """Return number of edges."""
        return len(self._edges)

    def __eq__(self, other) -> bool:
        if not isinstance(other, MultiEdgeGraph):
            return False
        return self._edges == other._edges
