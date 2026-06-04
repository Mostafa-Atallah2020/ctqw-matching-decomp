import numpy as np
import pytest

from ctqw_matching_decomp.core import MultiEdgeGraph


class TestMultiEdgeGraph:
    """Tests for the MultiEdgeGraph class."""

    def test_basic_creation(self):
        """Test basic graph creation with valid edges."""
        edges = {('00', '01'), ('01', '11')}
        G = MultiEdgeGraph(edges)
        assert G.n_qubits == 2
        assert G.n_vertices == 4
        assert G.n_edges == 2

    def test_3_qubit_graph(self):
        """Test 3-qubit graph creation."""
        edges = {('000', '001'), ('001', '011'), ('011', '111')}
        G = MultiEdgeGraph(edges)
        assert G.n_qubits == 3
        assert G.n_vertices == 8
        assert G.n_edges == 3

    def test_hamiltonian_shape(self):
        """Test that Hamiltonian has correct shape."""
        edges = {('00', '01'), ('01', '11')}
        G = MultiEdgeGraph(edges)
        H = G.hamiltonian
        assert H.shape == (4, 4)

    def test_hamiltonian_symmetry(self):
        """Test that Hamiltonian is symmetric."""
        edges = {('00', '01'), ('01', '11'), ('00', '11')}
        G = MultiEdgeGraph(edges)
        H = G.hamiltonian
        assert np.allclose(H, H.T)

    def test_hamiltonian_values(self):
        """Test that Hamiltonian has correct values."""
        edges = {('00', '01')}
        G = MultiEdgeGraph(edges)
        H = G.hamiltonian
        # Edge (00, 01) means indices (0, 1) should be 1
        assert H[0, 1] == 1.0
        assert H[1, 0] == 1.0
        # Other entries should be 0
        assert H[0, 0] == 0.0
        assert H[1, 1] == 0.0

    def test_hamming_distance_split(self):
        """Test splitting edges by Hamming distance."""
        edges = {('00', '01'), ('00', '11')}  # H=1 and H=2
        G = MultiEdgeGraph(edges)
        assert ('00', '01') in G.hamming_1_edges or ('01', '00') in G.hamming_1_edges
        assert ('00', '11') in G.hamming_gt1_edges or ('11', '00') in G.hamming_gt1_edges

    def test_hamming_cost(self):
        """Test total Hamming cost calculation."""
        edges = {('00', '01'), ('00', '11')}  # H=1 + H=2 = 3
        G = MultiEdgeGraph(edges)
        assert G.hamming_cost() == 3

    def test_avg_hamming_cost(self):
        """Test average Hamming cost calculation."""
        edges = {('00', '01'), ('00', '11')}  # H=1 + H=2 = 3, avg = 1.5
        G = MultiEdgeGraph(edges)
        assert G.avg_hamming_cost() == 1.5

    def test_list_input(self):
        """Test that list input works (converted to set)."""
        edges = [('00', '01'), ('01', '11')]
        G = MultiEdgeGraph(edges)
        assert G.n_edges == 2

    def test_empty_edges_raises(self):
        """Test that empty edges raise ValueError."""
        with pytest.raises(ValueError):
            MultiEdgeGraph(set())

    def test_invalid_edge_format_raises(self):
        """Test that invalid edge format raises ValueError."""
        with pytest.raises(ValueError):
            MultiEdgeGraph({('00',)})  # Only one element in tuple

    def test_mismatched_length_raises(self):
        """Test that mismatched bitstring lengths raise ValueError."""
        with pytest.raises(ValueError):
            MultiEdgeGraph({('00', '001')})

    def test_repr(self):
        """Test string representation."""
        edges = {('00', '01')}
        G = MultiEdgeGraph(edges)
        assert 'MultiEdgeGraph' in repr(G)
        assert 'n_qubits=2' in repr(G)

    def test_equality(self):
        """Test graph equality."""
        edges1 = {('00', '01')}
        edges2 = {('00', '01')}
        G1 = MultiEdgeGraph(edges1)
        G2 = MultiEdgeGraph(edges2)
        assert G1 == G2

    def test_len(self):
        """Test __len__ returns number of edges."""
        edges = {('00', '01'), ('01', '11'), ('00', '11')}
        G = MultiEdgeGraph(edges)
        assert len(G) == 3
