import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from src.core import MultiEdgeGraph, MatchingDecomposition, PauliDecomposition
from src.utils import get_exact_evolution_operator


class TestMatchingDecomposition:
    """Tests for the MatchingDecomposition class."""

    def test_basic_creation(self):
        """Test basic MatchingDecomposition creation."""
        edges = {('00', '01'), ('01', '11')}
        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)
        assert decomp.n_qubits == 2
        assert decomp.graph == G

    def test_requires_multi_edge_graph(self):
        """Test that MatchingDecomposition requires MultiEdgeGraph input."""
        edges = {('00', '01')}
        with pytest.raises(TypeError):
            MatchingDecomposition(edges)

    def test_matchings_computed(self):
        """Test that matchings are computed."""
        edges = {('00', '01'), ('01', '11'), ('00', '10')}
        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)
        assert decomp.num_matchings() > 0
        assert len(decomp.get_matchings()) == decomp.num_matchings()

    def test_build_circuit_returns_quantum_circuit(self):
        """Test that build_circuit returns a QuantumCircuit."""
        edges = {('00', '01'), ('01', '11')}
        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)
        qc = decomp.build_circuit(n_steps=1, delta_t=0.1)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

    def test_circuit_converges_to_exact(self):
        """Test that circuit converges to exact evolution with more steps."""
        edges = {('00', '01'), ('01', '11')}
        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)
        delta_t = 0.1

        exact_op = get_exact_evolution_operator(delta_t, G.hamiltonian)

        # More Trotter steps should give better approximation
        qc_1 = decomp.build_circuit(n_steps=1, delta_t=delta_t)
        qc_10 = decomp.build_circuit(n_steps=10, delta_t=delta_t)

        error_1 = np.linalg.norm(Operator(qc_1) - exact_op, ord=2)
        error_10 = np.linalg.norm(Operator(qc_10) - exact_op, ord=2)

        assert error_10 < error_1

    def test_3_qubit_circuit(self):
        """Test circuit generation for 3-qubit graph."""
        edges = {('000', '001'), ('001', '011'), ('011', '111')}
        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)
        qc = decomp.build_circuit(n_steps=2, delta_t=0.1)
        assert qc.num_qubits == 3


class TestPauliDecomposition:
    """Tests for the PauliDecomposition class."""

    def test_basic_creation(self):
        """Test basic PauliDecomposition creation."""
        edges = {('00', '01'), ('01', '11')}
        G = MultiEdgeGraph(edges)
        decomp = PauliDecomposition(G)
        assert decomp.n_qubits == 2
        assert decomp.graph == G

    def test_requires_multi_edge_graph(self):
        """Test that PauliDecomposition requires MultiEdgeGraph input."""
        edges = {('00', '01')}
        with pytest.raises(TypeError):
            PauliDecomposition(edges)

    def test_pauli_terms_computed(self):
        """Test that Pauli terms are computed."""
        edges = {('00', '01'), ('01', '11')}
        G = MultiEdgeGraph(edges)
        decomp = PauliDecomposition(G)
        assert len(decomp.pauli_terms) > 0

    def test_build_circuit_returns_quantum_circuit(self):
        """Test that build_circuit returns a QuantumCircuit."""
        edges = {('00', '01'), ('01', '11')}
        G = MultiEdgeGraph(edges)
        decomp = PauliDecomposition(G)
        qc = decomp.build_circuit(n_steps=1, delta_t=0.1)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

    def test_circuit_converges_to_exact(self):
        """Test that circuit converges to exact evolution with more steps."""
        edges = {('00', '01'), ('01', '11')}
        G = MultiEdgeGraph(edges)
        decomp = PauliDecomposition(G)
        delta_t = 0.1

        exact_op = get_exact_evolution_operator(delta_t, G.hamiltonian)

        # More Trotter steps should give better approximation
        qc_1 = decomp.build_circuit(n_steps=1, delta_t=delta_t)
        qc_10 = decomp.build_circuit(n_steps=10, delta_t=delta_t)

        error_1 = np.linalg.norm(Operator(qc_1) - exact_op, ord=2)
        error_10 = np.linalg.norm(Operator(qc_10) - exact_op, ord=2)

        assert error_10 < error_1

    def test_get_pauli_terms(self):
        """Test getting Pauli terms representation."""
        edges = {('00', '01')}
        G = MultiEdgeGraph(edges)
        decomp = PauliDecomposition(G)
        pauli_terms = decomp.get_pauli_terms()
        assert pauli_terms is not None
        assert len(pauli_terms) > 0


class TestDecompositionComparison:
    """Tests comparing Matching and Pauli decompositions."""

    def test_both_converge_to_same_result(self):
        """Test that both decompositions converge to the same exact evolution."""
        edges = {('00', '01'), ('01', '11'), ('00', '10')}
        G = MultiEdgeGraph(edges)

        matching_decomp = MatchingDecomposition(G)
        pauli_decomp = PauliDecomposition(G)

        delta_t = 0.1
        n_steps = 50  # High steps for convergence

        exact_op = get_exact_evolution_operator(delta_t, G.hamiltonian)

        matching_qc = matching_decomp.build_circuit(n_steps=n_steps, delta_t=delta_t)
        pauli_qc = pauli_decomp.build_circuit(n_steps=n_steps, delta_t=delta_t)

        matching_error = np.linalg.norm(Operator(matching_qc) - exact_op, ord=2)
        pauli_error = np.linalg.norm(Operator(pauli_qc) - exact_op, ord=2)

        # Both should have small errors at high step count
        assert matching_error < 0.1
        assert pauli_error < 0.1
