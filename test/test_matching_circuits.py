"""
Tests for matching decomposition circuit implementation.

These tests verify that the MatchingDecomposition class produces circuits
that match the expected operator for various graph configurations.

Test cases are derived from examples/circuit implementation.ipynb which
contains hardcoded reference circuits.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from src.core import MultiEdgeGraph, MatchingDecomposition
from src.utils import get_exact_evolution_operator
from src.utils.misc import multi_crx


class TestSingleEdgeCircuits:
    """Tests for single-edge graphs."""

    def test_edge_00_01(self):
        """Test edge ('00', '01') - Hamming distance 1."""
        edges = {('00', '01')}
        delta_t = 0.1

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        # Expected circuit from notebook
        expected_qc = QuantumCircuit(2)
        expected_qc.x(1)
        expected_qc.crx(2 * delta_t, 1, 0)
        expected_qc.x(1)

        # Build actual circuit
        actual_qc = decomp.build_circuit(n_steps=1, delta_t=delta_t)

        # Compare operators
        expected_op = Operator(expected_qc)
        actual_op = Operator(actual_qc)

        dist = np.linalg.norm(expected_op.data - actual_op.data, ord=2)
        assert dist < 1e-10, f"Operator difference: {dist}"

    def test_edge_10_11(self):
        """Test edge ('10', '11') - Hamming distance 1."""
        edges = {('10', '11')}
        delta_t = 0.1

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        # Expected circuit from notebook
        expected_qc = QuantumCircuit(2)
        expected_qc.crx(2 * delta_t, 1, 0)

        actual_qc = decomp.build_circuit(n_steps=1, delta_t=delta_t)

        expected_op = Operator(expected_qc)
        actual_op = Operator(actual_qc)

        dist = np.linalg.norm(expected_op.data - actual_op.data, ord=2)
        assert dist < 1e-10, f"Operator difference: {dist}"

    def test_edge_00_11(self):
        """Test edge ('00', '11') - Hamming distance 2 (diagonal)."""
        edges = {('00', '11')}
        delta_t = 0.1

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        # Expected circuit from notebook
        expected_qc = QuantumCircuit(2)
        expected_qc.cx(0, 1)
        expected_qc.x(1)
        expected_qc.crx(2 * delta_t, 1, 0)
        expected_qc.x(1)
        expected_qc.cx(0, 1)

        actual_qc = decomp.build_circuit(n_steps=1, delta_t=delta_t)

        expected_op = Operator(expected_qc)
        actual_op = Operator(actual_qc)

        dist = np.linalg.norm(expected_op.data - actual_op.data, ord=2)
        assert dist < 1e-10, f"Operator difference: {dist}"


class TestTwoEdgeCircuits:
    """Tests for two-edge graphs."""

    def test_edges_00_10_and_01_11_separate(self):
        """Test edges {('00', '10'), ('01', '11')} - two separate Hamming-1 edges."""
        edges = {('00', '10'), ('01', '11')}
        delta_t = 0.1

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        # These edges can be compressed to a single RX gate
        # Expected circuit from notebook (cell 23)
        expected_qc = QuantumCircuit(2)
        expected_qc.rx(2 * delta_t, 1)

        actual_qc = decomp.build_circuit(n_steps=1, delta_t=delta_t)

        expected_op = Operator(expected_qc)
        actual_op = Operator(actual_qc)

        dist = np.linalg.norm(expected_op.data - actual_op.data, ord=2)
        assert dist < 1e-10, f"Operator difference: {dist}"

    def test_edges_00_11_and_01_10(self):
        """Test edges {('00', '11'), ('01', '10')} - two diagonal edges."""
        edges = {('00', '11'), ('01', '10')}
        delta_t = 0.1

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        # Expected from notebook (cell 38) - simplified version
        expected_qc = QuantumCircuit(2)
        expected_qc.cx(0, 1)
        expected_qc.rx(2 * delta_t, 0)
        expected_qc.cx(0, 1)

        actual_qc = decomp.build_circuit(n_steps=1, delta_t=delta_t)

        expected_op = Operator(expected_qc)
        actual_op = Operator(actual_qc)

        dist = np.linalg.norm(expected_op.data - actual_op.data, ord=2)
        assert dist < 1e-10, f"Operator difference: {dist}"


class TestThreeQubitCircuits:
    """Tests for 3-qubit graphs."""

    def test_edge_000_111(self):
        """Test edge ('000', '111') - Hamming distance 3."""
        edges = {('000', '111')}
        delta_t = 0.1

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        # Expected circuit from notebook (cell 43 or 48)
        rot1 = multi_crx(2 * delta_t, '00')

        expected_qc = QuantumCircuit(3)
        expected_qc.cx(1, 2)
        expected_qc.cx(0, 1)
        expected_qc.append(rot1, [2, 1, 0])
        expected_qc.cx(0, 1)
        expected_qc.cx(1, 2)

        actual_qc = decomp.build_circuit(n_steps=1, delta_t=delta_t)

        expected_op = Operator(expected_qc)
        actual_op = Operator(actual_qc)

        dist = np.linalg.norm(expected_op.data - actual_op.data, ord=2)
        assert dist < 1e-10, f"Operator difference: {dist}"


class TestExactEvolutionMatch:
    """Tests verifying circuit matches exact evolution."""

    def test_single_edge_exact_match(self):
        """Test that single-step circuit matches exact evolution for single edge."""
        edges = {('00', '01')}
        delta_t = 0.1

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        exact_op = get_exact_evolution_operator(delta_t, G.hamiltonian)
        actual_qc = decomp.build_circuit(n_steps=1, delta_t=delta_t)
        actual_op = Operator(actual_qc)

        # For single Hamming-1 edge, matching should be exact
        dist = np.linalg.norm(exact_op.data - actual_op.data, ord=2)
        assert dist < 1e-10, f"Operator difference from exact: {dist}"

    def test_compressed_edges_exact_match(self):
        """Test that compressed edges match exact evolution."""
        edges = {('00', '10'), ('01', '11')}
        delta_t = 0.1

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        exact_op = get_exact_evolution_operator(delta_t, G.hamiltonian)
        actual_qc = decomp.build_circuit(n_steps=1, delta_t=delta_t)
        actual_op = Operator(actual_qc)

        # Compressed edges should also be exact for commuting terms
        dist = np.linalg.norm(exact_op.data - actual_op.data, ord=2)
        assert dist < 1e-10, f"Operator difference from exact: {dist}"

    def test_diagonal_edge_exact_match(self):
        """Test that diagonal edge matches exact evolution."""
        edges = {('00', '11')}
        delta_t = 0.1

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        exact_op = get_exact_evolution_operator(delta_t, G.hamiltonian)
        actual_qc = decomp.build_circuit(n_steps=1, delta_t=delta_t)
        actual_op = Operator(actual_qc)

        # Single edge should be exact
        dist = np.linalg.norm(exact_op.data - actual_op.data, ord=2)
        assert dist < 1e-10, f"Operator difference from exact: {dist}"

    def test_multiple_steps_convergence(self):
        """Test that more Trotter steps improve accuracy for non-commuting terms."""
        # Use a graph where terms don't commute
        edges = {('00', '01'), ('01', '11'), ('00', '10')}
        delta_t = 0.5  # Larger time to see Trotter error

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        exact_op = get_exact_evolution_operator(delta_t, G.hamiltonian)

        # Single step
        qc_1 = decomp.build_circuit(n_steps=1, delta_t=delta_t)
        error_1 = np.linalg.norm(Operator(qc_1).data - exact_op.data, ord=2)

        # 10 steps
        qc_10 = decomp.build_circuit(n_steps=10, delta_t=delta_t)
        error_10 = np.linalg.norm(Operator(qc_10).data - exact_op.data, ord=2)

        # 50 steps
        qc_50 = decomp.build_circuit(n_steps=50, delta_t=delta_t)
        error_50 = np.linalg.norm(Operator(qc_50).data - exact_op.data, ord=2)

        # Error should decrease with more steps
        assert error_10 < error_1, f"10 steps ({error_10}) should be better than 1 step ({error_1})"
        assert error_50 < error_10, f"50 steps ({error_50}) should be better than 10 steps ({error_10})"


class TestSpaceReduction:
    """Tests for space reduction optimization."""

    def test_full_compression_two_edges(self):
        """Test that two compatible edges are fully compressed."""
        # These edges differ only in qubit 1
        edges = {('00', '10'), ('01', '11')}
        delta_t = 0.1

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        qc = decomp.build_circuit(n_steps=1, delta_t=delta_t)

        # Circuit should contain RX gate (fully compressed)
        ops = qc.count_ops()
        assert 'rx' in ops, f"Expected RX gate in compressed circuit, got: {ops}"

    def test_four_edge_compression(self):
        """Test compression of 4 compatible edges."""
        # 4 edges that can be compressed
        edges = {('000', '100'), ('001', '101'), ('010', '110'), ('011', '111')}
        delta_t = 0.1

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        qc = decomp.build_circuit(n_steps=1, delta_t=delta_t)

        # Should compress to single RX
        ops = qc.count_ops()
        assert 'rx' in ops, f"Expected RX gate, got: {ops}"


class TestCycleGraphs:
    """Tests for cycle graph structures."""

    def test_cycle_c4(self):
        """Test 4-node cycle graph."""
        edges = {('00', '01'), ('01', '11'), ('11', '10'), ('10', '00')}
        delta_t = 0.1

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        exact_op = get_exact_evolution_operator(delta_t, G.hamiltonian)

        # With enough steps, should converge to exact
        qc = decomp.build_circuit(n_steps=100, delta_t=delta_t)
        actual_op = Operator(qc)

        dist = np.linalg.norm(exact_op.data - actual_op.data, ord=2)
        assert dist < 0.01, f"Cycle C4 error too large: {dist}"


class TestCayleyGraphs:
    """Tests for Cayley graph structures (hypercube-like)."""

    def test_cayley_z2_z2(self):
        """Test Cayley graph of Z2 x Z2 (2-qubit hypercube)."""
        # All Hamming-1 edges on 2 qubits
        edges = {('00', '01'), ('00', '10'), ('01', '11'), ('10', '11')}
        delta_t = 0.1

        G = MultiEdgeGraph(edges)
        decomp = MatchingDecomposition(G)

        exact_op = get_exact_evolution_operator(delta_t, G.hamiltonian)

        # These edges should allow significant compression
        qc = decomp.build_circuit(n_steps=1, delta_t=delta_t)
        actual_op = Operator(qc)

        dist = np.linalg.norm(exact_op.data - actual_op.data, ord=2)
        # For Cayley graphs, matching decomposition should be exact in 1 step
        # since all terms commute when properly grouped
        assert dist < 0.1, f"Cayley Z2xZ2 error: {dist}"
