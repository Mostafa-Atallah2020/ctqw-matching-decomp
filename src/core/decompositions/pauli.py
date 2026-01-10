from __future__ import annotations
import itertools
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp

if TYPE_CHECKING:
    from src.core.multi_edge_graph import MultiEdgeGraph


class PauliDecomposition:
    """
    Quantum circuit construction using Pauli decomposition for CTQW.

    This class implements the Pauli-based Trotterization approach for
    Continuous-Time Quantum Walks (CTQW). It decomposes the Hamiltonian
    into Pauli basis terms and builds quantum circuits using PauliEvolutionGate.

    Attributes:
        graph: MultiEdgeGraph instance
        n_qubits: Number of qubits in the system
        hamiltonian: The Hamiltonian matrix
        pauli_terms: List of (pauli_string, coefficient) tuples
    """

    def __init__(self, graph: MultiEdgeGraph):
        """
        Initialize PauliDecomposition with a MultiEdgeGraph.

        Args:
            graph: MultiEdgeGraph instance containing the graph

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
        self.hamiltonian = graph.hamiltonian
        self.n_qubits = graph.n_qubits
        self.pauli_terms = self._decompose_hamiltonian()

    def _decompose_hamiltonian(self) -> List[Tuple[str, float]]:
        """
        Decompose Hamiltonian into Pauli basis.

        Returns:
            List of (pauli_string, coefficient) tuples for non-zero terms
        """
        pauli_terms = []

        for pauli_string in ["".join(p) for p in itertools.product("IXYZ", repeat=self.n_qubits)]:
            P = Pauli(pauli_string)
            P_op = Operator(P).data
            coeff = np.trace(P_op.conj().T @ self.hamiltonian) / (2**self.n_qubits)

            real_coeff = float(np.real(coeff))
            if not np.isclose(real_coeff, 0, atol=1e-10):
                pauli_terms.append((pauli_string, real_coeff))

        return pauli_terms

    def build_circuit(
        self,
        n_steps: int,
        delta_t: float
    ) -> QuantumCircuit:
        """
        Build quantum circuit using Pauli decomposition with TRUE Trotter approximation.

        This creates a proper n_steps Trotter approximation by:
        1. Decomposing H into Pauli terms: H = sum_i c_i * P_i
        2. Approximating exp(-iHt) ≈ [prod_i exp(-i*c_i*P_i*t/n)]^n
        3. Each Trotter step evolves individual Pauli terms

        Args:
            n_steps: Number of Trotter steps
            delta_t: Total evolution time

        Returns:
            QuantumCircuit: Complete Pauli-based circuit with TRUE Trotter approximation
        """
        if not self.pauli_terms:
            return QuantumCircuit(self.n_qubits)

        qc = QuantumCircuit(self.n_qubits)

        for step in range(n_steps):
            # Evolve each Pauli term for time delta_t/n_steps
            for pauli_str, coeff in self.pauli_terms:
                # Create single-term Hamiltonian
                single_term = SparsePauliOp([pauli_str], [coeff])

                # Evolve this term
                evo_gate = PauliEvolutionGate(
                    single_term,
                    time=delta_t / n_steps
                )
                qc.append(evo_gate, range(self.n_qubits))

        return qc

    def get_pauli_terms(self) -> List[Tuple[str, float]]:
        """
        Return the Pauli decomposition terms.

        Returns:
            List of (pauli_string, coefficient) tuples
        """
        return self.pauli_terms

    def num_terms(self) -> int:
        """
        Return the number of non-zero Pauli terms.

        Returns:
            Number of Pauli terms in the decomposition
        """
        return len(self.pauli_terms)
