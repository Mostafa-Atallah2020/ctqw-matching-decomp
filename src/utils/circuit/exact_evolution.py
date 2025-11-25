import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm


def build_exact_evolution_circuit(n_qubits: int, delta_t: float, hamiltonian: np.ndarray) -> QuantumCircuit:
    """
    Build exact quantum circuit for time evolution using matrix exponential.

    Computes exp(-iHt) exactly and encodes it as a unitary gate.

    Args:
        n_qubits: Number of qubits
        delta_t: Total evolution time
        hamiltonian: Hamiltonian matrix (numpy array)

    Returns:
        QuantumCircuit: Exact evolution circuit
    """
    qc = QuantumCircuit(n_qubits)

    # Compute exact evolution operator
    exact_op = expm(-1j * hamiltonian * delta_t)
    exact_operator = Operator(exact_op)

    # Add as unitary gate
    qc.unitary(exact_operator, range(n_qubits), label='exact_evolution')

    return qc


def get_exact_evolution_operator(delta_t: float, hamiltonian: np.ndarray) -> Operator:
    """
    Compute the exact time evolution operator.

    Args:
        delta_t: Total evolution time
        hamiltonian: Hamiltonian matrix (numpy array)

    Returns:
        Operator: Exact evolution operator exp(-iHt)
    """
    exact_op = expm(-1j * hamiltonian * delta_t)
    return Operator(exact_op)
