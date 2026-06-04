"""ctqw_matching_decomp — Continuous-Time Quantum Walk Hamiltonian simulation toolkit.

Implements the matching decomposition for synthesizing CTQW Hamiltonian
evolution circuits, alongside the Pauli decomposition baseline, plus
graph-based Hamiltonian construction and circuit utilities.

Companion code for "A matching decomposition algorithm for simulating
quantum walk Hamiltonians" (Atallah et al.).
"""

from .core import MultiEdgeGraph, MatchingDecomposition, PauliDecomposition
from .utils import (
    compress_edges_iteratively,
    build_matching_circuit_iteratively,
    getOpsCirc,
    getGateOps,
    getHammingWt,
    build_exact_evolution_circuit,
    get_exact_evolution_operator,
    GraphDrawer,
)

__all__ = [
    "MultiEdgeGraph",
    "MatchingDecomposition",
    "PauliDecomposition",
    "compress_edges_iteratively",
    "build_matching_circuit_iteratively",
    "getOpsCirc",
    "getGateOps",
    "getHammingWt",
    "build_exact_evolution_circuit",
    "get_exact_evolution_operator",
    "GraphDrawer",
]
