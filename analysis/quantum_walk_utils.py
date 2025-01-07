# quantum_walk_utils.py

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

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


class Logger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message: str):
        """Log message with timestamp to file only."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        # Write directly to file, don't print to console
        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")
            f.flush()  # Ensure immediate writing to disk

    def log_metrics(self, method: str, cx_count: int, u3_count: int, depth: int):
        """Log circuit metrics."""
        self.log(f"{method} - CX: {cx_count}, U3: {u3_count}, Depth: {depth}")

    def log_comparison(
        self,
        graph_index: int,
        category: str,
        exact_diff: Tuple[int, int],
        pauli_diff: Tuple[int, int],
        is_bipartite: bool,
    ):
        """Log comparison results for a graph."""
        self.log(f"Graph {graph_index} Results:")
        self.log(f"Category: {category}")
        self.log(
            f"Matching vs Exact: {'win' if exact_diff[0] < 0 else 'lose' if exact_diff[0] > 0 else 'draw'} "
            f"(CX diff: {exact_diff[0]}, U3 diff: {exact_diff[1]})"
        )
        self.log(
            f"Matching vs Pauli: {'win' if pauli_diff[0] < 0 else 'lose' if pauli_diff[0] > 0 else 'draw'} "
            f"(CX diff: {pauli_diff[0]}, U3 diff: {pauli_diff[1]})"
        )
        self.log(f"Bipartite: {is_bipartite}")

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


class GraphProcessor:
    @staticmethod
    def graph_to_bitstring(graph: nx.Graph) -> Set[Tuple[str, str]]:
        num_nodes = len(graph.nodes)
        num_bits = len(bin(num_nodes - 1)) - 2
        node_to_bitstring = {node: format(node, f"0{num_bits}b") for node in graph.nodes}
        return {(node_to_bitstring[u], node_to_bitstring[v]) for u, v in graph.edges}

    @staticmethod
    def parse_graph_filename(filepath: str) -> Dict[str, str]:
        """
        Parse graph filename to extract metadata.
        Format: '<num>graph_<size>_<vertices>c.g6'
        Example: '100graph_50-50_8c.g6'
        """
        # Extract just the filename without path and extension
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]

        try:
            # Extract number from 'Ngraph' format
            import re

            num_match = re.match(r"(\d+)graph_", base_name)
            if not num_match:
                raise ValueError("Could not find number of graphs")
            num_graphs = int(num_match.group(1))

            # Split remaining parts
            parts = base_name.split("graph_")[1].split("_")
            size = parts[0]  # '50-50', 'BM', or 'bipartite'
            vertex_info = parts[1]  # e.g., '8c'
            vertices = vertex_info[:-1]  # '8'
            graph_type = vertex_info[-1]  # 'c'

            return {
                "num_graphs": num_graphs,
                "size": size,
                "vertices": vertices,
                "type": graph_type,
            }
        except Exception as e:
            raise ValueError(
                f"Invalid filename format. Expected '<num>graph_<size>_<vertices>c.g6', got: {filename}"
            )


class BaseAnalyzer:
    def __init__(self, n_qubits: int, delta_t: float, seed: int = 0):
        self.n_qubits = n_qubits
        self.delta_t = delta_t
        self.seed = seed

    def analyze_circuit(self, qc: QuantumCircuit, runs: int = 10) -> CircuitMetrics:
        """Analyze circuit with fixed transpilation settings over multiple runs and return the lowest counts."""
        min_metrics = CircuitMetrics(
            cx_count=float("inf"), u3_count=float("inf"), depth=float("inf")
        )

        for _ in range(runs):
            try:
                transpiled_qc = transpile(
                    qc,
                    basis_gates=["cx", "u3"],
                    optimization_level=3,
                    seed_transpiler=self.seed,
                    # routing_method="sabre",
                )

                counts = transpiled_qc.count_ops()
                current_metrics = CircuitMetrics(
                    cx_count=counts.get("cx", 0),
                    u3_count=counts.get("u3", 0),
                    depth=transpiled_qc.depth(),
                )

                # Update minimum metrics
                min_metrics.cx_count = min(min_metrics.cx_count, current_metrics.cx_count)
                min_metrics.u3_count = min(min_metrics.u3_count, current_metrics.u3_count)
                min_metrics.depth = min(min_metrics.depth, current_metrics.depth)

            except Exception as e:
                print(f"Transpilation error: {str(e)}")

        # Return the minimum metrics found
        return min_metrics if min_metrics.cx_count != float("inf") else None

    # def analyze_circuit(self, qc: QuantumCircuit, runs: int = 10) -> CircuitMetrics:
    #     """Analyze circuit with fixed transpilation settings over multiple runs and return the average counts."""
    #     total_metrics = CircuitMetrics(cx_count=0, u3_count=0, depth=0)

    #     for _ in range(runs):
    #         try:
    #             transpiled_qc = transpile(
    #                 qc,
    #                 basis_gates=['cx', 'u3'],
    #                 seed_transpiler=self.seed,
    #                 optimization_level=3
    #                 #routing_method='sabre'

    #             )

    #             counts = transpiled_qc.count_ops()
    #             current_metrics = CircuitMetrics(
    #                 cx_count=counts.get('cx', 0),
    #                 u3_count=counts.get('u3', 0),
    #                 depth=transpiled_qc.depth()
    #             )

    #             # Accumulate metrics
    #             total_metrics.cx_count += current_metrics.cx_count
    #             total_metrics.u3_count += current_metrics.u3_count
    #             total_metrics.depth += current_metrics.depth

    #         except Exception as e:
    #             print(f"Transpilation error: {str(e)}")

    #     # Calculate averages
    #     average_metrics = CircuitMetrics(
    #         cx_count=total_metrics.cx_count / runs,
    #         u3_count=total_metrics.u3_count / runs,
    #         depth=total_metrics.depth / runs
    #     )

    #     return average_metrics

    def analyze_matching(
        self, edges: Set[Tuple[str, str]], n_steps: int = 1
    ) -> Optional[CircuitMetrics]:
        """Analyze circuit using matching method."""
        try:
            static_G = StaticGraph(edges)
            intersecting_G = IntersectingEdgesGraph(edges)

            qc = QuantumCircuit(static_G.n_qubits)
            for _ in range(n_steps):
                for subgraph in intersecting_G.subgraphs:
                    G = MultiEdgeGraph(subgraph.edges)
                    sub_qc = G.get_qc(simplified=True)
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

    def analyze_matching(
        self, edges: Set[Tuple[str, str]], n_steps: int = 1
    ) -> Optional[CircuitMetrics]:
        try:
            static_G = StaticGraph(edges)
            intersecting_G = IntersectingEdgesGraph(edges)

            qc = QuantumCircuit(static_G.n_qubits)
            for _ in range(n_steps):
                for subgraph in intersecting_G.subgraphs:
                    G = MultiEdgeGraph(subgraph.edges)
                    sub_qc = G.get_qc(simplified=True)
                    qc = qc.compose(sub_qc)

            return self.analyze_circuit(qc)
        except Exception as e:
            return None

    def analyze_exact(self, edges: Set[Tuple[str, str]]) -> Optional[CircuitMetrics]:
        try:
            static_G = StaticGraph(edges)
            H = -1j * self.delta_t * static_G.get_adj_mat()
            U = Operator(expm(H))

            qc = QuantumCircuit(static_G.n_qubits)
            qc.unitary(U, range(static_G.n_qubits))

            return self.analyze_circuit(qc)
        except Exception as e:
            return None

    def analyze_pauli(self, edges: Set[Tuple[str, str]]) -> Optional[CircuitMetrics]:
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

            return self.analyze_circuit(qc)
        except Exception as e:
            return None


class ResultsManager:
    def __init__(self, base_dir: str, graph_info: Dict[str, str]):
        self.base_dir = base_dir
        self.graph_info = graph_info

    def get_output_path(self, category: str, ext: str) -> str:
        filename = f"{category}_{self.graph_info['size']}_{self.graph_info['vertices']}c"
        return os.path.join(self.base_dir, filename)

    def save_results(self, results: Dict, category: str):
        output_path = self.get_output_path(category, "txt")
        with open(output_path, "w") as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

    def save_categorized_graphs(self, results: List[Dict], original_graphs: List[nx.Graph]):
        """Save graphs to separate g6 files based on their categories."""
        # Create dictionary to store graphs by category
        from collections import defaultdict  # Add import here

        graphs_by_category = defaultdict(list)

        # Group graphs by their categories
        for result, graph in zip(results, original_graphs):
            category = result["category"]
            graphs_by_category[category].append(graph)

        # Save each category to a separate file
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
        #'data': os.path.join(base_dir, 'data'),
        "plots": os.path.join(base_dir, "plots"),
        "results": os.path.join(base_dir, "results"),
    }
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    return directories
