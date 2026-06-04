# quantum_walk_utils.py
"""
Utilities for CTQW analysis using the refactored decomposition classes.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from scipy.linalg import expm

# Use the new refactored decomposition classes
from ctqw_matching_decomp.core import MultiEdgeGraph, MatchingDecomposition, PauliDecomposition
from ctqw_matching_decomp.utils import get_exact_evolution_operator
from ctqw_matching_decomp.utils.graph import graph_to_bitstring_edges, parse_g6_filename


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
        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")
            f.flush()

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
        """Convert NetworkX graph to bitstring edge set."""
        return graph_to_bitstring_edges(graph)

    @staticmethod
    def parse_graph_filename(filepath: str) -> Dict[str, str]:
        """
        Parse graph filename to extract metadata.
        Format: '<num>graph_<size>_<vertices>c.g6'
        """
        metadata = parse_g6_filename(filepath)
        # Convert to legacy format expected by analysis scripts
        return {
            "num_graphs": metadata.get("n_graphs", 0),
            "size": metadata.get("type", "unknown"),
            "vertices": str(metadata.get("vertices", 0)),
            "type": metadata.get("suffix", "c"),
        }


class BaseAnalyzer:
    def __init__(self, n_qubits: int, delta_t: float, seed: int = 0):
        self.n_qubits = n_qubits
        self.delta_t = delta_t
        self.seed = seed

    def analyze_circuit(self, qc: QuantumCircuit, runs: int = 1) -> CircuitMetrics:
        """Analyze circuit with transpilation and return average metrics."""
        total_metrics = CircuitMetrics(cx_count=0, u3_count=0, depth=0)

        for _ in range(runs):
            try:
                transpiled_qc = transpile(
                    qc,
                    basis_gates=["cx", "u3"],
                    optimization_level=3,
                    seed_transpiler=self.seed,
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

        average_metrics = CircuitMetrics(
            cx_count=total_metrics.cx_count / runs,
            u3_count=total_metrics.u3_count / runs,
            depth=total_metrics.depth / runs,
        )

        return average_metrics

    def analyze_matching(
        self, edges: Set[Tuple[str, str]], n_steps: int = 1,
        heuristic: str = 'greedy'
    ) -> Optional[CircuitMetrics]:
        """Analyze circuit using MatchingDecomposition class."""
        try:
            # Create graph and decomposition using new classes
            G = MultiEdgeGraph(edges)
            decomp = MatchingDecomposition(G, heuristic=heuristic)

            # Build circuit with specified steps
            qc = decomp.build_circuit(n_steps=n_steps, delta_t=self.delta_t)

            metrics = self.analyze_circuit(qc)
            if metrics is None:
                raise Exception("Failed to analyze circuit")
            return metrics

        except Exception as e:
            print(f"Matching analysis error: {str(e)}")
            return None

    def analyze_exact(self, edges: Set[Tuple[str, str]]) -> Optional[CircuitMetrics]:
        """Analyze circuit using exact evolution."""
        try:
            G = MultiEdgeGraph(edges)
            exact_op = get_exact_evolution_operator(self.delta_t, G.hamiltonian)

            qc = QuantumCircuit(G.n_qubits)
            qc.unitary(exact_op, range(G.n_qubits))

            metrics = self.analyze_circuit(qc)
            if metrics is None:
                raise Exception("Failed to analyze circuit")
            return metrics

        except Exception as e:
            print(f"Exact analysis error: {str(e)}")
            return None

    def analyze_pauli(
        self, edges: Set[Tuple[str, str]], n_steps: int = 1
    ) -> Optional[CircuitMetrics]:
        """Analyze circuit using PauliDecomposition class."""
        try:
            # Create graph and decomposition using new classes
            G = MultiEdgeGraph(edges)
            decomp = PauliDecomposition(G)

            # Build circuit with specified steps (matching the notebook implementation)
            qc = decomp.build_circuit(n_steps=n_steps, delta_t=self.delta_t)

            metrics = self.analyze_circuit(qc)
            if metrics is None:
                raise Exception("Failed to analyze circuit")
            return metrics

        except Exception as e:
            print(f"Pauli analysis error: {str(e)}")
            return None


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
