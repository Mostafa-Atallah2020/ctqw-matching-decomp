import itertools
from typing import Dict, Union

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import RXGate
from qiskit.quantum_info import Statevector, state_fidelity


def multi_crx(angle, ctrl_state):
    """
    Create a multi-controlled RX gate based on the control pattern.

    Args:
        angle: Rotation angle in radians
        ctrl_state: Binary string specifying control states (e.g., '10')

    Returns:
        Controlled RX gate
    """
    n_ctrls = len(ctrl_state)
    if n_ctrls == 0:
        return RXGate(angle)
    gate = RXGate(angle).control(n_ctrls, ctrl_state=ctrl_state[::-1])
    return gate


def binary_tuple_to_int_tuple(binary_tuple):
    # Convert each binary string in the tuple to an integer
    int_tuple = tuple(int(binary_str, 2) for binary_str in binary_tuple)
    return int_tuple


def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two binary strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def get_cyclic_connections(connections, target):
    # Create a dictionary to store the connections
    conn_dict = {}
    for a, b in connections:
        if a not in conn_dict:
            conn_dict[a] = []
        if b not in conn_dict:
            conn_dict[b] = []
        conn_dict[a].append(b)
        conn_dict[b].append(a)

    # Find the starting index of the target node
    start_idx = None
    for i, (a, b) in enumerate(connections):
        if a == target or b == target:
            start_idx = i
            break

    # If the target node is not found, return the original list
    if start_idx is None:
        return connections

    # Initialize the result list and the visited set
    result = []
    visited = set()

    # Start from the target node and traverse the connections
    curr_node = target
    for _ in range(len(connections)):
        if curr_node in visited:
            break
        visited.add(curr_node)
        for neighbor in conn_dict[curr_node]:
            if neighbor not in visited:
                result.append((curr_node, neighbor))
                curr_node = neighbor
                break

    # If the length of the result is less than the input, append the remaining connections
    if len(result) < len(connections):
        result.extend(connections[len(result) :])

    return result


def lists_to_sets(*lists):
    # Use itertools.product to get all combinations of picking one element from each list
    combinations = list(itertools.product(*lists))

    # Convert each combination (which is a tuple) to a set
    sets = [set(comb) for comb in combinations]

    return sets


def graph_matchings_greedy(edges):
    subgraphs = []
    for edge in edges:
        placed = False
        for subgraph in subgraphs:
            if not any(set(edge) & set(e) for e in subgraph):
                subgraph.add(edge)
                placed = True
                break

        if not placed:
            subgraphs.append({edge})

    return subgraphs


def graph_matchings_parallel(edges):
    from collections import defaultdict

    def get_bit_flip_position(edge):
        """Determine which bit position differs between edge endpoints"""
        u, v = edge

        # Handle string vertices (convert binary strings to integers)
        if isinstance(u, str) and isinstance(v, str):
            try:
                u_int = int(u, 2)  # Convert binary string to integer
                v_int = int(v, 2)
            except ValueError:
                # If not binary strings, treat as arbitrary vertex labels
                return -2
        else:
            u_int, v_int = u, v

        # XOR to find differing bits
        diff = u_int ^ v_int
        if diff == 0:
            return -1  # Self-loop or identical vertices

        # Check if only one bit differs (Hamming distance = 1)
        if bin(diff).count("1") == 1:
            # Find position of the single differing bit
            position = (diff & -diff).bit_length() - 1
            return position
        else:
            return -2  # Multi-bit difference

    # Group edges by bit flip position
    edge_groups = defaultdict(list)
    for edge in edges:
        bit_pos = get_bit_flip_position(edge)
        edge_groups[bit_pos].append(edge)

    matchings = []

    # Process each edge group to create optimal matchings
    for bit_position, group_edges in edge_groups.items():
        # For single-bit flip edges, create maximum matchings
        if bit_position >= 0:
            while group_edges:
                current_matching = set()
                remaining_edges = []

                for edge in group_edges:
                    # Check if edge shares vertices with current matching
                    if not any(
                        set(edge) & set(existing_edge) for existing_edge in current_matching
                    ):
                        current_matching.add(edge)
                    else:
                        remaining_edges.append(edge)

                if current_matching:
                    matchings.append(current_matching)
                group_edges = remaining_edges

        # Handle multi-bit flip edges or non-binary vertices with greedy approach
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


def graph_to_bitstring_edges(graph):
    num_nodes = len(graph.nodes)
    num_bits = len(bin(num_nodes - 1)) - 2  # bin(x) gives '0bxxx', so we subtract 2
    node_to_bitstring = {node: format(node, f"0{num_bits}b") for node in graph.nodes}
    edges_bitstring = {(node_to_bitstring[u], node_to_bitstring[v]) for u, v in graph.edges}
    return edges_bitstring


def count_edges(G):
    return nx.number_of_edges(G)


def calculate_edge_density(G):
    # Can use nx.density(G) directly instead of this function
    return nx.density(G)


def is_bipartite(G):
    return nx.is_bipartite(G)


def find_diameter(G):
    if not nx.is_connected(G):
        return float("inf")
    return nx.diameter(G)


def find_max_clique(G):
    return len(max(nx.find_cliques(G), key=len, default=[]))


def average_clustering(G):
    return nx.average_clustering(G)


def estimate_group_size(G):
    # This is a custom metric - NetworkX doesn't have direct equivalent
    # Could use automorphism groups but would be much slower
    degree_sequence = [d for _, d in G.degree()]
    return max(degree_sequence.count(x) for x in set(degree_sequence))


def estimate_orbit_count(G):
    # Similar to above, NetworkX doesn't have direct equivalent
    # Could use nx.vf2pp_isomorphism but would be much slower
    return len(set(d for _, d in G.degree()))


def get_state(circuit=None, initial_state=None):
    """
    Evolve a quantum state through a circuit and extract the final state vector.

    Args:
        `circuit` (`QuantumCircuit`): The quantum circuit to evolve the state through.
                                  If None, an empty circuit will be created.
        `initial_state` (`list` or `Statevector`): Initial state. If None, |+⟩ state will be used.

    Returns:
        `qiskit.Statevector`: The final state vector after evolution
    """
    # If no circuit is provided, create an empty one with 1 qubit
    if circuit is None:
        circuit = QuantumCircuit(1)

    num_qubits = circuit.num_qubits

    # If no initial state is provided, use |+⟩ state
    if initial_state is None:
        # Create |+⟩ state by starting with |0⟩ and applying Hadamard to each qubit
        plus_circuit = QuantumCircuit(num_qubits)
        for qubit in range(num_qubits):
            plus_circuit.h(qubit)

        # Create the |+⟩ state by evolving |0⟩ through Hadamard gates
        zero_state = Statevector.from_label("0" * num_qubits)
        initial_state = zero_state.evolve(plus_circuit)
    elif not isinstance(initial_state, Statevector):
        initial_state = Statevector(initial_state)

    # Evolve the state through the circuit
    final_state = initial_state.evolve(circuit)

    return final_state


def count_gates(circuit, optimization_level=3):
    """
    Count the number of CX and U3 gates in a Qiskit quantum circuit after transpilation.

    Args:
        circuit (QuantumCircuit): The quantum circuit to analyze
        optimization_level (int): Optimization level for transpilation (0-3, default=3)

    Returns:
        tuple: A tuple containing (cx_count, u3_count)
    """
    # Transpile the circuit with the specified optimization level
    transpiled_circuit = transpile(
        circuit, basis_gates=["cx", "u3"], optimization_level=optimization_level
    )

    # Get the operation counts dictionary
    op_counts = transpiled_circuit.count_ops()

    # Get CX gate count (default to 0 if none found)
    cx_count = op_counts.get("cx", 0)

    # Get U3 gate count (default to 0 if none found)
    u3_count = op_counts.get("u3", 0)

    return cx_count, u3_count


def count_gates_direct_transpilation(circuit, optimization_level=3):
    """
    Count gates using DIRECT Qiskit transpilation only - NO estimation fallbacks.

    Args:
        circuit: QuantumCircuit to analyze
        optimization_level: Transpilation optimization level (0-3)

    Returns:
        Tuple of (cx_count, u3_count, success_flag, method_used, transpiled_circuit)
    """

    try:
        transpiled_circuit = transpile(
            circuit,
            basis_gates=["cx", "u3", "u", "rz", "ry", "rx", "x", "h", "p"],
            optimization_level=optimization_level,
            seed_transpiler=42,  # For reproducible results
        )

        op_counts = transpiled_circuit.count_ops()

        # Count gates
        cx_count = op_counts.get("cx", 0) + op_counts.get("cnot", 0)

        u3_count = (
            op_counts.get("u3", 0)
            + op_counts.get("u", 0)
            + op_counts.get("rz", 0)
            + op_counts.get("ry", 0)
            + op_counts.get("rx", 0)
            + op_counts.get("p", 0)
            + op_counts.get("x", 0)
            + op_counts.get("h", 0)
        )

        return cx_count, u3_count, True, "Direct-Transpile", transpiled_circuit

    except Exception as e:
        print(f"  Direct transpilation failed: {str(e)[:60]}...")

        # NO FALLBACK - Return failure
        return 0, 0, False, "FAILED", None


def compare_quantum_states(
    state1: Union[Statevector, np.ndarray],
    state2: Union[Statevector, np.ndarray],
    tolerance: float = 1e-10,
    verbose: bool = True,
) -> Dict:
    """
    Compare two quantum states by calculating fidelity and checking amplitude differences.

    Args:
        state1: First quantum state (Statevector or numpy array)
        state2: Second quantum state (Statevector or numpy array)
        tolerance: Numerical tolerance for amplitude comparison (default: 1e-10)
        verbose: Whether to print detailed comparison results (default: True)

    Returns:
        dict: Dictionary containing:
            - 'fidelity': State fidelity between the two states
            - 'amplitudes_match': Boolean indicating if amplitudes match within tolerance
            - 'max_amplitude_diff': Maximum absolute difference between corresponding amplitudes
            - 'mismatched_indices': List of indices where amplitudes don't match
            - 'amplitude_differences': Array of all amplitude differences
            - 'relative_errors': Array of relative errors for non-zero amplitudes
    """

    # Convert to numpy arrays if they're Statevector objects
    if hasattr(state1, "data"):
        amp1 = state1.data
    else:
        amp1 = np.array(state1)

    if hasattr(state2, "data"):
        amp2 = state2.data
    else:
        amp2 = np.array(state2)

    # Ensure arrays have the same length
    if len(amp1) != len(amp2):
        raise ValueError(f"States must have the same dimension. Got {len(amp1)} and {len(amp2)}")

    # Calculate fidelity
    fidelity = state_fidelity(state1, state2)

    # Calculate amplitude differences
    amplitude_diffs = np.abs(amp1 - amp2)
    max_amplitude_diff = np.max(amplitude_diffs)

    # Find mismatched indices
    mismatched_indices = np.where(amplitude_diffs > tolerance)[0].tolist()
    amplitudes_match = len(mismatched_indices) == 0

    # Calculate relative errors for non-zero amplitudes
    relative_errors = np.zeros_like(amplitude_diffs)
    non_zero_mask = np.abs(amp1) > tolerance
    relative_errors[non_zero_mask] = amplitude_diffs[non_zero_mask] / np.abs(amp1[non_zero_mask])

    # Create results dictionary
    results = {
        "fidelity": fidelity,
        "amplitudes_match": amplitudes_match,
        "max_amplitude_diff": max_amplitude_diff,
        "mismatched_indices": mismatched_indices,
        "amplitude_differences": amplitude_diffs,
        "relative_errors": relative_errors,
    }

    if verbose:
        print(f"State Comparison Results:")
        print(f"========================")
        print(f"Fidelity: {fidelity:.10f}")
        print(f"Amplitudes match (tolerance={tolerance}): {amplitudes_match}")
        print(f"Maximum amplitude difference: {max_amplitude_diff:.2e}")

        if not amplitudes_match:
            print(f"Number of mismatched amplitudes: {len(mismatched_indices)}")
            print(f"Mismatched indices: {mismatched_indices}")

            # Show detailed mismatches for first few indices
            max_show = min(5, len(mismatched_indices))
            print(f"\nDetailed mismatches (showing first {max_show}):")
            for i, idx in enumerate(mismatched_indices[:max_show]):
                state_label = format(idx, f"0{int(np.log2(len(amp1)))}b")
                print(
                    f"  |{state_label}⟩: {amp1[idx]:.6f} vs {amp2[idx]:.6f} "
                    f"(diff: {amplitude_diffs[idx]:.2e}, rel_err: {relative_errors[idx]:.2e})"
                )

            if len(mismatched_indices) > max_show:
                print(f"  ... and {len(mismatched_indices) - max_show} more mismatches")
        else:
            print("All amplitudes match within tolerance!")

    return results


def detailed_state_analysis(
    state1: Union[Statevector, np.ndarray],
    state2: Union[Statevector, np.ndarray],
    tolerance: float = 1e-10,
) -> None:
    """
    Perform detailed analysis of two quantum states, showing all non-zero amplitudes.

    Args:
        state1: First quantum state
        state2: Second quantum state
        tolerance: Threshold for considering amplitudes as non-zero
    """

    # Get comparison results
    results = compare_quantum_states(state1, state2, tolerance, verbose=False)

    # Convert to numpy arrays
    if hasattr(state1, "data"):
        amp1 = state1.data
    else:
        amp1 = np.array(state1)

    if hasattr(state2, "data"):
        amp2 = state2.data
    else:
        amp2 = np.array(state2)

    # Find non-zero amplitudes in either state
    non_zero_mask = (np.abs(amp1) > tolerance) | (np.abs(amp2) > tolerance)
    non_zero_indices = np.where(non_zero_mask)[0]

    n_qubits = int(np.log2(len(amp1)))

    print(f"Detailed State Analysis:")
    print(f"=======================")
    print(f"Fidelity: {results['fidelity']:.10f}")
    print(f"Number of qubits: {n_qubits}")
    print(f"Non-zero amplitudes (tolerance={tolerance}):")
    print(
        f"{'State':<{n_qubits+2}} {'Amplitude 1':<15} {'Amplitude 2':<15} {'Difference':<12} {'Match'}"
    )
    print("-" * (n_qubits + 50))

    for idx in non_zero_indices:
        state_label = format(idx, f"0{n_qubits}b")
        diff = np.abs(amp1[idx] - amp2[idx])
        match = "✓" if diff <= tolerance else "✗"
        print(f"|{state_label}⟩ {amp1[idx]:>14.6f} {amp2[idx]:>14.6f} {diff:>11.2e} {match:>5}")

    print("")


# Example usage function that works with your existing code structure
def analyze_circuit_comparison(qc_original, qc_simplified, tolerance=1e-10):
    """
    Analyze the comparison between original and simplified circuits.
    This function integrates with your existing code structure.

    Args:
        qc_original: Original quantum circuit
        qc_simplified: Simplified quantum circuit
        tolerance: Numerical tolerance for comparison
    """
    from src.misc import count_gates, get_state  # Import your functions

    # Get states
    state_original = get_state(qc_original)
    state_simplified = get_state(qc_simplified)

    # Get gate counts
    cx_count, u3_count = count_gates(qc_original)
    cx_count_simplified, u3_count_simplified = count_gates(qc_simplified)

    print(f"Circuit Comparison Analysis:")
    print(f"===========================")
    print(f"Original circuit - CX: {cx_count}, U3: {u3_count}")
    print(f"Simplified circuit - CX: {cx_count_simplified}, U3: {u3_count_simplified}")

    gate_diff_cx = cx_count - cx_count_simplified
    gate_diff_u3 = u3_count - u3_count_simplified
    print(
        f"CX gate reduction: {gate_diff_cx} ({gate_diff_cx / cx_count * 100 if cx_count > 0 else 0:.2f}%)"
    )
    print(
        f"U3 gate reduction: {gate_diff_u3} ({gate_diff_u3 / u3_count * 100 if u3_count > 0 else 0:.2f}%)"
    )
    print()

    # Detailed state comparison
    results = compare_quantum_states(state_original, state_simplified, tolerance, verbose=False)

    return results
