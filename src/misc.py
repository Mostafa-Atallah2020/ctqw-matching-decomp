import itertools
from collections import defaultdict
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
    """
    Create optimal parallel matchings by finding the best relabeling strategy.

    Tries multiple approaches and returns the one with minimum total hamming distance.

    Args:
        edges: Set/list of tuples representing edges (u, v) where u, v are binary strings

    Returns:
        List of sets, where each set is a matching using optimized binary labels
    """
    edges = list(edges)
    if not edges:
        return []

    # Get all vertices
    vertices = list(set(v for edge in edges for v in edge))

    # Determine bit length from the input edges (preserve original bit length)
    n_bits = len(vertices[0]) if vertices else 2

    # Ensure we have enough bits for all vertices
    min_bits_needed = max(2, (len(vertices) - 1).bit_length())
    if min_bits_needed > n_bits:
        n_bits = min_bits_needed

    def hamming_distance(u, v):
        """Calculate hamming distance between binary strings"""
        try:
            u_int = int(u, 2)
            v_int = int(v, 2)
            return bin(u_int ^ v_int).count("1")
        except ValueError:
            return len(u)  # Fallback for non-binary

    def calculate_total_cost(edge_list):
        """Calculate total hamming distance for a list of edges"""
        return sum(hamming_distance(u, v) for u, v in edge_list)

    def find_all_maximum_matchings(edge_list):
        """Find maximum matching using Edmonds' Blossom algorithm via NetworkX"""
        # Create graph
        G = nx.Graph()
        G.add_edges_from(edge_list)

        # Find maximum matching using Edmonds' Blossom
        max_matching = nx.max_weight_matching(G)

        # Convert to list format consistent with original function
        return [list(max_matching)]

    def create_relabeling_from_matching(matching, all_vertices):
        """Create vertex relabeling based on a matching"""
        vertex_mapping = {}
        used_labels = set()

        # Label matching edges to flip bit 0
        counter = 0
        for u, v in matching:
            if u not in vertex_mapping and v not in vertex_mapping:
                if n_bits == 1:
                    label1, label2 = "0", "1"
                else:
                    base = format(counter, f"0{n_bits-1}b")
                    label1 = base + "0"
                    label2 = base + "1"

                vertex_mapping[u] = label1
                vertex_mapping[v] = label2
                used_labels.update([label1, label2])
                counter += 1

        # Label remaining vertices
        remaining = [v for v in all_vertices if v not in vertex_mapping]
        label_int = 0
        for vertex in remaining:
            while True:
                label = format(label_int, f"0{n_bits}b")
                if label not in used_labels:
                    vertex_mapping[vertex] = label
                    used_labels.add(label)
                    break
                label_int += 1

        return vertex_mapping

    def try_simple_swaps():
        """Try simple vertex swaps to find better labelings"""
        # Start with identity mapping - preserve original labels if possible
        current_mapping = {}
        used_labels = set()

        # First, try to keep original labels
        for vertex in vertices:
            if vertex not in used_labels and len(vertex) == n_bits:
                current_mapping[vertex] = vertex
                used_labels.add(vertex)

        # Assign labels to remaining vertices
        label_int = 0
        for vertex in vertices:
            if vertex not in current_mapping:
                while True:
                    label = format(label_int, f"0{n_bits}b")
                    if label not in used_labels:
                        current_mapping[vertex] = label
                        used_labels.add(label)
                        break
                    label_int += 1

        best_mapping = current_mapping.copy()
        best_cost = calculate_total_cost(
            [(current_mapping[u], current_mapping[v]) for u, v in edges]
        )

        # Try all pairwise swaps
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                # Swap labels of vertices[i] and vertices[j]
                test_mapping = current_mapping.copy()
                test_mapping[vertices[i]], test_mapping[vertices[j]] = (
                    test_mapping[vertices[j]],
                    test_mapping[vertices[i]],
                )

                test_cost = calculate_total_cost(
                    [(test_mapping[u], test_mapping[v]) for u, v in edges]
                )

                if test_cost < best_cost:
                    best_cost = test_cost
                    best_mapping = test_mapping.copy()

        return best_mapping, best_cost

    def edges_to_matchings(relabeled_edges):
        """Convert edges to parallel matchings grouped by bit flip"""

        def get_bit_flip_position(u, v):
            u_int = int(u, 2)
            v_int = int(v, 2)
            diff = u_int ^ v_int

            if diff == 0:
                return -1
            elif bin(diff).count("1") == 1:
                return (diff & -diff).bit_length() - 1
            else:
                return -2

        # Group by bit flip position
        groups = defaultdict(list)
        for u, v in relabeled_edges:
            bit_pos = get_bit_flip_position(u, v)
            groups[bit_pos].append((u, v))

        # Create matchings for each group
        matchings = []
        for group_edges in groups.values():
            remaining = group_edges[:]
            while remaining:
                matching = set()
                used_vertices = set()
                next_remaining = []

                for u, v in remaining:
                    if u not in used_vertices and v not in used_vertices:
                        matching.add((u, v))
                        used_vertices.update([u, v])
                    else:
                        next_remaining.append((u, v))

                if matching:
                    matchings.append(matching)

                if len(next_remaining) == len(remaining):
                    break
                remaining = next_remaining

        return matchings

    # Strategy 1: Try all maximum matchings
    all_max_matchings = find_all_maximum_matchings(edges)
    best_cost = float("inf")
    best_strategy = None

    for matching in all_max_matchings:
        mapping = create_relabeling_from_matching(matching, vertices)
        relabeled_edges = [(mapping[u], mapping[v]) for u, v in edges]
        cost = calculate_total_cost(relabeled_edges)

        if cost < best_cost:
            best_cost = cost
            best_strategy = ("matching", mapping, relabeled_edges)

    # Strategy 2: Try simple swaps
    swap_mapping, swap_cost = try_simple_swaps()
    swap_edges = [(swap_mapping[u], swap_mapping[v]) for u, v in edges]

    if swap_cost < best_cost:
        best_cost = swap_cost
        best_strategy = ("swap", swap_mapping, swap_edges)

    # Use the best strategy found
    _, best_mapping, best_edges = best_strategy

    # Convert to matchings
    matchings = edges_to_matchings(best_edges)

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


def reduce_graph_space(edges, active_qubits=None):
    """
    Reduce the graph space by keeping only essential qubits.
    
    Allow merging only if:
    - Number of edges is a power of 2 (complete subcube)
    - All edges flip the SAME set of qubits
    - Variation occurs in exactly log2(num_edges) non-flipping qubits
    """
    if not edges:
        return edges, active_qubits if active_qubits is not None else []
    
    bitstring_length = len(next(iter(edges))[0])
    
    if active_qubits is None:
        active_qubits = list(range(bitstring_length))
    
    edge_list = list(edges)
    num_edges = len(edges)
    
    # Check if number of edges is a power of 2
    is_power_of_2 = (num_edges & (num_edges - 1)) == 0 and num_edges > 0
    
    # Find qubits that flip in EACH edge and check consistency
    flipping_qubits_per_edge = []
    for source, target in edge_list:
        edge_flips = set()
        for qubit_idx in active_qubits:
            pos = bitstring_length - 1 - qubit_idx
            if source[pos] != target[pos]:
                edge_flips.add(qubit_idx)
        flipping_qubits_per_edge.append(edge_flips)
    
    # Check if all edges flip the same qubits
    if not flipping_qubits_per_edge:
        return edges, active_qubits
    
    flipping_qubits = flipping_qubits_per_edge[0]
    all_same_flips = all(edge_flips == flipping_qubits for edge_flips in flipping_qubits_per_edge)
    
    if not all_same_flips:
        # Edges flip different qubits - cannot merge
        return edges, active_qubits
    
    if not flipping_qubits:
        return edges, active_qubits
    
    non_flipping_qubits = sorted(set(active_qubits) - flipping_qubits, reverse=True)
    
    from itertools import combinations
    import math
    
    # Check if edges can merge
    can_merge = False
    
    if is_power_of_2 and num_edges >= 2 and len(non_flipping_qubits) > 0:
        expected_varying_bits = int(math.log2(num_edges))
        
        if len(non_flipping_qubits) >= expected_varying_bits:
            # Check which non-flipping qubits vary
            varying_non_flipping_source = []
            varying_non_flipping_target = []
            
            for qubit_idx in non_flipping_qubits:
                pos = bitstring_length - 1 - qubit_idx
                source_values = set(source[pos] for source, target in edge_list)
                target_values = set(target[pos] for source, target in edge_list)
                
                if len(source_values) > 1:
                    varying_non_flipping_source.append(qubit_idx)
                if len(target_values) > 1:
                    varying_non_flipping_target.append(qubit_idx)
            
            # Check if both source and target vary in the same qubits
            # and exactly the expected number of qubits vary
            if (varying_non_flipping_source == varying_non_flipping_target and
                len(varying_non_flipping_source) == expected_varying_bits):
                can_merge = True
    
    if can_merge:
        # Allow merging: find minimal edges with maximal bits
        min_distinct_edges = float('inf')
        best_qubits = list(flipping_qubits)
        
        # Try all subsets
        for flip_size in range(len(flipping_qubits), 0, -1):
            for flip_subset in combinations(sorted(flipping_qubits), flip_size):
                for non_flip_size in range(len(non_flipping_qubits) + 1):
                    for non_flip_subset in combinations(sorted(non_flipping_qubits), non_flip_size):
                        test_qubits = sorted(list(flip_subset) + list(non_flip_subset))
                        test_positions = sorted([bitstring_length - 1 - q for q in test_qubits])
                        
                        test_projected = set()
                        valid_projection = True
                        
                        for source, target in edge_list:
                            source_proj = ''.join(source[pos] for pos in test_positions)
                            target_proj = ''.join(target[pos] for pos in test_positions)
                            
                            if source_proj == target_proj:
                                valid_projection = False
                                break
                            
                            if source_proj <= target_proj:
                                test_projected.add((source_proj, target_proj))
                            else:
                                test_projected.add((target_proj, source_proj))
                        
                        if valid_projection:
                            distinct_count = len(test_projected)
                            if distinct_count < min_distinct_edges:
                                min_distinct_edges = distinct_count
                                best_qubits = test_qubits
                            elif distinct_count == min_distinct_edges and len(test_qubits) > len(best_qubits):
                                best_qubits = test_qubits
    else:
        # No merging allowed: return original edges with all bits
        return edges, active_qubits
    
    qubits_to_keep = sorted(best_qubits)
    positions_to_keep = sorted([bitstring_length - 1 - q for q in qubits_to_keep])
    
    reduced_edges = set()
    for source, target in edges:
        source_proj = ''.join(source[pos] for pos in positions_to_keep)
        target_proj = ''.join(target[pos] for pos in positions_to_keep)
        
        if source_proj <= target_proj:
            reduced_edges.add((source_proj, target_proj))
        else:
            reduced_edges.add((target_proj, source_proj))
    
    return reduced_edges, qubits_to_keep