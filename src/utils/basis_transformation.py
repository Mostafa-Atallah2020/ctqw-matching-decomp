from qiskit import QuantumCircuit, QuantumRegister
from typing import Dict, List, Tuple, Set
from functools import reduce


def compose_transformations(tr1: Dict[str, str], tr2: Dict[str, str]) -> Dict[str, str]:
    """
    Compose two transformations: TR2 ∘ TR1 (apply TR1 first, then TR2).

    Args:
        tr1: First transformation as a dictionary {state: state}
        tr2: Second transformation as a dictionary {state: state}

    Returns:
        Dictionary representing the composed transformation
    """
    n_qubits = len(next(iter(tr1.keys())))
    all_states = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]

    complete_tr1 = {state: tr1.get(state, state) for state in all_states}
    complete_tr2 = {state: tr2.get(state, state) for state in all_states}

    return {state: complete_tr2[complete_tr1[state]] for state in all_states}


def compose_multiple_transformations(transformations: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Compose multiple transformations from left to right.

    Args:
        transformations: List of transformation dictionaries

    Returns:
        Dictionary representing the composed transformation
    """
    return reduce(compose_transformations, transformations)


def get_number_of_qubits(transformation: Dict[str, str]) -> int:
    """Get the number of qubits from a transformation dictionary."""
    return len(next(iter(transformation.keys())))


def get_all_states(n_qubits: int) -> List[str]:
    """Generate all possible basis states for n qubits."""
    return [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]


def complete_transformation(transformation: Dict[str, str], n_qubits: int = None) -> Dict[str, str]:
    """
    Get the full transformation dictionary including unchanged states.

    Args:
        transformation: Partial transformation dictionary (may not include identity mappings)
        n_qubits: Number of qubits (if None, inferred from transformation keys)

    Returns:
        Complete transformation dictionary with all 2^n states

    Example:
        >>> tr = {'10': '11', '11': '10'}
        >>> complete_transformation(tr, 2)
        {'00': '00', '01': '01', '10': '11', '11': '10'}
    """
    if not transformation:
        raise ValueError("Transformation dictionary cannot be empty")

    if n_qubits is None:
        n_qubits = get_number_of_qubits(transformation)

    all_states = get_all_states(n_qubits)

    return {state: transformation.get(state, state) for state in all_states}


def extract_swaps(transformation: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    Extract swap pairs from transformation dictionary.

    Args:
        transformation: Dictionary {initial_state: final_state}

    Returns:
        List of tuples representing state swaps
    """

    def find_cycles(trans: Dict[str, str], processed: Set[str]) -> List[List[str]]:
        cycles = []
        for state in trans:
            if state not in processed and trans[state] != state:
                cycle = []
                current = state
                while current not in processed:
                    cycle.append(current)
                    processed.add(current)
                    current = trans.get(current, current)
                    if current == state:
                        break
                if len(cycle) > 1:
                    cycles.append(cycle)
        return cycles

    processed = set()
    cycles = find_cycles(transformation, processed)

    # Convert cycles to swap pairs
    swaps = []
    for cycle in cycles:
        if len(cycle) == 2:
            swaps.append((cycle[0], cycle[1]))
        else:
            # Decompose longer cycles into swaps
            for i in range(len(cycle) - 1):
                swaps.append((cycle[i], cycle[i + 1]))

    return swaps


def get_differing_qubits(state1: str, state2: str) -> List[int]:
    """
    Find qubit positions that differ between two states (indexed right to left).

    Args:
        state1: First binary state string
        state2: Second binary state string

    Returns:
        List of qubit indices that differ
    """
    n_qubits = len(state1)
    return [n_qubits - 1 - i for i in range(n_qubits) if state1[i] != state2[i]]


def get_common_pattern(state1: str, state2: str, value: str = "1") -> List[int]:
    """
    Find qubit positions where both states have the same value (indexed right to left).

    Args:
        state1: First binary state string
        state2: Second binary state string
        value: Value to look for ('0' or '1')

    Returns:
        List of qubit indices with common value
    """
    n_qubits = len(state1)
    return [n_qubits - 1 - i for i in range(n_qubits) if state1[i] == state2[i] == value]


def create_single_qubit_swap_circuit(state1: str, state2: str, n_qubits: int) -> QuantumCircuit:
    """
    Create circuit for swapping two states that differ by one qubit.

    Args:
        state1: First state
        state2: Second state
        n_qubits: Number of qubits

    Returns:
        QuantumCircuit implementing the swap
    """
    qc = QuantumCircuit(n_qubits)

    diff_qubits = get_differing_qubits(state1, state2)
    target = diff_qubits[0]

    common_ones = get_common_pattern(state1, state2, "1")
    common_zeros = get_common_pattern(state1, state2, "0")

    # Remove target from common patterns if present
    common_ones = [q for q in common_ones if q != target]
    common_zeros = [q for q in common_zeros if q != target]

    # Apply X gates for zero controls
    for q in common_zeros:
        qc.x(q)

    # Apply controlled gate
    if len(common_ones) == 0 and len(common_zeros) == 0:
        qc.x(target)
    elif len(common_ones) == 1 and len(common_zeros) == 0:
        qc.cx(common_ones[0], target)
    elif len(common_ones) + len(common_zeros) == 1:
        control = common_ones[0] if common_ones else common_zeros[0]
        qc.cx(control, target)
    elif len(common_ones) + len(common_zeros) == 2:
        controls = common_ones + common_zeros
        qc.ccx(controls[0], controls[1], target)
    else:
        controls = common_ones + common_zeros
        qc.mcx(controls, target)

    # Remove X gates for zero controls
    for q in common_zeros:
        qc.x(q)

    return qc


def create_two_qubit_swap_circuit(state1: str, state2: str, n_qubits: int) -> QuantumCircuit:
    """
    Create circuit for swapping two states that differ by two qubits.

    Args:
        state1: First state
        state2: Second state
        n_qubits: Number of qubits

    Returns:
        QuantumCircuit implementing the swap
    """
    qc = QuantumCircuit(n_qubits)

    diff_qubits = get_differing_qubits(state1, state2)
    common_ones = get_common_pattern(state1, state2, "1")

    if len(common_ones) >= 1:
        # Use Toffoli-based swap
        control = common_ones[0]
        target1, target2 = diff_qubits

        qc.ccx(control, target1, target2)
        qc.x(target2)
        qc.ccx(control, target2, target1)
        qc.x(target2)
        qc.ccx(control, target1, target2)
    else:
        # Use X gates to create proper control pattern
        common_zeros = get_common_pattern(state1, state2, "0")
        target1, target2 = diff_qubits

        # Apply X gates for zero controls
        for q in common_zeros:
            qc.x(q)

        if common_zeros:
            control = common_zeros[0]
            qc.ccx(control, target1, target2)
            qc.x(target2)
            qc.ccx(control, target2, target1)
            qc.x(target2)
            qc.ccx(control, target1, target2)

        # Remove X gates
        for q in common_zeros:
            qc.x(q)

    return qc


def create_swap_circuit(state1: str, state2: str, n_qubits: int) -> QuantumCircuit:
    """
    Create circuit for swapping two specific states.

    Args:
        state1: First state
        state2: Second state
        n_qubits: Number of qubits

    Returns:
        QuantumCircuit implementing the swap
    """
    diff_qubits = get_differing_qubits(state1, state2)

    if len(diff_qubits) == 1:
        return create_single_qubit_swap_circuit(state1, state2, n_qubits)
    elif len(diff_qubits) == 2:
        return create_two_qubit_swap_circuit(state1, state2, n_qubits)
    else:
        # Multiple qubits differ - use general approach
        qc = QuantumCircuit(n_qubits)
        # For complex swaps, decompose into simpler operations
        # This is a simplified version - may need optimization
        for target in diff_qubits:
            common_ones = [
                n_qubits - 1 - i
                for i in range(n_qubits)
                if state1[i] == "1" and (n_qubits - 1 - i) != target
            ]
            if len(common_ones) == 1:
                qc.cx(common_ones[0], target)
            elif len(common_ones) >= 2:
                qc.mcx(common_ones[:2], target)
        return qc


def transformation_to_circuit(transformation: Dict[str, str]) -> QuantumCircuit:
    """
    Generate a Qiskit circuit for a given transformation.

    Args:
        transformation: Dictionary {initial_state: final_state}

    Returns:
        QuantumCircuit implementing the transformation
    """
    n_qubits = get_number_of_qubits(transformation)
    swaps = extract_swaps(transformation)

    # Create individual circuits for each swap
    swap_circuits = [create_swap_circuit(s1, s2, n_qubits) for s1, s2 in swaps]

    # Compose all circuits
    qc = QuantumCircuit(n_qubits)
    for circuit in swap_circuits:
        qc.compose(circuit, inplace=True)

    return qc


def create_transformation_from_swaps(swaps: List[Tuple[str, str]], n_qubits: int) -> Dict[str, str]:
    """
    Create a transformation dictionary from a list of swaps.

    Args:
        swaps: List of tuples (state1, state2) to swap
        n_qubits: Number of qubits

    Returns:
        Dictionary representing the transformation
    """
    all_states = get_all_states(n_qubits)
    transformation = {state: state for state in all_states}

    for state1, state2 in swaps:
        transformation[state1] = state2
        transformation[state2] = state1

    return transformation


def print_transformation(
    transformation: Dict[str, str], title: str = "Transformation", show_unchanged: bool = False
):
    """
    Pretty print a transformation.

    Args:
        transformation: Dictionary representing the transformation
        title: Title to print
        show_unchanged: If True, also print unchanged states
    """
    print(f"\n{title}:")
    for state, target in sorted(transformation.items()):
        if state != target or show_unchanged:
            arrow = "→" if state != target else "↦"
            print(f"|{state}⟩ {arrow} |{target}⟩")


def verify_transformation(circuit: QuantumCircuit, expected: Dict[str, str]) -> bool:
    """
    Verify that a circuit implements the expected transformation.
    (This is a placeholder - actual verification would require simulation)

    Args:
        circuit: QuantumCircuit to verify
        expected: Expected transformation dictionary

    Returns:
        True if verification passes (placeholder always returns True)
    """
    # In a real implementation, this would simulate the circuit
    # and verify it matches the expected transformation
    return True


# Example usage
if __name__ == "__main__":
    # Example 1: Complete a partial transformation
    print("=" * 60)
    print("Example 1: Complete partial transformation")
    tr_partial = {"10": "11", "11": "10"}
    print(f"\nPartial transformation: {tr_partial}")

    tr_complete = complete_transformation(tr_partial, 2)
    print(f"\nComplete transformation: {tr_complete}")
    print_transformation(
        tr_complete, "Complete transformation (showing all states)", show_unchanged=True
    )

    # Example 2: Compose two transformations
    print("\n" + "=" * 60)
    print("Example 2: Compose transformations")
    tr1 = {"101": "011", "011": "101"}
    tr2 = {"100": "110", "110": "100", "101": "111", "111": "101"}

    # Complete them first
    tr1_complete = complete_transformation(tr1, 3)
    tr2_complete = complete_transformation(tr2, 3)

    print(f"\nTR1 (partial): {tr1}")
    print(f"TR1 (complete): {tr1_complete}")
    print(f"\nTR2 (partial): {tr2}")
    print(f"TR2 (complete): {tr2_complete}")

    composed = compose_transformations(tr1, tr2)
    print_transformation(composed, "Composed transformation (TR2 ∘ TR1)")

    circuit = transformation_to_circuit(composed)
    print("\nCircuit:")
    print(circuit)

    # Example 3: Show difference between partial and complete
    print("\n" + "=" * 60)
    print("Example 3: Partial vs Complete transformation display")

    tr_simple = {"000": "010", "010": "000"}
    print(f"\nPartial transformation dict: {tr_simple}")

    print_transformation(tr_simple, "Partial (only changed states)")

    tr_simple_complete = complete_transformation(tr_simple, 3)
    print_transformation(tr_simple_complete, "Complete (all states)", show_unchanged=True)

    # Example 4: Compose multiple and show complete result
    print("\n" + "=" * 60)
    print("Example 4: Multiple composition with complete output")

    tr_a = {"000": "010", "010": "000"}
    tr_b = {"010": "110", "110": "010"}

    composed_ab = compose_transformations(tr_a, tr_b)
    composed_ab_complete = complete_transformation(composed_ab, 3)

    print("\nComposed transformation (complete):")
    print(composed_ab_complete)
    print_transformation(composed_ab_complete, "All states", show_unchanged=True)

    # Example 5: Error handling
    print("\n" + "=" * 60)
    print("Example 5: Using complete_transformation with inference")

    tr_infer = {"0000": "1000", "1000": "0000", "0001": "1001", "1001": "0001"}
    tr_infer_complete = complete_transformation(tr_infer)  # n_qubits inferred as 4

    print(f"\nInferred {get_number_of_qubits(tr_infer_complete)} qubits")
    print(f"Total states in complete transformation: {len(tr_infer_complete)}")
    print_transformation(tr_infer_complete, "4-qubit transformation")
