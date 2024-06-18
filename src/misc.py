from qiskit.circuit.library import RXGate


def multi_crx(angle, ctrl_state):
    n_ctrls = len(ctrl_state)
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
