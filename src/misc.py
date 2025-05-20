import itertools

import networkx as nx
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


def lists_to_sets(*lists):
    # Use itertools.product to get all combinations of picking one element from each list
    combinations = list(itertools.product(*lists))

    # Convert each combination (which is a tuple) to a set
    sets = [set(comb) for comb in combinations]

    return sets


def graph_matchings(edges):
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
