import itertools
from collections import defaultdict

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
    return sum(len(neighbors) for neighbors in G.adj.values()) // 2


def calculate_edge_density(edge_count, vertex_count):
    """Calculate edge density of a graph."""
    max_possible_edges = (vertex_count * (vertex_count - 1)) / 2
    return edge_count / max_possible_edges


def is_bipartite(G):
    color = {}
    for start_node in G.nodes():
        if start_node not in color:
            stack = [(start_node, 0)]
            while stack:
                node, c = stack.pop()
                if node in color:
                    if color[node] != c:
                        return False
                else:
                    color[node] = c
                    stack.extend((neighbor, 1 - c) for neighbor in G.adj[node])
    return True


def find_diameter(G):
    if not nx.is_connected(G):
        return float("inf")

    def bfs(start):
        distances = {start: 0}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in G.adj[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        return max(distances.values())

    return max(bfs(node) for node in G.nodes())


def find_max_clique(G):
    def is_clique(nodes):
        return all(v in G.adj[u] for u in nodes for v in nodes if u != v)

    def backtrack(candidates, clique):
        if not candidates:
            return clique
        v = max(candidates, key=lambda x: len(G.adj[x]))
        candidates.remove(v)
        new_clique = clique | {v}
        for u in list(candidates):
            if not all(u in G.adj[w] for w in new_clique):
                candidates.remove(u)
        return max((backtrack(candidates.copy(), new_clique), clique), key=len)

    return len(backtrack(set(G.nodes()), set()))


def average_clustering(G):
    def local_clustering(node):
        neighbors = list(G.adj[node])
        if len(neighbors) < 2:
            return 0
        links = sum(1 for u in neighbors for v in neighbors if u < v and v in G.adj[u])
        possible_links = len(neighbors) * (len(neighbors) - 1) / 2
        return links / possible_links if possible_links > 0 else 0

    if len(G.nodes()) == 0:
        return 0
    return sum(local_clustering(node) for node in G.nodes()) / len(G.nodes())


def estimate_group_size(G):
    degree_counts = defaultdict(int)
    for node in G.nodes():
        degree_counts[len(G.adj[node])] += 1
    return max(degree_counts.values()) if degree_counts else 0


def estimate_orbit_count(G):
    return len(set(len(G.adj[node]) for node in G.nodes()))


def collect_graph_properties(filename, n_vertices):
    properties = {
        "edge_counts": [],
        "edge_densities": [],
        "bipartite_counts": defaultdict(int),
        "diameters": [],
        "clique_numbers": [],
        "max_degrees": [],
        "clustering_coefficients": [],
        "group_sizes": [],
        "orbit_counts": [],
    }

    with open(filename, "r") as file:
        for line in file:
            try:
                G = nx.from_graph6_bytes(line.strip().encode("utf-8"))
                edge_count = count_edges(G)

                properties["edge_counts"].append(edge_count)
                properties["edge_densities"].append(calculate_edge_density(edge_count, n_vertices))
                properties["bipartite_counts"][is_bipartite(G)] += 1

                if nx.is_connected(G):
                    properties["diameters"].append(find_diameter(G))

                properties["clique_numbers"].append(find_max_clique(G))
                properties["max_degrees"].append(max(G.degree(node) for node in G.nodes()))
                properties["clustering_coefficients"].append(average_clustering(G))
                properties["group_sizes"].append(estimate_group_size(G))
                properties["orbit_counts"].append(estimate_orbit_count(G))

            except nx.NetworkXError as e:
                print(f"Error processing graph: {e}")

    return properties
