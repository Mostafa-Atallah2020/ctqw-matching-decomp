import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply


class Node:
    def __init__(self, level: int):
        self.value = None  # Will be set later based on label length
        self.level = level
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.zigzag_connections = []
        self.x: float = 0.0
        self.y: float = 0.0

    def set_position(self, x: float, y: float):
        self.x = x
        self.y = y

    def add_zigzag_connection(self, node: "Node"):
        if node not in self.zigzag_connections:
            self.zigzag_connections.append(node)


class OptimalLabeler:
    def __init__(self, num_vertices: int):
        self.num_vertices = num_vertices
        self.label_length = self._calculate_min_bits()
        self.used_labels = {"0" * self.label_length}
        self.vertex_to_label = {0: "0" * self.label_length}

    def _calculate_min_bits(self) -> int:
        num_bits = len(bin(self.num_vertices)[2:])
        return num_bits + (1 if self.num_vertices > 2**num_bits else 0)

    def get_hamming_distance(self, label1: str, label2: str) -> int:
        return sum(c1 != c2 for c1, c2 in zip(label1, label2))

    def get_available_labels(self) -> List[str]:
        all_possible = [format(i, f"0{self.label_length}b") for i in range(2**self.label_length)]
        return [label for label in all_possible if label not in self.used_labels]

    def find_best_label(
        self, vertex: int, neighbors: List[int], edges: List[Tuple[int, int]]
    ) -> str:
        available_labels = self.get_available_labels()
        if not available_labels:
            self.label_length += 1
            self.vertex_to_label = {
                k: v.zfill(self.label_length) for k, v in self.vertex_to_label.items()
            }
            self.used_labels = {label.zfill(self.label_length) for label in self.used_labels}
            available_labels = self.get_available_labels()

        best_score = float("inf")
        best_label = available_labels[0]

        for label in available_labels:
            score = 0
            for neighbor in neighbors:
                if neighbor in self.vertex_to_label:
                    neighbor_label = self.vertex_to_label[neighbor]
                    ham_dist = self.get_hamming_distance(label, neighbor_label)
                    score += (ham_dist - 1) ** 2 if ham_dist > 1 else 0

            if score < best_score:
                best_score = score
                best_label = label

        return best_label

    def optimize_labels(
        self, edges: List[Tuple[int, int]], starting_vertex: int = 0
    ) -> Dict[int, str]:
        adj_list = defaultdict(list)
        for v1, v2 in edges:
            adj_list[v1].append(v2)
            adj_list[v2].append(v1)

        queue = [(starting_vertex, [])]
        visited = {starting_vertex}

        while queue:
            vertex, neighbors = queue.pop(0)
            if vertex not in self.vertex_to_label:
                best_label = self.find_best_label(vertex, neighbors, edges)
                self.vertex_to_label[vertex] = best_label
                self.used_labels.add(best_label)

            for neighbor in adj_list[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(
                        (neighbor, [v for v in adj_list[neighbor] if v in self.vertex_to_label])
                    )

        return self.vertex_to_label

    def verify_labeling(self, edges: List[Tuple[int, int]]) -> Tuple[bool, Dict[str, int]]:
        if len(set(self.vertex_to_label.values())) != len(self.vertex_to_label):
            return False, {}

        dist_count = defaultdict(int)
        for v1, v2 in edges:
            if v1 in self.vertex_to_label and v2 in self.vertex_to_label:
                dist = self.get_hamming_distance(self.vertex_to_label[v1], self.vertex_to_label[v2])
                dist_count[dist] += 1

        return True, dict(dist_count)


class HardWeldedTree:
    def __init__(self, height: int, fixed_labels: Optional[Dict[int, str]] = None):
        if height < 2:
            raise ValueError("Height must be at least 2")

        self.height = height
        self.num_vertices = self._calculate_num_vertices()
        self.entrance = None
        self.exit = None
        self.all_nodes = []
        self.middle_layer_1 = []
        self.middle_layer_2 = []

        # Create tree structure first
        self._create_tree()

        # Then assign labels
        if fixed_labels:
            self._assign_labels(fixed_labels)
        else:
            labeler = OptimalLabeler(self.num_vertices)
            edges = self._create_edge_list()
            vertex_labels = labeler.optimize_labels(edges)
            is_valid, _ = labeler.verify_labeling(edges)
            if not is_valid:
                raise ValueError("Invalid labeling: duplicate labels found")
            self._assign_labels(vertex_labels)

    def _calculate_num_vertices(self) -> int:
        num = 2  # Entrance + Exit
        for i in range(self.height - 1):
            num += 2**i  # Upper tree
        for i in range(self.height - 1):
            num += 2**i  # Lower tree
        num += 2 * (2**self.height)  # Two middle layers
        return num

    def _create_upper_tree(self):
        self.entrance = Node(0)
        self.entrance.set_position(0, 1.0)
        self.all_nodes.append(self.entrance)
        current_level = [self.entrance]

        for level in range(self.height):
            next_level = []
            nodes_in_level = 2 ** (level + 1)
            x_spacing = 1.0 / (nodes_in_level + 1)
            y_position = 1.0 - ((level + 1) / (self.height + 1))

            for j, parent in enumerate(current_level):
                left_child = Node(level + 1)
                right_child = Node(level + 1)

                left_child.set_position(-0.5 + (2 * j + 1) * x_spacing, y_position)
                right_child.set_position(-0.5 + (2 * j + 2) * x_spacing, y_position)

                parent.left = left_child
                parent.right = right_child

                next_level.extend([left_child, right_child])
                self.all_nodes.extend([left_child, right_child])

            current_level = next_level
            if level == self.height - 1:
                self.middle_layer_1 = current_level

    def _create_lower_tree(self):
        nodes_in_level = 2**self.height
        x_spacing = 1.0 / (nodes_in_level + 1)
        y_position = 0.4

        for i in range(nodes_in_level):
            node = Node(self.height + 1)
            node.set_position(-0.5 + (i + 1) * x_spacing, y_position)
            self.middle_layer_2.append(node)
            self.all_nodes.append(node)

        current_level = self.middle_layer_2
        for level in range(self.height - 1):
            next_level = []
            nodes_in_level = 2 ** (self.height - level - 1)
            x_spacing = 1.0 / (nodes_in_level + 1)
            y_position = 0.4 - ((level + 1) / (self.height + 1))

            for j in range(0, len(current_level), 2):
                child = Node(self.height + level + 2)
                child.set_position(-0.5 + (j // 2 + 1) * x_spacing, y_position)

                current_level[j].left = child
                if j + 1 < len(current_level):
                    current_level[j + 1].right = child

                next_level.append(child)
                self.all_nodes.append(child)

            current_level = next_level

        self.exit = Node(2 * self.height)
        self.exit.set_position(0, 0)
        self.all_nodes.append(self.exit)

        for node in current_level:
            node.left = self.exit

    def _create_zigzag_connections(self):
        n = len(self.middle_layer_1)
        for i in range(n):
            node1 = self.middle_layer_1[i]
            next_idx = (i + 1) % n
            next_next_idx = (i + 2) % n
            node1.add_zigzag_connection(self.middle_layer_2[next_idx])
            node1.add_zigzag_connection(self.middle_layer_2[next_next_idx])

    def _create_edge_list(self) -> List[Tuple[int, int]]:
        edges = []
        for node in self.all_nodes:
            if node.left:
                edges.append((self.all_nodes.index(node), self.all_nodes.index(node.left)))
            if node.right:
                edges.append((self.all_nodes.index(node), self.all_nodes.index(node.right)))

        for node in self.middle_layer_1:
            for zigzag_node in node.zigzag_connections:
                edges.append((self.all_nodes.index(node), self.all_nodes.index(zigzag_node)))

        return edges

    def _assign_labels(self, vertex_labels: Dict[int, str]):
        for vertex_id, label in vertex_labels.items():
            if vertex_id < len(self.all_nodes):
                self.all_nodes[vertex_id].value = label

    def _create_tree(self):
        self._create_upper_tree()
        self._create_lower_tree()
        self._create_zigzag_connections()

    def get_edges(self) -> Set[Tuple[str, str]]:
        edges = set()
        for node in self.all_nodes:
            if node.left:
                edges.add((node.value, node.left.value))
            if node.right:
                edges.add((node.value, node.right.value))

        for node in self.middle_layer_1:
            for zigzag_node in node.zigzag_connections:
                edges.add((node.value, zigzag_node.value))
                edges.add((zigzag_node.value, node.value))
        return edges

    def get_vertex_labels(self) -> Dict[int, str]:
        return {i: node.value for i, node in enumerate(self.all_nodes)}

    def draw(self, figsize=None):
        if figsize is None:
            width = min(10, 3 + self.height)
            height = min(12, 4 + self.height)
            figsize = (width, height)

        plt.figure(figsize=figsize)

        for node in self.all_nodes:
            if node.left:
                plt.plot([node.x, node.left.x], [node.y, node.left.y], "b-", linewidth=1)
            if node.right:
                plt.plot([node.x, node.right.x], [node.y, node.right.y], "b-", linewidth=1)

        for node in self.middle_layer_1:
            for zigzag_node in node.zigzag_connections:
                plt.plot(
                    [node.x, zigzag_node.x], [node.y, zigzag_node.y], "r--", linewidth=1, alpha=0.6
                )

        label_size = max(6, 8 - (self.height - 2))
        for node in self.all_nodes:
            plt.plot(node.x, node.y, "ko", markersize=8)
            plt.text(
                node.x + 0.05,
                node.y,
                node.value,
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=label_size,
            )

            if node == self.entrance or node == self.exit:
                label = "Entrance" if node == self.entrance else "Exit"
                y_offset = 0.05 if node == self.entrance else -0.05
                plt.text(
                    node.x,
                    node.y + y_offset,
                    label,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=label_size + 1,
                )

        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        return plt.gcf()


class BinaryWeldedTree:
    def __init__(self, height: int):
        self.height = height
        self.bits_per_part = height  # Each part gets 'height' bits
        self.total_bits = 2 * height  # Total bits for each label
        self.entrance = None
        self.exit = None
        self.all_nodes: List[Node] = []
        self._create_maze()

    def _create_label(self, top_part: int, bottom_part: int) -> str:
        """Create a binary label with the correct number of bits for each part"""
        top_binary = format(top_part, f"0{self.bits_per_part}b")
        bottom_binary = format(bottom_part, f"0{self.bits_per_part}b")
        return top_binary + bottom_binary

    def _create_branch(
        self, start_level: int, is_entrance: bool, connect_to: Optional[List[Node]] = None
    ) -> List[Node]:
        """Create a branch (either entrance or exit part)"""
        current_level = []

        if is_entrance:
            # Create entrance node (all zeros)
            self.entrance = Node(self._create_label(0, 0), start_level)
            self.entrance.set_position(0, 1.0)
            current_level = [self.entrance]
            self.all_nodes.append(self.entrance)
        else:
            # Create exit node (zeros for top, ones for bottom)
            max_bottom = (1 << self.bits_per_part) - 1  # All ones for bottom part
            self.exit = Node(self._create_label(0, max_bottom), start_level)
            self.exit.set_position(0, 0)
            current_level = [self.exit]
            self.all_nodes.append(self.exit)

        level_num = start_level + (1 if is_entrance else -1)
        levels_to_create = self.height - 1

        for i in range(levels_to_create):
            next_level = []
            nodes_in_level = 2 ** (i + 1)
            x_spacing = 1.0 / (nodes_in_level + 1)
            total_levels = self.height * 2 - 1
            y_position = 1.0 - ((i + 1) / total_levels) if is_entrance else (i + 1) / total_levels

            for j, parent in enumerate(current_level):
                for child_idx in range(2):
                    # Calculate position value for label
                    pos = j * 2 + child_idx + 1
                    if is_entrance:
                        label = self._create_label(pos, 0)  # Top part gets position, bottom zeros
                    else:
                        label = self._create_label(0, pos)  # Top part zeros, bottom gets position

                    # Create child node
                    child = Node(label, level_num)

                    # Set position
                    x_pos = -0.5 + ((2 * j + child_idx) + 1) * x_spacing
                    child.set_position(x_pos, y_position)

                    # Connect nodes
                    if is_entrance:
                        if child_idx == 0:
                            parent.left = child
                        else:
                            parent.right = child
                    else:
                        if child_idx == 0:
                            child.left = parent
                        else:
                            child.right = parent

                    next_level.append(child)
                    self.all_nodes.append(child)

            current_level = next_level

        # Connect to middle layer if provided
        if connect_to:
            if is_entrance:
                for i, node in enumerate(current_level):
                    node.left = connect_to[i * 2]
                    node.right = connect_to[i * 2 + 1]
            else:
                for i, node in enumerate(current_level):
                    connect_to[i * 2].left = node
                    connect_to[i * 2 + 1].right = node

        return current_level

    def _create_middle_layer(self, level: int) -> List[Node]:
        """Create the shared middle layer"""
        nodes_in_level = 2**self.height
        x_spacing = 1.0 / (nodes_in_level + 1)
        y_position = 0.5

        middle_nodes = []
        for i in range(nodes_in_level):
            # Create unique middle layer labels
            top_part = (i // 2) + 1
            bottom_part = (i % 2) + 1
            label = self._create_label(top_part, bottom_part)

            node = Node(label, level)
            node.set_position(-0.5 + (i + 1) * x_spacing, y_position)
            middle_nodes.append(node)
            self.all_nodes.append(node)

        return middle_nodes

    def _create_maze(self):
        # Create shared middle layer first
        middle_layer = self._create_middle_layer(self.height)

        # Create top part, connecting to middle layer
        self._create_branch(0, True, middle_layer)

        # Create bottom part, connecting to middle layer
        self._create_branch(self.height * 2 - 1, False, middle_layer)

    def get_edges(self) -> Set[Tuple[str, str]]:
        """Get all edges in the format required for the graph code"""
        edges = set()
        for node in self.all_nodes:
            if node.left:
                edges.add((node.value, node.left.value))
            if node.right:
                edges.add((node.value, node.right.value))
        return edges

    def draw(self, figsize=(10, 12)):
        plt.figure(figsize=figsize)

        # Draw edges
        for node in self.all_nodes:
            if node.left:
                plt.plot([node.x, node.left.x], [node.y, node.left.y], "b-", linewidth=1)
            if node.right:
                plt.plot([node.x, node.right.x], [node.y, node.right.y], "b-", linewidth=1)

        # Draw nodes
        for node in self.all_nodes:
            plt.plot(node.x, node.y, "ko", markersize=10)
            # Add binary label
            plt.text(
                node.x + 0.02,
                node.y,
                node.value,
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=8,
            )
            # Add Entrance/Exit labels if applicable
            if node == self.entrance or node == self.exit:
                label = "Entrance" if node == self.entrance else "Exit"
                y_offset = 0.05 if node == self.entrance else -0.05
                plt.text(
                    node.x,
                    node.y + y_offset,
                    label,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=10,
                )

        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        return plt.gcf()


class QuantumWalkBWT:
    """
    Quantum Walk simulation on Binary Welded Tree.
    Implements the continuous-time quantum walk based on the paper:
    "Exponential algorithmic speedup by a quantum walk" by Childs et al.
    (arXiv:quant-ph/0209131)
    """

    def __init__(self, tree: Optional[HardWeldedTree] = None, height: int = 3):
        """
        Initialize a quantum walk on a Binary Welded Tree.

        Args:
            tree: An existing HardWeldedTree instance, or None to create a new one
            height: Height of the tree if creating a new one
        """
        if tree is None:
            self.tree = HardWeldedTree(height)
        else:
            self.tree = tree

        self.height = self.tree.height
        self.n_vertices = len(self.tree.all_nodes)
        self.entrance_idx = self.tree.all_nodes.index(self.tree.entrance)
        self.exit_idx = self.tree.all_nodes.index(self.tree.exit)

        # Construct Hamiltonian (adjacency matrix)
        self.H = self._construct_hamiltonian()

        # For theoretical comparison
        self.n = self.height  # Match notation from paper

    def _construct_hamiltonian(self):
        """
        Construct the Hamiltonian for the quantum walk.

        The Hamiltonian is based on the adjacency matrix of the graph,
        with special coupling at the "defect" (the weld between trees).

        Returns:
            scipy.sparse.csc_matrix: Sparse Hamiltonian matrix
        """
        # Initialize sparse matrix
        H = sparse.lil_matrix((self.n_vertices, self.n_vertices), dtype=np.complex128)

        # Get edges from the tree
        edges = self.tree.get_edges()

        # Build Hamiltonian based on graph connectivity
        node_to_idx = {node.value: i for i, node in enumerate(self.tree.all_nodes)}

        for u, v in edges:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]

            # Check if this is a connection between middle layers (the "defect")
            is_middle_connection = False
            for node in self.tree.middle_layer_1:
                if node.value == u and any(znode.value == v for znode in node.zigzag_connections):
                    is_middle_connection = True
                    break
                if node.value == v and any(znode.value == u for znode in node.zigzag_connections):
                    is_middle_connection = True
                    break

            # Set coupling strength based on whether it's part of the "defect"
            if is_middle_connection:
                # γ = 1/√2 at the defect as per the paper
                H[u_idx, v_idx] = 1 / np.sqrt(2)
                H[v_idx, u_idx] = 1 / np.sqrt(2)
            else:
                # Regular connections have coupling 1
                H[u_idx, v_idx] = 1
                H[v_idx, u_idx] = 1

        return H.tocsc()  # Convert to CSC format for efficient operations

    def time_evolution(self, t: float):
        """
        Evolve the quantum state from entrance node for time t.

        Args:
            t (float): Time to evolve the system

        Returns:
            numpy.ndarray: Evolved quantum state vector
        """
        # Initial state: localized at entrance
        psi_0 = np.zeros(self.n_vertices, dtype=np.complex128)
        psi_0[self.entrance_idx] = 1.0

        # Time evolution: e^(-iHt)|ψ₀⟩
        # Use scipy's expm_multiply for efficient computation
        psi_t = expm_multiply(-1j * self.H * t, psi_0)

        return psi_t

    def exit_probability(self, t: float):
        """
        Calculate probability of being at exit node at time t.

        Args:
            t (float): Time to measure the exit probability

        Returns:
            float: Probability of finding the quantum walker at the exit node
        """
        psi_t = self.time_evolution(t)
        prob = np.abs(psi_t[self.exit_idx]) ** 2
        return prob

    def column_subspace_hamiltonian(self):
        """
        Construct the effective Hamiltonian in the column subspace.

        This corresponds to the column-based analysis in the paper, which shows
        that the quantum walk dynamics can be understood by considering a simpler
        effective Hamiltonian acting on column states.

        Returns:
            numpy.ndarray: Effective Hamiltonian in the column subspace
        """
        # Create a (2n+1)×(2n+1) Hamiltonian for the column subspace
        n = self.height
        H_column = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.complex128)

        # Set non-zero elements - nearest-neighbor couplings
        for j in range(2 * n):
            if j == n - 1:  # At the defect
                H_column[j, j + 1] = np.sqrt(2)
                H_column[j + 1, j] = np.sqrt(2)
            else:
                H_column[j, j + 1] = 1
                H_column[j + 1, j] = 1

        return H_column

    def theoretical_exit_probability(self, t: float):
        """
        Calculate theoretical exit probability based on the correct formula:

        ∑_E |⟨E|1⟩|² |⟨E|2n⟩|² + ∑_{E≠E'} [(1-e^(-i(E-E')τ))/(i(E-E')τ)] ⟨2n|E⟩⟨E|1⟩⟨1|E'⟩⟨E'|2n⟩

        Args:
            t (float): Time to calculate the theoretical exit probability

        Returns:
            float: Theoretical probability
        """
        # Use the column subspace Hamiltonian to simplify calculations
        H_col = self.column_subspace_hamiltonian()

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(H_col)

        # Define positions of entrance (|1⟩) and exit (|2n⟩) nodes in column representation
        # Note: In a column subspace of size (2n+1), indices range from 0 to 2n
        entrance_idx = 0  # First column
        exit_idx = 2 * self.n  # Last column

        # First term: sum_E |⟨E|1⟩|² |⟨E|2n⟩|²
        first_term = 0.0
        for i in range(len(eigenvalues)):
            # Get amplitudes for entrance and exit columns
            entrance_amp = eigenvectors[entrance_idx, i]  # ⟨1|E⟩
            exit_amp = eigenvectors[exit_idx, i]  # ⟨2n|E⟩

            # Add contribution to first term
            first_term += abs(entrance_amp) ** 2 * abs(exit_amp) ** 2

        # Second term: sum_{E≠E'} [(1-e^(-i(E-E')τ))/(i(E-E')τ)] ⟨2n|E⟩⟨E|1⟩⟨1|E'⟩⟨E'|2n⟩
        second_term = 0.0 + 0.0j  # Complex accumulator
        for i in range(len(eigenvalues)):
            for j in range(len(eigenvalues)):
                if i != j:  # E ≠ E'
                    # Calculate energy difference
                    energy_diff = eigenvalues[i] - eigenvalues[j]

                    # Calculate phase factor with careful handling of small denominators
                    if abs(energy_diff * t) < 1e-10:
                        # Use Taylor expansion for small arguments to avoid division by zero
                        phase_factor = 1.0 - 1j * energy_diff * t / 2.0
                    else:
                        phase_factor = (1.0 - np.exp(-1j * energy_diff * t)) / (
                            1j * energy_diff * t
                        )

                    # Calculate amplitude products
                    entrance_exit_i = (
                        eigenvectors[exit_idx, i] * eigenvectors[entrance_idx, i]
                    )  # ⟨2n|E⟩⟨E|1⟩
                    entrance_exit_j = (
                        eigenvectors[entrance_idx, j] * eigenvectors[exit_idx, j]
                    )  # ⟨1|E'⟩⟨E'|2n⟩

                    # Add contribution to second term
                    second_term += phase_factor * entrance_exit_i * entrance_exit_j

        # Total probability is the sum of both terms (take real part to ensure a real result)
        total_probability = first_term + np.real(second_term)

        return float(total_probability)

    def exit_probability_derivative(self, t: float, delta: float = 1e-6):
        """
        Calculate the numerical derivative of the exit probability function.

        Args:
            t: Time point at which to calculate the derivative
            delta: Small time increment for numerical differentiation

        Returns:
            float: Approximate derivative of exit probability at time t
        """
        # Central difference approximation
        p_plus = self.theoretical_exit_probability(t + delta)
        p_minus = self.theoretical_exit_probability(t - delta)

        return (p_plus - p_minus) / (2 * delta)

    def analytical_hitting_time(
        self, t_min: float = 0.1, t_max: Optional[float] = None, num_initial_points: int = 5
    ) -> Tuple[float, float]:
        """
        Find the analytical hitting time by solving for the time when the derivative
        of the exit probability equals zero.

        Args:
            t_min: Minimum time to consider
            t_max: Maximum time to consider (defaults to 2*π*(2n+1)/2)
            num_initial_points: Number of initial points to try for optimization

        Returns:
            tuple: (hitting_time, max_probability)
        """
        if t_max is None:
            # Use twice the expected optimal time as search range
            t_max = 2 * np.pi * (2 * self.n + 1) / 2

        # Objective function: We want to find where derivative = 0 (root finding)
        # Convert to minimization by returning absolute value of derivative
        def objective(t):
            if t <= 0:  # Avoid negative times
                return float("inf")
            return abs(self.exit_probability_derivative(float(t)))

        # We'll try multiple starting points to avoid local minima
        theoretical_t = np.pi * (2 * self.n + 1) / 2  # Expected hitting time from theory

        # Create initial points spread around the theoretical value
        initial_points = np.linspace(t_min, t_max, num_initial_points)
        if theoretical_t > t_min and theoretical_t < t_max:
            # Make sure theoretical value is one of our starting points
            initial_points = np.sort(np.append(initial_points, theoretical_t))

        best_result = None
        best_objective = float("inf")

        # Try optimization from each starting point
        for t0 in initial_points:
            result = minimize(
                objective,
                t0,
                method="Nelder-Mead",
                bounds=[(t_min, t_max)],
                options={"xatol": 1e-8, "fatol": 1e-8},
            )

            if result.success and result.fun < best_objective:
                best_objective = result.fun
                best_result = result

        if best_result is None:
            print("Warning: Optimization failed to find a hitting time.")
            # Fallback to the theoretical expectation
            hitting_time = theoretical_t
        else:
            hitting_time = float(best_result.x[0])

        # Calculate the maximum probability at the hitting time
        max_probability = self.theoretical_exit_probability(hitting_time)

        return hitting_time, max_probability

    def optimize_hitting_time_gradient(
        self, t_min: float = 0.1, t_max: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Find the analytical hitting time using gradient descent to maximize exit probability.

        Args:
            t_min: Minimum time to consider
            t_max: Maximum time to consider (defaults to 2*π*(2n+1)/2)

        Returns:
            tuple: (hitting_time, max_probability)
        """
        if t_max is None:
            # Use twice the expected optimal time as search range
            t_max = 2 * np.pi * (2 * self.n + 1) / 2

        # Objective function: Negative of exit probability (for minimization)
        def objective(t):
            return -self.theoretical_exit_probability(float(t[0]))

        # Gradient function (numerical approximation)
        def gradient(t):
            return np.array([-self.exit_probability_derivative(float(t[0]))])

        # Initial guess based on theoretical prediction
        t0 = np.array([np.pi * (2 * self.n + 1) / 2])

        # Run optimization
        result = minimize(
            objective,
            t0,
            method="L-BFGS-B",
            jac=gradient,
            bounds=[(t_min, t_max)],
            options={"ftol": 1e-8, "gtol": 1e-8},
        )

        if not result.success:
            print("Warning: Gradient-based optimization failed to find a hitting time.")
            print(f"Reason: {result.message}")
            # Fallback to the theoretical expectation
            hitting_time = float(t0[0])
        else:
            hitting_time = float(result.x[0])

        # Calculate the maximum probability at the hitting time
        max_probability = self.theoretical_exit_probability(hitting_time)

        return hitting_time, max_probability

    def find_hitting_time_numerical(self, num_samples=300, t_max=None) -> Tuple[float, float]:
        """
        Find the hitting time (time of maximum exit probability) using numerical sampling.

        Args:
            num_samples: Number of time points to sample
            t_max: Maximum time to consider (if None, estimated automatically)

        Returns:
            Tuple containing:
            - numerical_hitting_time: Time of maximum probability
            - max_numerical_prob: Maximum probability
        """
        # Estimate optimal time if not provided
        if t_max is None:
            # Theoretical estimate from paper: π(2n+1)/2
            theoretical_optimal_t = np.pi * (2 * self.n + 1) / 2
            t_max = 2 * theoretical_optimal_t

        # Create time points with more samples around expected peak
        t_values_standard = np.linspace(0, t_max, num_samples)
        t_values_peak = np.linspace(
            0.8 * theoretical_optimal_t, 1.2 * theoretical_optimal_t, num_samples // 2
        )
        t_values = np.sort(np.unique(np.concatenate([t_values_standard, t_values_peak])))

        # Calculate theoretical exit probabilities
        theo_probs = []
        for t in t_values:
            theo_prob = self.theoretical_exit_probability(t)
            theo_probs.append(theo_prob)

        # Find maximum probability and corresponding time
        max_idx = np.argmax(theo_probs)
        hitting_time = t_values[max_idx]
        max_prob = theo_probs[max_idx]

        return hitting_time, max_prob

    def plot_exit_probability(self, t_max=None, samples=200, ax=None, show_hitting_times=True):
        """
        Plot the exit probability over time, comparing numerical and theoretical values.

        Args:
            t_max (float, optional): Maximum time for the plot
            samples (int): Number of time points to sample
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            show_hitting_times (bool): Whether to mark hitting times

        Returns:
            matplotlib.figure.Figure: Figure containing the plot
            tuple: (numerical_hitting_time, max_numerical_prob, theoretical_hitting_time, max_theoretical_prob)
        """
        if t_max is None:
            # Use the expected optimal time as reference
            optimal_t = np.pi * (2 * self.n + 1) / 2
            t_max = 2 * optimal_t

        # Sample more points around the expected peak
        t_values_standard = np.linspace(0, t_max, samples)
        t_values_peak = np.linspace(0.9 * optimal_t, 1.1 * optimal_t, samples // 4)
        t_values = np.sort(np.unique(np.concatenate([t_values_standard, t_values_peak])))

        # Calculate probabilities
        print(f"Calculating exit probabilities for {len(t_values)} time points...")
        start_time = time.time()

        numerical_probs = []
        theoretical_probs = []
        for t in t_values:
            numerical_probs.append(self.exit_probability(t))
            theoretical_probs.append(self.theoretical_exit_probability(t))

        elapsed = time.time() - start_time
        print(f"Calculations completed in {elapsed:.2f} seconds.")

        # Create new figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure

        # Plot results
        ax.plot(t_values, numerical_probs, "b-", linewidth=2, label="Numerical")
        ax.plot(t_values, theoretical_probs, "r--", linewidth=2, label="Theoretical")

        # Calculate peak points
        max_num_idx = np.argmax(numerical_probs)
        max_num_t = t_values[max_num_idx]
        max_num_prob = numerical_probs[max_num_idx]

        # Find theoretical maximum
        theo_max_idx = np.argmax(theoretical_probs)
        theo_hitting_time = t_values[theo_max_idx]
        theo_max_prob = theoretical_probs[theo_max_idx]

        if show_hitting_times:
            # Find analytical hitting time
            analytical_t, analytical_p = self.analytical_hitting_time()
            gradient_t, gradient_p = self.optimize_hitting_time_gradient()

            # Highlight optimal times
            # Formula-based
            optimal_t = np.pi * (2 * self.n + 1) / 2
            ax.axvline(
                x=optimal_t, color="k", linestyle="-.", label=f"Formula: t = {optimal_t:.2f}"
            )

            # Analytical
            ax.axvline(
                x=analytical_t,
                color="r",
                linestyle=":",
                label=f"Root finding: t = {analytical_t:.2f}",
            )

            # Gradient-based
            ax.axvline(
                x=gradient_t, color="g", linestyle="--", label=f"Gradient: t = {gradient_t:.2f}"
            )

            # Mark maximum points
            ax.plot(max_num_t, max_num_prob, "bo", markersize=8)
            ax.plot(theo_hitting_time, theo_max_prob, "ro", markersize=8)
            ax.plot(analytical_t, analytical_p, "rx", markersize=10)
            ax.plot(gradient_t, gradient_p, "gx", markersize=10)

            # Add annotations
            ax.annotate(
                f"Num max: {max_num_prob:.4f} at t = {max_num_t:.2f}",
                xy=(max_num_t, max_num_prob),
                xytext=(max_num_t + 0.1 * t_max, max_num_prob - 0.1),
                arrowprops=dict(arrowstyle="->"),
            )

            ax.annotate(
                f"Theo max: {theo_max_prob:.4f} at t = {theo_hitting_time:.2f}",
                xy=(theo_hitting_time, theo_max_prob),
                xytext=(theo_hitting_time - 0.1 * t_max, theo_max_prob - 0.15),
                arrowprops=dict(arrowstyle="->"),
            )

        ax.set_title(f"Exit Probability for BWT Quantum Walk (height = {self.height})")
        ax.set_xlabel("Time t")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Configure axes
        ax.set_xlim(0, t_max)
        ax.set_ylim(0, 1.05 * max(max_num_prob, max(theoretical_probs)))

        plt.tight_layout()

        # Return both the figure and the hitting time information
        return fig, (max_num_t, max_num_prob, theo_hitting_time, theo_max_prob)

    def visualize_quantum_state(self, t, figsize=(10, 10)):
        """
        Visualize the quantum state on the BWT at a specific time.

        Args:
            t (float): Time at which to visualize the quantum state
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Figure containing the visualization
        """
        # Get the quantum state at time t
        psi_t = self.time_evolution(t)

        # Extract positions and probabilities
        positions = []
        probabilities = []
        for i, node in enumerate(self.tree.all_nodes):
            positions.append((node.x, node.y))
            probabilities.append(np.abs(psi_t[i]) ** 2)

        # Draw the graph
        fig = plt.figure(figsize=figsize)

        # Draw edges
        for node in self.tree.all_nodes:
            if node.left:
                plt.plot([node.x, node.left.x], [node.y, node.left.y], "b-", linewidth=1, alpha=0.3)
            if node.right:
                plt.plot(
                    [node.x, node.right.x], [node.y, node.right.y], "b-", linewidth=1, alpha=0.3
                )

        for node in self.tree.middle_layer_1:
            for zigzag_node in node.zigzag_connections:
                plt.plot(
                    [node.x, zigzag_node.x], [node.y, zigzag_node.y], "r--", linewidth=1, alpha=0.3
                )

        # Scale marker sizes by probability
        max_size = 500
        for i, ((x, y), prob) in enumerate(zip(positions, probabilities)):
            plt.scatter(x, y, s=prob * max_size, c="b", alpha=0.7)

            # Mark entrance and exit
            if i == self.entrance_idx:
                plt.scatter(x, y, s=100, facecolors="none", edgecolors="g", linewidth=2)
                plt.text(x, y + 0.05, "Entrance", ha="center", fontsize=10)
            elif i == self.exit_idx:
                plt.scatter(x, y, s=100, facecolors="none", edgecolors="r", linewidth=2)
                plt.text(x, y - 0.05, "Exit", ha="center", fontsize=10)

        plt.title(f"Quantum State Visualization at t = {t:.2f}")
        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()

        return fig
