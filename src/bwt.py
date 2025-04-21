import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
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

    def theoretical_exit_probability(self, t: float):
        """
        Calculate theoretical exit probability based on the paper's formula.
        For t ≈ π(2n+1)/2, it reaches its peak.

        Args:
            t (float): Time to calculate the theoretical exit probability

        Returns:
            float: Theoretical probability based on the formulas from the paper
        """
        # For t near the optimal time, use the simplified formula
        optimal_t = np.pi * (2 * self.n + 1) / 2

        if abs(t - optimal_t) < 0.1 * np.pi:
            # The formula simplifies to (4/(2n+1)²) * sin⁴(nπ/(2n+1))
            # As n increases, sin(nπ/(2n+1)) approaches 1
            sin_term = np.sin(self.n * np.pi / (2 * self.n + 1))
            return (4 / (2 * self.n + 1) ** 2) * sin_term**4
        else:
            # For non-optimal times, use the full formula with phase factors
            result = 0
            for m in range(1, 2 * self.n + 1):
                # Eigenvalues: E_m = 2*cos(mπ/(2n+1))
                E_m = 2 * np.cos(m * np.pi / (2 * self.n + 1))
                sin_term = np.sin(m * np.pi / (2 * self.n + 1))
                phase = np.exp(-1j * E_m * t)

                # Contribution from this eigenvalue
                amplitude = (2 / (2 * self.n + 1)) * phase * ((-1) ** (m + 1)) * (sin_term**2)
                result += amplitude

            return np.abs(result) ** 2

    def column_subspace_hamiltonian(self):
        """
        Construct the effective Hamiltonian in the column subspace.

        This corresponds to the column-based analysis in the paper, which shows
        that the quantum walk dynamics can be understood by considering a simpler
        effective Hamiltonian acting on column states.

        Returns:
            numpy.ndarray: Effective Hamiltonian in the column subspace
        """
        # Create a 2n×2n Hamiltonian for the column subspace
        n = self.height
        H_column = np.zeros((2 * n, 2 * n), dtype=np.complex128)

        # Set non-zero elements - nearest-neighbor couplings
        for j in range(2 * n - 1):
            if j == n - 1:  # At the defect
                H_column[j, j + 1] = np.sqrt(2)
                H_column[j + 1, j] = np.sqrt(2)
            else:
                H_column[j, j + 1] = 1
                H_column[j + 1, j] = 1

        return H_column

    def plot_exit_probability(self, t_max=None, samples=200, ax=None):
        """
        Plot the exit probability over time, comparing numerical and theoretical values.

        Args:
            t_max (float, optional): Maximum time for the plot
            samples (int): Number of time points to sample
            ax (matplotlib.axes.Axes, optional): Axes to plot on

        Returns:
            matplotlib.figure.Figure: Figure containing the plot
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

        # Highlight optimal time
        optimal_t = np.pi * (2 * self.n + 1) / 2
        ax.axvline(
            x=optimal_t, color="g", linestyle=":", label=f"Optimal time: t ≈ {optimal_t:.2f}"
        )

        # Calculate peak points
        max_num_idx = np.argmax(numerical_probs)
        max_num_t = t_values[max_num_idx]
        max_num_prob = numerical_probs[max_num_idx]

        ax.annotate(
            f"Max: {max_num_prob:.4f} at t = {max_num_t:.2f}",
            xy=(max_num_t, max_num_prob),
            xytext=(max_num_t + 0.1 * t_max, max_num_prob - 0.1),
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
        return fig

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


def compare_theory_numerical(height=3, num_samples=200):
    """
    Compare theoretical and numerical exit probabilities for BWT quantum walk.

    Args:
        height (int): Height of the Binary Welded Tree
        num_samples (int): Number of time points to sample

    Returns:
        dict: Results of the comparison
    """
    print(f"=== BWT Quantum Walk Comparison (height={height}) ===")

    # Initialize quantum walk
    start_time = time.time()
    print("Initializing Binary Welded Tree and Hamiltonian...")
    qwalk = QuantumWalkBWT(height=height)
    print(f"Initialization completed in {time.time() - start_time:.2f} seconds")
    print(f"Number of vertices in BWT: {qwalk.n_vertices}")

    # Calculate theoretical optimal time
    optimal_t = np.pi * (2 * height + 1) / 2
    print(f"Theoretical optimal time: {optimal_t:.4f}")

    # Calculate exit probability at optimal time
    print("\nCalculating exit probability at optimal time...")
    num_prob = qwalk.exit_probability(optimal_t)
    theo_prob = qwalk.theoretical_exit_probability(optimal_t)

    print(f"  Numerical: {num_prob:.6f}")
    print(f"  Theoretical: {theo_prob:.6f}")
    print(f"  Absolute difference: {abs(num_prob - theo_prob):.6f}")
    print(f"  Relative difference: {abs(num_prob - theo_prob)/theo_prob*100:.2f}%")

    # Plot exit probability over time
    print("\nGenerating exit probability plot...")
    t_max = 2 * optimal_t

    # Sample more points around the expected peak for better resolution
    t_values_standard = np.linspace(0, t_max, num_samples)
    t_values_around_peak = np.linspace(0.8 * optimal_t, 1.2 * optimal_t, num_samples // 2)
    t_values = np.sort(np.unique(np.concatenate([t_values_standard, t_values_around_peak])))

    # Calculate probabilities
    print(f"Calculating probabilities for {len(t_values)} time points...")
    start_time = time.time()
    num_probs = [qwalk.exit_probability(t) for t in t_values]
    theo_probs = [qwalk.theoretical_exit_probability(t) for t in t_values]
    print(f"Calculations completed in {time.time() - start_time:.2f} seconds")

    # Find peak values
    max_num_idx = np.argmax(num_probs)
    max_num_t = t_values[max_num_idx]
    max_num_prob = num_probs[max_num_idx]

    max_theo_idx = np.argmax(theo_probs)
    max_theo_t = t_values[max_theo_idx]
    max_theo_prob = theo_probs[max_theo_idx]

    print("\nPeak exit probabilities:")
    print(f"  Numerical peak: {max_num_prob:.6f} at time {max_num_t:.4f}")
    print(f"  Theoretical peak: {max_theo_prob:.6f} at time {max_theo_t:.4f}")
    print(f"  Time difference: {abs(max_num_t - max_theo_t):.4f}")
    print(f"  Probability difference: {abs(max_num_prob - max_theo_prob):.6f}")

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Main plot
    ax1.plot(t_values, num_probs, "b-", linewidth=2, label="Numerical")
    ax1.plot(t_values, theo_probs, "r--", linewidth=2, label="Theoretical")
    ax1.axvline(
        x=optimal_t, color="g", linestyle=":", label=f"Expected optimal time: t = {optimal_t:.4f}"
    )

    # Mark peak points
    ax1.plot(max_num_t, max_num_prob, "bo", markersize=8)
    ax1.plot(max_theo_t, max_theo_prob, "ro", markersize=8)

    ax1.annotate(
        f"Max numerical: {max_num_prob:.4f} at t = {max_num_t:.2f}",
        xy=(max_num_t, max_num_prob),
        xytext=(max_num_t - 0.2 * t_max, max_num_prob - 0.2 * max_num_prob),
        arrowprops=dict(arrowstyle="->"),
    )

    ax1.annotate(
        f"Max theoretical: {max_theo_prob:.4f} at t = {max_theo_t:.2f}",
        xy=(max_theo_t, max_theo_prob),
        xytext=(max_theo_t + 0.2 * t_max, max_theo_prob - 0.2 * max_theo_prob),
        arrowprops=dict(arrowstyle="->"),
    )

    ax1.set_title(f"Exit Probability for BWT Quantum Walk (height = {height})")
    ax1.set_ylabel("Probability")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    ax1.set_xlim(0, t_max)
    ax1.set_ylim(0, 1.05 * max(max_num_prob, max_theo_prob))

    # Difference plot
    diff = np.array(num_probs) - np.array(theo_probs)
    ax2.plot(t_values, diff, "k-", linewidth=1.5)
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax2.set_title("Difference (Numerical - Theoretical)")
    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Difference")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, t_max)

    plt.tight_layout()

    # Generate visualizations at key times
    print("\nGenerating quantum state visualizations...")
    visualization_times = [
        0,  # Initial state
        optimal_t * 0.25,  # Quarter way
        optimal_t * 0.5,  # Halfway
        optimal_t * 0.75,  # Three-quarters way
        optimal_t,  # At optimal time
        max_num_t,  # At numerical peak
    ]

    visualizations = []
    for t in visualization_times:
        print(f"  Visualizing state at t = {t:.2f}...")
        vis_fig = qwalk.visualize_quantum_state(t)
        visualizations.append((t, vis_fig))

    return {
        "height": height,
        "n_vertices": qwalk.n_vertices,
        "optimal_time_theoretical": optimal_t,
        "optimal_time_numerical": max_num_t,
        "peak_prob_theoretical": max_theo_prob,
        "peak_prob_numerical": max_num_prob,
        "time_difference": abs(max_num_t - optimal_t),
        "prob_difference": abs(max_num_prob - max_theo_prob),
        "relative_time_diff": abs(max_num_t - optimal_t) / optimal_t,
        "relative_prob_diff": abs(max_num_prob - max_theo_prob) / max_theo_prob,
        "probability_plot": fig,
        "state_visualizations": visualizations,
        "qwalk": qwalk,
    }


def analyze_scaling(heights=[2, 3, 4, 5]):
    """Analyze how optimal time and peak probability scale with BWT height."""
    print("\n=== BWT Quantum Walk Scaling Analysis ===")
    results = []

    for h in heights:
        print(f"\nProcessing BWT with height {h}...")
        result = compare_theory_numerical(height=h, num_samples=100)
        results.append(result)

    # Create summary table
    print("\n=== Scaling Analysis Summary ===")
    print(
        f"{'Height':<8} {'Vertices':<10} {'Theo Time':<12} {'Num Time':<12} {'Theo Prob':<12} {'Num Prob':<12} {'Rel Time Diff':<15} {'Rel Prob Diff':<15}"
    )
    print(
        f"{'------':<8} {'--------':<10} {'---------':<12} {'--------':<12} {'---------':<12} {'--------':<12} {'-------------':<15} {'-------------':<15}"
    )

    for r in results:
        print(
            f"{r['height']:<8} {r['n_vertices']:<10} {r['optimal_time_theoretical']:<12.4f} "
            f"{r['optimal_time_numerical']:<12.4f} {r['peak_prob_theoretical']:<12.6f} "
            f"{r['peak_prob_numerical']:<12.6f} {r['relative_time_diff']*100:<15.2f}% "
            f"{r['relative_prob_diff']*100:<15.2f}%"
        )

    # Create scaling plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Extract data
    h_values = [r["height"] for r in results]
    n_vertices = [r["n_vertices"] for r in results]
    theo_times = [r["optimal_time_theoretical"] for r in results]
    num_times = [r["optimal_time_numerical"] for r in results]
    theo_probs = [r["peak_prob_theoretical"] for r in results]
    num_probs = [r["peak_prob_numerical"] for r in results]

    # Plot time scaling
    ax1.plot(h_values, theo_times, "go-", linewidth=2, label="Theoretical")
    ax1.plot(h_values, num_times, "bo-", linewidth=2, label="Numerical")

    # Linear fit for theoretical times
    x_fit = np.linspace(min(h_values), max(h_values), 100)
    y_fit = np.pi * (2 * x_fit + 1) / 2  # Theoretical formula
    ax1.plot(x_fit, y_fit, "r--", label=r"$\pi(2n+1)/2)")

    ax1.set_title("Optimal Time vs. BWT Height")
    ax1.set_xlabel("Height (n)")
    ax1.set_ylabel("Optimal Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot probability scaling
    ax2.plot(h_values, theo_probs, "go-", linewidth=2, label="Theoretical")
    ax2.plot(h_values, num_probs, "bo-", linewidth=2, label="Numerical")

    # Theoretical scaling: 4/(2n+1)^2
    y_fit_prob = 4 / (2 * x_fit + 1) ** 2
    ax2.plot(x_fit, y_fit_prob, "r--", label=r"$4/(2n+1)^2")

    # Log-log inset for probability scaling
    axins = ax2.inset_axes([0.55, 0.55, 0.4, 0.4])
    axins.loglog(h_values, theo_probs, "go-", linewidth=2)
    axins.loglog(h_values, num_probs, "bo-", linewidth=2)
    axins.loglog(x_fit, y_fit_prob, "r--")
    axins.set_title("Log-Log Scale")
    axins.grid(True, alpha=0.3)

    ax2.set_title("Peak Exit Probability vs. BWT Height")
    ax2.set_xlabel("Height (n)")
    ax2.set_ylabel("Peak Probability")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    return fig, results
