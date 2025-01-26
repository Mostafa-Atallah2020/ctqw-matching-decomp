from typing import Dict, Set, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


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
