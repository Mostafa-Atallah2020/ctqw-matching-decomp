import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple, Set
from collections import defaultdict


class Node:
    def __init__(self, value: str, level: int):
        self.value = value
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


class HardWeldedTree:
    def __init__(self, height: int):
        if height < 2:
            raise ValueError("Height must be at least 2")
        self.height = height

        # Calculate total number of vertices
        self.num_vertices = 0
        # Entrance + Exit
        self.num_vertices += 2
        # Upper tree (excluding middle layer)
        for i in range(self.height - 1):
            self.num_vertices += 2**i
        # Lower tree (excluding middle layer)
        for i in range(self.height - 1):
            self.num_vertices += 2**i
        # Two middle layers
        self.num_vertices += 2 * (2**self.height)

        # Calculate number of bits needed
        self.bits_needed = len(bin(self.num_vertices - 1)[2:])  # -1 because we start from 0
        self.current_label = 0
        self.entrance = None
        self.exit = None
        self.all_nodes = []
        self.middle_layer_1 = []
        self.middle_layer_2 = []
        self._create_tree()

    def _get_next_label(self) -> str:
        """Get next unique label"""
        label = format(self.current_label, f"0{self.bits_needed}b")
        self.current_label += 1
        return label

    def _create_unique_label(self, level: int, position: int, is_top: bool) -> str:
        """Create a unique binary label with exactly total_bits length"""
        label = None
        counter = 0
        while label is None or label in self.used_labels:
            if is_top:
                # For top part, fill first half with unique bits, second half with zeros
                top_part = (level * (2**self.height) + position + counter) % (2**self.bits_per_part)
                label = format(top_part, f"0{self.bits_per_part}b") + "0" * self.bits_per_part
            else:
                # For bottom part, fill first half with zeros, second half with unique bits
                bottom_part = (level * (2**self.height) + position + counter) % (
                    2**self.bits_per_part
                )
                label = "0" * self.bits_per_part + format(bottom_part, f"0{self.bits_per_part}b")
            counter += 1
        self.used_labels.add(label)
        return label

    def _create_middle_layer_label(self, layer_num: int, position: int) -> str:
        """Create unique middle layer label with exactly total_bits length"""
        label = None
        counter = 0
        while label is None or label in self.used_labels:
            if layer_num == 1:
                # First middle layer: unique first half, incremental second half
                top_part = ((2**self.height) + position + counter) % (2**self.bits_per_part)
                bottom_part = position % (2**self.bits_per_part)
            else:
                # Second middle layer: incremental first half, unique second half
                top_part = position % (2**self.bits_per_part)
                bottom_part = ((2**self.height) + position + counter) % (2**self.bits_per_part)
            label = format(top_part, f"0{self.bits_per_part}b") + format(
                bottom_part, f"0{self.bits_per_part}b"
            )
            counter += 1
        self.used_labels.add(label)
        return label

    def _create_upper_tree(self):
        """Create upper tree including first middle layer"""
        # Create entrance node
        self.entrance = Node(self._get_next_label(), 0)
        self.entrance.set_position(0, 1.0)
        self.all_nodes.append(self.entrance)
        current_level = [self.entrance]

        # Create intermediate levels
        for level in range(self.height):
            next_level = []
            nodes_in_level = 2 ** (level + 1)
            x_spacing = 1.0 / (nodes_in_level + 1)
            y_position = 1.0 - ((level + 1) / (self.height + 1))

            for j, parent in enumerate(current_level):
                # Create left and right children with unique labels
                left_child = Node(self._get_next_label(), level + 1)
                right_child = Node(self._get_next_label(), level + 1)

                # Set positions
                left_child.set_position(-0.5 + (2 * j + 1) * x_spacing, y_position)
                right_child.set_position(-0.5 + (2 * j + 2) * x_spacing, y_position)

                # Connect nodes
                parent.left = left_child
                parent.right = right_child

                next_level.extend([left_child, right_child])
                self.all_nodes.extend([left_child, right_child])

            current_level = next_level
            if level == self.height - 1:
                self.middle_layer_1 = current_level

    def _create_lower_tree(self):
        """Create lower tree including second middle layer"""
        nodes_in_level = 2**self.height
        x_spacing = 1.0 / (nodes_in_level + 1)
        y_position = 0.4

        # Create middle layer 2
        for i in range(nodes_in_level):
            node = Node(self._get_next_label(), self.height + 1)
            node.set_position(-0.5 + (i + 1) * x_spacing, y_position)
            self.middle_layer_2.append(node)
            self.all_nodes.append(node)

        # Create remaining levels
        current_level = self.middle_layer_2
        for level in range(self.height - 1):
            next_level = []
            nodes_in_level = 2 ** (self.height - level - 1)
            x_spacing = 1.0 / (nodes_in_level + 1)
            y_position = 0.4 - ((level + 1) / (self.height + 1))

            for j in range(0, len(current_level), 2):
                child = Node(self._get_next_label(), self.height + level + 2)
                child.set_position(-0.5 + (j // 2 + 1) * x_spacing, y_position)

                current_level[j].left = child
                if j + 1 < len(current_level):
                    current_level[j + 1].right = child

                next_level.append(child)
                self.all_nodes.append(child)

            current_level = next_level

        # Create exit node
        self.exit = Node(self._get_next_label(), 2 * self.height)
        self.exit.set_position(0, 0)
        self.all_nodes.append(self.exit)

        # Connect last level to exit
        for node in current_level:
            node.left = self.exit

    def _create_zigzag_connections(self):
        """Create zigzag connections between middle layers"""
        n = len(self.middle_layer_1)
        for i in range(n):
            node1 = self.middle_layer_1[i]
            next_idx = (i + 1) % n
            next_next_idx = (i + 2) % n

            node1.add_zigzag_connection(self.middle_layer_2[next_idx])
            node1.add_zigzag_connection(self.middle_layer_2[next_next_idx])

    def _create_tree(self):
        self._create_upper_tree()
        self._create_lower_tree()
        self._create_zigzag_connections()

    def get_edges(self) -> Set[Tuple[str, str]]:
        """Get all edges including zigzag connections"""
        edges = set()

        # Regular tree edges
        for node in self.all_nodes:
            if node.left:
                edges.add((node.value, node.left.value))
            if node.right:
                edges.add((node.value, node.right.value))

        # Zigzag connections (bidirectional)
        for node in self.middle_layer_1:
            for zigzag_node in node.zigzag_connections:
                edges.add((node.value, zigzag_node.value))
                edges.add((zigzag_node.value, node.value))

        return edges

    def draw(self, figsize=None):
        """Draw the tree with perfect mirroring between upper and lower parts."""
        if figsize is None:
            width = min(10, 3 + self.height)
            height = min(12, 4 + self.height)
            figsize = (width, height)

        plt.figure(figsize=figsize)

        # Basic parameters
        total_height = 1.0
        middle_y = total_height / 2
        middle_gap = 0.1

        # Position entrance and exit
        self.entrance.x = 0
        self.entrance.y = 1.0
        self.exit.x = 0
        self.exit.y = 0.0

        # Position middle layers
        for i, node in enumerate(self.middle_layer_1):
            width = len(self.middle_layer_1)
            x = -1 + 2 * (i + 1) / (width + 1)
            node.x = x
            node.y = middle_y + middle_gap / 2

        for i, node in enumerate(self.middle_layer_2):
            width = len(self.middle_layer_2)
            x = -1 + 2 * (i + 1) / (width + 1)
            node.x = x
            node.y = middle_y - middle_gap / 2

        # Group nodes by level
        upper_levels = {}
        lower_levels = {}
        for node in self.all_nodes:
            if node in [self.entrance, self.exit] + self.middle_layer_1 + self.middle_layer_2:
                continue
            if node.level < self.height:
                upper_levels.setdefault(node.level, []).append(node)
            elif node.level > self.height:
                lower_levels.setdefault(node.level, []).append(node)

        # Position upper tree nodes
        for level, nodes in upper_levels.items():
            y = 1.0 - (level / self.height) * (1.0 - (middle_y + middle_gap / 2))
            width = len(nodes)
            for i, node in enumerate(nodes):
                node.x = -1 + 2 * (i + 1) / (width + 1)
                node.y = y

        # Position lower tree nodes with mirroring
        for level, nodes in lower_levels.items():
            mirror_level = 2 * self.height - level
            y = (mirror_level / self.height) * (middle_y - middle_gap / 2)
            width = len(nodes)
            for i, node in enumerate(nodes):
                node.x = -1 + 2 * (i + 1) / (width + 1)
                node.y = y

        # Draw edges
        for node in self.all_nodes:
            if node.left:
                plt.plot([node.x, node.left.x], [node.y, node.left.y], "b-", linewidth=1)
            if node.right:
                plt.plot([node.x, node.right.x], [node.y, node.right.y], "b-", linewidth=1)

        # Draw zigzag connections
        for node in self.middle_layer_1:
            for zigzag_node in node.zigzag_connections:
                plt.plot(
                    [node.x, zigzag_node.x], [node.y, zigzag_node.y], "r--", linewidth=1, alpha=0.6
                )

        # Draw nodes and labels
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
