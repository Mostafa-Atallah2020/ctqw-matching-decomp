import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Set


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
        """Add a zigzag connection to another node"""
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
        self.bits_per_part = height
        self.total_bits = 2 * height
        self.entrance = None
        self.exit = None
        self.all_nodes = []
        self.middle_layer_1 = []
        self.middle_layer_2 = []
        self._create_tree()

    def _get_middle_layer_labels(self, layer_num: int) -> List[str]:
        """Generate labels for middle layer nodes.
        layer_num: 1 for first middle layer, 2 for second middle layer"""
        num_nodes = 2**self.height
        labels = []
        for i in range(num_nodes):
            if layer_num == 1:
                # First middle layer: combine unique top bits with sequential bottom bits
                top_part = (i + 1) % (2**self.height)  # Ensure non-zero top part
                bottom_part = i % (2**self.height)
            else:
                # Second middle layer: use different pattern
                top_part = (i + 2) % (2**self.height)  # Offset to avoid duplicates
                bottom_part = (i + 1) % (2**self.height)
            labels.append(self._create_label(top_part, bottom_part))
        return labels

    def _create_label(self, top_part: int, bottom_part: int) -> str:
        """Create a binary label with correct number of bits for each part"""
        top_bits = format(top_part, f"0{self.bits_per_part}b")
        bottom_bits = format(bottom_part, f"0{self.bits_per_part}b")
        return top_bits + bottom_bits

    def _create_upper_tree(self):
        """Create upper tree including first middle layer"""
        # Create entrance (all zeros)
        self.entrance = Node("0" * self.total_bits, 0)
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
                # Create left and right children
                if level == self.height - 1:  # Middle layer 1
                    middle_labels = self._get_middle_layer_labels(1)
                    left_idx = 2 * j
                    right_idx = 2 * j + 1
                    left_label = middle_labels[left_idx]
                    right_label = middle_labels[right_idx]
                else:
                    left_label = self._create_label(2 * j + 1, 0)
                    right_label = self._create_label(2 * j + 2, 0)

                left_child = Node(left_label, level + 1)
                right_child = Node(right_label, level + 1)

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
        # Create middle layer first
        nodes_in_level = 2**self.height
        x_spacing = 1.0 / (nodes_in_level + 1)
        y_position = 0.4  # Lower than first middle layer

        # Create middle layer nodes with distinct labels
        middle_labels = self._get_middle_layer_labels(2)
        for i, label in enumerate(middle_labels):
            node = Node(label, self.height + 1)
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
                value = self._create_label(0, j // 2 + 1)
                child = Node(value, self.height + level + 2)
                child.set_position(-0.5 + (j // 2 + 1) * x_spacing, y_position)

                # Connect nodes
                current_level[j].left = child
                if j + 1 < len(current_level):
                    current_level[j + 1].right = child

                next_level.append(child)
                self.all_nodes.append(child)

            current_level = next_level

        # Create exit node
        max_value = 2**self.bits_per_part - 1
        self.exit = Node(self._create_label(0, max_value), 2 * self.height)
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
        """Create complete tree structure"""
        self._create_upper_tree()
        self._create_lower_tree()
        self._create_zigzag_connections()

    def get_edges(self) -> Set[Tuple[str, str]]:
        """Get all edges in the required format including zigzag connections"""
        edges = set()

        # Collect all regular tree edges
        for node in self.all_nodes:
            if node.left:
                edges.add((node.value, node.left.value))
            if node.right:
                edges.add((node.value, node.right.value))

        # Collect all zigzag connections from both middle layers
        for node in self.middle_layer_1:
            if hasattr(node, "zigzag_connections"):
                for zigzag_node in node.zigzag_connections:
                    edges.add((node.value, zigzag_node.value))
                    # Add reverse connection as well
                    edges.add((zigzag_node.value, node.value))

        # Print all edges for verification
        # print("\nTree edges:")
        # for edge in sorted(edges):
        #     print(f"{edge[0]} -> {edge[1]}")

        return edges

    def draw(self, figsize=(12, 15)):
        plt.figure(figsize=figsize)

        # Draw regular edges
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

        # Draw nodes
        for node in self.all_nodes:
            plt.plot(node.x, node.y, "ko", markersize=10)
            plt.text(
                node.x + 0.02,
                node.y,
                node.value,
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=8,
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
                    fontsize=10,
                )

        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        return plt.gcf()
