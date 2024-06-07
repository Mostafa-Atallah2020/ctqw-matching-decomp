import matplotlib.pyplot as plt
import networkx as nx


class GraphDrawer:
    def __init__(self, n, edges):
        """
        Initialize the GraphDrawer with 2^n nodes and given edges.

        :param n: The exponent to determine the number of nodes (2^n).
        :param edges: A set of tuples representing the edges of the graph.
        """
        self.n = n
        self.edges = edges
        self.nodes = set(range(2**n))
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(edges)
        self.positions = self._generate_positions()

    def _generate_positions(self):
        """
        Generate positions for 2^n nodes in a (2^n - 1) x 2 grid.

        :return: A dictionary with node positions.
        """
        positions = {}
        for i in range(2**self.n):
            row = i // 2
            col = i % 2
            positions[i] = (col, row)
        return positions

    def show(self):
        """
        Draw the graph with the generated positions.
        """
        nx.draw(
            self.graph,
            self.positions,
            with_labels=True,
            node_color="skyblue",
            node_size=700,
            edge_color="gray",
            font_size=15,
            font_weight="bold",
        )
        plt.show()
