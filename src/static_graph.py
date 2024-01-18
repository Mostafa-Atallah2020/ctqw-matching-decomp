import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class StaticGraph:
    def __init__(self, nodes, edges) -> None:
        self.nodes = nodes
        self.edges = edges
        #self.adj_mat = self.__get_adj_mat()
        # Create a graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    def __add__(self, other):
        return StaticGraph(self.nodes | other.nodes, self.edges | other.edges)

    def get_adj_mat(self):
        n = len(self.nodes)
        adj_mat = np.zeros((n, n))
        for i in self.nodes:
            for j in self.nodes:
                if (i,j) in self.edges:
                    adj_mat[i,j] = 1
                    adj_mat[j,i] = 1

        return adj_mat

    def draw(self):
        # Draw the graph
        pos = nx.spring_layout(self.graph)  # Define the layout for the nodes
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=700,
            node_color="skyblue",
            font_size=10,
            font_color="black",
            font_weight="bold",
            edge_color="gray",
            linewidths=1,
            alpha=0.7,
        )

        # Show the plot
        plt.show()
