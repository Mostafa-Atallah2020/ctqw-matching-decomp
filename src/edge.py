from itertools import permutations


class Edge:
    """
    Represents an edge in a binary hypercube graph.
    Each edge connects two nodes represented as binary strings.
    """

    def __init__(self, edge):
        self.edge = edge
        self.start, self.end = edge
        self.__validate()
        self.type = self.__edge_type()
        self.hamming_distance = self.__hamming_distance()
        self.differing_positions = self.__get_differing_positions()

    def __validate(self):
        if not isinstance(self.edge, tuple) or len(self.start) != len(self.end):
            raise ValueError("Each edge must be a tuple of two binary strings of equal length.")
        if not all(bit in "01" for bit in self.start + self.end):
            raise ValueError("Each node must be a binary string of 0s and 1s.")

    def __hamming_distance(self):
        return sum(c1 != c2 for c1, c2 in zip(self.start, self.end))

    def __edge_type(self):
        return "Axial" if self.__hamming_distance() == 1 else "Diagonal"

    def __get_differing_positions(self):
        return [
            idx for idx, (bit_i, bit_j) in enumerate(zip(self.start, self.end)) if bit_i != bit_j
        ]

    def __hash__(self):
        edge_pair = tuple(sorted([self.start, self.end]))
        return hash(edge_pair)

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.start == other.start and self.end == other.end) or (
            self.start == other.end and self.end == other.start
        )

    def get_all_projections(self):
        """
        Generate projections that share either start or end point with the original diagonal edge.
        Returns a list of Edge objects.
        """
        if self.type == "Axial":
            return [self]

        projections = []
        positions = self.differing_positions

        # For paths from start point
        for i in range(len(positions)):
            # Create intermediate node by changing one bit
            current = list(self.start)
            current[positions[i]] = self.end[positions[i]]
            intermediate = "".join(current)
            projections.append(Edge((self.start, intermediate)))

        # For paths to end point
        for i in range(len(positions)):
            # Create intermediate node by changing all bits except one
            current = list(self.end)
            current[positions[i]] = self.start[positions[i]]
            intermediate = "".join(current)
            projections.append(Edge((intermediate, self.end)))

        return projections

    def __str__(self):
        return f"Edge({self.start} -> {self.end})"

    def __repr__(self):
        return (
            f"Edge(start='{self.start}', end='{self.end}', type='{self.type}', "
            f"hamming_distance={self.hamming_distance})"
        )
