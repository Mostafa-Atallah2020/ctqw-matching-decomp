class Edge:
    def __init__(self, edge):
        self.edge = edge
        self.start, self.end = edge
        self.__validate()
        self.type = self.__edge_type()
        self.hamming_distance = self.__hamming_distance()
        self.differing_positions = self.__get_differing_positions()
        self.connections = [
            (self.differing_positions[k], self.differing_positions[k + 1])
            for k in range(len(self.differing_positions) - 1)
        ]

    def __validate(self):
        if not isinstance((self.start, self.end), tuple) or len(self.start) != len(self.end):
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

    def get_parallel_candidates(self):
        """Generate possible edges by transforming one bit at a time."""
        possible_edges = []
        start_bits = list(self.start)

        for i in range(len(self.differing_positions)):
            # Create intermediate node by changing bits up to position i
            intermediate = start_bits.copy()
            for j in range(i + 1):
                pos = self.differing_positions[j]
                intermediate[pos] = self.end[pos]

            # Add edge from start to intermediate
            possible_edges.append(Edge((self.start, "".join(intermediate))))

            # Add edge from intermediate to end
            possible_edges.append(Edge(("".join(intermediate), self.end)))

        return list(set(possible_edges))

    def get_all_projections(self):
        """Generate all possible projections between start and end nodes."""
        projections = []
        n = len(self.differing_positions)

        # Add original edge
        projections.append(self)

        # Generate all possible intermediate nodes
        for i in range(1, n):
            # Get all combinations of i positions
            current = list(self.start)
            for j in range(i):
                pos = self.differing_positions[j]
                current[pos] = self.end[pos]
                new_node = "".join(current)

                # Add edges to and from intermediate node
                projections.append(Edge((self.start, new_node)))
                projections.append(Edge((new_node, self.end)))

        return list(set(projections))
