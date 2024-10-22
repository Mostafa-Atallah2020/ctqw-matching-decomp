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
            raise ValueError(
                "Each node in an edge must be a binary string consisting of 0s and 1s."
            )

    def __hamming_distance(self):
        """Calculate the Hamming distance between the two nodes of the edge."""
        return sum(c1 != c2 for c1, c2 in zip(self.start, self.end))

    def __edge_type(self):
        """Determine if the edge is parallel or diagonal."""
        if self.__hamming_distance() == 1:
            return "Axial"
        else:
            return "Diagonal"

    def __get_differing_positions(self):
        # List to store positions where bits differ
        differing_positions = []

        # Identify positions where bits differ
        for idx, (bit_i, bit_j) in enumerate(zip(self.start, self.end)):
            if bit_i != bit_j:
                differing_positions.append(idx)

        return differing_positions

    def to_possible_edges(self):
        """generate a possible set of edge candidates."""
        # Find possible parallel edges
        possible_edges = []

        def generate_edges(i, j, positions):
            if not positions:
                e = Edge(("".join(i), "".join(j)))
                possible_edges.append(e)
                return

            pos = positions[0]
            new_i = i[:]
            new_j = j[:]
            new_j[pos] = "0"
            generate_edges(new_i, new_j, positions[1:])
            new_j[pos] = "1"
            generate_edges(new_i, new_j, positions[1:])

        # Generate edges based on differing positions
        generate_edges(list(self.start), list(self.start), self.differing_positions)

        return possible_edges

    def get_all_projections(self):
        """Generate all possible projections of an edge onto a hypercube."""
        projections = []
        n = len(self.differing_positions)

        # Add the original edge
        projections.append(self)

        # Generate all intermediate nodes
        for i in range(1, n):
            for combo in self.__combinations(self.differing_positions, i):
                new_node = list(self.start)
                for pos in combo:
                    new_node[pos] = self.end[pos]
                new_node_str = "".join(new_node)
                projections.append(Edge((self.start, new_node_str)))
                projections.append(Edge((new_node_str, self.end)))

        # Add the edge from start to start
        projections.append(Edge((self.start, self.start)))

        return projections

    def __combinations(self, iterable, r):
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = list(range(r))
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1
            yield tuple(pool[i] for i in indices)
