class Edge:
    def __init__(self, edge):
        self.start, self.end = edge
        self.__validate()
        self.type = self.__edge_type()
        self.hamming_distance = self.__hamming_distance()
        # Create CNOT tuples from the differing positions
        self.differing_positions = self.__get_differing_positions()
        self.cnots = [
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
            return "parallel"
        else:
            return "diagonal"

    def __get_differing_positions(self):
        # List to store positions where bits differ
        differing_positions = []

        # Identify positions where bits differ
        for idx, (bit_i, bit_j) in enumerate(zip(self.start, self.end)):
            if bit_i != bit_j:
                differing_positions.append(idx)

        return differing_positions

    def to_possible_edges(self):
        """Convert a diagonal edge into a set of parallel edges."""
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

        # Initialize lists for the left and right strings
        i_list = list(self.start)
        j_list = list(self.start)

        # Generate edges based on differing positions
        generate_edges(i_list, j_list, self.differing_positions)

        return possible_edges

    def get_parallel_candidates(self):
        """Exclude edges from possible edges that do not satisfy the condition and return parallel candidates."""

        parallel_candidates = []

        # Calculate known_bits from CNOTs
        known_bits = ""
        for e in self.cnots:
            k, l = e
            xor = int(self.end[k]) ^ int(self.end[l])
            known_bits += str(xor)

        # Filter possible edges and add valid parallel candidates
        for e in self.to_possible_edges():
            if e.hamming_distance == 1 and known_bits in e.end:
                parallel_candidates.append(e)

        return parallel_candidates
