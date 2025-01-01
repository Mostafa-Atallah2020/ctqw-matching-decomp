from itertools import combinations

class Edge:
    """
    Represents an edge in a binary hypercube graph.
    Each edge connects two nodes represented as binary strings.
    """
    def __init__(self, edge):
        """
        Initialize an edge with two binary string nodes.
        
        Args:
            edge (tuple): A tuple of two binary strings (start_node, end_node)
        """
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
        """Validate that the edge consists of valid binary strings of equal length."""
        if not isinstance(self.edge, tuple) or len(self.start) != len(self.end):
            raise ValueError("Each edge must be a tuple of two binary strings of equal length.")
        if not all(bit in "01" for bit in self.start + self.end):
            raise ValueError("Each node must be a binary string of 0s and 1s.")

    def __hamming_distance(self):
        """Calculate the Hamming distance between the start and end nodes."""
        return sum(c1 != c2 for c1, c2 in zip(self.start, self.end))

    def __edge_type(self):
        """Determine if the edge is Axial (Hamming distance 1) or Diagonal."""
        return "Axial" if self.__hamming_distance() == 1 else "Diagonal"

    def __get_differing_positions(self):
        """Find positions where bits differ between start and end nodes."""
        return [
            idx for idx, (bit_i, bit_j) in enumerate(zip(self.start, self.end)) 
            if bit_i != bit_j
        ]

    def __hash__(self):
        """Make Edge objects hashable for set operations."""
        return hash((self.start, self.end))

    def __eq__(self, other):
        """Define equality for Edge objects."""
        if not isinstance(other, Edge):
            return False
        return (self.start == other.start and self.end == other.end) or \
               (self.start == other.end and self.end == other.start)

    def get_parallel_candidates(self):
        """
        Generate possible parallel paths by transforming bits sequentially.
        Returns a list of Edge objects representing possible parallel paths.
        """
        candidates = []
        n = len(self.differing_positions)
        
        # Generate all possible orderings of bit flips
        for r in range(1, n):
            # Create intermediate node by changing bits up to position r
            current = list(self.start)
            for i in range(r):
                pos = self.differing_positions[i]
                current[pos] = self.end[pos]
            intermediate = "".join(current)
            
            # Add edges to and from intermediate node
            candidates.append(Edge((self.start, intermediate)))
            candidates.append(Edge((intermediate, self.end)))
        
        return list(set(candidates))

    def get_all_projections(self):
        """
        Generate all possible projections between start and end nodes.
        Returns a list of Edge objects representing all possible paths.
        """
        projections = set()
        n = len(self.differing_positions)
        
        # Add original edge
        projections.add(self)
        
        # Generate all possible intermediate nodes
        for r in range(1, n):  # r is the number of bits to flip
            for pos_combo in combinations(self.differing_positions, r):
                # Create intermediate node
                current = list(self.start)
                for pos in pos_combo:
                    current[pos] = self.end[pos]
                intermediate = "".join(current)
                
                # Add edges to and from intermediate node
                projections.add(Edge((self.start, intermediate)))
                projections.add(Edge((intermediate, self.end)))
        
        return list(projections)

    def get_path_distance(self):
        """Calculate the minimum number of steps needed to traverse this edge."""
        return self.hamming_distance

    def __str__(self):
        """String representation of the edge."""
        return f"Edge({self.start} -> {self.end})"

    def __repr__(self):
        """Detailed representation of the edge."""
        return f"Edge(start='{self.start}', end='{self.end}', type='{self.type}', " \
               f"hamming_distance={self.hamming_distance})"