import random
from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from sympy import symbols

from src import MCRX, Edge, Expression, GraphDrawer
from src.mcrx_simplifier import MCRXCascadeSimplifier
from src.misc import (
    binary_tuple_to_int_tuple,
    graph_matchings_greedy,
    graph_matchings_parallel,
    hamming_distance,
)


class Graph:
    def __init__(self, edges) -> None:
        self.edges = set()
        self._validate_edges(edges)

        if self.edges:
            first_edge = next(iter(self.edges))
            if isinstance(first_edge[0], str):
                self.n_qubits = len(first_edge[0])
            else:
                self.n_qubits = len(bin(max(max(self.edges)))) - 2
        else:
            raise ValueError("Empty edge set")

        self.nodes = set(range(2**self.n_qubits))
        self.rot_angle = np.pi / 2
        self.__int_edges = set([self._edge_to_int_tuple(e) for e in self.edges])
        self.set_hamming_1, self.set_hamming_greater_1 = self.__split_by_hamming_distance()

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.__int_edges)

    def _edge_to_int_tuple(self, edge):
        if isinstance(edge[0], str):
            return tuple(int(v, 2) for v in edge)
        return edge

    def _validate_edges(self, edges):
        if isinstance(edges, set):
            self.edges = edges
        elif isinstance(edges, list):
            self.edges = set(edges)
        else:
            raise ValueError("Edges must be a set or list.")

        for e in self.edges:
            if not (isinstance(e, tuple) and len(e) == 2):
                raise ValueError("Each edge should be a tuple of length 2")

    def __split_by_hamming_distance(self):
        set_hamming_1 = set()
        set_hamming_greater_1 = set()

        for edge in self.__int_edges:
            dist = bin(edge[0] ^ edge[1]).count("1")
            if dist == 1:
                set_hamming_1.add(edge)
            else:
                set_hamming_greater_1.add(edge)

        return set_hamming_1, set_hamming_greater_1

    def __repr__(self):
        return f"StaticGraph(edges={self.edges})"


class PowerOf2EdgeGraph(Graph):
    def __init__(self, edges):
        super().__init__(edges)
        self.subgraphs = self.__decompose_into_subgraphs()

    def __decompose_into_subgraphs(self):
        edges = list(self.graph.edges)
        subgraphs = []

        while edges:
            k = int(np.log2(len(edges)))
            subgraph_size = 2**k

            # Select a random edge as a starting point
            start_edge = random.choice(edges)
            subgraph_edges = [start_edge]
            edges.remove(start_edge)

            # Greedily add edges that don't share vertices with existing edges
            for _ in range(subgraph_size - 1):
                if not edges:
                    break
                for edge in edges:
                    if all(len(set(edge) & set(e)) == 0 for e in subgraph_edges):
                        subgraph_edges.append(edge)
                        edges.remove(edge)
                        break

            subgraphs.append(nx.Graph(subgraph_edges))

        return subgraphs


class StaticGraph:
    def __init__(self, edges) -> None:
        self.edges = set()
        self._validate_edges(edges)

        self.n_qubits = len(next(iter(self.edges))[0])
        self.nodes = set(range(2**self.n_qubits))
        self.rot_angle = np.pi / 2
        self.__int_edges = set([binary_tuple_to_int_tuple(t) for t in self.edges])
        self.set_hamming_1, self.set_hamming_greater_1 = self.__split_by_hamming_distance()

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    def __repr__(self):
        return "StaticGraph(%s)" % (self.edges)

    def __split_by_hamming_distance(self):
        """Split edges into sets based on Hamming distance."""
        set_hamming_1 = set()
        set_hamming_greater_1 = set()

        for edge in self.edges:
            dist = hamming_distance(edge[0], edge[1])
            if dist == 1:
                set_hamming_1.add(edge)
            else:
                set_hamming_greater_1.add(edge)

        return set_hamming_1, set_hamming_greater_1

    def _validate_edges(self, edges):
        if not isinstance(edges, set):
            raise ValueError("Edges must be a set.")

        for e in edges:
            if isinstance(e, Edge):
                self.edges.add(e.edge)
            elif isinstance(e, tuple):
                self.edges.add(e)
            else:
                raise ValueError("Edge type should be a tuple or Edge")

    def __add__(self, other):
        return StaticGraph(self.nodes | other.nodes, self.edges | other.edges)

    def get_adj_mat(self):
        vertex_to_index = {v: i for i, v in enumerate(self.nodes)}
        num_vertices = len(self.nodes)
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

        for edge in self.__int_edges:
            v1, v2 = edge
            if v1 in vertex_to_index and v2 in vertex_to_index:
                i, j = vertex_to_index[v1], vertex_to_index[v2]
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1  # For undirected graph

        return adj_matrix

    def get_statevector(self):
        vec = [0 for i in range(len(self.nodes))]
        for edge in self.__int_edges:
            a, b = edge
            if vec[a] == 0:
                vec[a] = symbols(f"alpha{a}")
            if vec[b] == 0:
                vec[b] = symbols(f"alpha{b}")

        return np.array(vec)

    def draw(self):
        GraphDrawer(self.n_qubits, self.__int_edges).show()


class DynamicGraph:
    def __init__(self, graph_sequence) -> None:
        """
        Initialize the DynamicGraph with a sequence of (graph, time) tuples.
        Ensures all graphs have the same number of qubits.

        Parameters:
        graph_sequence (list of tuples): Each tuple contains a graph object and a corresponding time.
        """
        self.graph_sequence = graph_sequence
        self.n_qubits = self.graph_sequence[0][0].n_qubits

        # Validate that all graphs have the same number of qubits
        for graph, _ in self.graph_sequence:
            if graph.n_qubits != self.n_qubits:
                raise ValueError("All graphs in the sequence must have the same number of qubits")

    def time_evo_op(self, t_steps=1):
        """
        Compute the time evolution operator for the sequence of graphs.

        Returns:
        Operator: The resulting time evolution operator.
        """
        time_evo_op = np.eye(2**self.n_qubits, dtype=complex)

        for _ in range(t_steps):
            for graph, time in self.graph_sequence:
                adj_matrix = graph.get_adj_mat()
                unitary = expm(-1j * adj_matrix * time)
                time_evo_op = np.dot(unitary, time_evo_op)

        time_evo_op = Operator(time_evo_op)
        return time_evo_op

    def draw(self):
        for i, (graph, delta_t) in enumerate(self.graph_sequence):
            print(f"{graph} | Time = {delta_t}")
            graph.draw()
            plt.show()


class IntersectingEdgesGraph(StaticGraph):
    def __init__(self, edges, matchings=""):
        """
        Initialize IntersectingEdgesGraph with specified matching algorithm.

        Args:
            edges: The edges for the graph
            matchings: Matching algorithm to use. Options:
                - 'greedy': Use graph_matchings_greedy function
                - 'parallel': Use graph_matchings_parallel function
        """
        super().__init__(edges)
        self.matchings = matchings
        self.subgraphs = self.__decompose_into_subgraphs()

    def __decompose_into_subgraphs(self):
        decomposed_subgraphs = []

        # Select the appropriate matching function
        if self.matchings == "greedy":
            subgraphs = graph_matchings_greedy(self.edges)
        elif self.matchings == "parallel":
            subgraphs = graph_matchings_parallel(self.edges)
        else:
            raise ValueError(
                f"Invalid matchings value: '{self.matchings}'. "
                f"Valid options are: 'greedy', 'parallel'"
            )

        for sg in subgraphs:
            g = MultiEdgeGraph(sg)
            decomposed_subgraphs.append(g)
        return decomposed_subgraphs


class MultiEdgeGraph(StaticGraph):
    def __new__(cls, edges):
        temp_instance = super().__new__(cls)
        StaticGraph.__init__(temp_instance, edges)

        if len(temp_instance.set_hamming_greater_1) == 0:
            return NonDiagonalEdgeGraph(edges)
        else:
            return DiagonalEdgeGraph(edges)


class ParallelEdgeGraph(StaticGraph):
    def __init__(self, edges):
        self._validate_parallel_edges(edges)
        super().__init__(edges)
        self.target = None
        self.vars = self.__get_vars()
        self.expr = Expression(self.__get_expr())

    def _validate_parallel_edges(self, edges):
        for edge in edges:
            i, j = edge
            if hamming_distance(i, j) != 1:
                raise ValueError("The Graph is not a Parallel Edge Graph.")

    def get_qc(self, simplified=False, angle=None):
        if self.expr.expr.simplify() == True:
            qc = QuantumCircuit(self.n_qubits)
            qc.rx(angle if angle is not None else self.rot_angle, self.target)
            return qc
        else:
            mcrx = MCRX(
                self.n_qubits,
                self.expr,
                self.target,
                angle if angle is not None else self.rot_angle,
            )
            if simplified:
                return mcrx.simplify().qc
            else:
                return mcrx.qc

    def __get_vars(self):
        sym_vars = []
        for i in range(self.n_qubits):
            sym_name = f"x_{i}"
            sym_vars.append(symbols(sym_name, latex=True))

        return sym_vars

    def __get_expr(self):
        expr = False
        for i, j in self.edges:
            subexpr = True
            for k in range(self.n_qubits):
                if (i[k] == j[k]) and (j[k] == "0"):
                    subexpr = subexpr & ~self.vars[k]
                elif (i[k] == j[k]) and (j[k] == "1"):
                    subexpr = subexpr & self.vars[k]
                else:
                    self.target = k
            expr = expr | subexpr

        return expr


class NonDiagonalEdgeGraph(StaticGraph):
    """
    A graph class for handling non-diagonal edges in quantum circuits.
    Non-diagonal edges represent transitions between quantum states that differ by
    a Hamming distance of 1.
    """

    def __init__(self, edges):
        super().__init__(edges)
        self._validate_edge_distances()
        self._validate_vertex_usage()
        self.edge_sets = self.__split_tuples_by_changing_bit()
        self.targets, self.exprs = self.__get_targets_exprs()

    def _validate_edge_distances(self):
        """Ensure all edges have Hamming distance of 1."""
        for edge in self.edges:
            dist = sum(b1 != b2 for b1, b2 in zip(edge[0], edge[1]))
            if dist != 1:
                raise ValueError(f"Edge {edge} has Hamming distance {dist}, expected 1")

    def _validate_vertex_usage(self):
        """Ensure each vertex appears in at most 2 edges (for implementable quantum circuits)."""
        vertex_count = defaultdict(int)
        for edge in self.edges:
            vertex_count[edge[0]] += 1
            vertex_count[edge[1]] += 1

        for vertex, count in vertex_count.items():
            if count > 2:
                raise ValueError(f"Vertex {vertex} appears in {count} edges, maximum allowed is 2")

    def get_qc(self, simplified=False, angle=None):
        """
        Generate a quantum circuit implementing the graph transformations.

        Args:
            simplified (bool): Whether to simplify the resulting circuit

        Returns:
            QuantumCircuit: The constructed quantum circuit
        """
        qc_dict = {}

        # Build subcircuits for each edge set
        for idx, edges in self.edge_sets.items():
            G = ParallelEdgeGraph(edges)
            qc = G.get_qc(
                simplified=simplified, angle=angle if angle is not None else self.rot_angle
            )
            qc_dict[idx] = qc

        # Combine subcircuits in order
        circ = QuantumCircuit(self.n_qubits)
        for idx in sorted(qc_dict.keys()):
            qc = qc_dict[idx]
            circ = circ.compose(qc, range(self.n_qubits))

        return circ

    def __split_tuples_by_changing_bit(self):
        """
        Group edges by which qubit position changes.
        Returns:
            dict: Maps bit position to set of edges that change that bit
        """
        subsets = {}

        for t in self.edges:
            # Find position where bits differ
            for i, (b1, b2) in enumerate(zip(t[0], t[1])):
                if b1 != b2:
                    if i not in subsets:
                        subsets[i] = set()
                    subsets[i].add(t)
                    break  # Only one bit changes per edge

        return subsets

    def __get_targets_exprs(self):
        """
        Extract target qubits and expressions for MCRX gates.

        Returns:
            tuple: (list of target qubits, list of control expressions)
        """
        targets = []
        exprs = []

        for idx, edges in self.edge_sets.items():
            G = ParallelEdgeGraph(edges)
            target = G.target
            expr = G.expr

            if target not in targets:
                targets.append(target)
            if expr not in exprs:
                exprs.append(expr)

        return targets, exprs


class DiagonalEdgeGraph(StaticGraph):
    def __init__(self, edges):
        super().__init__(edges)
        self.candidates = self._get_candidates()
        self.best_candidate = self._get_best_candidate()
        self.connections = self._get_filtered_connections()

    def _get_filtered_connections(self):
        """Get connections for diagonal edges"""
        raw_connections = []

        # Get connections from each diagonal edge
        for edge in self.set_hamming_greater_1:
            edge_obj = Edge(edge)
            # Get differing positions
            diff_positions = []
            for i, (b1, b2) in enumerate(zip(edge_obj.start, edge_obj.end)):
                if b1 != b2:
                    diff_positions.append(i)

            if len(diff_positions) > 1:
                # Create connections from first differing position to all others
                first_pos = diff_positions[0]
                for pos in diff_positions[1:]:
                    raw_connections.append((first_pos, pos))

        return sorted(list(set(raw_connections)))

    def _get_candidates(self):
        """
        Get valid candidates for transformation.
        Each candidate will have exactly 2 edges:
        - The common Hamming-1 edge(s)
        - One individual projection edge from the diagonal edge decomposition
        """
        # print("Starting _get_candidates")
        if not self.set_hamming_greater_1:
            print("No diagonal edges found")
            return []

        valid_candidates = []
        hamming1_edges = self.set_hamming_1.copy()  # Common edges
        # print(f"Hamming-1 edges: {hamming1_edges}")
        # print(f"Diagonal edges: {self.set_hamming_greater_1}")

        # For each diagonal edge, collect all individual projection edges
        for diagonal_edge in self.set_hamming_greater_1:
            # print(f"\nProcessing diagonal edge: {diagonal_edge}")
            start, end = diagonal_edge

            # Find positions where bits differ
            diff_positions = []
            for i, (b1, b2) in enumerate(zip(start, end)):
                if b1 != b2:
                    diff_positions.append(i)
            # print(f"Differing positions: {diff_positions}")

            # For each differing position, create individual projection edges
            for target_qubit in diff_positions:
                # print(f"\nTrying target qubit {target_qubit}")

                # Create projection for start node
                start_proj = list(start)
                start_proj[target_qubit] = end[target_qubit]
                start_intermediate = "".join(start_proj)
                projection_edge_1 = (start, start_intermediate)

                # Create projection for end node
                end_proj = list(end)
                end_proj[target_qubit] = start[target_qubit]
                end_intermediate = "".join(end_proj)
                projection_edge_2 = (end_intermediate, end)

                # Create candidates: common edges + individual projection edge
                for projection_edge in [projection_edge_1, projection_edge_2]:
                    try:
                        # Each candidate: common Hamming-1 edges + one projection edge
                        combined_edges = hamming1_edges | {projection_edge}
                        # print(f"Candidate edges: {combined_edges}")

                        # Validate that all edges have Hamming distance 1
                        valid_edges = True
                        for edge in combined_edges:
                            dist = sum(1 for a, b in zip(edge[0], edge[1]) if a != b)
                            # print(f"Edge {edge}: Hamming distance = {dist}")
                            if dist != 1:
                                valid_edges = False
                                break

                        if valid_edges:
                            candidate_graph = NonDiagonalEdgeGraph(combined_edges)
                            valid_candidates.append(candidate_graph)
                            # print("Successfully created candidate")
                        else:
                            # print("Skipped candidate due to invalid Hamming distances")
                            pass

                    except ValueError as e:
                        print(f"Failed to create candidate: {str(e)}")
                        continue

        # print(f"\nFinal number of valid candidates: {len(valid_candidates)}")
        return valid_candidates

    def _get_best_candidate(self):
        """Select the best candidate based on the minimum number of variables."""
        if not self.candidates:
            return None

        min_vars = float("inf")
        best = None

        for candidate in self.candidates:
            try:
                variables = set()
                for expr in candidate.exprs:
                    variables.update(expr.simplify().vars)

                var_count = len(variables)
                if var_count < min_vars:
                    min_vars = var_count
                    best = candidate

            except Exception:
                continue

        return best

    def get_qc(self, simplified=False, angle=None):
        """Generate quantum circuit for the graph."""
        if not self.best_candidate:
            return QuantumCircuit(self.n_qubits)

        # Get the base circuit from the best candidate
        unsimplified_qc = self.best_candidate.get_qc(angle=angle if angle is not None else self.rot_angle)
        n_qubits = self.best_candidate.n_qubits

        # Build the circuit
        circ = QuantumCircuit(n_qubits)
        
        # Analyze what the rotation prepares vs what we want
        gate_configs = self._get_general_transformation_configuration()
        
        # Add forward gates (pre-rotation preparation if needed)
        for gate_config in gate_configs['forward']:
            if gate_config['type'] == 'X':
                circ.x(gate_config['qubit'])
            elif gate_config['type'] == 'CNOT':
                circ.cx(gate_config['control'], gate_config['target'])

        # Add the rotation circuit 
        if simplified:
            try:
                simplifier = MCRXCascadeSimplifier(verbose=False)
                simplified_qc, _ = simplifier.simplify(unsimplified_qc)
                circ.append(simplified_qc, range(n_qubits))
            except:
                circ.append(unsimplified_qc, range(n_qubits))
        else:
            circ.append(unsimplified_qc, range(n_qubits))

        # Add reverse gates (post-rotation transformation)
        for gate_config in gate_configs['reverse']:
            if gate_config['type'] == 'X':
                circ.x(gate_config['qubit'])
            elif gate_config['type'] == 'CNOT':
                circ.cx(gate_config['control'], gate_config['target'])

        return circ.decompose()

    def _get_general_transformation_configuration(self):
        """
        General method to determine transformation based on boolean edge analysis.
        
        Uses self.best_candidate.edges to understand what rotation prepares,
        and self.set_hamming_greater_1 to understand what we want.
        
        Returns:
            dict: Configuration with 'forward' and 'reverse' gate lists
        """
        
        if not self.best_candidate or not hasattr(self.best_candidate, 'edges'):
            return {'forward': [], 'reverse': []}
        
        if not self.set_hamming_greater_1:
            return {'forward': [], 'reverse': []}
        
        # Get what the rotation circuit prepares
        rotation_edges = self.best_candidate.edges
        rotation_states = self._extract_states_from_edges(rotation_edges)
        
        # Get what we want (the diagonal edge)
        target_edges = self.set_hamming_greater_1
        target_states = self._extract_states_from_edges(target_edges)
        
        # Compute the boolean transformation needed
        transformation = self._compute_boolean_transformation(rotation_states, target_states)
        
        return transformation

    def _extract_states_from_edges(self, edges):
        """
        Extract the computational basis states from a set of edges.
        
        Args:
            edges (set): Set of edges like {('000', '010'), ('001', '011')}
            
        Returns:
            set: Set of unique computational basis states
        """
        
        states = set()
        for edge in edges:
            states.add(edge[0])  # Start state
            states.add(edge[1])  # End state
        
        return states

    def _compute_boolean_transformation(self, rotation_states, target_states):
        """
        Compute the boolean transformation needed to map rotation states to target states.
        
        Args:
            rotation_states (set): States that rotation circuit prepares
            target_states (set): States we want to achieve
            
        Returns:
            dict: Gate configuration for the transformation
        """
        
        # Convert to sorted lists for consistent ordering
        rotation_list = sorted(list(rotation_states))
        target_list = sorted(list(target_states))
        
        # Must have same number of states
        if len(rotation_list) != len(target_list):
            print(f"Warning: Different number of states - rotation: {len(rotation_list)}, target: {len(target_list)}")
            return {'forward': [], 'reverse': []}
        
        # For 2-state case, compute direct mapping
        if len(rotation_list) == 2:
            return self._compute_two_state_transformation(rotation_list, target_list)
        
        # For more complex cases, use general boolean logic
        return self._compute_general_boolean_transformation(rotation_list, target_list)

    def _compute_two_state_transformation(self, rotation_states, target_states):
        """
        Compute transformation for two-state case using boolean logic.
        
        Args:
            rotation_states (list): [state1, state2] from rotation
            target_states (list): [state1, state2] that we want
            
        Returns:
            dict: Gate configuration
        """
        
        state1_rot, state2_rot = rotation_states[0], rotation_states[1]
        state1_target, state2_target = target_states[0], target_states[1]
        
        # print(f"Mapping: {state1_rot}→{state1_target}, {state2_rot}→{state2_target}")
        
        # Compute the boolean operations needed for each mapping
        operations = []
        
        # Analyze the mapping using XOR logic
        mapping1 = self._compute_state_mapping(state1_rot, state1_target)
        mapping2 = self._compute_state_mapping(state2_rot, state2_target)
        
        # Find operations that work for both mappings
        operations = self._find_common_operations(mapping1, mapping2, state1_rot, state2_rot)
        
        return {
            'forward': [],
            'reverse': operations
        }

    def _compute_state_mapping(self, start_state, end_state):
        """
        Compute what boolean operations are needed to map start_state to end_state.
        
        Args:
            start_state (str): Starting binary string like '010'
            end_state (str): Target binary string like '111'
            
        Returns:
            dict: Information about the required transformation
        """
        
        if len(start_state) != len(end_state):
            return {'valid': False}
        
        # Find which qubits need to be flipped
        flips_needed = []
        for i, (bit1, bit2) in enumerate(zip(start_state, end_state)):
            if bit1 != bit2:
                flips_needed.append(i)
        
        return {
            'valid': True,
            'start': start_state,
            'end': end_state,
            'flips': flips_needed,
            'xor_pattern': ''.join('1' if b1 != b2 else '0' for b1, b2 in zip(start_state, end_state))
        }

    def _find_common_operations(self, mapping1, mapping2, state1, state2):
        """
        Find gate operations that correctly transform both mappings.
        
        Args:
            mapping1 (dict): Mapping info for first state pair
            mapping2 (dict): Mapping info for second state pair
            state1 (str): First rotation state
            state2 (str): Second rotation state
            
        Returns:
            list: List of gate operations
        """
        
        if not mapping1['valid'] or not mapping2['valid']:
            return []
        
        operations = []
        n_qubits = len(state1)
        
        # Method 1: Try simple X gates on qubits that need flipping
        simple_solution = self._try_simple_x_gates(mapping1, mapping2, state1, state2)
        if simple_solution:
            return simple_solution
        
        # Method 2: Try controlled operations
        controlled_solution = self._try_controlled_operations(mapping1, mapping2, state1, state2)
        if controlled_solution:
            return controlled_solution
        
        # Method 3: General approach using multiple CNOTs and X gates
        return self._generate_general_solution(mapping1, mapping2, state1, state2)

    def _try_simple_x_gates(self, mapping1, mapping2, state1, state2):
        """
        Try to solve with simple X gates on individual qubits.
        
        Returns:
            list: Gate operations if successful, empty list otherwise
        """
        
        # Check if same qubits need flipping in both mappings
        flips1 = set(mapping1['flips'])
        flips2 = set(mapping2['flips'])
        
        if flips1 == flips2:
            # Simple case: same qubits need flipping in both mappings
            return [{'type': 'X', 'qubit': qubit} for qubit in sorted(flips1)]
        
        return []

    def _try_controlled_operations(self, mapping1, mapping2, state1, state2):
        """
        Try to solve with controlled operations (CNOTs + X gates).
        
        Returns:
            list: Gate operations if successful, empty list otherwise
        """
        
        operations = []
        n_qubits = len(state1)
        
        # For each qubit position, determine if it needs conditional flipping
        for qubit in range(n_qubits):
            # Check if this qubit behaves differently in the two mappings
            flip_in_mapping1 = qubit in mapping1['flips']
            flip_in_mapping2 = qubit in mapping2['flips']
            
            if flip_in_mapping1 != flip_in_mapping2:
                # This qubit needs conditional flipping
                # Find a control qubit that distinguishes the two states
                control_qubit = self._find_control_qubit(state1, state2, qubit)
                if control_qubit is not None:
                    if flip_in_mapping1 and state1[control_qubit] == '1':
                        # Flip when control is 1
                        operations.append({'type': 'CNOT', 'control': control_qubit, 'target': qubit})
                    elif flip_in_mapping1 and state1[control_qubit] == '0':
                        # Flip when control is 0 (use X-CNOT-X pattern)
                        operations.extend([
                            {'type': 'X', 'qubit': control_qubit},
                            {'type': 'CNOT', 'control': control_qubit, 'target': qubit},
                            {'type': 'X', 'qubit': control_qubit}
                        ])
            elif flip_in_mapping1 and flip_in_mapping2:
                # Both mappings need this qubit flipped - simple X gate
                operations.append({'type': 'X', 'qubit': qubit})
        
        # Verify this solution works
        if self._verify_operations(operations, [(state1, mapping1['end']), (state2, mapping2['end'])]):
            return operations
        
        return []

    def _find_control_qubit(self, state1, state2, target_qubit):
        """
        Find a qubit that can be used as control to distinguish between state1 and state2.
        
        Returns:
            int: Control qubit index, or None if not found
        """
        
        for i in range(len(state1)):
            if i != target_qubit and state1[i] != state2[i]:
                return i
        
        return None

    def _generate_general_solution(self, mapping1, mapping2, state1, state2):
        """
        Generate a general solution using systematic approach.
        
        Returns:
            list: Gate operations
        """
        
        operations = []
        n_qubits = len(state1)
        
        # Strategy: Use the first differing qubit as a "selector"
        selector_qubit = None
        for i in range(n_qubits):
            if state1[i] != state2[i]:
                selector_qubit = i
                break
        
        if selector_qubit is None:
            # States are identical - shouldn't happen for diagonal edges
            return []
        
        # For each other qubit, determine the conditional operation needed
        for target_qubit in range(n_qubits):
            if target_qubit == selector_qubit:
                continue
                
            # Determine what happens to this qubit in each mapping
            flip1 = target_qubit in mapping1['flips']
            flip2 = target_qubit in mapping2['flips']
            
            if flip1 != flip2:
                # Need conditional flip based on selector qubit
                if (flip1 and state1[selector_qubit] == '1') or (flip2 and state2[selector_qubit] == '1'):
                    operations.append({'type': 'CNOT', 'control': selector_qubit, 'target': target_qubit})
            elif flip1 and flip2:
                # Always flip
                operations.append({'type': 'X', 'qubit': target_qubit})
        
        # Handle the selector qubit itself
        if selector_qubit in mapping1['flips']:
            operations.append({'type': 'X', 'qubit': selector_qubit})
        
        return operations

    def _verify_operations(self, operations, state_mappings):
        """
        Verify that the operations correctly implement the state mappings.
        
        Args:
            operations (list): List of gate operations
            state_mappings (list): List of (start_state, end_state) tuples
            
        Returns:
            bool: True if operations work correctly
        """
        
        for start_state, expected_end_state in state_mappings:
            current_state = start_state
            
            # Apply each operation
            for op in operations:
                if op['type'] == 'X':
                    qubit = op['qubit']
                    state_list = list(current_state)
                    state_list[qubit] = '1' if state_list[qubit] == '0' else '0'
                    current_state = ''.join(state_list)
                    
                elif op['type'] == 'CNOT':
                    control, target = op['control'], op['target']
                    state_list = list(current_state)
                    if state_list[control] == '1':
                        state_list[target] = '1' if state_list[target] == '0' else '0'
                    current_state = ''.join(state_list)
            
            if current_state != expected_end_state:
                return False
        
        return True

    def _compute_general_boolean_transformation(self, rotation_states, target_states):
        """
        Compute transformation for cases with more than 2 states.
        
        Returns:
            dict: Gate configuration
        """
        
        # For now, implement a simple approach
        # This can be extended for more complex multi-state cases
        
        operations = []
        
        # Analyze all state pairs and find common patterns
        for i, (rot_state, target_state) in enumerate(zip(rotation_states, target_states)):
            mapping = self._compute_state_mapping(rot_state, target_state)
            
            # For each required flip, add appropriate operation
            for qubit in mapping['flips']:
                # Simple approach: just add X gate (this may need refinement)
                if {'type': 'X', 'qubit': qubit} not in operations:
                    operations.append({'type': 'X', 'qubit': qubit})
        
        return {
            'forward': [],
            'reverse': operations
        }

    def _debug_transformation_analysis(self):
        """
        Debug method to understand the transformation needed.
        """
        
        print("=== Transformation Analysis Debug ===")
        
        if not self.best_candidate or not hasattr(self.best_candidate, 'edges'):
            print("No best_candidate or edges found")
            return
        
        rotation_edges = self.best_candidate.edges
        rotation_states = self._extract_states_from_edges(rotation_edges)
        
        target_edges = self.set_hamming_greater_1
        target_states = self._extract_states_from_edges(target_edges)
        
        print(f"Rotation circuit prepares states: {sorted(rotation_states)}")
        print(f"Target diagonal edge states: {sorted(target_states)}")
        
        transformation = self._compute_boolean_transformation(rotation_states, target_states)
        print(f"Required transformation: {transformation}")
        
        print("=" * 50)

    # Usage example:
    # graph._debug_transformation_analysis()

    def __del__(self):
        """Clean up any remaining resources"""
        attrs = ["candidates", "best_candidate", "connections"]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
