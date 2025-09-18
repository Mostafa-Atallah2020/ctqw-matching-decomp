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
        """Generate quantum circuit ensuring each candidate contributes its target state."""
        
        target_edges = list(self.set_hamming_greater_1)
        
        if len(target_edges) == 1:
            # Single edge case - use exact working implementation
            return self._get_single_edge_circuit(simplified, angle)
        else:
            # Multi-edge case - ensure each candidate contributes target state properly
            return self._get_single_state_focus_circuit(simplified, angle)

    def _get_single_edge_circuit(self, simplified=False, angle=None):
        """Single edge case - exact working implementation."""
        
        if not self.best_candidate:
            return QuantumCircuit(self.n_qubits)

        unsimplified_qc = self.best_candidate.get_qc(angle=angle if angle is not None else self.rot_angle)
        n_qubits = self.best_candidate.n_qubits

        circ = QuantumCircuit(n_qubits)
        
        gate_configs = self._get_general_transformation_configuration()
        
        # Forward gates
        for gate_config in gate_configs['forward']:
            if gate_config['type'] == 'X':
                circ.x(gate_config['qubit'])
            elif gate_config['type'] == 'CNOT':
                circ.cx(gate_config['control'], gate_config['target'])

        # Rotation circuit
        if simplified:
            try:
                simplifier = MCRXCascadeSimplifier(verbose=False)
                simplified_qc, _ = simplifier.simplify(unsimplified_qc)
                circ.append(simplified_qc, range(n_qubits))
            except:
                circ.append(unsimplified_qc, range(n_qubits))
        else:
            circ.append(unsimplified_qc, range(n_qubits))

        # Reverse gates
        for gate_config in gate_configs['reverse']:
            if gate_config['type'] == 'X':
                circ.x(gate_config['qubit'])
            elif gate_config['type'] == 'CNOT':
                circ.cx(gate_config['control'], gate_config['target'])

        return circ.decompose()

    def _get_single_state_focus_circuit(self, simplified=False, angle=None):
        """Focus each candidate on producing exactly its target state."""
        
        target_edges = list(self.set_hamming_greater_1)
        all_target_states = self._extract_states_from_edges(self.set_hamming_greater_1)
        
        # print(f"Target edges: {target_edges}")
        # print(f"All target states to cover: {sorted(all_target_states)}")
        
        # Create focused assignments - each candidate produces one specific target state
        focused_assignments = self._create_focused_assignments(all_target_states)
        
        if not focused_assignments:
            return QuantumCircuit(self.n_qubits)
        
        n_qubits = len(list(all_target_states)[0])
        circ = QuantumCircuit(n_qubits)
        
        # print(f"Using {len(focused_assignments)} focused assignments")
        
        # Apply each focused assignment
        for i, assignment in enumerate(focused_assignments):
            candidate = assignment['candidate']
            target_state = assignment['target_state']
            transformation = assignment['transformation']
            
            # print(f"Assignment {i+1}: {candidate.edges} → focus on state {target_state}")
            # print(f"  Transformation: {transformation}")
            
            # Apply pre-rotation transformation
            for gate_config in transformation['forward']:
                if gate_config['type'] == 'X':
                    circ.x(gate_config['qubit'])
                elif gate_config['type'] == 'CNOT':
                    circ.cx(gate_config['control'], gate_config['target'])
            
            # Add candidate's rotation circuit
            try:
                candidate_circuit = candidate.get_qc(angle=angle if angle is not None else self.rot_angle)
                
                if simplified:
                    try:
                        simplifier = MCRXCascadeSimplifier(verbose=False)
                        simplified_qc, _ = simplifier.simplify(candidate_circuit)
                        circ.append(simplified_qc, range(n_qubits))
                    except:
                        circ.append(candidate_circuit, range(n_qubits))
                else:
                    circ.append(candidate_circuit, range(n_qubits))
                    
            except Exception as e:
                print(f"Error adding candidate {i}: {e}")
                continue
            
            # Apply post-rotation transformation
            for gate_config in transformation['reverse']:
                if gate_config['type'] == 'X':
                    circ.x(gate_config['qubit'])
                elif gate_config['type'] == 'CNOT':
                    circ.cx(gate_config['control'], gate_config['target'])

        return circ.decompose()

    def _create_focused_assignments(self, all_target_states):
        """
        Create assignments where each candidate is focused on producing one target state.
        
        Args:
            all_target_states (set): All target states to cover
            
        Returns:
            list: Focused assignments with transformations
        """
        
        target_states_list = sorted(list(all_target_states))
        assignments = []
        used_candidates = set()
        
        # print(f"Creating focused assignments for: {target_states_list}")
        
        for target_state in target_states_list:
            assignment = self._create_focused_assignment_for_state(target_state, used_candidates)
            
            if assignment:
                assignments.append(assignment)
                used_candidates.add(id(assignment['candidate']))
                # print(f"Target {target_state} → candidate {assignment['candidate'].edges}")
            else:
                print(f"Could not create assignment for {target_state}")
        
        return assignments

    def _create_focused_assignment_for_state(self, target_state, used_candidates):
        """
        Create focused assignment for a specific target state.
        
        The goal is to find a candidate and transformation such that the candidate
        produces a superposition that includes the target state prominently.
        
        Args:
            target_state (str): Target state like '111'
            used_candidates (set): Already used candidates
            
        Returns:
            dict: Assignment with 'candidate', 'target_state', 'transformation'
        """
        
        best_assignment = None
        best_score = -1
        
        for candidate in self.candidates:
            if (id(candidate) in used_candidates or 
                not hasattr(candidate, 'edges') or not candidate.edges):
                continue
            
            candidate_states = self._extract_states_from_edges(candidate.edges)
            
            # Evaluate this candidate for the target state
            evaluation = self._evaluate_candidate_for_target_focus(candidate_states, target_state)
            
            if evaluation and evaluation['score'] > best_score:
                best_score = evaluation['score']
                best_assignment = {
                    'candidate': candidate,
                    'target_state': target_state,
                    'transformation': evaluation['transformation']
                }
        
        return best_assignment

    def _evaluate_candidate_for_target_focus(self, candidate_states, target_state):
        """
        Evaluate how well a candidate can be focused on producing the target state.
        
        Args:
            candidate_states (set): States from candidate edges
            target_state (str): Target state to focus on
            
        Returns:
            dict: Evaluation with 'score' and 'transformation'
        """
        
        candidate_states_list = sorted(list(candidate_states))
        
        # Case 1: Candidate already contains target state
        if target_state in candidate_states:
            # Find the other state in the candidate
            other_state = None
            for state in candidate_states:
                if state != target_state:
                    other_state = state
                    break
            
            if other_state:
                # Try to transform other_state to target_state (to emphasize target_state)
                transformation = self._compute_target_emphasis_transformation(
                    candidate_states_list, target_state
                )
                
                return {
                    'score': 100,  # High score for already containing target
                    'transformation': transformation
                }
        
        # Case 2: Try to transform candidate to produce target state
        transformation = self._find_transformation_to_target(candidate_states_list, target_state)
        
        if transformation:
            # Calculate distance-based score
            min_distance = min(self._hamming_distance(cs, target_state) for cs in candidate_states)
            score = max(0, 50 - min_distance * 10)
            
            return {
                'score': score,
                'transformation': transformation
            }
        
        return None

    def _compute_target_emphasis_transformation(self, candidate_states, target_state):
        """
        Compute transformation to emphasize the target state in the superposition.
        
        Strategy: Transform the candidate so that both states in the superposition
        are related to the target state.
        
        Args:
            candidate_states (list): Candidate states like ['011', '111']
            target_state (str): Target state like '111'
            
        Returns:
            dict: Transformation configuration
        """
        
        if target_state not in candidate_states:
            return {'forward': [], 'reverse': []}
        
        # If candidate already contains target state, try to map the other state closer to target
        other_states = [s for s in candidate_states if s != target_state]
        
        if not other_states:
            return {'forward': [], 'reverse': []}
        
        other_state = other_states[0]
        
        # Try to find a simple transformation that makes both states useful
        # For now, use a simple approach - no transformation if target already present
        return {'forward': [], 'reverse': []}

    def _find_transformation_to_target(self, candidate_states, target_state):
        """
        Find transformation to make candidate produce target state.
        
        Args:
            candidate_states (list): Candidate states like ['000', '100']
            target_state (str): Target state like '111'
            
        Returns:
            dict: Transformation or None
        """
        
        # Find the candidate state closest to target
        closest_state = min(candidate_states, 
                        key=lambda cs: self._hamming_distance(cs, target_state))
        
        # Create transformation to map closest_state to target_state
        qubits_to_flip = []
        for i, (bit1, bit2) in enumerate(zip(closest_state, target_state)):
            if bit1 != bit2:
                qubits_to_flip.append(i)
        
        # If too many flips needed, this is not a good candidate
        if len(qubits_to_flip) > 2:
            return None
        
        operations = [{'type': 'X', 'qubit': qubit} for qubit in qubits_to_flip]
        
        return {'forward': [], 'reverse': operations}

    def _hamming_distance(self, state1, state2):
        """Calculate Hamming distance between two binary strings."""
        if len(state1) != len(state2):
            return float('inf')
        return sum(c1 != c2 for c1, c2 in zip(state1, state2))

    # Alternative approach: Use working single-edge logic for each individual state
    def _create_single_state_assignments(self, all_target_states):
        """
        Alternative: Treat each target state as a single-edge problem.
        
        For each target state, find a candidate and use the working single-edge
        transformation logic to ensure that state is produced.
        
        Args:
            all_target_states (set): All target states
            
        Returns:
            list: Assignments using single-edge approach
        """
        
        assignments = []
        used_candidates = set()
        
        for target_state in sorted(all_target_states):
            # Create a virtual "edge" for this single state
            virtual_edge = (target_state, target_state)  # Same state twice
            
            # Find best candidate for this virtual edge
            best_candidate = None
            best_score = -1
            
            for candidate in self.candidates:
                if (id(candidate) in used_candidates or 
                    not hasattr(candidate, 'edges') or not candidate.edges):
                    continue
                
                candidate_states = self._extract_states_from_edges(candidate.edges)
                
                # Score based on overlap with target state
                if target_state in candidate_states:
                    score = 100
                else:
                    min_distance = min(self._hamming_distance(cs, target_state) for cs in candidate_states)
                    score = max(0, 20 - min_distance * 5)
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate:
                # Use single-edge transformation logic
                candidate_states = self._extract_states_from_edges(best_candidate.edges)
                target_states_for_transformation = {target_state}
                
                # Create transformation using the working boolean logic
                transformation = self._compute_boolean_transformation(candidate_states, target_states_for_transformation)
                
                assignments.append({
                    'candidate': best_candidate,
                    'target_state': target_state,
                    'transformation': transformation
                })
                
                used_candidates.add(id(best_candidate))
        
        return assignments

    # Keep all exact working methods
    def _get_general_transformation_configuration(self):
        """EXACT copy of working single-edge implementation."""
        
        if not self.best_candidate or not hasattr(self.best_candidate, 'edges'):
            return {'forward': [], 'reverse': []}
        
        if not self.set_hamming_greater_1:
            return {'forward': [], 'reverse': []}
        
        rotation_edges = self.best_candidate.edges
        rotation_states = self._extract_states_from_edges(rotation_edges)
        
        target_edges = self.set_hamming_greater_1
        target_states = self._extract_states_from_edges(target_edges)
        
        transformation = self._compute_boolean_transformation(rotation_states, target_states)
        
        return transformation

    def _extract_states_from_edges(self, edges):
        """Extract computational basis states from edges."""
        
        states = set()
        for edge in edges:
            states.add(edge[0])
            states.add(edge[1])
        
        return states

    def _compute_boolean_transformation(self, rotation_states, target_states):
        """EXACT copy of working implementation."""
        
        rotation_list = sorted(list(rotation_states))
        target_list = sorted(list(target_states))
        
        # if len(rotation_list) != len(target_list):
        #     print(f"Warning: Different number of states - rotation: {len(rotation_list)}, target: {len(target_list)}")
        #     return {'forward': [], 'reverse': []}
        
        if len(rotation_list) == 2:
            return self._compute_two_state_transformation(rotation_list, target_list)
        
        return self._compute_general_boolean_transformation(rotation_list, target_list)

    def _compute_two_state_transformation(self, rotation_states, target_states):
        """EXACT copy of working implementation."""
        
        state1_rot, state2_rot = rotation_states[0], rotation_states[1]
        state1_target, state2_target = target_states[0], target_states[1]
        
        # print(f"Mapping: {state1_rot}→{state1_target}, {state2_rot}→{state2_target}")
        
        mapping1 = self._compute_state_mapping(state1_rot, state1_target)
        mapping2 = self._compute_state_mapping(state2_rot, state2_target)
        
        operations = self._find_common_operations(mapping1, mapping2, state1_rot, state2_rot)
        
        return {
            'forward': [],
            'reverse': operations
        }

    def _compute_state_mapping(self, start_state, end_state):
        """EXACT copy of working implementation."""
        
        if len(start_state) != len(end_state):
            return {'valid': False}
        
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
        """EXACT copy of working implementation."""
        
        if not mapping1['valid'] or not mapping2['valid']:
            return []
        
        # Try simple X gates first
        simple_solution = self._try_simple_x_gates(mapping1, mapping2, state1, state2)
        if simple_solution:
            return simple_solution
        
        # Try controlled operations
        controlled_solution = self._try_controlled_operations(mapping1, mapping2, state1, state2)
        if controlled_solution:
            return controlled_solution
        
        # General approach
        return self._generate_general_solution(mapping1, mapping2, state1, state2)

    def _try_simple_x_gates(self, mapping1, mapping2, state1, state2):
        """EXACT copy of working implementation."""
        
        flips1 = set(mapping1['flips'])
        flips2 = set(mapping2['flips'])
        
        if flips1 == flips2:
            return [{'type': 'X', 'qubit': qubit} for qubit in sorted(flips1)]
        
        return []

    def _try_controlled_operations(self, mapping1, mapping2, state1, state2):
        """EXACT copy of working implementation."""
        
        operations = []
        n_qubits = len(state1)
        
        for qubit in range(n_qubits):
            flip_in_mapping1 = qubit in mapping1['flips']
            flip_in_mapping2 = qubit in mapping2['flips']
            
            if flip_in_mapping1 != flip_in_mapping2:
                control_qubit = self._find_control_qubit(state1, state2, qubit)
                if control_qubit is not None:
                    if flip_in_mapping1 and state1[control_qubit] == '1':
                        operations.append({'type': 'CNOT', 'control': control_qubit, 'target': qubit})
                    elif flip_in_mapping1 and state1[control_qubit] == '0':
                        operations.extend([
                            {'type': 'X', 'qubit': control_qubit},
                            {'type': 'CNOT', 'control': control_qubit, 'target': qubit},
                            {'type': 'X', 'qubit': control_qubit}
                        ])
            elif flip_in_mapping1 and flip_in_mapping2:
                operations.append({'type': 'X', 'qubit': qubit})
        
        if self._verify_operations(operations, [(state1, mapping1['end']), (state2, mapping2['end'])]):
            return operations
        
        return []

    def _find_control_qubit(self, state1, state2, target_qubit):
        """EXACT copy of working implementation."""
        
        for i in range(len(state1)):
            if i != target_qubit and state1[i] != state2[i]:
                return i
        
        return None

    def _generate_general_solution(self, mapping1, mapping2, state1, state2):
        """EXACT copy of working implementation."""
        
        operations = []
        n_qubits = len(state1)
        
        selector_qubit = None
        for i in range(n_qubits):
            if state1[i] != state2[i]:
                selector_qubit = i
                break
        
        if selector_qubit is None:
            return []
        
        for target_qubit in range(n_qubits):
            if target_qubit == selector_qubit:
                continue
                
            flip1 = target_qubit in mapping1['flips']
            flip2 = target_qubit in mapping2['flips']
            
            if flip1 != flip2:
                if (flip1 and state1[selector_qubit] == '1') or (flip2 and state2[selector_qubit] == '1'):
                    operations.append({'type': 'CNOT', 'control': selector_qubit, 'target': target_qubit})
            elif flip1 and flip2:
                operations.append({'type': 'X', 'qubit': target_qubit})
        
        if selector_qubit in mapping1['flips']:
            operations.append({'type': 'X', 'qubit': selector_qubit})
        
        return operations

    def _verify_operations(self, operations, state_mappings):
        """EXACT copy of working implementation."""
        
        for start_state, expected_end_state in state_mappings:
            current_state = start_state
            
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
        """EXACT copy of working implementation."""
        
        operations = []
        
        for i, (rot_state, target_state) in enumerate(zip(rotation_states, target_states)):
            mapping = self._compute_state_mapping(rot_state, target_state)
            
            for qubit in mapping['flips']:
                if {'type': 'X', 'qubit': qubit} not in operations:
                    operations.append({'type': 'X', 'qubit': qubit})
        
        return {
            'forward': [],
            'reverse': operations
        }

    def __del__(self):
        """Clean up any remaining resources"""
        attrs = ["candidates", "best_candidate", "connections"]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
