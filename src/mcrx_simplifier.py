# File: src/mcrx_simplifier.py
"""
MCRX cascade simplification implementing the algorithm from the tex file.
Uses SymPy for boolean algebra as described in the paper.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict
import re
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
import sympy as sp
from sympy.logic import simplify_logic
from sympy.logic.boolalg import And, Or, Not, Xor
from sympy import symbols

from .pattern_analyzer import ControlPatternAnalyzer, PatternRelationship
from .misc import multi_crx


class MCRXGateInfo:
    """Information about a single MCRX gate."""
    
    def __init__(self, theta: float, target: int, ctrl_qubits: List[int], ctrl_state: str):
        self.theta = theta
        self.target = target
        self.ctrl_qubits = ctrl_qubits
        self.ctrl_state = ctrl_state
        self.pattern = ctrl_state
        self.n_controls = len(ctrl_qubits)
    
    def __repr__(self):
        return f"MCRXGateInfo(θ={self.theta:.3f}, target={self.target}, ctrl={self.ctrl_qubits}, pattern='{self.pattern}')"


class BooleanOptimizer:
    """Boolean algebra optimization using SymPy for the algorithm from tex file."""
    
    def __init__(self):
        self.symbol_cache = {}
    
    def get_symbols(self, n_bits: int) -> List[sp.Symbol]:
        """Get SymPy symbols for n bits."""
        if n_bits not in self.symbol_cache:
            self.symbol_cache[n_bits] = [sp.Symbol(f'x{i}') for i in range(n_bits)]
        return self.symbol_cache[n_bits]
    
    def state_to_expr(self, state: int, n_bits: int, symbols: List[sp.Symbol]) -> sp.Basic:
        """
        Convert computational basis state to SymPy boolean expression.
        From tex: Each state corresponds to a conjunctive term.
        
        Args:
            state: Computational basis state (integer)
            n_bits: Number of bits
            symbols: SymPy symbols for variables
            
        Returns:
            Boolean expression for this state
        """
        state_pattern = format(state, f'0{n_bits}b')
        expr_terms = []
        
        for i, bit in enumerate(state_pattern):
            if bit == '1':
                expr_terms.append(symbols[i])
            else:  # bit == '0'
                expr_terms.append(Not(symbols[i]))
        
        if not expr_terms:
            return sp.true
        elif len(expr_terms) == 1:
            return expr_terms[0]
        else:
            return And(*expr_terms)
    
    def states_to_or_expr(self, states: List[int], n_bits: int) -> Tuple[sp.Basic, List[sp.Symbol]]:
        """
        Convert list of computational basis states to OR expression.
        From tex: Create boolean function for states with same rotation coefficient.
        
        Args:
            states: List of computational basis states
            n_bits: Number of control bits
            
        Returns:
            Tuple of (OR expression, symbols)
        """
        if not states:
            return sp.false, []
        
        symbols = self.get_symbols(n_bits)
        
        if len(states) == 1:
            return self.state_to_expr(states[0], n_bits, symbols), symbols
        
        # Multiple states - create OR of all state expressions
        state_exprs = []
        for state in states:
            state_expr = self.state_to_expr(state, n_bits, symbols)
            state_exprs.append(state_expr)
        
        or_expr = Or(*state_exprs)
        return or_expr, symbols
    
    def simplify_boolean_expr(self, expr: sp.Basic) -> sp.Basic:
        """
        Simplify boolean expression using SymPy.
        From tex: Apply boolean algebra simplification.
        """
        try:
            simplified = simplify_logic(expr)
            return simplified
        except Exception:
            return expr
    
    def analyze_expr_complexity(self, expr: sp.Basic) -> Dict[str, int]:
        """Analyze complexity of boolean expression."""
        expr_str = str(expr)
        
        return {
            'and_ops': expr_str.count('&'),
            'or_ops': expr_str.count('|'),
            'not_ops': expr_str.count('~'),
            'xor_ops': expr_str.count('Xor'),
            'variables': len(expr.free_symbols),
            'total_length': len(expr_str)
        }
    
    def expr_to_logical_circuit(self, expr: sp.Basic, symbols: List[sp.Symbol], 
                               angle: float, target: int, ctrl_qubits: List[int], 
                               n_qubits: int) -> QuantumCircuit:
        """
        Convert simplified boolean expression to logical circuit.
        From tex: Implement optimized boolean function as quantum circuit.
        """
        circuit = QuantumCircuit(n_qubits)
        
        if expr == sp.true:
            # Always true - unconditional rotation
            circuit.rx(angle, target)
            return circuit
        elif expr == sp.false:
            # Never true - no rotation
            return circuit
        
        # Analyze expression structure and create logical circuit
        self._implement_expression(expr, symbols, angle, target, ctrl_qubits, circuit)
        return circuit
    
    def _implement_expression(self, expr: sp.Basic, symbols: List[sp.Symbol],
                             angle: float, target: int, ctrl_qubits: List[int],
                             circuit: QuantumCircuit):
        """Implement boolean expression as quantum circuit."""
        
        if expr.is_Symbol:
            # Single variable: x_i → CRX(ctrl[i], target)
            var_index = self._get_symbol_index(expr, symbols)
            if var_index < len(ctrl_qubits):
                circuit.crx(angle, ctrl_qubits[var_index], target)
            
        elif isinstance(expr, Not) and expr.args[0].is_Symbol:
            # Negated variable: ~x_i → X + CRX + X
            var_index = self._get_symbol_index(expr.args[0], symbols)
            if var_index < len(ctrl_qubits):
                circuit.x(ctrl_qubits[var_index])
                circuit.crx(angle, ctrl_qubits[var_index], target)
                circuit.x(ctrl_qubits[var_index])
            
        elif isinstance(expr, And):
            # AND expression: implement as multi-controlled gate
            self._implement_and_expression(expr, symbols, angle, target, ctrl_qubits, circuit)
            
        elif isinstance(expr, Or):
            # OR expression: check for known patterns
            if self._is_xor_pattern(expr, symbols):
                self._implement_xor_pattern(expr, symbols, angle, target, ctrl_qubits, circuit)
            else:
                # General OR: implement each term
                for term in expr.args:
                    self._implement_expression(term, symbols, angle, target, ctrl_qubits, circuit)
            
        elif isinstance(expr, Xor):
            # XOR expression: implement using logical form
            self._implement_xor_expression(expr, symbols, angle, target, ctrl_qubits, circuit)
        
        else:
            # Complex expression: convert to DNF and implement
            try:
                dnf_expr = sp.to_dnf(expr)
                if isinstance(dnf_expr, Or):
                    for term in dnf_expr.args:
                        self._implement_expression(term, symbols, angle, target, ctrl_qubits, circuit)
                else:
                    self._implement_expression(dnf_expr, symbols, angle, target, ctrl_qubits, circuit)
            except Exception:
                # Fallback: implement directly
                self._implement_general_expression(expr, symbols, angle, target, ctrl_qubits, circuit)
    
    def _get_symbol_index(self, symbol: sp.Symbol, symbols: List[sp.Symbol]) -> int:
        """Get index of symbol in symbol list."""
        try:
            return symbols.index(symbol)
        except ValueError:
            # Extract from symbol name (e.g., 'x2' -> 2)
            symbol_name = str(symbol)
            if symbol_name.startswith('x'):
                return int(symbol_name[1:])
            return 0
    
    def _implement_and_expression(self, expr: And, symbols: List[sp.Symbol],
                                 angle: float, target: int, ctrl_qubits: List[int],
                                 circuit: QuantumCircuit):
        """Implement AND expression as multi-controlled gate."""
        controls = []
        control_states = []
        
        for arg in expr.args:
            if arg.is_Symbol:
                var_index = self._get_symbol_index(arg, symbols)
                if var_index < len(ctrl_qubits):
                    controls.append(ctrl_qubits[var_index])
                    control_states.append('1')
            elif isinstance(arg, Not) and arg.args[0].is_Symbol:
                var_index = self._get_symbol_index(arg.args[0], symbols)
                if var_index < len(ctrl_qubits):
                    controls.append(ctrl_qubits[var_index])
                    control_states.append('0')
        
        if controls:
            ctrl_state = ''.join(control_states)
            mcrx_gate = multi_crx(angle, ctrl_state)
            circuit.append(mcrx_gate, controls + [target])
    
    def _is_xor_pattern(self, expr: Or, symbols: List[sp.Symbol]) -> bool:
        """
        Check if OR expression represents XOR pattern.
        From tex: Detect (x0 & ~x1) | (~x0 & x1) = x0 ⊕ x1
        """
        if len(expr.args) != 2:
            return False
        
        arg1, arg2 = expr.args
        
        # Both should be AND expressions with 2 terms
        if not (isinstance(arg1, And) and isinstance(arg2, And) and 
                len(arg1.args) == 2 and len(arg2.args) == 2):
            return False
        
        try:
            # Check if it simplifies to XOR
            simplified = simplify_logic(expr)
            return isinstance(simplified, Xor) or 'Xor' in str(simplified)
        except Exception:
            return False
    
    def _implement_xor_pattern(self, expr: Or, symbols: List[sp.Symbol],
                              angle: float, target: int, ctrl_qubits: List[int],
                              circuit: QuantumCircuit):
        """
        Implement XOR pattern using logical form.
        From tex: XOR logical form = CNOT + CRX + CNOT
        """
        if len(ctrl_qubits) >= 2:
            # XOR logical form for first two qubits
            circuit.cx(ctrl_qubits[0], ctrl_qubits[1])
            circuit.crx(angle, ctrl_qubits[1], target)
            circuit.cx(ctrl_qubits[0], ctrl_qubits[1])
    
    def _implement_xor_expression(self, expr: Xor, symbols: List[sp.Symbol],
                                 angle: float, target: int, ctrl_qubits: List[int],
                                 circuit: QuantumCircuit):
        """Implement XOR expression using logical form."""
        if len(expr.args) == 2 and len(ctrl_qubits) >= 2:
            # Two-way XOR: implement as CNOT + CRX + CNOT
            circuit.cx(ctrl_qubits[0], ctrl_qubits[1])
            circuit.crx(angle, ctrl_qubits[1], target)
            circuit.cx(ctrl_qubits[0], ctrl_qubits[1])
        else:
            # Multi-way XOR: convert to OR of AND terms
            expanded = sp.to_dnf(expr)
            self._implement_expression(expanded, symbols, angle, target, ctrl_qubits, circuit)
    
    def _implement_general_expression(self, expr: sp.Basic, symbols: List[sp.Symbol],
                                    angle: float, target: int, ctrl_qubits: List[int],
                                    circuit: QuantumCircuit):
        """Fallback implementation for complex expressions."""
        # For general expressions, evaluate for all possible states
        n_vars = len(symbols)
        
        for i in range(2**n_vars):
            assignment = {symbols[j]: bool((i >> j) & 1) for j in range(n_vars)}
            
            try:
                if expr.subs(assignment):
                    # This state satisfies the expression
                    state_pattern = format(i, f'0{n_vars}b')
                    
                    # Create multi-controlled gate for this state
                    controls = ctrl_qubits[:n_vars]
                    mcrx_gate = multi_crx(angle, state_pattern)
                    circuit.append(mcrx_gate, controls + [target])
            except Exception:
                continue


class MCRXCascadeSimplifier:
    """
    MCRX cascade simplification implementing Algorithm 1 from the tex file.
    """
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.pattern_analyzer = ControlPatternAnalyzer()
        self.boolean_optimizer = BooleanOptimizer()
    
    def simplify(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Main simplification method implementing Algorithm 1 from the tex file.
        
        Args:
            circuit: Input quantum circuit containing only MCRX gates
            
        Returns:
            Optimized quantum circuit
            
        Raises:
            ValueError: If circuit contains non-MCRX gates, mixed targets, or mixed angles
            RuntimeError: If circuit is empty or invalid
        """
        # Validate input
        if circuit.num_qubits == 0:
            raise RuntimeError("Cannot simplify empty circuit (0 qubits)")
        
        if len(circuit.data) == 0:
            raise RuntimeError("Cannot simplify circuit with no gates")
        
        # Extract and validate MCRX gates
        mcrx_gates = self._validate_and_extract_mcrx_gates(circuit)
        self._validate_same_target_and_angle(mcrx_gates)
        
        # Apply Algorithm 1 from tex file
        return self._apply_algorithm_1(mcrx_gates, circuit.num_qubits)
    
    def _validate_and_extract_mcrx_gates(self, circuit: QuantumCircuit) -> List[MCRXGateInfo]:
        """Extract and validate MCRX gates from circuit."""
        if not circuit.data:
            raise ValueError("Circuit contains no gates")
        
        mcrx_gates = []
        
        for i, instruction in enumerate(circuit.data):
            gate_info = self._extract_gate_info(instruction, circuit)
            if gate_info is None:
                raise ValueError(
                    f"Gate {i+1} is not an MCRX gate: '{instruction.operation.name}'. "
                    f"All gates must be multi-controlled RX gates"
                )
            mcrx_gates.append(gate_info)
        
        return mcrx_gates
    
    def _extract_gate_info(self, instruction: Instruction, circuit: QuantumCircuit) -> Optional[MCRXGateInfo]:
        """Extract gate information from circuit instruction."""
        op_name = instruction.operation.name.lower()
        
        # Check if it's an MCRX gate
        mcrx_patterns = ['mcrx', 'ccrx', 'multi_crx', 'mcx_rotation']
        if not any(pattern in op_name for pattern in mcrx_patterns):
            return None
        
        # Get qubits
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        target = qubits[-1]
        ctrl_qubits = qubits[:-1]
        
        # Get rotation angle
        if hasattr(instruction.operation, 'params') and instruction.operation.params:
            theta = float(instruction.operation.params[0])
        else:
            raise ValueError(f"Gate '{instruction.operation.name}' has no rotation angle parameter")
        
        # Extract control state
        ctrl_state = self._extract_control_state(instruction, len(ctrl_qubits))
        
        return MCRXGateInfo(theta, target, ctrl_qubits, ctrl_state)
    
    def _extract_control_state(self, instruction: Instruction, n_controls: int) -> str:
        """Extract control state pattern from gate instruction."""
        op_name = instruction.operation.name
        
        # Try to extract from operation attributes
        if hasattr(instruction.operation, 'ctrl_state'):
            ctrl_state = instruction.operation.ctrl_state
            if isinstance(ctrl_state, str):
                return ctrl_state
            elif isinstance(ctrl_state, int):
                return format(ctrl_state, f'0{n_controls}b')
        
        # Try to extract from gate name (e.g., 'mcrx_o10' -> '10')
        pattern_match = re.search(r'_o(\d+)', op_name)
        if pattern_match:
            binary_str = pattern_match.group(1)
            if len(binary_str) <= n_controls:
                return binary_str.zfill(n_controls)
        
        # Default to all-ones control
        return '1' * n_controls
    
    def _validate_same_target_and_angle(self, mcrx_gates: List[MCRXGateInfo]) -> None:
        """Validate all gates have same target and angle."""
        if len(mcrx_gates) <= 1:
            return
        
        reference_target = mcrx_gates[0].target
        reference_angle = mcrx_gates[0].theta
        
        for i, gate in enumerate(mcrx_gates[1:], 1):
            if gate.target != reference_target:
                raise ValueError(
                    f"MCRX gates must all have the same target qubit. "
                    f"Gate {i+1} targets qubit {gate.target}, but gate 1 targets qubit {reference_target}."
                )
            
            if abs(gate.theta - reference_angle) > self.tolerance:
                raise ValueError(
                    f"MCRX gates must all have the same rotation angle. "
                    f"Gate {i+1} has angle {gate.theta:.6f}, but gate 1 has angle {reference_angle:.6f}."
                )
    
    def _apply_algorithm_1(self, mcrx_gates: List[MCRXGateInfo], n_qubits: int) -> QuantumCircuit:
        """
        Apply Algorithm 1 from tex file: SimplifySameTargetMCRXCascade.
        
        From tex:
        1. Initialize rotation coefficient array r[2^n] = 0
        2. For each computational basis state S ∈ {0,1,...,2^n-1}:
           For each control pattern C_i:
               If state S satisfies control pattern C_i:
                   r[S] ← r[S] + 1
        3. Group states by rotation coefficient
        4. For each unique coefficient k:
           Create control condition for OR of states in S_k
           Add gate: CRX_{∨_{S∈S_k} S, t}(kθ)
        """
        if len(mcrx_gates) == 1:
            # Single gate - no optimization needed
            circuit = QuantumCircuit(n_qubits)
            gate = mcrx_gates[0]
            mcrx_gate = multi_crx(gate.theta, gate.pattern)
            circuit.append(mcrx_gate, gate.ctrl_qubits + [gate.target])
            return circuit
        
        # Step 1: Initialize rotation coefficient array
        n_controls = max(gate.n_controls for gate in mcrx_gates)
        n_states = 2 ** n_controls
        rotation_coefficients = [0] * n_states
        
        # Step 2: Calculate rotation coefficients for each state
        for state in range(n_states):
            state_bits = format(state, f'0{n_controls}b')
            for gate in mcrx_gates:
                if self._state_satisfies_pattern(state_bits, gate, n_controls):
                    rotation_coefficients[state] += 1
        
        # Step 3: Group states by rotation coefficient
        coeff_groups = defaultdict(list)
        for state, coeff in enumerate(rotation_coefficients):
            if coeff > 0:
                coeff_groups[coeff].append(state)
        
        # Step 4: Create optimized gates using boolean algebra
        circuit = QuantumCircuit(n_qubits)
        theta = mcrx_gates[0].theta
        target = mcrx_gates[0].target
        ctrl_qubits = mcrx_gates[0].ctrl_qubits
        
        for coeff, states in coeff_groups.items():
            if not states:
                continue
            
            total_angle = coeff * theta
            
            if len(states) == 1:
                # Single state - direct implementation
                state_pattern = format(states[0], f'0{n_controls}b')
                mcrx_gate = multi_crx(total_angle, state_pattern)
                circuit.append(mcrx_gate, ctrl_qubits + [target])
            
            else:
                # Multiple states - use boolean algebra
                or_expr, symbols = self.boolean_optimizer.states_to_or_expr(states, n_controls)
                simplified_expr = self.boolean_optimizer.simplify_boolean_expr(or_expr)
                
                # Convert to logical circuit
                logical_circuit = self.boolean_optimizer.expr_to_logical_circuit(
                    simplified_expr, symbols, total_angle, target, ctrl_qubits, n_qubits
                )
                
                # Compose with main circuit
                circuit = circuit.compose(logical_circuit)
        
        return circuit
    
    def _state_satisfies_pattern(self, state_bits: str, gate: MCRXGateInfo, n_controls: int) -> bool:
        """Check if computational basis state satisfies gate's control pattern."""
        if len(state_bits) < gate.n_controls:
            return False
        
        # Align pattern with state bits (use rightmost bits to match pattern)
        if gate.n_controls < n_controls:
            # Pattern is shorter - pad with don't cares or align appropriately
            # For now, use leftmost bits
            relevant_bits = state_bits[:gate.n_controls]
        else:
            relevant_bits = state_bits
        
        return relevant_bits == gate.pattern
    
    def analyze_optimization_potential(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Analyze optimization potential using the algorithm."""
        if circuit.num_qubits == 0:
            raise RuntimeError("Cannot analyze empty circuit (0 qubits)")
        
        if len(circuit.data) == 0:
            raise RuntimeError("Cannot analyze circuit with no gates")
        
        mcrx_gates = self._validate_and_extract_mcrx_gates(circuit)
        self._validate_same_target_and_angle(mcrx_gates)
        
        # Apply algorithm analysis
        patterns = [gate.pattern for gate in mcrx_gates]
        n_controls = max(gate.n_controls for gate in mcrx_gates)
        n_states = 2 ** n_controls
        rotation_coefficients = [0] * n_states
        
        # Calculate rotation coefficients
        for state in range(n_states):
            state_bits = format(state, f'0{n_controls}b')
            for gate in mcrx_gates:
                if self._state_satisfies_pattern(state_bits, gate, n_controls):
                    rotation_coefficients[state] += 1
        
        # Group by coefficients
        coeff_groups = defaultdict(list)
        for state, coeff in enumerate(rotation_coefficients):
            if coeff > 0:
                coeff_groups[coeff].append(state)
        
        # Estimate gates after optimization
        estimated_gates = 0
        boolean_functions = []
        
        for coeff, states in coeff_groups.items():
            if len(states) == 1:
                estimated_gates += 1
                state_pattern = format(states[0], f'0{n_controls}b')
                boolean_functions.append(f"State: {state_pattern}")
            else:
                # Use SymPy to get actual boolean function
                or_expr, symbols = self.boolean_optimizer.states_to_or_expr(states, n_controls)
                simplified_expr = self.boolean_optimizer.simplify_boolean_expr(or_expr)
                
                # Estimate gates based on simplified expression
                if simplified_expr.is_Symbol:
                    estimated_gates += 1
                elif isinstance(simplified_expr, Xor) or 'Xor' in str(simplified_expr):
                    estimated_gates += 3  # CNOT + CRX + CNOT
                elif isinstance(simplified_expr, And):
                    estimated_gates += 1  # Multi-controlled gate
                else:
                    # Conservative estimate
                    complexity = self.boolean_optimizer.analyze_expr_complexity(simplified_expr)
                    estimated_gates += max(1, complexity['and_ops'] + complexity['or_ops'])
                
                boolean_functions.append(str(simplified_expr))
        
        return {
            'status': 'valid',
            'original_gates': len(mcrx_gates),
            'estimated_optimized': estimated_gates,
            'potential_reduction': max(0, len(mcrx_gates) - estimated_gates),
            'reduction_percentage': max(0, (len(mcrx_gates) - estimated_gates) / len(mcrx_gates) * 100),
            'patterns': patterns,
            'unique_patterns': len(set(patterns)),
            'boolean_functions': boolean_functions,
            'rotation_coefficients': {i: coeff for i, coeff in enumerate(rotation_coefficients) if coeff > 0},
            'coefficient_groups': {k: v for k, v in coeff_groups.items()},
            'target_qubit': mcrx_gates[0].target,
            'rotation_angle': mcrx_gates[0].theta
        }
