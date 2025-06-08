# File: src/mcrx_simplifier.py
"""
MCRX cascade simplification implementing logical forms with CNOT optimization tricks.
Uses SymPy for boolean algebra and implements specific logical forms from tex file.
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


class LogicalFormOptimizer:
    """
    Implement specific logical forms from tex file using CNOT optimization tricks.
    """
    
    def __init__(self):
        self.symbol_cache = {}
    
    def get_symbols(self, n_bits: int) -> List[sp.Symbol]:
        """Get SymPy symbols for n bits."""
        if n_bits not in self.symbol_cache:
            self.symbol_cache[n_bits] = [sp.Symbol(f'x{i}') for i in range(n_bits)]
        return self.symbol_cache[n_bits]
    
    def state_to_expr(self, state: int, n_bits: int, symbols: List[sp.Symbol]) -> sp.Basic:
        """Convert computational basis state to SymPy boolean expression."""
        state_pattern = format(state, f'0{n_bits}b')
        expr_terms = []
        
        for i, bit in enumerate(state_pattern):
            if bit == '1':
                expr_terms.append(symbols[i])
            else:
                expr_terms.append(Not(symbols[i]))
        
        if not expr_terms:
            return sp.true
        elif len(expr_terms) == 1:
            return expr_terms[0]
        else:
            return And(*expr_terms)
    
    def states_to_or_expr(self, states: List[int], n_bits: int) -> Tuple[sp.Basic, List[sp.Symbol]]:
        """Convert list of computational basis states to OR expression."""
        if not states:
            return sp.false, []
        
        symbols = self.get_symbols(n_bits)
        
        if len(states) == 1:
            return self.state_to_expr(states[0], n_bits, symbols), symbols
        
        state_exprs = []
        for state in states:
            state_expr = self.state_to_expr(state, n_bits, symbols)
            state_exprs.append(state_expr)
        
        or_expr = Or(*state_exprs)
        return or_expr, symbols
    
    def simplify_boolean_expr(self, expr: sp.Basic) -> sp.Basic:
        """Simplify boolean expression using SymPy."""
        try:
            simplified = simplify_logic(expr)
            return simplified
        except Exception:
            return expr
    
    def implement_logical_form(self, expr: sp.Basic, symbols: List[sp.Symbol], 
                              angle: float, target: int, ctrl_qubits: List[int], 
                              n_qubits: int) -> QuantumCircuit:
        """
        Implement logical form using CNOT optimization tricks from tex file.
        """
        circuit = QuantumCircuit(n_qubits)
        
        if expr == sp.true:
            circuit.rx(angle, target)
            return circuit
        elif expr == sp.false:
            return circuit
        
        # Check for specific logical forms from tex file
        if self._is_two_qubit_xor_form(expr, symbols):
            # Implement: ['10', '01'] → x₀ ⊕ x₁ using CNOT trick
            self._implement_two_qubit_xor_form(circuit, angle, target, ctrl_qubits)
            
        elif self._is_three_qubit_complex_form(expr, symbols):
            # Implement: ['110', '101'] → x₀ ∧ (x₁ ⊕ x₂) using CNOT trick
            self._implement_three_qubit_complex_form(circuit, angle, target, ctrl_qubits)
            
        elif self._is_maximal_simplification_form(expr, symbols):
            # Implement: ['11', '10'] → x₀ (single control)
            self._implement_maximal_simplification_form(circuit, expr, symbols, angle, target, ctrl_qubits)
            
        else:
            # General implementation
            self._implement_general_form(circuit, expr, symbols, angle, target, ctrl_qubits)
        
        return circuit
    
    def _is_two_qubit_xor_form(self, expr: sp.Basic, symbols: List[sp.Symbol]) -> bool:
        """
        Check if expression matches x₀ ⊕ x₁ pattern from tex file.
        This corresponds to patterns ['10', '01'].
        """
        if len(symbols) != 2:
            return False
        
        try:
            # Check if expression is equivalent to x0 XOR x1
            x0, x1 = symbols[0], symbols[1]
            xor_expr = Xor(x0, x1)
            
            # Also check expanded form: (x0 & ~x1) | (~x0 & x1)
            expanded_xor = Or(And(x0, Not(x1)), And(Not(x0), x1))
            
            simplified = simplify_logic(expr)
            
            return (simplified == xor_expr or 
                   simplified == expanded_xor or
                   str(simplified) == str(xor_expr) or
                   'Xor' in str(simplified))
        except Exception:
            return False
    
    def _is_three_qubit_complex_form(self, expr: sp.Basic, symbols: List[sp.Symbol]) -> bool:
        """
        Check if expression matches x₀ ∧ (x₁ ⊕ x₂) pattern from tex file.
        This corresponds to patterns ['110', '101'].
        """
        if len(symbols) != 3:
            return False
        
        try:
            x0, x1, x2 = symbols[0], symbols[1], symbols[2]
            # Pattern: x₀ ∧ (x₁ ⊕ x₂)
            target_expr = And(x0, Xor(x1, x2))
            
            # Also check expanded form
            expanded = And(x0, Or(And(x1, Not(x2)), And(Not(x1), x2)))
            
            simplified = simplify_logic(expr)
            
            return (simplified == target_expr or 
                   simplified == expanded or
                   str(simplified) == str(target_expr))
        except Exception:
            return False
    
    def _is_maximal_simplification_form(self, expr: sp.Basic, symbols: List[sp.Symbol]) -> bool:
        """
        Check if expression simplifies to single variable.
        This corresponds to patterns like ['11', '10'] → x₀.
        """
        try:
            simplified = simplify_logic(expr)
            return simplified.is_Symbol
        except Exception:
            return False
    
    def _implement_two_qubit_xor_form(self, circuit: QuantumCircuit, angle: float, 
                                     target: int, ctrl_qubits: List[int]):
        """
        Implement XOR logical form from tex file:
        1. CNOT from qubit 0 to qubit 1 (compute x₀ ⊕ x₁ in qubit 1)
        2. Controlled RX on target controlled by qubit 1  
        3. CNOT from qubit 0 to qubit 1 (restore qubit 1)
        """
        if len(ctrl_qubits) >= 2:
            q0, q1 = ctrl_qubits[0], ctrl_qubits[1]
            
            # Step 1: Compute XOR
            circuit.cx(q0, q1)
            
            # Step 2: Apply controlled rotation
            circuit.crx(angle, q1, target)
            
            # Step 3: Restore
            circuit.cx(q0, q1)
    
    def _implement_three_qubit_complex_form(self, circuit: QuantumCircuit, angle: float,
                                          target: int, ctrl_qubits: List[int]):
        """
        Implement complex 3-qubit logical form from tex file:
        1. CNOT from qubit 1 to qubit 2 (compute x₁ ⊕ x₂ in qubit 2)
        2. Multi-controlled RX with pattern '11' (qubits 0,2)
        3. CNOT from qubit 1 to qubit 2 (restore qubit 2)
        """
        if len(ctrl_qubits) >= 3:
            q0, q1, q2 = ctrl_qubits[0], ctrl_qubits[1], ctrl_qubits[2]
            
            # Step 1: Compute x₁ ⊕ x₂ in qubit 2
            circuit.cx(q1, q2)
            
            # Step 2: Apply 2-controlled RX with qubits 0,2
            mcrx_gate = multi_crx(angle, '11')
            circuit.append(mcrx_gate, [q0, q2, target])
            
            # Step 3: Restore qubit 2
            circuit.cx(q1, q2)
    
    def _implement_maximal_simplification_form(self, circuit: QuantumCircuit, expr: sp.Basic,
                                             symbols: List[sp.Symbol], angle: float,
                                             target: int, ctrl_qubits: List[int]):
        """
        Implement maximal simplification: single variable control.
        """
        simplified = simplify_logic(expr)
        if simplified.is_Symbol:
            # Find which variable it simplified to
            var_index = symbols.index(simplified)
            if var_index < len(ctrl_qubits):
                circuit.crx(angle, ctrl_qubits[var_index], target)
    
    def _implement_general_form(self, circuit: QuantumCircuit, expr: sp.Basic,
                               symbols: List[sp.Symbol], angle: float,
                               target: int, ctrl_qubits: List[int]):
        """
        General implementation for other boolean expressions.
        """
        if expr.is_Symbol:
            var_index = self._get_symbol_index(expr, symbols)
            if var_index < len(ctrl_qubits):
                circuit.crx(angle, ctrl_qubits[var_index], target)
                
        elif isinstance(expr, Not) and expr.args[0].is_Symbol:
            var_index = self._get_symbol_index(expr.args[0], symbols)
            if var_index < len(ctrl_qubits):
                circuit.x(ctrl_qubits[var_index])
                circuit.crx(angle, ctrl_qubits[var_index], target)
                circuit.x(ctrl_qubits[var_index])
                
        elif isinstance(expr, And):
            self._implement_and_expression(circuit, expr, symbols, angle, target, ctrl_qubits)
            
        elif isinstance(expr, Or):
            # For general OR, implement each term
            for term in expr.args:
                self._implement_general_form(circuit, term, symbols, angle, target, ctrl_qubits)
    
    def _get_symbol_index(self, symbol: sp.Symbol, symbols: List[sp.Symbol]) -> int:
        """Get index of symbol in symbol list."""
        try:
            return symbols.index(symbol)
        except ValueError:
            symbol_name = str(symbol)
            if symbol_name.startswith('x'):
                return int(symbol_name[1:])
            return 0
    
    def _implement_and_expression(self, circuit: QuantumCircuit, expr: And,
                                 symbols: List[sp.Symbol], angle: float,
                                 target: int, ctrl_qubits: List[int]):
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


class MCRXCascadeSimplifier:
    """
    MCRX cascade simplification implementing Algorithm 1 with logical form optimizations.
    """
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.pattern_analyzer = ControlPatternAnalyzer()
        self.logical_optimizer = LogicalFormOptimizer()
    
    def simplify(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Main simplification method implementing Algorithm 1 with logical forms."""
        if circuit.num_qubits == 0:
            raise RuntimeError("Cannot simplify empty circuit (0 qubits)")
        
        if len(circuit.data) == 0:
            raise RuntimeError("Cannot simplify circuit with no gates")
        
        mcrx_gates = self._validate_and_extract_mcrx_gates(circuit)
        self._validate_same_target_and_angle(mcrx_gates)
        
        return self._apply_algorithm_1_with_logical_forms(mcrx_gates, circuit.num_qubits)
    
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
        
        mcrx_patterns = ['mcrx', 'ccrx', 'multi_crx', 'mcx_rotation']
        if not any(pattern in op_name for pattern in mcrx_patterns):
            return None
        
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        target = qubits[-1]
        ctrl_qubits = qubits[:-1]
        
        if hasattr(instruction.operation, 'params') and instruction.operation.params:
            theta = float(instruction.operation.params[0])
        else:
            raise ValueError(f"Gate '{instruction.operation.name}' has no rotation angle parameter")
        
        ctrl_state = self._extract_control_state(instruction, len(ctrl_qubits))
        
        return MCRXGateInfo(theta, target, ctrl_qubits, ctrl_state)
    
    def _extract_control_state(self, instruction: Instruction, n_controls: int) -> str:
        """Extract control state pattern from gate instruction."""
        op_name = instruction.operation.name
        
        if hasattr(instruction.operation, 'ctrl_state'):
            ctrl_state = instruction.operation.ctrl_state
            if isinstance(ctrl_state, str):
                return ctrl_state
            elif isinstance(ctrl_state, int):
                return format(ctrl_state, f'0{n_controls}b')
        
        pattern_match = re.search(r'_o(\d+)', op_name)
        if pattern_match:
            binary_str = pattern_match.group(1)
            if len(binary_str) <= n_controls:
                return binary_str.zfill(n_controls)
        
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
    
    def _apply_algorithm_1_with_logical_forms(self, mcrx_gates: List[MCRXGateInfo], n_qubits: int) -> QuantumCircuit:
        """
        Apply Algorithm 1 from tex file with logical form optimizations.
        """
        if len(mcrx_gates) == 1:
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
        
        # Step 4: Create optimized gates using logical forms
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
                # Multiple states - use logical form optimization
                or_expr, symbols = self.logical_optimizer.states_to_or_expr(states, n_controls)
                simplified_expr = self.logical_optimizer.simplify_boolean_expr(or_expr)
                
                # Use logical form implementation with CNOT tricks
                logical_circuit = self.logical_optimizer.implement_logical_form(
                    simplified_expr, symbols, total_angle, target, ctrl_qubits, n_qubits
                )
                
                circuit = circuit.compose(logical_circuit)
        
        return circuit
    
    def _state_satisfies_pattern(self, state_bits: str, gate: MCRXGateInfo, n_controls: int) -> bool:
        """Check if computational basis state satisfies gate's control pattern."""
        if len(state_bits) < gate.n_controls:
            return False
        
        if gate.n_controls < n_controls:
            relevant_bits = state_bits[:gate.n_controls]
        else:
            relevant_bits = state_bits
        
        return relevant_bits == gate.pattern
    
    def analyze_optimization_potential(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Analyze optimization potential using the algorithm with logical form detection."""
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
        
        # Analyze logical forms and estimate gates
        estimated_gates = 0
        boolean_functions = []
        logical_forms = []
        
        for coeff, states in coeff_groups.items():
            if len(states) == 1:
                estimated_gates += 1
                state_pattern = format(states[0], f'0{n_controls}b')
                boolean_functions.append(f"State: {state_pattern}")
                logical_forms.append("Direct implementation")
            else:
                # Analyze with logical form optimizer
                or_expr, symbols = self.logical_optimizer.states_to_or_expr(states, n_controls)
                simplified_expr = self.logical_optimizer.simplify_boolean_expr(or_expr)
                
                # Detect specific logical forms
                if self.logical_optimizer._is_two_qubit_xor_form(simplified_expr, symbols):
                    estimated_gates += 3  # CNOT + CRX + CNOT
                    logical_forms.append("Two-qubit XOR logical form")
                elif self.logical_optimizer._is_three_qubit_complex_form(simplified_expr, symbols):
                    estimated_gates += 3  # CNOT + CCRX + CNOT  
                    logical_forms.append("Three-qubit complex logical form")
                elif self.logical_optimizer._is_maximal_simplification_form(simplified_expr, symbols):
                    estimated_gates += 1  # Single control
                    logical_forms.append("Maximal simplification")
                else:
                    # General case
                    complexity = len(str(simplified_expr))
                    estimated_gates += max(1, complexity // 10)
                    logical_forms.append("General boolean form")
                
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
            'logical_forms': logical_forms,
            'rotation_coefficients': {i: coeff for i, coeff in enumerate(rotation_coefficients) if coeff > 0},
            'coefficient_groups': {k: v for k, v in coeff_groups.items()},
            'target_qubit': mcrx_gates[0].target,
            'rotation_angle': mcrx_gates[0].theta
        }
