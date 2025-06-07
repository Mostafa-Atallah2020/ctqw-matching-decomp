# File: src/mcrx_simplifier.py
"""
MCRX cascade simplification using the complete theoretical framework.
Implements the general optimization algorithm with integrated Boolean function optimization
and quantum circuit synthesis.

CORRECTED VERSION: Fixed pattern interpretation to match multi_crx convention and TeX file examples.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union
from collections import defaultdict, Counter
from qiskit import QuantumCircuit

from .pattern_analyzer import ControlPatternAnalyzer, PatternRelationship, BooleanFunction


class MCRXGateInfo:
    """Information about a single MCRX gate."""
    
    def __init__(self, theta: float, target: int, ctrl_qubits: List[int], ctrl_state: str):
        self.theta = theta
        self.target = target
        self.ctrl_qubits = ctrl_qubits
        self.ctrl_state = ctrl_state
        self.pattern = ctrl_state
    
    def __repr__(self):
        return f"MCRXGateInfo(θ={self.theta:.3f}, target={self.target}, pattern='{self.pattern}')"


class QuantumCircuitSynthesizer:
    """
    Quantum circuit synthesis for optimized Boolean functions.
    
    CORRECTED VERSION: Fixed variable-to-qubit mapping to match TeX file examples.
    """
    
    def __init__(self):
        pass
    
    def synthesize_circuit(self, optimization_result: Dict, base_angle: float, 
                          ctrl_qubits: List[int], target: int, n_qubits: int) -> QuantumCircuit:
        """
        Synthesize quantum circuit for optimized Boolean function.
        
        Args:
            optimization_result: Result from Boolean function optimization
            base_angle: Base rotation angle
            ctrl_qubits: Control qubit indices (ordered: [x0_qubit, x1_qubit, x2_qubit, ...])
            target: Target qubit index
            n_qubits: Total number of qubits in circuit
            
        Returns:
            Optimized quantum circuit implementing the Boolean function
        """
        circuit = QuantumCircuit(n_qubits)
        implementation = optimization_result['implementation']
        
        print(f"      Synthesizing: {implementation['description']}")
        
        if implementation['type'] == 'none':
            # No gate needed
            pass
            
        elif implementation['type'] == 'constant':
            # Unconditional rotation
            circuit.rx(base_angle, target)
            print(f"        → RX({base_angle:.4f}, {target})")
            
        elif implementation['type'] == 'single_control':
            # Single controlled gate
            var_index = implementation['control_qubit']  # This is the variable index (x0, x1, etc.)
            if var_index < len(ctrl_qubits):
                control_qubit = ctrl_qubits[var_index]  # Map to actual qubit
                circuit.crx(base_angle, control_qubit, target)
                print(f"        → CRX({base_angle:.4f}, {control_qubit}, {target}) [variable x{var_index}]")
            else:
                print(f"        → Error: Variable x{var_index} not available in ctrl_qubits {ctrl_qubits}")
            
        elif implementation['type'] == 'negated_control':
            # Negated control: X + CRX + X
            var_index = implementation['control_qubit']
            if var_index < len(ctrl_qubits):
                control_qubit = ctrl_qubits[var_index]
                circuit.x(control_qubit)
                circuit.crx(base_angle, control_qubit, target)
                circuit.x(control_qubit)
                print(f"        → X({control_qubit}) + CRX({base_angle:.4f}, {control_qubit}, {target}) + X({control_qubit}) [variable ¬x{var_index}]")
            
        elif implementation['type'] == 'xor_control':
            # XOR logical form: CNOT + CRX + CNOT
            # For x0 ⊕ x1, we compute x0 XOR x1 and use result as control
            if len(ctrl_qubits) >= 2:
                q0, q1 = ctrl_qubits[0], ctrl_qubits[1]  # x0 and x1
                circuit.cx(q0, q1)  # Compute x0 ⊕ x1 in q1
                circuit.crx(base_angle, q1, target)  # Controlled rotation
                circuit.cx(q0, q1)  # Restore q1
                print(f"        → XOR form: CX({q0},{q1}) + CRX({base_angle:.4f},{q1},{target}) + CX({q0},{q1}) [x0 ⊕ x1]")
            
        elif implementation['type'] == 'xnor_control':
            # XNOR logical form: CNOT + X + CRX + X + CNOT
            if len(ctrl_qubits) >= 2:
                q0, q1 = ctrl_qubits[0], ctrl_qubits[1]
                circuit.cx(q0, q1)  # Compute XOR
                circuit.x(q1)       # NOT to get XNOR
                circuit.crx(base_angle, q1, target)  # Controlled rotation
                circuit.x(q1)       # Restore NOT
                circuit.cx(q0, q1)  # Restore XOR
                print(f"        → XNOR form: CX + X + CRX + X + CX [x0 ⊙ x1]")
            
        elif implementation['type'] == 'complex_overlap':
            # Complex overlapping form: x0 ∧ (x1 ⊕ x2) 
            if len(ctrl_qubits) >= 3:
                expr = optimization_result['simplified_expression']
                if 'x0 ∧ (x1 ⊕ x2)' in expr:
                    q0, q1, q2 = ctrl_qubits[0], ctrl_qubits[1], ctrl_qubits[2]
                    circuit.cx(q1, q2)  # Compute x1 ⊕ x2 in q2
                    from .misc import multi_crx
                    mcrx_gate = multi_crx(base_angle, '11')
                    circuit.append(mcrx_gate, [q0, q2, target])  # x0 ∧ (x1 ⊕ x2)
                    circuit.cx(q1, q2)  # Restore q2
                    print(f"        → Complex form: CX({q1},{q2}) + MCRX('11',[{q0},{q2},{target}]) + CX({q1},{q2}) [x0 ∧ (x1 ⊕ x2)]")
            
        elif implementation['type'] == 'multi_control':
            # Multi-controlled gate
            from .misc import multi_crx
            control_pattern = '1' * len(ctrl_qubits)  # All-ones control
            mcrx_gate = multi_crx(base_angle, control_pattern)
            circuit.append(mcrx_gate, ctrl_qubits + [target])
            print(f"        → Multi-control: MCRX({base_angle:.4f}, '{control_pattern}') on {ctrl_qubits + [target]}")
            
        elif implementation['type'] == 'multi_term':
            # Multi-term OR implementation (general case)
            circuit = self._synthesize_multi_term(optimization_result, base_angle, ctrl_qubits, target, n_qubits)
            print(f"        → Multi-term OR implementation")
            
        else:
            # General case - implement each state separately
            circuit = self._synthesize_general_case(optimization_result, base_angle, ctrl_qubits, target, n_qubits)
            print(f"        → General case implementation")
        
        return circuit
    
    def _synthesize_multi_term(self, optimization_result: Dict, base_angle: float,
                              ctrl_qubits: List[int], target: int, n_qubits: int) -> QuantumCircuit:
        """Synthesize circuit for multi-term Boolean expressions."""
        circuit = QuantumCircuit(n_qubits)
        
        # This is a placeholder for more sophisticated multi-term synthesis
        # In practice, this would implement each term separately and combine them
        # For now, we'll use a simplified approach
        
        expr = optimization_result['simplified_expression']
        if '∨' in expr:
            # Split into terms and implement each
            terms = expr.split(' ∨ ')
            for i, term in enumerate(terms):
                # Create a subcircuit for each term
                # This is simplified - in practice you'd optimize the combination
                term_circuit = self._synthesize_single_term(term.strip(), base_angle / len(terms), ctrl_qubits, target, n_qubits)
                circuit = circuit.compose(term_circuit)
        
        return circuit
    
    def _synthesize_single_term(self, term: str, angle: float, ctrl_qubits: List[int], 
                               target: int, n_qubits: int) -> QuantumCircuit:
        """Synthesize circuit for a single Boolean term."""
        circuit = QuantumCircuit(n_qubits)
        
        # Parse the term and create appropriate controls
        # This is a simplified implementation
        if '∧' in term:
            # AND term - create multi-controlled gate
            # Extract variables and their polarities
            variables = []
            if 'x0' in term:
                variables.append(ctrl_qubits[0])
            if 'x1' in term:
                variables.append(ctrl_qubits[1])
            
            if len(variables) == 1:
                circuit.crx(angle, variables[0], target)
            elif len(variables) == 2:
                from .misc import multi_crx
                mcrx_gate = multi_crx(angle, '11')
                circuit.append(mcrx_gate, variables + [target])
        
        return circuit
    
    def _synthesize_general_case(self, optimization_result: Dict, base_angle: float,
                                ctrl_qubits: List[int], target: int, n_qubits: int) -> QuantumCircuit:
        """Synthesize circuit for general case."""
        circuit = QuantumCircuit(n_qubits)
        
        # Get the Boolean function and implement state by state
        # This is the fallback implementation
        
        # For now, create a single multi-controlled gate as fallback
        if len(ctrl_qubits) > 0:
            from .misc import multi_crx
            control_pattern = '1' * len(ctrl_qubits)
            mcrx_gate = multi_crx(base_angle, control_pattern)
            circuit.append(mcrx_gate, ctrl_qubits + [target])
        
        return circuit


class MCRXCascadeSimplifier:
    """
    MCRX cascade simplification using the complete theoretical framework.
    
    CORRECTED VERSION: Fixed pattern interpretation to match multi_crx convention.
    """
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.pattern_analyzer = ControlPatternAnalyzer()
        self.circuit_synthesizer = QuantumCircuitSynthesizer()
    
    def simplify(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Main simplification method implementing the complete theoretical framework.
        
        Algorithm from theory:
        1. Extract MCRX gates and group by target qubit and angle
        2. For each group, apply the general optimization algorithm:
           a. Initialize rotation coefficient array r[2^n] = 0
           b. For each computational basis state, count satisfied patterns
           c. Group states by rotation coefficient
           d. Optimize Boolean function for each coefficient group
           e. Synthesize optimal quantum circuit
        3. Combine optimized subcircuits
        """
        print("\n" + "="*80)
        print("MCRX CASCADE SIMPLIFICATION - CORRECTED THEORETICAL FRAMEWORK")
        print("="*80)
        
        # Extract MCRX gates from circuit
        mcrx_gates = self._extract_mcrx_gates(circuit)
        
        if not mcrx_gates:
            print("No MCRX gates found - returning original circuit")
            return circuit.copy()
        
        print(f"Extracted {len(mcrx_gates)} MCRX gates")
        
        # Group by target qubit and angle
        groups = self._group_by_target_and_angle(mcrx_gates)
        print(f"Organized into {len(groups)} optimization groups")
        
        # Create optimized circuit
        optimized_circuit = QuantumCircuit(circuit.num_qubits)
        
        # Process each group using the general algorithm
        total_original_gates = 0
        total_optimized_gates = 0
        
        for group_id, ((target, angle), gates) in enumerate(groups.items()):
            print(f"\n--- Group {group_id + 1}: Target={target}, Angle={angle:.4f} ---")
            
            if len(gates) == 1:
                print("Single gate - no optimization possible")
                self._add_single_gate(optimized_circuit, gates[0])
                total_original_gates += 1
                total_optimized_gates += 1
            else:
                print(f"Optimizing {len(gates)} gates with patterns: {[g.pattern for g in gates]}")
                optimized_subcircuit, optimization_stats = self._apply_general_algorithm(gates)
                optimized_circuit = optimized_circuit.compose(optimized_subcircuit)
                
                total_original_gates += len(gates)
                total_optimized_gates += optimization_stats['gates_created']
        
        print(f"\n--- Optimization Summary ---")
        print(f"Original gates: {total_original_gates}")
        print(f"Optimized gates: {total_optimized_gates}")
        if total_original_gates > 0:
            reduction_pct = ((total_original_gates - total_optimized_gates) / total_original_gates) * 100
            print(f"Gate reduction: {total_original_gates - total_optimized_gates} ({reduction_pct:.1f}%)")
        
        print("="*80)
        return optimized_circuit
    
    def _extract_mcrx_gates(self, circuit: QuantumCircuit) -> List[MCRXGateInfo]:
        """Extract MCRX gates from quantum circuit with comprehensive pattern decoding."""
        mcrx_gates = []
        
        print("\n--- MCRX Gate Extraction ---")
        
        for i, instruction in enumerate(circuit.data):
            op_name = instruction.operation.name.lower()
            
            # Recognize MCRX gate patterns
            if any(pattern in op_name for pattern in ['mcrx', 'multi_crx', 'ccrx', 'controlled_rx']):
                print(f"Processing gate {i}: {op_name}")
                
                # Extract gate information
                qubits = [circuit.find_bit(q).index for q in instruction.qubits]
                target = qubits[-1]
                ctrl_qubits = qubits[:-1]
                
                # Get rotation angle
                if hasattr(instruction.operation, 'params') and instruction.operation.params:
                    theta = float(instruction.operation.params[0])
                else:
                    theta = np.pi
                
                # Decode control state pattern
                ctrl_state = self._decode_control_pattern(instruction, op_name, len(ctrl_qubits))
                
                gate_info = MCRXGateInfo(theta, target, ctrl_qubits, ctrl_state)
                mcrx_gates.append(gate_info)
                
                print(f"  → {gate_info}")
        
        return mcrx_gates
    
    def _decode_control_pattern(self, instruction, op_name: str, n_ctrl: int) -> str:
        """
        Decode control pattern from gate instruction.
        
        CORRECTED VERSION: Handle multi_crx pattern interpretation correctly.
        """
        
        # Method 1: Check ctrl_state attribute (most reliable)
        if hasattr(instruction.operation, 'ctrl_state'):
            if isinstance(instruction.operation.ctrl_state, str):
                return instruction.operation.ctrl_state
            else:
                ctrl_state_int = instruction.operation.ctrl_state
                # Direct binary conversion
                pattern = format(ctrl_state_int, f'0{n_ctrl}b')
                print(f"    Decoded ctrl_state {ctrl_state_int} → '{pattern}' (binary)")
                return pattern
        
        # Method 2: Parse from gate name
        elif 'ccrx_o' in op_name:
            parts = op_name.split('_o')
            if len(parts) > 1:
                try:
                    pattern_num = int(parts[1])
                    pattern = format(pattern_num, f'0{n_ctrl}b')
                    print(f"    Decoded gate name {op_name} → '{pattern}' (binary)")
                    return pattern
                except ValueError:
                    pass
        
        # Method 3: Check for other naming patterns
        elif 'rx_' in op_name:
            # Look for patterns like 'rx_10', 'rx_01', etc.
            parts = op_name.split('_')
            for part in parts:
                if len(part) == n_ctrl and all(c in '01' for c in part):
                    print(f"    Decoded from name pattern: '{part}'")
                    return part
        
        # Default: all-ones pattern
        print(f"    Using default all-ones pattern for {op_name}")
        return '1' * n_ctrl
    
    def _group_by_target_and_angle(self, gates: List[MCRXGateInfo]) -> Dict[Tuple[int, float], List[MCRXGateInfo]]:
        """Group gates by target qubit and rotation angle."""
        groups = defaultdict(list)
        
        for gate in gates:
            # Round angle to handle floating point precision
            rounded_angle = round(gate.theta / self.tolerance) * self.tolerance
            key = (gate.target, rounded_angle)
            groups[key].append(gate)
        
        return dict(groups)
    
    def _apply_general_algorithm(self, gates: List[MCRXGateInfo]) -> Tuple[QuantumCircuit, Dict]:
        """
        Apply the general optimization algorithm from theoretical framework.
        
        CORRECTED VERSION: Fixed pattern matching to use direct conversion.
        """
        if not gates:
            return QuantumCircuit(0), {'gates_created': 0}
        
        # Get basic information
        first_gate = gates[0]
        base_theta = first_gate.theta
        target = first_gate.target
        ctrl_qubits = first_gate.ctrl_qubits
        n_ctrl = len(ctrl_qubits)
        n_qubits = max(max([gate.target] + gate.ctrl_qubits) for gate in gates) + 1
        
        print(f"  Applying General Algorithm:")
        print(f"    Target: {target}, Controls: {ctrl_qubits}")
        print(f"    Base angle: {base_theta:.4f}")
        print(f"    Patterns: {[g.pattern for g in gates]}")
        
        # Step 1: Initialize rotation coefficient array r[2^n] = 0
        n_states = 2 ** n_ctrl
        rotation_coefficients = [0] * n_states
        
        # Step 2: For each computational basis state, count pattern satisfaction
        print(f"    Step 2: Analyzing {n_states} computational basis states")
        
        for state_int in range(n_states):
            state_pattern = format(state_int, f'0{n_ctrl}b')
            
            for gate in gates:
                # CORRECTED: Direct pattern matching (no bit reversal)
                if state_pattern == gate.pattern:
                    rotation_coefficients[state_int] += 1
            
            if rotation_coefficients[state_int] > 0:
                print(f"      State |{state_pattern}⟩ (#{state_int}): {rotation_coefficients[state_int]} rotation(s)")
        
        # Step 3: Group states by rotation coefficient
        print(f"    Step 3: Grouping states by rotation coefficient")
        coefficient_groups = defaultdict(list)
        for state_int in range(n_states):
            if rotation_coefficients[state_int] > 0:
                coefficient_groups[rotation_coefficients[state_int]].append(state_int)
        
        print(f"      Coefficient groups: {dict(coefficient_groups)}")
        
        # Step 4: Optimize and synthesize for each coefficient group
        print(f"    Step 4: Boolean optimization and circuit synthesis")
        circuit = QuantumCircuit(n_qubits)
        gates_created = 0
        
        for coefficient, states in coefficient_groups.items():
            if coefficient > 0:
                print(f"      Processing coefficient {coefficient}:")
                print(f"        States: {[format(s, f'0{n_ctrl}b') for s in states]} (integers: {states})")
                
                # Calculate rotation angle for this group
                rotation_angle = coefficient * base_theta
                print(f"        Rotation angle: {coefficient} × {base_theta:.4f} = {rotation_angle:.4f}")
                
                # Step 4a: Optimize Boolean function for these states
                # Create a direct Boolean function from states (no pattern conversion)
                from .pattern_analyzer import BooleanFunction
                boolean_func = BooleanFunction(states, n_ctrl)
                optimization_result = self.pattern_analyzer.boolean_optimizer.optimize_function(boolean_func)
                
                print(f"        Boolean function: {optimization_result['simplified_expression']}")
                print(f"        Optimization method: {optimization_result['simplification_method']}")
                print(f"        Implementation: {optimization_result['implementation']['description']}")
                print(f"        Quantum cost: {optimization_result['quantum_cost']}")
                
                # Step 4b: Synthesize optimal quantum circuit
                optimized_gate_circuit = self.circuit_synthesizer.synthesize_circuit(
                    optimization_result, rotation_angle, ctrl_qubits, target, n_qubits
                )
                
                circuit = circuit.compose(optimized_gate_circuit)
                gates_created += len(optimized_gate_circuit.data)
        
        optimization_stats = {
            'gates_created': gates_created,
            'coefficient_groups': len(coefficient_groups),
            'total_states_processed': sum(len(states) for states in coefficient_groups.values())
        }
        
        return circuit, optimization_stats
    
    def _add_single_gate(self, circuit: QuantumCircuit, gate: MCRXGateInfo):
        """Add a single MCRX gate to the circuit."""
        from .misc import multi_crx
        mcrx_gate = multi_crx(gate.theta, gate.pattern)
        circuit.append(mcrx_gate, gate.ctrl_qubits + [gate.target])
    
    def analyze_optimization_potential(self, circuit: QuantumCircuit) -> Dict[str, any]:
        """
        Comprehensive analysis of optimization potential using theoretical framework.
        
        Returns detailed analysis including Boolean function optimization potential,
        estimated gate reductions, and quantum cost improvements.
        """
        print("\n--- Optimization Potential Analysis ---")
        
        mcrx_gates = self._extract_mcrx_gates(circuit)
        
        if not mcrx_gates:
            return {
                'total_mcrx_gates': 0,
                'optimizable_groups': 0,
                'potential_gate_reduction': 0,
                'potential_quantum_cost_reduction': 0,
                'optimization_potential': 'NONE',
                'group_details': [],
                'notes': 'No MCRX gates found in circuit'
            }
        
        groups = self._group_by_target_and_angle(mcrx_gates)
        
        analysis = {
            'total_mcrx_gates': len(mcrx_gates),
            'optimizable_groups': 0,
            'potential_gate_reduction': 0,
            'potential_quantum_cost_reduction': 0,
            'group_details': [],
            'optimization_potential': 'MINIMAL',
            'theoretical_framework_analysis': True
        }
        
        for (target, angle), group_gates in groups.items():
            if len(group_gates) > 1:
                analysis['optimizable_groups'] += 1
                
                print(f"  Analyzing group: target={target}, gates={len(group_gates)}")
                
                # Apply theoretical analysis
                patterns = [g.pattern for g in group_gates]
                optimization_analysis = self.pattern_analyzer.analyze_patterns_for_optimization(patterns)
                optimization_result = optimization_analysis['optimization_result']
                
                # Estimate gate reduction
                original_quantum_cost = len(group_gates) * 10  # Rough estimate for multi-controlled gates
                optimized_quantum_cost = optimization_result['quantum_cost']
                potential_cost_reduction = max(0, original_quantum_cost - optimized_quantum_cost)
                
                # Estimate gate count reduction based on Boolean optimization
                complexity_reduction = optimization_result['complexity_reduction']
                potential_gate_reduction = max(0, len(group_gates) - 1)
                
                analysis['potential_gate_reduction'] += potential_gate_reduction
                analysis['potential_quantum_cost_reduction'] += potential_cost_reduction
                
                group_detail = {
                    'target': target,
                    'angle': angle,
                    'gate_count': len(group_gates),
                    'patterns': patterns,
                    'boolean_function': optimization_result['simplified_expression'],
                    'optimization_method': optimization_result['simplification_method'],
                    'implementation_type': optimization_result['implementation']['type'],
                    'estimated_gate_reduction': potential_gate_reduction,
                    'estimated_cost_reduction': potential_cost_reduction,
                    'optimization_quality': optimization_result['optimization_quality']
                }
                
                analysis['group_details'].append(group_detail)
                
                print(f"    Boolean function: {optimization_result['simplified_expression']}")
                print(f"    Implementation: {optimization_result['implementation']['type']}")
                print(f"    Estimated reduction: {potential_gate_reduction} gates, {potential_cost_reduction} quantum cost")
        
        # Overall assessment based on theoretical analysis
        total_reduction_ratio = analysis['potential_gate_reduction'] / max(analysis['total_mcrx_gates'], 1)
        
        if total_reduction_ratio >= 0.6:
            analysis['optimization_potential'] = 'HIGH'
        elif total_reduction_ratio >= 0.3:
            analysis['optimization_potential'] = 'MEDIUM'
        elif total_reduction_ratio > 0:
            analysis['optimization_potential'] = 'LOW'
        
        analysis['notes'] = (
            "Analysis based on complete theoretical framework including Boolean function optimization. "
            "Actual results depend on specific circuit structure and pattern complexity."
        )
        
        return analysis
    
    def get_optimization_statistics(self, original_circuit: QuantumCircuit, 
                                   optimized_circuit: QuantumCircuit) -> Dict[str, any]:
        """
        Compute detailed optimization statistics comparing original and optimized circuits.
        """
        def count_gate_types(circuit):
            gate_counts = defaultdict(int)
            for instruction in circuit.data:
                gate_name = instruction.operation.name.lower()
                if any(pattern in gate_name for pattern in ['mcrx', 'multi_crx', 'ccrx']):
                    gate_counts['MCRX'] += 1
                elif gate_name in ['cx', 'cnot']:
                    gate_counts['CX'] += 1
                elif gate_name in ['u3', 'u', 'rx', 'ry', 'rz'] or 'rx' in gate_name:
                    gate_counts['U3'] += 1
                elif gate_name in ['x', 'y', 'z']:
                    gate_counts['Pauli'] += 1
                else:
                    gate_counts['Other'] += 1
            return dict(gate_counts)
        
        orig_gates = count_gate_types(original_circuit)
        opt_gates = count_gate_types(optimized_circuit)
        
        statistics = {
            'original_gates': orig_gates,
            'optimized_gates': opt_gates,
            'original_depth': original_circuit.depth(),
            'optimized_depth': optimized_circuit.depth(),
            'reductions': {},
            'improvement_ratios': {}
        }
        
        # Calculate reductions
        all_gate_types = set(orig_gates.keys()) | set(opt_gates.keys())
        for gate_type in all_gate_types:
            orig_count = orig_gates.get(gate_type, 0)
            opt_count = opt_gates.get(gate_type, 0)
            reduction = orig_count - opt_count
            statistics['reductions'][gate_type] = {
                'absolute': reduction,
                'percentage': (reduction / max(orig_count, 1)) * 100
            }
        
        # Overall improvements
        orig_total = sum(orig_gates.values())
        opt_total = sum(opt_gates.values())
        
        statistics['total_gate_reduction'] = {
            'absolute': orig_total - opt_total,
            'percentage': ((orig_total - opt_total) / max(orig_total, 1)) * 100
        }
        
        statistics['depth_reduction'] = {
            'absolute': statistics['original_depth'] - statistics['optimized_depth'],
            'percentage': ((statistics['original_depth'] - statistics['optimized_depth']) / 
                          max(statistics['original_depth'], 1)) * 100
        }
        
        return statistics