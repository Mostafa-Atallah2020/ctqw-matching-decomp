"""
MCRX Simplifier - Fixed Pattern Reading Implementation

CRITICAL FIXES:
1. Read pattern strings LEFT-TO-RIGHT (position 0 = qubit 0, position 1 = qubit 1)
2. Label qubits TOP-TO-BOTTOM (q_0 at top, q_1 below, etc.)
3. Control state extraction from Qiskit gates
4. Proper subset pattern handling (don't pad with '0's)
5. CX trick implementation with qubit mapping
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict
import itertools
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate
import sympy as sp
from sympy.logic import simplify_logic
from sympy.logic.boolalg import And, Or, Not, Xor
from .misc import multi_crx


class ControlPattern:
    """Represents a control pattern for MCRX gates."""

    def __init__(self, pattern: str, coefficient: int = 1, active_qubits: List[int] = None):
        self.pattern = pattern
        self.coefficient = coefficient
        self.n_controls = len(pattern)
        self.active_qubits = active_qubits or list(range(self.n_controls))

    def satisfies_state(self, state: int, total_qubits: int) -> bool:
        """Check if computational basis state satisfies this control pattern."""
        state_binary = format(state, f"0{total_qubits}b")

        # Check only active qubits
        for i, qubit_idx in enumerate(self.active_qubits):
            required_bit = self.pattern[i]
            actual_bit = state_binary[qubit_idx]
            if required_bit != actual_bit:
                return False

        return True

    def to_full_pattern(self, total_qubits: int, default_value: str = "X") -> str:
        """Convert to full pattern string for all qubits."""
        full_pattern = [default_value] * total_qubits

        for i, qubit_idx in enumerate(self.active_qubits):
            if qubit_idx < total_qubits:
                full_pattern[qubit_idx] = self.pattern[i]

        return "".join(full_pattern)

    def __repr__(self):
        return f"ControlPattern('{self.pattern}', coeff={self.coefficient}, qubits={self.active_qubits})"

    def __eq__(self, other):
        return (
            isinstance(other, ControlPattern)
            and self.pattern == other.pattern
            and self.active_qubits == other.active_qubits
        )


class BooleanExpressionAnalyzer:
    """Analyzes Boolean expressions for control patterns."""

    def __init__(self, n_controls: int):
        self.n_controls = n_controls
        self.symbols = [sp.Symbol(f"x{i}") for i in range(n_controls)]

    def pattern_to_boolean_expr(self, pattern: ControlPattern) -> sp.Basic:
        """Convert control pattern to Boolean expression."""
        terms = []

        for i, qubit_idx in enumerate(pattern.active_qubits):
            if qubit_idx < len(self.symbols):
                bit = pattern.pattern[i]
                if bit == "1":
                    terms.append(self.symbols[qubit_idx])
                elif bit == "0":
                    terms.append(sp.Not(self.symbols[qubit_idx]))

        if not terms:
            return sp.true
        elif len(terms) == 1:
            return terms[0]
        else:
            return sp.And(*terms)

    def patterns_to_combined_expr(self, patterns: List[ControlPattern]) -> sp.Basic:
        """Convert list of patterns to combined Boolean OR expression."""
        if not patterns:
            return sp.false

        pattern_exprs = []
        for pattern in patterns:
            expr = self.pattern_to_boolean_expr(pattern)
            # Add multiple copies if coefficient > 1
            for _ in range(pattern.coefficient):
                pattern_exprs.append(expr)

        if len(pattern_exprs) == 1:
            return pattern_exprs[0]
        else:
            return sp.Or(*pattern_exprs)

    def simplify_expression(self, expr: sp.Basic) -> sp.Basic:
        """Simplify Boolean expression."""
        try:
            return simplify_logic(expr)
        except Exception:
            return expr

    def analyze_expression(self, expr: sp.Basic) -> Dict[str, Any]:
        """Analyze Boolean expression complexity and properties."""
        expr_str = str(expr)

        # Check for specific patterns
        is_single_var = expr in self.symbols
        is_negated_var = any(expr == sp.Not(s) for s in self.symbols)
        is_constant = expr in [sp.true, sp.false]

        # Check if expression is equivalent to original (no simplification)
        is_simplified = len(expr_str) < 50  # Heuristic

        return {
            "expression": expr,
            "simplified_str": str(expr),
            "is_constant": is_constant,
            "is_single_variable": is_single_var,
            "is_negated_variable": is_negated_var,
            "is_simplified": is_simplified,
            "num_variables": len([s for s in self.symbols if s in expr.free_symbols]),
            "total_operations": expr_str.count("&") + expr_str.count("|") + expr_str.count("~"),
        }


class XORPatternDetector:
    """Detects XOR patterns for CX trick optimization."""

    @staticmethod
    def detect_xor_pairs(
        patterns: List[ControlPattern],
    ) -> List[Tuple[ControlPattern, ControlPattern, List[int]]]:
        """
        Detect XOR pattern pairs with the differing qubit positions.

        Returns:
            List of (pattern1, pattern2, differing_positions)
        """
        xor_pairs = []

        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i + 1 :], i + 1):
                differing_positions = XORPatternDetector._get_xor_positions(pattern1, pattern2)
                if differing_positions:
                    xor_pairs.append((pattern1, pattern2, differing_positions))

        return xor_pairs

    @staticmethod
    def _get_xor_positions(
        pattern1: ControlPattern, pattern2: ControlPattern
    ) -> Optional[List[int]]:
        """
        Get positions where patterns differ for XOR detection.

        Returns:
            List of qubit indices where patterns differ, or None if not XOR
        """
        # Must have same active qubits
        if pattern1.active_qubits != pattern2.active_qubits:
            return None

        # Find positions where bits differ
        differing_positions = []
        common_positions = []

        for i, qubit_idx in enumerate(pattern1.active_qubits):
            bit1 = pattern1.pattern[i]
            bit2 = pattern2.pattern[i]

            if bit1 != bit2:
                differing_positions.append(qubit_idx)
            else:
                common_positions.append(qubit_idx)

        # XOR pattern: exactly 2 positions differ, others are the same
        if len(differing_positions) == 2:
            return differing_positions

        return None


class MCRXCascadeSimplifier:
    """
     MCRX cascade simplifier with correct pattern reading.
    """

    def __init__(self, tolerance: float = 1e-10, verbose: bool = False):
        self.tolerance = tolerance
        self.verbose = verbose

    def simplify(self, circuit: QuantumCircuit) -> Tuple[QuantumCircuit, Dict[str, Any]]:
        """Main simplification method with  pattern extraction."""
        if self.verbose:
            print("🔬 Starting  pattern reading MCRX simplification...")

        # Step 1: Validate and extract circuit information
        target_qubit, rotation_angle, all_ctrl_qubits = self._validate_circuit(circuit)
        n_controls = len(all_ctrl_qubits)

        if self.verbose:
            print(
                f"✓ Circuit validated: target={target_qubit}, angle={rotation_angle:.4f}, controls={n_controls}"
            )

        # Step 2:  pattern extraction
        patterns = self._extract_patterns(
            circuit, target_qubit, rotation_angle, all_ctrl_qubits
        )

        if self.verbose:
            print(f"✓ Extracted {len(patterns)} patterns:")
            for p in patterns:
                print(f"  {p}")

        # Step 3: Boolean expression analysis
        bool_analyzer = BooleanExpressionAnalyzer(n_controls)
        original_expr = bool_analyzer.patterns_to_combined_expr(patterns)
        simplified_expr = bool_analyzer.simplify_expression(original_expr)
        expr_analysis = bool_analyzer.analyze_expression(simplified_expr)

        if self.verbose:
            print(f"✓ Original Boolean: {original_expr}")
            print(f"✓ Simplified Boolean: {simplified_expr}")

        # Step 4: XOR detection
        xor_detector = XORPatternDetector()
        xor_pairs = xor_detector.detect_xor_pairs(patterns)

        if self.verbose:
            print(f"✓ XOR pairs detected: {len(xor_pairs)}")
            for pattern1, pattern2, diff_pos in xor_pairs:
                print(f"  XOR: {pattern1.pattern} ⊕ {pattern2.pattern} at positions {diff_pos}")

        # Step 5: Check if optimization is valid
        has_different_control_sets = self._has_different_control_sets(patterns)

        if self.verbose:
            print(f"✓ Different control sets: {has_different_control_sets}")

        # Step 6: Apply optimization (FIXED for subset patterns)
        if has_different_control_sets:
            # CRITICAL: Cannot optimize when patterns have different control sets
            optimized_circuit = circuit.copy()
            optimization_method = "no_optimization_different_controls"
        elif xor_pairs and self._should_apply_cx_trick(xor_pairs[0], expr_analysis):
            optimized_circuit = self._apply_cx_trick(
                xor_pairs[0], target_qubit, all_ctrl_qubits, circuit.num_qubits, rotation_angle
            )
            optimization_method = "CX_trick"
        elif expr_analysis["is_single_variable"] or expr_analysis["is_negated_variable"]:
            # Only apply if all patterns have same control set
            if self._all_patterns_same_control_set(patterns):
                optimized_circuit = self._apply_single_control(
                    simplified_expr,
                    target_qubit,
                    all_ctrl_qubits,
                    circuit.num_qubits,
                    rotation_angle,
                    bool_analyzer,
                )
                optimization_method = "single_control"
            else:
                optimized_circuit = circuit.copy()
                optimization_method = "no_optimization_mixed_controls"
        elif len(patterns) == 1:  # Only one unique pattern (may have coefficient > 1)
            optimized_circuit = self._apply_identical_pattern(
                patterns[0], target_qubit, all_ctrl_qubits, circuit.num_qubits, rotation_angle
            )
            optimization_method = "identical_patterns"
        else:
            # Check if Boolean expression actually simplifies
            if str(original_expr) == str(simplified_expr):
                # No simplification possible
                optimized_circuit = circuit.copy()
                optimization_method = "no_optimization"
            else:
                # Complex simplification - only if same control sets
                if self._all_patterns_same_control_set(patterns):
                    optimized_circuit = circuit.copy()  # For now, don't optimize complex cases
                    optimization_method = "complex_case"
                else:
                    optimized_circuit = circuit.copy()
                    optimization_method = "no_optimization_mixed_controls"

        # Step 7: Compile information
        optimization_info = {
            "original_patterns": [f"{p.pattern} (qubits: {p.active_qubits})" for p in patterns],
            "original_boolean_expr": str(original_expr),
            "simplified_boolean_expr": str(simplified_expr),
            "expression_analysis": expr_analysis,
            "xor_pairs": [(p1.pattern, p2.pattern, diff_pos) for p1, p2, diff_pos in xor_pairs],
            "has_different_control_sets": has_different_control_sets,
            "gate_reduction": len(circuit.data) - len(optimized_circuit.data),
            "optimization_method": optimization_method,
            "uses_cnot_tricks": optimization_method == "CX_trick",
            "optimization_blocked_reason": self._get_optimization_blocked_reason(
                optimization_method, has_different_control_sets
            ),
        }

        if self.verbose:
            print(f"✓ Optimization: {optimization_method}")
            if optimization_info["optimization_blocked_reason"]:
                print(f"✓ Blocked reason: {optimization_info['optimization_blocked_reason']}")
            print(f"✓ Gate reduction: {optimization_info['gate_reduction']}")

        return optimized_circuit, optimization_info

    def _validate_circuit(self, circuit: QuantumCircuit) -> Tuple[int, float, List[int]]:
        """Validate circuit and extract information."""
        if circuit.num_qubits == 0 or len(circuit.data) == 0:
            raise ValueError("Circuit must have qubits and gates")

        # Find all control qubits and target qubit
        all_qubits = set()
        target_candidates = set()

        for instruction in circuit.data:
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            all_qubits.update(qubits)
            target_candidates.add(qubits[-1])  # Last qubit is target

        # Validate all gates target the same qubit
        if len(target_candidates) != 1:
            raise ValueError("All gates must target the same qubit")

        target_qubit = target_candidates.pop()
        ctrl_qubits = sorted([q for q in all_qubits if q != target_qubit])

        # Validate angles
        first_angle = float(circuit.data[0].operation.params[0])
        for i, instruction in enumerate(circuit.data):
            if not hasattr(instruction.operation, "params") or not instruction.operation.params:
                raise ValueError(f"Gate {i+1}: Missing rotation angle")

            angle = float(instruction.operation.params[0])
            if abs(angle - first_angle) > self.tolerance:
                raise ValueError(f"Gate {i+1}: All gates must have same angle")

        return target_qubit, first_angle, ctrl_qubits

    def _extract_patterns(
        self,
        circuit: QuantumCircuit,
        target_qubit: int,
        rotation_angle: float,
        all_ctrl_qubits: List[int],
    ) -> List[ControlPattern]:
        """Extract patterns with correct left-to-right reading."""
        pattern_data = defaultdict(int)

        for instruction in circuit.data:
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            gate_ctrl_qubits = qubits[:-1]  # All except target

            # Extract pattern correctly
            pattern_str, active_qubits = self._extract_pattern_string(
                instruction, gate_ctrl_qubits
            )

            # Create key for grouping
            key = (pattern_str, tuple(active_qubits))
            pattern_data[key] += 1

        # Create ControlPattern objects
        patterns = []
        for (pattern_str, active_qubits_tuple), count in pattern_data.items():
            patterns.append(ControlPattern(pattern_str, count, list(active_qubits_tuple)))

        return patterns

    def _extract_pattern_string(
        self, instruction, gate_ctrl_qubits: List[int]
    ) -> Tuple[str, List[int]]:
        """Extract pattern string reading LEFT-TO-RIGHT."""
        op_name = instruction.operation.name
        n_gate_controls = len(gate_ctrl_qubits)

        # Default pattern (all controls must be |1⟩)
        default_pattern = "1" * n_gate_controls

        # Try to extract ctrl_state
        ctrl_state_str = default_pattern

        if hasattr(instruction.operation, "ctrl_state"):
            ctrl_state = instruction.operation.ctrl_state
            if isinstance(ctrl_state, str):
                # Reverse Qiskit's ctrl_state to get our left-to-right convention
                ctrl_state_str = ctrl_state[::-1]
            elif isinstance(ctrl_state, int):
                # Convert int to binary and reverse
                ctrl_state_str = format(ctrl_state, f"0{n_gate_controls}b")[::-1]

        # Parse operation name if needed
        if ctrl_state_str == default_pattern:
            import re

            pattern_match = re.search(r"_o(\d+)", op_name)
            if pattern_match:
                state_number = int(pattern_match.group(1))
                ctrl_state_str = format(state_number, f"0{n_gate_controls}b")[::-1]

        # Return pattern and active qubits in TOP-TO-BOTTOM order
        active_qubits = sorted(gate_ctrl_qubits)  # Sort to ensure top-to-bottom

        # Map ctrl_state_str to the sorted qubit order
        if gate_ctrl_qubits != active_qubits:
            # Reorder pattern to match sorted qubits
            reordered_pattern = ["0"] * len(active_qubits)
            for i, qubit in enumerate(gate_ctrl_qubits):
                sorted_index = active_qubits.index(qubit)
                reordered_pattern[sorted_index] = ctrl_state_str[i]
            ctrl_state_str = "".join(reordered_pattern)

        return ctrl_state_str, active_qubits

    def _get_optimization_blocked_reason(
        self, optimization_method: str, has_different_control_sets: bool
    ) -> Optional[str]:
        """Get reason why optimization was blocked."""
        if optimization_method == "no_optimization_different_controls":
            return "Patterns have different control sets - rotations are cumulative, cannot use Boolean simplification"
        elif optimization_method == "no_optimization_mixed_controls":
            return "Mixed control sets prevent safe optimization"
        elif optimization_method == "no_optimization":
            return "Boolean expression does not simplify"
        return None

    def _has_different_control_sets(self, patterns: List[ControlPattern]) -> bool:
        """
        Check if patterns have different control sets.

        CRITICAL: When patterns control different qubits, we cannot apply
        Boolean simplification because rotations are cumulative.

        Example: ['11'] on qubits [0,1] + ['1'] on qubit [0]
        - Boolean says: (x0 & x1) | x0 = x0
        - But physically: RX(θ) on |11⟩ + RX(θ) on |1X⟩ ≠ RX(θ) on |1X⟩
        """
        if len(patterns) <= 1:
            return False

        # Get all unique control sets
        control_sets = set()
        for pattern in patterns:
            control_sets.add(tuple(sorted(pattern.active_qubits)))

        return len(control_sets) > 1

    def _all_patterns_same_control_set(self, patterns: List[ControlPattern]) -> bool:
        """Check if all patterns have the same control qubit set."""
        return not self._has_different_control_sets(patterns)

    def _should_apply_cx_trick(self, xor_pair_info: Tuple, expr_analysis: Dict) -> bool:
        """Determine if CX trick should be applied."""
        pattern1, pattern2, diff_positions = xor_pair_info

        # Apply CX trick if:
        # 1. We have exactly 2 differing positions
        # 2. The expression doesn't simplify to something simpler
        return (
            len(diff_positions) == 2
            and not expr_analysis["is_single_variable"]
            and not expr_analysis["is_negated_variable"]
        )

    def _apply_cx_trick(
        self,
        xor_pair_info: Tuple,
        target_qubit: int,
        all_ctrl_qubits: List[int],
        n_qubits: int,
        rotation_angle: float,
    ) -> QuantumCircuit:
        """Apply CX trick with correct qubit mapping."""
        pattern1, pattern2, diff_positions = xor_pair_info
        circuit = QuantumCircuit(n_qubits)

        if len(diff_positions) == 2:
            pos1, pos2 = diff_positions

            # Apply CX trick: CX + controlled rotation + CX
            circuit.cx(pos1, pos2)

            # Create control condition for remaining qubits
            # Find common control pattern
            common_controls = []
            for i, qubit_idx in enumerate(pattern1.active_qubits):
                if qubit_idx not in diff_positions:
                    bit = pattern1.pattern[i]
                    common_controls.append((qubit_idx, bit))

            # Apply controlled rotation
            if common_controls:
                # Multi-controlled case
                common_pattern = ""
                common_qubits = []
                for qubit_idx, bit in common_controls:
                    common_qubits.append(qubit_idx)
                    common_pattern += bit

                # Add the second differing qubit as additional control
                common_qubits.append(pos2)
                # Determine required state for pos2 based on XOR logic
                common_pattern += "1"  # XOR requires this to be 1

                gate = multi_crx(rotation_angle, common_pattern)
                circuit.append(gate, common_qubits + [target_qubit])
            else:
                # Simple case: just control on pos2
                circuit.crx(rotation_angle, pos2, target_qubit)

            circuit.cx(pos1, pos2)

        return circuit

    def _apply_single_control(
        self,
        simplified_expr: sp.Basic,
        target_qubit: int,
        all_ctrl_qubits: List[int],
        n_qubits: int,
        rotation_angle: float,
        bool_analyzer: BooleanExpressionAnalyzer,
    ) -> QuantumCircuit:
        """Apply single control optimization."""
        circuit = QuantumCircuit(n_qubits)

        # Find which control qubit
        for i, symbol in enumerate(bool_analyzer.symbols):
            if simplified_expr == symbol:
                # Control qubit i must be |1⟩
                circuit.crx(rotation_angle, all_ctrl_qubits[i], target_qubit)
                return circuit
            elif simplified_expr == sp.Not(symbol):
                # Control qubit i must be |0⟩
                circuit.x(all_ctrl_qubits[i])
                circuit.crx(rotation_angle, all_ctrl_qubits[i], target_qubit)
                circuit.x(all_ctrl_qubits[i])
                return circuit

        return circuit

    def _apply_identical_pattern(
        self,
        pattern: ControlPattern,
        target_qubit: int,
        all_ctrl_qubits: List[int],
        n_qubits: int,
        rotation_angle: float,
    ) -> QuantumCircuit:
        """Apply identical pattern optimization."""
        circuit = QuantumCircuit(n_qubits)

        # Single gate with multiplied angle
        total_angle = rotation_angle * pattern.coefficient
        gate = multi_crx(total_angle, pattern.pattern)
        circuit.append(gate, pattern.active_qubits + [target_qubit])

        return circuit

    def analyze_patterns(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Analyze patterns with FIXED extraction."""
        try:
            target_qubit, rotation_angle, all_ctrl_qubits = self._validate_circuit(circuit)
            patterns = self._extract_patterns(
                circuit, target_qubit, rotation_angle, all_ctrl_qubits
            )

            # Boolean analysis
            bool_analyzer = BooleanExpressionAnalyzer(len(all_ctrl_qubits))
            original_expr = bool_analyzer.patterns_to_combined_expr(patterns)
            simplified_expr = bool_analyzer.simplify_expression(original_expr)
            expr_analysis = bool_analyzer.analyze_expression(simplified_expr)

            # XOR detection
            xor_detector = XORPatternDetector()
            xor_pairs = xor_detector.detect_xor_pairs(patterns)

            # Check for different control sets
            has_different_control_sets = self._has_different_control_sets(patterns)

            return {
                "status": "success",
                "original_patterns": [f"{p.pattern} (qubits: {p.active_qubits})" for p in patterns],
                "pattern_coefficients": {p.pattern: p.coefficient for p in patterns},
                "original_boolean_expr": str(original_expr),
                "simplified_boolean_expr": str(simplified_expr),
                "expression_analysis": expr_analysis,
                "xor_pairs": [(p1.pattern, p2.pattern, diff_pos) for p1, p2, diff_pos in xor_pairs],
                "has_different_control_sets": has_different_control_sets,
                "target_qubit": target_qubit,
                "rotation_angle": rotation_angle,
                "ctrl_qubits": all_ctrl_qubits,
                "can_apply_cx_trick": len(xor_pairs) > 0 and not has_different_control_sets,
                "simplifies": str(original_expr) != str(simplified_expr),
                "can_optimize": not has_different_control_sets,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}


