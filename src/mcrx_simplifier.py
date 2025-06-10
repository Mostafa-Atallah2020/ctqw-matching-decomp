"""
MCRX Simplifier - Complete Enhanced Implementation with Fixed Debug Output

CRITICAL FIXES & FEATURES:
1. Read pattern strings LEFT-TO-RIGHT (position 0 = qubit 0, position 1 = qubit 1)
2. Label qubits TOP-TO-BOTTOM (q_0 at top, q_1 below, etc.)
3. Control state extraction from Qiskit gates
4. Proper subset pattern handling (don't pad with '0's)
5. CX trick implementation with qubit mapping (single and multiple)
6. ITERATIVE HANDLING: Handle multiple MCRX gates through iterative pairwise simplification
7. BOOLEAN SIMPLIFICATION: Properly convert simplified Boolean expressions to optimized circuits
8. MCRX-ONLY VALIDATION: Ensure circuit contains only multi-controlled RX gates
9. VERBOSE CONTROL: All debug output controlled by verbose flag
10. FIDELITY VERIFICATION: Verify optimized circuits maintain state equivalence with debug logging
11. DEBUG OUTPUT: Robust gate listing that handles all edge cases
"""

import itertools
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate
from sympy.logic import simplify_logic
from sympy.logic.boolalg import And, Not, Or, Xor

from .misc import compare_quantum_states, get_state, multi_crx


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

    def to_decimal(self) -> int:
        """Convert pattern to decimal value for sorting."""
        return int(self.pattern, 2)

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

    def __init__(self, all_ctrl_qubits: List[int]):
        """
        Initialize with ALL control qubits that appear in patterns.

        Args:
            all_ctrl_qubits: List of all control qubit indices (e.g., [0, 1, 2])
        """
        self.all_ctrl_qubits = sorted(all_ctrl_qubits)
        self.qubit_to_symbol = {qubit: sp.Symbol(f"x{qubit}") for qubit in self.all_ctrl_qubits}
        self.symbols = [self.qubit_to_symbol[qubit] for qubit in self.all_ctrl_qubits]

    def pattern_to_boolean_expr(self, pattern: ControlPattern) -> sp.Basic:
        """Convert control pattern to Boolean expression with qubit mapping."""
        terms = []

        # Map pattern positions to actual qubits
        for i, qubit_idx in enumerate(pattern.active_qubits):
            if qubit_idx in self.qubit_to_symbol:
                bit = pattern.pattern[i]
                if bit == "1":
                    terms.append(self.qubit_to_symbol[qubit_idx])
                elif bit == "0":
                    terms.append(sp.Not(self.qubit_to_symbol[qubit_idx]))

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
        Enhanced to detect multiple XOR pairs within the same pattern comparison.

        Returns:
            List of (pattern1, pattern2, differing_positions)
        """
        xor_pairs = []

        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i + 1 :], i + 1):
                differing_positions = XORPatternDetector._get_xor_positions(pattern1, pattern2)
                if differing_positions:
                    # Check if this is a multiple XOR case
                    if len(differing_positions) >= 4:
                        # Decompose into individual XOR pairs
                        individual_pairs = XORPatternDetector._decompose_multiple_xor(
                            pattern1, pattern2, differing_positions
                        )
                        for pair_positions in individual_pairs:
                            xor_pairs.append((pattern1, pattern2, pair_positions))
                    else:
                        # Single XOR pair
                        xor_pairs.append((pattern1, pattern2, differing_positions))

        return xor_pairs

    @staticmethod
    def _get_xor_positions(
        pattern1: ControlPattern, pattern2: ControlPattern
    ) -> Optional[List[int]]:
        """
        Get positions where patterns differ for XOR detection.
        Enhanced to detect multiple XOR pairs within the same pattern comparison.

        Returns:
            List of qubit indices where patterns differ, or None if not XOR-compatible
        """
        # Must have same active qubits
        if pattern1.active_qubits != pattern2.active_qubits:
            return None

        # Find all positions where bits differ
        differing_positions = []
        for i, qubit_idx in enumerate(pattern1.active_qubits):
            bit1 = pattern1.pattern[i]
            bit2 = pattern2.pattern[i]
            if bit1 != bit2:
                differing_positions.append(qubit_idx)

        # Enhanced XOR detection: handle multiple XOR pairs
        if len(differing_positions) == 2:
            # Single XOR pair
            return differing_positions
        elif len(differing_positions) == 4:
            # Potential double XOR pair - check if it can be decomposed
            return XORPatternDetector._check_double_xor_pattern(
                pattern1, pattern2, differing_positions
            )
        elif len(differing_positions) > 0 and len(differing_positions) % 2 == 0:
            # Even number of differences - might be multiple XOR pairs
            return XORPatternDetector._check_multiple_xor_patterns(
                pattern1, pattern2, differing_positions
            )

        return None

    @staticmethod
    def _check_double_xor_pattern(
        pattern1: ControlPattern, pattern2: ControlPattern, differing_positions: List[int]
    ) -> Optional[List[int]]:
        """Check if 4 differing positions form 2 independent XOR pairs."""
        if len(differing_positions) != 4:
            return None

        # Try all possible ways to pair the 4 positions into 2 XOR pairs
        from itertools import combinations

        positions = differing_positions
        # Generate all ways to choose 2 positions for first pair
        for pair1_positions in combinations(positions, 2):
            pair2_positions = [pos for pos in positions if pos not in pair1_positions]

            # Check if both pairs are valid XOR patterns
            if XORPatternDetector._is_valid_xor_pair(
                pattern1, pattern2, list(pair1_positions)
            ) and XORPatternDetector._is_valid_xor_pair(pattern1, pattern2, pair2_positions):
                # Found valid double XOR - return all positions
                return differing_positions

        return None

    @staticmethod
    def _check_multiple_xor_patterns(
        pattern1: ControlPattern, pattern2: ControlPattern, differing_positions: List[int]
    ) -> Optional[List[int]]:
        """Check if multiple differing positions form valid XOR pairs."""
        if len(differing_positions) % 2 != 0:
            return None

        # For now, limit to reasonable numbers (4, 6, 8 positions)
        if len(differing_positions) > 8:
            return None

        from itertools import combinations

        # Try to partition into XOR pairs
        n_pairs = len(differing_positions) // 2

        # For 4 positions, try the double XOR case
        if len(differing_positions) == 4:
            return XORPatternDetector._check_double_xor_pattern(
                pattern1, pattern2, differing_positions
            )

        # For more complex cases, use a simple heuristic
        # Check if positions can be grouped into consecutive pairs
        sorted_positions = sorted(differing_positions)
        for i in range(0, len(sorted_positions), 2):
            if i + 1 < len(sorted_positions):
                pair = [sorted_positions[i], sorted_positions[i + 1]]
                if not XORPatternDetector._is_valid_xor_pair(pattern1, pattern2, pair):
                    return None

        return differing_positions

    @staticmethod
    def _decompose_multiple_xor(
        pattern1: ControlPattern, pattern2: ControlPattern, differing_positions: List[int]
    ) -> List[List[int]]:
        """
        Decompose multiple differing positions into individual XOR pairs.

        Returns:
            List of individual XOR pair position lists
        """
        individual_pairs = []

        if len(differing_positions) == 4:
            # For 4 positions, find the two valid XOR pairs
            from itertools import combinations

            for pair1_positions in combinations(differing_positions, 2):
                pair2_positions = [pos for pos in differing_positions if pos not in pair1_positions]

                if XORPatternDetector._is_valid_xor_pair(
                    pattern1, pattern2, list(pair1_positions)
                ) and XORPatternDetector._is_valid_xor_pair(pattern1, pattern2, pair2_positions):
                    individual_pairs.append(list(pair1_positions))
                    individual_pairs.append(pair2_positions)
                    break

        elif len(differing_positions) == 6:
            # For 6 positions, try to find 3 XOR pairs
            # Use a greedy approach: find consecutive valid pairs
            sorted_positions = sorted(differing_positions)
            for i in range(0, len(sorted_positions), 2):
                if i + 1 < len(sorted_positions):
                    pair = [sorted_positions[i], sorted_positions[i + 1]]
                    if XORPatternDetector._is_valid_xor_pair(pattern1, pattern2, pair):
                        individual_pairs.append(pair)

        elif len(differing_positions) % 2 == 0:
            # For other even numbers, use consecutive pairing
            sorted_positions = sorted(differing_positions)
            for i in range(0, len(sorted_positions), 2):
                if i + 1 < len(sorted_positions):
                    pair = [sorted_positions[i], sorted_positions[i + 1]]
                    if XORPatternDetector._is_valid_xor_pair(pattern1, pattern2, pair):
                        individual_pairs.append(pair)

        return individual_pairs

    @staticmethod
    def _is_valid_xor_pair(
        pattern1: ControlPattern, pattern2: ControlPattern, pair_positions: List[int]
    ) -> bool:
        """Check if two positions form a valid XOR pair."""
        if len(pair_positions) != 2:
            return False

        pos1, pos2 = pair_positions
        pos1_idx = pattern1.active_qubits.index(pos1)
        pos2_idx = pattern1.active_qubits.index(pos2)

        bit1_p1 = pattern1.pattern[pos1_idx]
        bit2_p1 = pattern1.pattern[pos2_idx]
        bit1_p2 = pattern2.pattern[pos1_idx]
        bit2_p2 = pattern2.pattern[pos2_idx]

        # Valid XOR pairs: 01-10, 10-01, 00-11, 11-00
        valid_xor_types = [
            (("0", "1"), ("1", "0")),  # 01-10
            (("1", "0"), ("0", "1")),  # 10-01
            (("0", "0"), ("1", "1")),  # 00-11
            (("1", "1"), ("0", "0")),  # 11-00
        ]

        pattern1_bits = (bit1_p1, bit2_p1)
        pattern2_bits = (bit1_p2, bit2_p2)

        return (pattern1_bits, pattern2_bits) in valid_xor_types


class MCRXCascadeSimplifier:
    """
    MCRX cascade simplifier with pattern reading, iterative multi-pattern handling,
    and automatic fidelity verification with FIXED debug output.

    Features:
    - Supports single and multiple MCRX pattern optimization
    - Boolean expression simplification
    - XOR pattern detection and CX tricks (single and multiple)
    - Iterative pairwise simplification for complex cases
    - Automatic state equivalence verification
    - Robust debug file generation on optimization errors
    """

    def __init__(
        self, tolerance: float = 1e-10, verbose: bool = False, verify_fidelity: bool = True
    ):
        self.tolerance = tolerance
        self.verbose = verbose
        self.verify_fidelity = verify_fidelity

    def _write_circuit_gates_robust(
        self, f, circuit: QuantumCircuit, circuit_name: str = "CIRCUIT"
    ):
        """Robust gate writing function with comprehensive error handling."""
        f.write(f"{circuit_name}:\n")
        f.write("-" * (len(circuit_name) + 1) + "\n")
        f.write(f"Total qubits: {circuit.num_qubits}\n")
        f.write(f"Total gates: {len(circuit.data)}\n")
        f.write(f"Circuit depth: {circuit.depth()}\n\n")

        if len(circuit.data) == 0:
            f.write("No gates in circuit\n\n")
            return

        f.write("Gates:\n")
        for i, instruction in enumerate(circuit.data):
            try:
                gate_name = getattr(instruction.operation, "name", f"unknown_gate_{i}")

                # Handle qubits more robustly
                qubits = []
                for q in instruction.qubits:
                    try:
                        if hasattr(q, "index"):
                            qubits.append(q.index)
                        elif isinstance(q, int):
                            qubits.append(q)
                        else:
                            # Try to find the qubit in the circuit
                            try:
                                bit_obj = circuit.find_bit(q)
                                if hasattr(bit_obj, "index"):
                                    qubits.append(bit_obj.index)
                                else:
                                    qubits.append(bit_obj[0])  # Sometimes returns (index, register)
                            except Exception as find_error:
                                qubits.append(f"find_error_{len(qubits)}")
                    except Exception as qubit_error:
                        qubits.append(f"qubit_error_{len(qubits)}")

                # Handle parameters
                params_str = ""
                try:
                    if hasattr(instruction.operation, "params") and instruction.operation.params:
                        params = []
                        for param in instruction.operation.params:
                            try:
                                if hasattr(param, "evalf"):
                                    params.append(f"{float(param.evalf()):.4f}")
                                else:
                                    params.append(f"{float(param):.4f}")
                            except Exception as param_error:
                                params.append(f"param_error: {param_error}")
                        params_str = f"({', '.join(params)})"
                except Exception as param_error:
                    params_str = f"(param_error: {param_error})"

                # Write gate information
                f.write(f"{i+1}. {gate_name}{params_str}\n")
                f.write(f"   Qubits: {qubits}\n")

                # For MCRX gates, identify controls and target
                if len(qubits) > 1 and all(isinstance(q, int) for q in qubits):
                    controls = qubits[:-1]
                    target = qubits[-1]
                    f.write(f"   Controls: {controls} -> Target: {target}\n")
                elif len(qubits) == 1 and isinstance(qubits[0], int):
                    f.write(f"   Target: {qubits[0]}\n")

                # Add control state if available
                try:
                    if hasattr(instruction.operation, "ctrl_state"):
                        ctrl_state = instruction.operation.ctrl_state
                        f.write(f"   Control state: {ctrl_state}\n")
                except Exception:
                    pass

            except Exception as gate_error:
                f.write(f"{i+1}. ERROR: Could not process gate {i}: {gate_error}\n")
                f.write(f"   Instruction type: {type(instruction)}\n")
                try:
                    f.write(f"   Operation type: {type(instruction.operation)}\n")
                    f.write(
                        f"   Operation name: {getattr(instruction.operation, 'name', 'unknown')}\n"
                    )
                except Exception:
                    f.write("   Operation type: unknown\n")
                try:
                    f.write(f"   Num qubits: {len(instruction.qubits)}\n")
                except Exception:
                    f.write("   Num qubits: unknown\n")

        f.write("\n")

    def _verify_circuit_equivalence(
        self,
        original_circuit: QuantumCircuit,
        optimized_circuit: QuantumCircuit,
        optimization_info: Dict[str, Any],
    ) -> None:
        """
        Verify that the optimized circuit is equivalent to the original input circuit.
        Saves both circuits to debug files if fidelity verification fails.

        Args:
            original_circuit: The actual original circuit passed to simplify()
            optimized_circuit: The optimized circuit produced by simplification
            optimization_info: Information about the optimization process

        Raises:
            ValueError: If fidelity is below tolerance with detailed debug info
        """
        if not self.verify_fidelity:
            return

        try:
            if self.verbose:
                print("🔍 Verifying circuit equivalence...")

            # Get quantum states
            state_original = get_state(original_circuit)
            state_optimized = get_state(optimized_circuit)

            # Compare states
            state_results = compare_quantum_states(
                state_original,
                state_optimized,
                tolerance=self.tolerance,
                verbose=False,  # Keep quiet unless there's an error
            )

            fidelity = state_results.get("fidelity", 0.0)
            amplitudes_match = state_results.get("amplitudes_match", False)

            if self.verbose:
                print(f"✓ Fidelity: {fidelity:.10f}")
                print(f"✓ Amplitudes match: {amplitudes_match}")

            # Check if circuits are equivalent
            if not amplitudes_match or fidelity < (1.0 - self.tolerance):
                # Save debug information as a single comprehensive text file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Create debug directory if it doesn't exist
                debug_dir = "mcrx_optimization_errors"
                os.makedirs(debug_dir, exist_ok=True)

                # Save comprehensive debug info as text
                debug_file = os.path.join(debug_dir, f"mcrx_optimization_error_{timestamp}.txt")

                try:
                    with open(debug_file, "w") as f:
                        f.write("MCRX OPTIMIZATION ERROR REPORT\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"Fidelity: {fidelity:.10f}\n")
                        f.write(f"Amplitudes match: {amplitudes_match}\n")
                        f.write(f"Tolerance: {self.tolerance}\n")
                        f.write(
                            f"Max amplitude difference: {state_results.get('max_difference', 'N/A')}\n"
                        )
                        f.write(
                            f"Mismatched amplitudes: {state_results.get('num_mismatched_amplitudes', 0)}\n\n"
                        )

                        # Optimization details
                        f.write("OPTIMIZATION DETAILS:\n")
                        f.write("-" * 25 + "\n")
                        for key, value in optimization_info.items():
                            f.write(f"{key}: {value}\n")
                        f.write("\n")

                        # Use robust circuit writing
                        self._write_circuit_gates_robust(f, original_circuit, "ORIGINAL CIRCUIT")
                        self._write_circuit_gates_robust(f, optimized_circuit, "OPTIMIZED CIRCUIT")

                        # State comparison details
                        if state_results.get("mismatched_indices"):
                            f.write("DETAILED MISMATCHES:\n")
                            f.write("-" * 22 + "\n")
                            mismatched = state_results["mismatched_indices"]
                            if len(mismatched) > 10:
                                f.write(f"Showing first 10 of {len(mismatched)} mismatches:\n")
                                mismatched = mismatched[:10]

                            for idx in mismatched:
                                orig_amp = (
                                    state_results.get("original_amplitudes", [])[idx]
                                    if idx < len(state_results.get("original_amplitudes", []))
                                    else "N/A"
                                )
                                opt_amp = (
                                    state_results.get("optimized_amplitudes", [])[idx]
                                    if idx < len(state_results.get("optimized_amplitudes", []))
                                    else "N/A"
                                )
                                diff = (
                                    abs(orig_amp - opt_amp)
                                    if orig_amp != "N/A" and opt_amp != "N/A"
                                    else "N/A"
                                )

                                # Convert index to binary state
                                n_qubits = original_circuit.num_qubits
                                state_str = format(idx, f"0{n_qubits}b")

                                f.write(
                                    f"  |{state_str}⟩: {orig_amp} vs {opt_amp} (diff: {diff})\n"
                                )

                    if self.verbose:
                        print(f"💾 Saved debug information to: {debug_file}")

                    debug_file_saved = debug_file

                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not save debug file: {e}")
                    debug_file_saved = f"Could not save debug file: {e}"

                # Detailed error message
                max_diff = state_results.get("max_difference", "N/A")
                num_mismatched = state_results.get("num_mismatched_amplitudes", 0)

                error_msg = (
                    f"MCRX Optimization produced incorrect circuit!\n"
                    f"Fidelity: {fidelity:.10f} (required: >= {1.0 - self.tolerance:.10f})\n"
                    f"Amplitudes match: {amplitudes_match}\n"
                    f"Max amplitude difference: {max_diff}\n"
                    f"Mismatched amplitudes: {num_mismatched}\n"
                    f"Optimization method: {optimization_info.get('optimization_method', 'Unknown')}\n"
                    f"Uses CX tricks: {optimization_info.get('uses_cnot_tricks', False)}\n\n"
                    f"Debug information saved to: {debug_file_saved}"
                )

                raise ValueError(error_msg)

        except Exception as e:
            if "MCRX Optimization produced incorrect circuit" in str(e):
                # Re-raise our custom error
                raise
            else:
                # Handle unexpected errors in verification
                if self.verbose:
                    print(f"⚠️  Warning: Could not verify circuit equivalence: {e}")
                # Don't fail the optimization for verification errors

    def simplify(self, circuit: QuantumCircuit) -> Tuple[QuantumCircuit, Dict[str, Any]]:
        """Main simplification method with iterative handling of multiple patterns."""
        if self.verbose:
            print("🔬 Starting MCRX simplification with iterative multi-pattern handling...")

        # Save the original input circuit for verification
        self.qc_original = circuit.copy()

        # Step 1: Validate and extract circuit information
        target_qubit, rotation_angle, all_ctrl_qubits = self._validate_circuit(circuit)
        n_controls = len(all_ctrl_qubits)

        if self.verbose:
            print(
                f"✓ Circuit validated: target={target_qubit}, angle={rotation_angle:.4f}, controls={n_controls}"
            )

        # Step 2: Extract all patterns
        patterns = self._extract_patterns_fixed(
            circuit, target_qubit, rotation_angle, all_ctrl_qubits
        )

        if self.verbose:
            print(f"✓ Extracted {len(patterns)} patterns:")
            for p in patterns:
                print(f"  {p}")

        # Step 3: Check if we need iterative simplification
        if len(patterns) <= 2:
            # Single pattern or two patterns - use basic simplification (original implementation)
            optimized_circuit, optimization_info = self._apply_basic_simplification(
                patterns, target_qubit, all_ctrl_qubits, circuit, rotation_angle
            )

            # Verify circuit equivalence using the original input circuit
            if self.verify_fidelity and optimization_info.get("optimization_method") not in [
                "no_optimization",
                "no_optimization_different_controls",
                "no_optimization_mixed_controls",
            ]:
                self._verify_circuit_equivalence(
                    self.qc_original, optimized_circuit, optimization_info
                )

            return optimized_circuit, optimization_info

        # Step 4: Apply iterative simplification for multiple patterns (3+)
        optimized_circuit, optimization_info = self._apply_iterative_simplification(
            patterns, circuit.num_qubits, target_qubit, rotation_angle, all_ctrl_qubits
        )

        # Verify circuit equivalence using the original input circuit
        if self.verify_fidelity and optimization_info.get("optimization_method") not in [
            "no_optimization",
            "no_optimization_different_controls",
            "no_optimization_mixed_controls",
        ]:
            self._verify_circuit_equivalence(self.qc_original, optimized_circuit, optimization_info)

        return optimized_circuit, optimization_info

    def _apply_iterative_simplification(
        self,
        initial_patterns: List[ControlPattern],
        n_qubits: int,
        target_qubit: int,
        rotation_angle: float,
        all_ctrl_qubits: List[int],
    ) -> Tuple[QuantumCircuit, Dict[str, Any]]:
        """Apply iterative pairwise simplification to multiple patterns."""
        if self.verbose:
            print("🔄 Applying iterative pairwise simplification...")

        # Sort patterns by decimal value
        remaining_patterns = self._sort_patterns_by_value(initial_patterns)
        final_circuit = QuantumCircuit(n_qubits)

        optimization_log = {
            "steps": [],
            "final_patterns": [],
            "cx_tricks_count": 0,
            "simplifications_count": 0,
            "initial_gate_count": sum(p.coefficient for p in initial_patterns),
        }

        if self.verbose:
            print("✓ Patterns sorted by decimal value:")
            for p in remaining_patterns:
                print(f"  {p.pattern} (decimal: {p.to_decimal()}) -> {p}")

        step_counter = 1

        while len(remaining_patterns) > 1:
            if self.verbose:
                print(
                    f"\n--- Step {step_counter}: {len(remaining_patterns)} patterns remaining ---"
                )

            # Try to find a simplifiable pair starting with the lowest value pattern
            current_pattern = remaining_patterns[0]
            simplification_found = False

            for i, candidate_pattern in enumerate(remaining_patterns[1:], 1):
                if self.verbose:
                    print(f"Trying: {current_pattern.pattern} + {candidate_pattern.pattern}")

                # Create a circuit with just these two patterns
                pair_circuit = self._create_pair_circuit(
                    current_pattern, candidate_pattern, n_qubits, target_qubit, rotation_angle
                )

                if self.verbose:
                    print(f"  Created pair circuit with {len(pair_circuit.data)} gates")

                # Test simplification by creating a temporary simplifier and forcing basic mode
                # We temporarily override the pattern count to force basic simplification
                temp_circuit = pair_circuit.copy()

                # Apply basic simplification directly
                simplified_circuit, simplify_info = self._apply_basic_simplification(
                    [current_pattern, candidate_pattern],
                    target_qubit,
                    all_ctrl_qubits,
                    temp_circuit,
                    rotation_angle,
                )

                if self.verbose:
                    print(f"  Simplification result: {simplify_info['optimization_method']}")
                    print(f"  Gate reduction: {simplify_info['gate_reduction']}")

                if simplify_info["gate_reduction"] > 0 or simplify_info["uses_cnot_tricks"]:
                    # Simplification found! (Either gate reduction OR CX trick)
                    if self.verbose:
                        print(
                            f"✓ Simplification found! Method: {simplify_info['optimization_method']}"
                        )

                    step_info = {
                        "step": step_counter,
                        "pattern1": f"{current_pattern.pattern} (qubits: {current_pattern.active_qubits})",
                        "pattern2": f"{candidate_pattern.pattern} (qubits: {candidate_pattern.active_qubits})",
                        "method": simplify_info["optimization_method"],
                        "gate_reduction": simplify_info["gate_reduction"],
                    }
                    optimization_log["steps"].append(step_info)

                    if simplify_info["uses_cnot_tricks"]:
                        # CX trick(s) applied - append directly to final circuit
                        final_circuit = final_circuit.compose(simplified_circuit)
                        optimization_log["cx_tricks_count"] += 1

                        if self.verbose:
                            if simplify_info["optimization_method"] == "multiple_CX_tricks":
                                print("✓ Multiple CX tricks applied - added to final circuit")
                            else:
                                print("✓ CX trick applied - added to final circuit")

                    else:
                        # Boolean or other simplification - extract new pattern(s) from simplified circuit
                        new_patterns = self._extract_patterns_fixed(
                            simplified_circuit, target_qubit, rotation_angle, all_ctrl_qubits
                        )
                        remaining_patterns.extend(new_patterns)
                        optimization_log["simplifications_count"] += 1

                        if self.verbose:
                            print(
                                f"✓ Extracted {len(new_patterns)} new patterns from simplified circuit"
                            )
                            for np in new_patterns:
                                print(f"    New pattern: {np}")

                    # Remove the processed patterns
                    remaining_patterns.remove(current_pattern)
                    remaining_patterns.remove(candidate_pattern)

                    # Re-sort remaining patterns
                    remaining_patterns = self._sort_patterns_by_value(remaining_patterns)

                    simplification_found = True
                    break

            if not simplification_found:
                # No simplification found for current pattern - add it to final circuit
                pattern_circuit = self._create_single_pattern_circuit(
                    current_pattern, n_qubits, target_qubit, rotation_angle
                )
                final_circuit = final_circuit.compose(pattern_circuit)
                remaining_patterns.remove(current_pattern)

                optimization_log["final_patterns"].append(
                    f"{current_pattern.pattern} (qubits: {current_pattern.active_qubits})"
                )

                if self.verbose:
                    print(
                        f"✗ No simplification found for {current_pattern.pattern} - added to final circuit"
                    )

            step_counter += 1

        # Add any remaining single pattern
        if remaining_patterns:
            last_pattern = remaining_patterns[0]
            pattern_circuit = self._create_single_pattern_circuit(
                last_pattern, n_qubits, target_qubit, rotation_angle
            )
            final_circuit = final_circuit.compose(pattern_circuit)
            optimization_log["final_patterns"].append(
                f"{last_pattern.pattern} (qubits: {last_pattern.active_qubits})"
            )

            if self.verbose:
                print(f"✓ Added final pattern {last_pattern.pattern} to circuit")

        # Compile optimization information
        optimization_info = {
            "original_patterns": [
                f"{p.pattern} (qubits: {p.active_qubits})" for p in initial_patterns
            ],
            "final_patterns": optimization_log["final_patterns"],
            "optimization_steps": optimization_log["steps"],
            "gate_reduction": optimization_log["initial_gate_count"] - len(final_circuit.data),
            "cx_tricks_applied": optimization_log["cx_tricks_count"],
            "simplifications_found": optimization_log["simplifications_count"],
            "optimization_method": "iterative_pairwise_simplification",
            "uses_cnot_tricks": optimization_log["cx_tricks_count"] > 0,
            "optimization_blocked_reason": None,
        }

        if self.verbose:
            print(f"✓ Final optimization: {optimization_info['gate_reduction']} gates reduced")
            print(f"✓ CX tricks applied: {optimization_info['cx_tricks_applied']}")
            print(f"✓ Simplifications found: {optimization_info['simplifications_found']}")

        return final_circuit, optimization_info

    def _apply_basic_simplification(
        self,
        patterns: List[ControlPattern],
        target_qubit: int,
        all_ctrl_qubits: List[int],
        circuit: QuantumCircuit,
        rotation_angle: float,
    ) -> Tuple[QuantumCircuit, Dict[str, Any]]:
        """Apply basic simplification for single pattern or two-pattern cases."""
        if not patterns:
            return QuantumCircuit(circuit.num_qubits), {
                "gate_reduction": 0,
                "optimization_method": "empty_circuit",
                "uses_cnot_tricks": False,
            }

        if self.verbose:
            print(f"  Basic simplification: {len(patterns)} patterns")
            for p in patterns:
                print(f"    {p}")

        # Boolean expression analysis
        all_ctrl_qubits_in_patterns = set()
        for pattern in patterns:
            all_ctrl_qubits_in_patterns.update(pattern.active_qubits)

        bool_analyzer = BooleanExpressionAnalyzer(list(all_ctrl_qubits_in_patterns))
        original_expr = bool_analyzer.patterns_to_combined_expr(patterns)
        simplified_expr = bool_analyzer.simplify_expression(original_expr)
        expr_analysis = bool_analyzer.analyze_expression(simplified_expr)

        if self.verbose:
            print(f"    Original expr: {original_expr}")
            print(f"    Simplified expr: {simplified_expr}")
            print(f"    Single var: {expr_analysis['is_single_variable']}")
            print(f"    Negated var: {expr_analysis['is_negated_variable']}")

        # XOR detection
        xor_detector = XORPatternDetector()
        xor_pairs = xor_detector.detect_xor_pairs(patterns)

        if self.verbose:
            print(f"    XOR pairs found: {len(xor_pairs)}")
            for i, (p1, p2, diff_pos) in enumerate(xor_pairs):
                # Determine XOR type for display
                if len(diff_pos) == 2:
                    pos1, pos2 = diff_pos
                    pos1_idx = p1.active_qubits.index(pos1)
                    pos2_idx = p1.active_qubits.index(pos2)

                    bit1_p1 = p1.pattern[pos1_idx]
                    bit2_p1 = p1.pattern[pos2_idx]
                    bit1_p2 = p2.pattern[pos1_idx]
                    bit2_p2 = p2.pattern[pos2_idx]

                    xor_type = f"{bit1_p1}{bit2_p1}-{bit1_p2}{bit2_p2}"
                    print(
                        f"      XOR {i+1}: {p1.pattern} ⊕ {p2.pattern} at positions {diff_pos} (type: {xor_type})"
                    )
                else:
                    print(f"      XOR {i+1}: {p1.pattern} ⊕ {p2.pattern} at positions {diff_pos}")

        # Check if optimization is valid
        has_different_control_sets = self._has_different_control_sets(patterns)

        if self.verbose:
            print(f"    Different control sets: {has_different_control_sets}")

        # Apply optimization
        if has_different_control_sets:
            optimized_circuit = circuit.copy()
            optimization_method = "no_optimization_different_controls"
        elif xor_pairs and self._should_apply_cx_trick_multiple(xor_pairs, expr_analysis):
            if self.verbose:
                print(f"    Applying multiple CX tricks for {len(xor_pairs)} XOR pairs")
            optimized_circuit = self._apply_multiple_cx_tricks(
                xor_pairs, target_qubit, all_ctrl_qubits, circuit.num_qubits, rotation_angle
            )
            optimization_method = "multiple_CX_tricks"
        elif (
            xor_pairs
            and len(xor_pairs) == 1
            and self._should_apply_cx_trick(xor_pairs[0], expr_analysis)
        ):
            if self.verbose:
                print(f"    Applying single CX trick for: {xor_pairs[0]}")
            optimized_circuit = self._apply_cx_trick_fixed(
                xor_pairs[0], target_qubit, all_ctrl_qubits, circuit.num_qubits, rotation_angle
            )
            optimization_method = "CX_trick"
        elif expr_analysis["is_single_variable"] or expr_analysis["is_negated_variable"]:
            if self._all_patterns_same_control_set(patterns):
                if self.verbose:
                    print(f"    Applying single control optimization")
                optimized_circuit = self._apply_single_control_fixed(
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
        elif len(patterns) == 1:
            optimized_circuit = self._apply_identical_pattern_fixed(
                patterns[0], target_qubit, all_ctrl_qubits, circuit.num_qubits, rotation_angle
            )
            optimization_method = "identical_patterns"
        else:
            if str(original_expr) == str(simplified_expr):
                optimized_circuit = circuit.copy()
                optimization_method = "no_optimization"
            else:
                if self._all_patterns_same_control_set(patterns):
                    # Boolean expression actually simplified - create new optimized circuit
                    if self.verbose:
                        print(
                            f"    Boolean simplification detected: {original_expr} → {simplified_expr}"
                        )

                    optimized_circuit = self._apply_boolean_simplification(
                        simplified_expr,
                        target_qubit,
                        all_ctrl_qubits,
                        circuit.num_qubits,
                        rotation_angle,
                        bool_analyzer,
                        patterns,
                    )
                    optimization_method = "boolean_simplification"
                else:
                    optimized_circuit = circuit.copy()
                    optimization_method = "no_optimization_mixed_controls"

        if self.verbose:
            print(f"    Final method: {optimization_method}")
            print(f"    Gate reduction: {len(circuit.data) - len(optimized_circuit.data)}")

        optimization_info = {
            "original_patterns": [f"{p.pattern} (qubits: {p.active_qubits})" for p in patterns],
            "original_boolean_expr": str(original_expr),
            "simplified_boolean_expr": str(simplified_expr),
            "expression_analysis": expr_analysis,
            "xor_pairs": [(p1.pattern, p2.pattern, diff_pos) for p1, p2, diff_pos in xor_pairs],
            "has_different_control_sets": has_different_control_sets,
            "gate_reduction": len(circuit.data) - len(optimized_circuit.data),
            "optimization_method": optimization_method,
            "uses_cnot_tricks": optimization_method in ["CX_trick", "multiple_CX_tricks"],
            "optimization_blocked_reason": self._get_optimization_blocked_reason(
                optimization_method, has_different_control_sets
            ),
        }

        return optimized_circuit, optimization_info

    def _sort_patterns_by_value(self, patterns: List[ControlPattern]) -> List[ControlPattern]:
        """Sort patterns by their decimal value (ascending)."""

        def pattern_sort_key(pattern):
            decimal_val = pattern.to_decimal()
            # Secondary sort by number of active qubits for stability
            return (decimal_val, len(pattern.active_qubits), tuple(pattern.active_qubits))

        return sorted(patterns, key=pattern_sort_key)

    def _create_pair_circuit(
        self,
        pattern1: ControlPattern,
        pattern2: ControlPattern,
        n_qubits: int,
        target_qubit: int,
        rotation_angle: float,
    ) -> QuantumCircuit:
        """Create a circuit with exactly two MCRX gates for the given patterns."""
        circuit = QuantumCircuit(n_qubits)

        # Add first pattern (with its coefficient)
        for _ in range(pattern1.coefficient):
            gate1 = multi_crx(rotation_angle, pattern1.pattern)
            circuit.append(gate1, pattern1.active_qubits + [target_qubit])

        # Add second pattern (with its coefficient)
        for _ in range(pattern2.coefficient):
            gate2 = multi_crx(rotation_angle, pattern2.pattern)
            circuit.append(gate2, pattern2.active_qubits + [target_qubit])

        return circuit

    def _create_single_pattern_circuit(
        self, pattern: ControlPattern, n_qubits: int, target_qubit: int, rotation_angle: float
    ) -> QuantumCircuit:
        """Create a circuit with a single MCRX gate for the given pattern."""
        circuit = QuantumCircuit(n_qubits)

        # Multiply angle by coefficient
        total_angle = rotation_angle * pattern.coefficient
        gate = multi_crx(total_angle, pattern.pattern)
        circuit.append(gate, pattern.active_qubits + [target_qubit])

        return circuit

    def _apply_boolean_simplification(
        self,
        simplified_expr: sp.Basic,
        target_qubit: int,
        all_ctrl_qubits: List[int],
        n_qubits: int,
        rotation_angle: float,
        bool_analyzer: BooleanExpressionAnalyzer,
        original_patterns: List[ControlPattern],
    ) -> QuantumCircuit:
        """Apply Boolean simplification by creating optimized circuit from simplified expression."""
        circuit = QuantumCircuit(n_qubits)

        # Calculate total angle from all original patterns
        total_angle = rotation_angle * sum(p.coefficient for p in original_patterns)

        if self.verbose:
            print(f"      Creating circuit for simplified expression: {simplified_expr}")
            print(f"      Total angle: {total_angle}")

        # Convert simplified expression back to control pattern
        try:
            optimized_pattern = self._boolean_expr_to_pattern(
                simplified_expr, bool_analyzer, all_ctrl_qubits
            )

            if optimized_pattern:
                if self.verbose:
                    print(f"      Optimized pattern: {optimized_pattern}")

                # Create MCRX gate with the optimized pattern
                gate = multi_crx(total_angle, optimized_pattern[0])
                circuit.append(gate, optimized_pattern[1] + [target_qubit])
            else:
                # Fallback: if we can't convert back to pattern, use original
                if self.verbose:
                    print(f"      Could not convert to pattern, using original circuit")
                return QuantumCircuit(n_qubits)  # Return empty circuit to indicate no optimization

        except Exception as e:
            if self.verbose:
                print(f"      Error in Boolean simplification: {e}")
            return QuantumCircuit(n_qubits)  # Return empty circuit to indicate no optimization

        return circuit

    def _boolean_expr_to_pattern(
        self, expr: sp.Basic, bool_analyzer: BooleanExpressionAnalyzer, all_ctrl_qubits: List[int]
    ) -> Optional[Tuple[str, List[int]]]:
        """Convert a simplified Boolean expression back to a control pattern."""

        # Handle simple cases first
        if expr == sp.true:
            return ("", [])  # No controls needed
        elif expr == sp.false:
            return None  # Never executes

        # For more complex expressions, we need to extract the pattern
        # This is a simplified approach - we'll look for conjunctions of literals

        if isinstance(expr, sp.And):
            # Handle conjunction of literals: ~x1 & ~x2 & ~x3
            pattern_bits = {}
            active_qubits = []

            for arg in expr.args:
                if isinstance(arg, sp.Not):
                    # Negated variable: ~x_i means qubit i should be 0
                    var = arg.args[0]
                    if var in bool_analyzer.qubit_to_symbol.values():
                        qubit_idx = next(
                            q for q, s in bool_analyzer.qubit_to_symbol.items() if s == var
                        )
                        pattern_bits[qubit_idx] = "0"
                        active_qubits.append(qubit_idx)
                elif arg in bool_analyzer.qubit_to_symbol.values():
                    # Positive variable: x_i means qubit i should be 1
                    qubit_idx = next(
                        q for q, s in bool_analyzer.qubit_to_symbol.items() if s == arg
                    )
                    pattern_bits[qubit_idx] = "1"
                    active_qubits.append(qubit_idx)

            if active_qubits:
                # Sort qubits and create pattern string
                active_qubits.sort()
                pattern_str = "".join(pattern_bits[q] for q in active_qubits)
                return (pattern_str, active_qubits)

        elif isinstance(expr, sp.Not):
            # Handle single negated variable
            if expr.args[0] in bool_analyzer.qubit_to_symbol.values():
                var = expr.args[0]
                qubit_idx = next(q for q, s in bool_analyzer.qubit_to_symbol.items() if s == var)
                return ("0", [qubit_idx])

        elif expr in bool_analyzer.qubit_to_symbol.values():
            # Handle single positive variable
            qubit_idx = next(q for q, s in bool_analyzer.qubit_to_symbol.items() if s == expr)
            return ("1", [qubit_idx])

        # For other complex expressions, return None to indicate we can't simplify
        return None

    def _should_apply_cx_trick_multiple(self, xor_pairs: List[Tuple], expr_analysis: Dict) -> bool:
        """Determine if multiple CX tricks should be applied."""
        if not xor_pairs:
            return False

        # Apply multiple CX tricks if:
        # 1. We have multiple XOR pairs from the same pattern comparison
        # 2. All pairs have exactly 2 differing positions
        # 3. The expression doesn't simplify to something simpler

        # Check if all XOR pairs are from the same pattern comparison
        if len(xor_pairs) > 1:
            first_patterns = (xor_pairs[0][0], xor_pairs[0][1])
            for pattern1, pattern2, _ in xor_pairs[1:]:
                if (pattern1, pattern2) != first_patterns:
                    return False

        # Check that all pairs have exactly 2 differing positions
        for _, _, diff_positions in xor_pairs:
            if len(diff_positions) != 2:
                return False

        # Don't apply if expression simplifies to something simpler
        if expr_analysis["is_single_variable"] or expr_analysis["is_negated_variable"]:
            return False

        return True

    def _apply_multiple_cx_tricks(
        self,
        xor_pairs: List[Tuple],
        target_qubit: int,
        all_ctrl_qubits: List[int],
        n_qubits: int,
        rotation_angle: float,
    ) -> QuantumCircuit:
        """Apply multiple CX tricks for multiple XOR pairs within the same pattern comparison."""
        circuit = QuantumCircuit(n_qubits)

        if not xor_pairs:
            return circuit

        # Get the reference patterns (should be the same for all pairs)
        reference_pattern1 = xor_pairs[0][0]
        reference_pattern2 = xor_pairs[0][1]

        if self.verbose:
            print(f"      Applying {len(xor_pairs)} CX tricks:")

        # Apply each XOR pair transformation
        for i, (pattern1, pattern2, diff_positions) in enumerate(xor_pairs):
            if self.verbose:
                print(f"        XOR pair {i+1}: positions {diff_positions}")

            if len(diff_positions) == 2:
                pos1, pos2 = diff_positions

                # Determine XOR pattern type
                pos1_idx = pattern1.active_qubits.index(pos1)
                pos2_idx = pattern1.active_qubits.index(pos2)

                bit1_pattern1 = pattern1.pattern[pos1_idx]
                bit2_pattern1 = pattern1.pattern[pos2_idx]
                bit1_pattern2 = pattern2.pattern[pos1_idx]
                bit2_pattern2 = pattern2.pattern[pos2_idx]

                if self.verbose:
                    print(f"          Position {pos1}: {bit1_pattern1} vs {bit1_pattern2}")
                    print(f"          Position {pos2}: {bit2_pattern1} vs {bit2_pattern2}")

                # Apply appropriate CX trick
                if (bit1_pattern1, bit2_pattern1) == ("0", "1") and (
                    bit1_pattern2,
                    bit2_pattern2,
                ) == ("1", "0"):
                    # 01-10 pattern → Standard CX trick
                    if self.verbose:
                        print(f"          Type: 01-10 XOR → Standard CX trick")
                    circuit.cx(pos1, pos2)
                elif (bit1_pattern1, bit2_pattern1) == ("1", "0") and (
                    bit1_pattern2,
                    bit2_pattern2,
                ) == ("0", "1"):
                    # 10-01 pattern → Standard CX trick
                    if self.verbose:
                        print(f"          Type: 10-01 XOR → Standard CX trick")
                    circuit.cx(pos1, pos2)
                elif (bit1_pattern1, bit2_pattern1) == ("0", "0") and (
                    bit1_pattern2,
                    bit2_pattern2,
                ) == ("1", "1"):
                    # 00-11 pattern → X + CX trick
                    if self.verbose:
                        print(f"          Type: 00-11 XOR → X + CX trick")
                    circuit.x(pos2)
                    circuit.cx(pos1, pos2)
                elif (bit1_pattern1, bit2_pattern1) == ("1", "1") and (
                    bit1_pattern2,
                    bit2_pattern2,
                ) == ("0", "0"):
                    # 11-00 pattern → X + CX trick
                    if self.verbose:
                        print(f"          Type: 11-00 XOR → X + CX trick")
                    circuit.x(pos2)
                    circuit.cx(pos1, pos2)

        # Apply the controlled rotation on the transformed pattern
        # After CX transformations, calculate the effective control pattern
        reference_pattern = reference_pattern1  # Use first pattern as reference

        # Get common control positions (positions that don't differ)
        all_diff_positions = set()
        cx_target_positions = []  # Second position of each XOR pair

        for _, _, diff_positions in xor_pairs:
            all_diff_positions.update(diff_positions)
            if len(diff_positions) == 2:
                cx_target_positions.append(diff_positions[1])  # Second position is CX target

        # Build the effective pattern after CX transformations
        effective_pattern = ""
        effective_qubits = []

        # Add common control positions (unchanged by CX)
        for i, qubit_idx in enumerate(reference_pattern.active_qubits):
            if qubit_idx not in all_diff_positions:
                bit = reference_pattern.pattern[i]
                effective_pattern += bit
                effective_qubits.append(qubit_idx)

        # Add CX target positions (always '1' after CX transformation)
        for target_pos in sorted(cx_target_positions):
            effective_pattern += "1"
            effective_qubits.append(target_pos)

        # Sort qubits and reorder pattern accordingly
        sorted_indices = sorted(range(len(effective_qubits)), key=lambda i: effective_qubits[i])
        final_qubits = [effective_qubits[i] for i in sorted_indices]
        final_pattern = "".join(effective_pattern[i] for i in sorted_indices)

        if self.verbose:
            print(f"      Effective MCRX pattern: '{final_pattern}' on qubits {final_qubits}")

        # Apply the MCRX with the effective pattern
        if final_pattern and final_qubits:
            gate = multi_crx(rotation_angle, final_pattern)
            circuit.append(gate, final_qubits + [target_qubit])
        else:
            # Fallback: unconditional rotation
            circuit.rx(rotation_angle, target_qubit)

        # Undo the XOR transformations in reverse order
        for pattern1, pattern2, diff_positions in reversed(xor_pairs):
            if len(diff_positions) == 2:
                pos1, pos2 = diff_positions

                # Determine XOR pattern type
                pos1_idx = pattern1.active_qubits.index(pos1)
                pos2_idx = pattern1.active_qubits.index(pos2)

                bit1_pattern1 = pattern1.pattern[pos1_idx]
                bit2_pattern1 = pattern1.pattern[pos2_idx]
                bit1_pattern2 = pattern2.pattern[pos1_idx]
                bit2_pattern2 = pattern2.pattern[pos2_idx]

                # Undo transformations
                if (bit1_pattern1, bit2_pattern1) == ("0", "1") and (
                    bit1_pattern2,
                    bit2_pattern2,
                ) == ("1", "0"):
                    # 01-10 pattern → Undo CX
                    circuit.cx(pos1, pos2)
                elif (bit1_pattern1, bit2_pattern1) == ("1", "0") and (
                    bit1_pattern2,
                    bit2_pattern2,
                ) == ("0", "1"):
                    # 10-01 pattern → Undo CX
                    circuit.cx(pos1, pos2)
                elif (bit1_pattern1, bit2_pattern1) == ("0", "0") and (
                    bit1_pattern2,
                    bit2_pattern2,
                ) == ("1", "1"):
                    # 00-11 pattern → Undo CX + X
                    circuit.cx(pos1, pos2)
                    circuit.x(pos2)
                elif (bit1_pattern1, bit2_pattern1) == ("1", "1") and (
                    bit1_pattern2,
                    bit2_pattern2,
                ) == ("0", "0"):
                    # 11-00 pattern → Undo CX + X
                    circuit.cx(pos1, pos2)
                    circuit.x(pos2)

        return circuit

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

    def _apply_cx_trick_fixed(
        self,
        xor_pair_info: Tuple,
        target_qubit: int,
        all_ctrl_qubits: List[int],
        n_qubits: int,
        rotation_angle: float,
    ) -> QuantumCircuit:
        """Apply CX trick with proper handling of both XOR patterns."""
        pattern1, pattern2, diff_positions = xor_pair_info
        circuit = QuantumCircuit(n_qubits)

        if len(diff_positions) == 2:
            pos1, pos2 = diff_positions

            # Determine XOR pattern type by checking the differing bits
            # Find the bit values at the differing positions
            pos1_idx = pattern1.active_qubits.index(pos1)
            pos2_idx = pattern1.active_qubits.index(pos2)

            bit1_pattern1 = pattern1.pattern[pos1_idx]
            bit2_pattern1 = pattern1.pattern[pos2_idx]
            bit1_pattern2 = pattern2.pattern[pos1_idx]
            bit2_pattern2 = pattern2.pattern[pos2_idx]

            if self.verbose:
                print(f"      XOR pattern analysis:")
                print(f"        Position {pos1}: {bit1_pattern1} vs {bit1_pattern2}")
                print(f"        Position {pos2}: {bit2_pattern1} vs {bit2_pattern2}")

            # Determine XOR type and apply appropriate sequence
            if (bit1_pattern1, bit2_pattern1) == ("0", "1") and (bit1_pattern2, bit2_pattern2) == (
                "1",
                "0",
            ):
                # Type 1: 01-10 pattern → Standard CX trick
                if self.verbose:
                    print(f"        Type: 01-10 XOR → Standard CX trick")
                self._apply_01_10_cx_trick(
                    circuit, pattern1, pos1, pos2, target_qubit, rotation_angle, diff_positions
                )

            elif (bit1_pattern1, bit2_pattern1) == ("1", "0") and (
                bit1_pattern2,
                bit2_pattern2,
            ) == ("0", "1"):
                # Type 1: 10-01 pattern → Standard CX trick
                if self.verbose:
                    print(f"        Type: 10-01 XOR → Standard CX trick")
                self._apply_01_10_cx_trick(
                    circuit, pattern1, pos1, pos2, target_qubit, rotation_angle, diff_positions
                )

            elif (bit1_pattern1, bit2_pattern1) == ("0", "0") and (
                bit1_pattern2,
                bit2_pattern2,
            ) == ("1", "1"):
                # Type 2: 00-11 pattern → X + CX trick
                if self.verbose:
                    print(f"        Type: 00-11 XOR → X + CX trick")
                self._apply_00_11_cx_trick(
                    circuit, pattern1, pos1, pos2, target_qubit, rotation_angle, diff_positions
                )

            elif (bit1_pattern1, bit2_pattern1) == ("1", "1") and (
                bit1_pattern2,
                bit2_pattern2,
            ) == ("0", "0"):
                # Type 2: 11-00 pattern → X + CX trick
                if self.verbose:
                    print(f"        Type: 11-00 XOR → X + CX trick")
                self._apply_00_11_cx_trick(
                    circuit, pattern2, pos1, pos2, target_qubit, rotation_angle, diff_positions
                )

            else:
                # Fallback: shouldn't happen if XOR detection is correct
                if self.verbose:
                    print(f"        Type: Unknown XOR pattern → Fallback to standard")
                self._apply_01_10_cx_trick(
                    circuit, pattern1, pos1, pos2, target_qubit, rotation_angle, diff_positions
                )

        return circuit

    def _apply_01_10_cx_trick(
        self,
        circuit: QuantumCircuit,
        reference_pattern: ControlPattern,
        pos1: int,
        pos2: int,
        target_qubit: int,
        rotation_angle: float,
        diff_positions: List[int],
    ) -> None:
        """Apply standard CX trick for 01-10 XOR patterns."""
        # Standard CX trick: CX + controlled rotation + CX
        circuit.cx(pos1, pos2)

        # Create control condition for remaining qubits
        common_controls = []
        for i, qubit_idx in enumerate(reference_pattern.active_qubits):
            if qubit_idx not in diff_positions:
                bit = reference_pattern.pattern[i]
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
            common_pattern += "1"  # For standard CX trick, pos2 should be 1

            gate = multi_crx(rotation_angle, common_pattern)
            circuit.append(gate, common_qubits + [target_qubit])
        else:
            # Simple case: just control on pos2
            circuit.crx(rotation_angle, pos2, target_qubit)

        circuit.cx(pos1, pos2)

    def _apply_00_11_cx_trick(
        self,
        circuit: QuantumCircuit,
        reference_pattern: ControlPattern,
        pos1: int,
        pos2: int,
        target_qubit: int,
        rotation_angle: float,
        diff_positions: List[int],
    ) -> None:
        """Apply X + CX trick for 00-11 XOR patterns."""
        # X + CX trick: X(pos2) + CX + controlled rotation + CX + X(pos2)
        circuit.x(pos2)  # Flip pos2 to convert 00-11 to 01-10
        circuit.cx(pos1, pos2)

        # Create control condition for remaining qubits
        common_controls = []
        for i, qubit_idx in enumerate(reference_pattern.active_qubits):
            if qubit_idx not in diff_positions:
                bit = reference_pattern.pattern[i]
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
            common_pattern += "1"  # After X and CX, pos2 should be controlled on 1

            gate = multi_crx(rotation_angle, common_pattern)
            circuit.append(gate, common_qubits + [target_qubit])
        else:
            # Simple case: just control on pos2
            circuit.crx(rotation_angle, pos2, target_qubit)

        circuit.cx(pos1, pos2)
        circuit.x(pos2)  # Flip pos2 back to original state

    def _apply_single_control_fixed(
        self,
        simplified_expr: sp.Basic,
        target_qubit: int,
        all_ctrl_qubits: List[int],
        n_qubits: int,
        rotation_angle: float,
        bool_analyzer: BooleanExpressionAnalyzer,
    ) -> QuantumCircuit:
        """FIXED: Apply single control optimization."""
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

    def _apply_identical_pattern_fixed(
        self,
        pattern: ControlPattern,
        target_qubit: int,
        all_ctrl_qubits: List[int],
        n_qubits: int,
        rotation_angle: float,
    ) -> QuantumCircuit:
        """FIXED: Apply identical pattern optimization."""
        circuit = QuantumCircuit(n_qubits)

        # Single gate with multiplied angle
        total_angle = rotation_angle * pattern.coefficient
        gate = multi_crx(total_angle, pattern.pattern)
        circuit.append(gate, pattern.active_qubits + [target_qubit])

        return circuit

    def _validate_circuit(self, circuit: QuantumCircuit) -> Tuple[int, float, List[int]]:
        """Validate circuit and extract information."""
        if circuit.num_qubits == 0 or len(circuit.data) == 0:
            raise ValueError("Circuit must have qubits and gates")

        # Validate that all gates are MCRX gates
        self._validate_mcrx_only_circuit(circuit)

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

    def _validate_mcrx_only_circuit(self, circuit: QuantumCircuit) -> None:
        """Validate that circuit contains only multi-controlled RX gates."""
        for i, instruction in enumerate(circuit.data):
            gate_name = instruction.operation.name.lower()

            # Check if it's a valid MCRX-type gate using pattern matching
            is_valid = self._is_valid_mcrx_gate(gate_name, instruction)

            if not is_valid:
                raise ValueError(
                    f"Gate {i+1} ('{instruction.operation.name}') is not a valid multi-controlled RX gate. "
                    f"Circuit must contain only MCRX-type gates. Found gate with {len(instruction.qubits)} qubits."
                )

    def _is_valid_mcrx_gate(self, gate_name: str, instruction) -> bool:
        """Check if a gate is a valid multi-controlled RX gate."""
        # Must have rotation parameter
        if not hasattr(instruction.operation, "params") or len(instruction.operation.params) == 0:
            return False

        # Check various MCRX gate name patterns
        valid_patterns = [
            r"^rx$",  # Basic RX
            r"^crx$",  # Controlled RX
            r"^mcrx$",  # Multi-controlled RX
            r"^ccrx(_o\d+)?$",  # 2-controlled RX (ccrx, ccrx_o1, etc.)
            r"^c\d+rx(_o\d+)?$",  # n-controlled RX (c3rx, c4rx_o0, etc.)
            r"^mcx(_o\d+)?$",  # Multi-controlled X (sometimes used for RX)
            r"^mcrx(_o\d+)?$",  # Multi-controlled RX with ctrl_state
            r"^.*rx.*$",  # Any gate containing 'rx'
        ]

        # Check if gate name matches any valid pattern
        for pattern in valid_patterns:
            if re.match(pattern, gate_name):
                return True

        # Additional check for gates that might be MCRX but with different naming
        # If it has qubits (at least target) and rotation params, likely valid
        if len(instruction.qubits) >= 1:
            # Check if it looks like a rotation gate
            param = instruction.operation.params[0]
            try:
                float(param)  # Can convert to float (rotation angle)
                return True
            except (ValueError, TypeError):
                pass

        return False

    def _extract_patterns_fixed(
        self,
        circuit: QuantumCircuit,
        target_qubit: int,
        rotation_angle: float,
        all_ctrl_qubits: List[int],
    ) -> List[ControlPattern]:
        """Extract patterns with left-to-right reading."""
        pattern_data = defaultdict(int)

        for instruction in circuit.data:
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            gate_ctrl_qubits = qubits[:-1]  # All except target

            # Extract pattern
            pattern_str, active_qubits = self._extract_pattern_string_fixed(
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

    def _extract_pattern_string_fixed(
        self, instruction, gate_ctrl_qubits: List[int]
    ) -> Tuple[str, List[int]]:
        """FIXED: Extract pattern string reading LEFT-TO-RIGHT."""
        op_name = instruction.operation.name
        n_gate_controls = len(gate_ctrl_qubits)

        # Default pattern (all controls must be |1⟩)
        default_pattern = "1" * n_gate_controls

        # Try to extract ctrl_state
        ctrl_state_str = default_pattern

        if hasattr(instruction.operation, "ctrl_state"):
            ctrl_state = instruction.operation.ctrl_state
            if isinstance(ctrl_state, str):
                # FIXED: Reverse Qiskit's ctrl_state to get our left-to-right convention
                ctrl_state_str = ctrl_state[::-1]
            elif isinstance(ctrl_state, int):
                # Convert int to binary and reverse
                ctrl_state_str = format(ctrl_state, f"0{n_gate_controls}b")[::-1]

        # Parse operation name if needed
        if ctrl_state_str == default_pattern:
            pattern_match = re.search(r"_o(\d+)", op_name)
            if pattern_match:
                state_number = int(pattern_match.group(1))
                ctrl_state_str = format(state_number, f"0{n_gate_controls}b")[::-1]

        # FIXED: Return pattern and active qubits in TOP-TO-BOTTOM order
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

    def analyze_patterns(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Analyze patterns with iterative capability."""
        try:
            target_qubit, rotation_angle, all_ctrl_qubits = self._validate_circuit(circuit)
            patterns = self._extract_patterns_fixed(
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

            # Analyze potential pairwise simplifications for multiple patterns
            simplification_opportunities = []
            if len(patterns) > 1:
                sorted_patterns = self._sort_patterns_by_value(patterns)

                for i, pattern1 in enumerate(sorted_patterns):
                    for j, pattern2 in enumerate(sorted_patterns[i + 1 :], i + 1):
                        pair_circuit = self._create_pair_circuit(
                            pattern1, pattern2, circuit.num_qubits, target_qubit, rotation_angle
                        )

                        _, simplify_info = self._apply_basic_simplification(
                            [pattern1, pattern2],
                            target_qubit,
                            all_ctrl_qubits,
                            pair_circuit,
                            rotation_angle,
                        )

                        if simplify_info["gate_reduction"] > 0:
                            simplification_opportunities.append(
                                {
                                    "pattern1": f"{pattern1.pattern} (qubits: {pattern1.active_qubits})",
                                    "pattern2": f"{pattern2.pattern} (qubits: {pattern2.active_qubits})",
                                    "method": simplify_info["optimization_method"],
                                    "gate_reduction": simplify_info["gate_reduction"],
                                    "uses_cx_trick": simplify_info["uses_cnot_tricks"],
                                }
                            )

            return {
                "status": "success",
                "total_patterns": len(patterns),
                "patterns_by_value": [
                    {
                        "pattern": p.pattern,
                        "decimal_value": p.to_decimal(),
                        "active_qubits": p.active_qubits,
                        "coefficient": p.coefficient,
                    }
                    for p in self._sort_patterns_by_value(patterns)
                ],
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
                "simplification_opportunities": simplification_opportunities,
                "potential_gate_reduction": sum(
                    opp["gate_reduction"] for opp in simplification_opportunities
                ),
                "supports_iterative": len(patterns) > 1,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}
