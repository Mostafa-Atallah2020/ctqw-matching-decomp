# File: src/pattern_analyzer.py
"""
Control pattern analysis for MCRX optimization with integrated Boolean function optimization.
Analyzes relationships between control patterns and optimizes Boolean functions for quantum circuit synthesis.

CORRECTED VERSION: Fixed pattern interpretation to match multi_crx convention and TeX file examples.
"""

from enum import Enum
from typing import List, Dict, Tuple, Set, Optional, Union
from collections import defaultdict, Counter
import itertools


class PatternRelationship(Enum):
    """Types of relationships between control patterns."""
    DISJOINT = "disjoint"
    OVERLAPPING = "overlapping"
    SUBSET = "subset"
    IDENTICAL = "identical"


class BooleanFunction:
    """Represents a Boolean function with optimization capabilities."""
    
    def __init__(self, satisfying_states: List[int], n_variables: int):
        self.satisfying_states = set(satisfying_states)
        self.n_variables = n_variables
        self.simplified_expression = None
        self.implementation_strategy = None
        
    def __repr__(self):
        return f"BooleanFunction(states={sorted(self.satisfying_states)}, vars={self.n_variables})"
    
    def to_truth_table(self) -> List[bool]:
        """Convert to truth table representation."""
        return [i in self.satisfying_states for i in range(2**self.n_variables)]
    
    def to_dnf_expression(self) -> str:
        """Convert to Disjunctive Normal Form expression."""
        if not self.satisfying_states:
            return "FALSE"
        
        terms = []
        for state in sorted(self.satisfying_states):
            state_bits = format(state, f'0{self.n_variables}b')
            term_parts = []
            for i, bit in enumerate(state_bits):
                if bit == '1':
                    term_parts.append(f"x{self.n_variables-1-i}")
                else:
                    term_parts.append(f"¬x{self.n_variables-1-i}")
            terms.append("(" + " ∧ ".join(term_parts) + ")")
        
        return " ∨ ".join(terms)


class BooleanOptimizer:
    """
    Advanced Boolean function optimization for quantum circuit synthesis.
    
    CORRECTED VERSION: Fixed variable ordering to match TeX file examples.
    """
    
    def __init__(self):
        pass
    
    def optimize_function(self, boolean_func: BooleanFunction) -> Dict[str, any]:
        """
        Optimize a Boolean function for quantum implementation.
        
        Args:
            boolean_func: Boolean function to optimize
            
        Returns:
            Dictionary with optimization results including simplified expression,
            implementation strategy, and complexity metrics
        """
        if not boolean_func.satisfying_states:
            return self._create_false_result()
        
        # Apply multiple optimization strategies
        karnaugh_result = self._karnaugh_optimization(boolean_func)
        pattern_result = self._pattern_recognition(boolean_func)
        espresso_result = self._espresso_style_minimization(boolean_func)
        
        # Choose best optimization
        best_result = self._select_best_optimization(karnaugh_result, pattern_result, espresso_result)
        
        # Determine quantum implementation strategy
        implementation = self._determine_quantum_implementation(best_result, boolean_func)
        
        return {
            'original_function': boolean_func.to_dnf_expression(),
            'simplified_expression': best_result['expression'],
            'simplification_method': best_result['method'],
            'implementation': implementation,
            'complexity_reduction': best_result['complexity_reduction'],
            'quantum_cost': implementation['quantum_cost'],
            'optimization_quality': best_result['quality']
        }
    
    def _karnaugh_optimization(self, boolean_func: BooleanFunction) -> Dict[str, any]:
        """Apply Karnaugh Map style optimization."""
        n_vars = boolean_func.n_variables
        states = boolean_func.satisfying_states
        
        if n_vars == 1:
            return self._optimize_1_variable(states)
        elif n_vars == 2:
            return self._optimize_2_variable(states)
        elif n_vars == 3:
            return self._optimize_3_variable(states)
        elif n_vars == 4:
            return self._optimize_4_variable(states)
        else:
            return self._optimize_general_karnaugh(states, n_vars)
    
    def _optimize_1_variable(self, states: Set[int]) -> Dict[str, any]:
        """Optimize 1-variable Boolean function."""
        if states == {0}:
            return {'expression': '¬x0', 'method': 'karnaugh_1var', 'complexity_reduction': 0, 'quality': 'optimal'}
        elif states == {1}:
            return {'expression': 'x0', 'method': 'karnaugh_1var', 'complexity_reduction': 0, 'quality': 'optimal'}
        elif states == {0, 1}:
            return {'expression': 'TRUE', 'method': 'karnaugh_1var', 'complexity_reduction': 1, 'quality': 'optimal'}
        else:
            return {'expression': 'FALSE', 'method': 'karnaugh_1var', 'complexity_reduction': 0, 'quality': 'optimal'}
    
    def _optimize_2_variable(self, states: Set[int]) -> Dict[str, any]:
        """
        Optimize 2-variable Boolean function using Karnaugh map patterns.
        
        CORRECTED: Variable ordering to match TeX file examples.
        State encoding: x0 is MSB (most significant bit)
        State 0 = '00' = x0=0, x1=0
        State 1 = '01' = x0=0, x1=1  
        State 2 = '10' = x0=1, x1=0
        State 3 = '11' = x0=1, x1=1
        """
        states_set = set(states)
        
        patterns = {
            frozenset({0}): ('¬x0 ∧ ¬x1', 0),
            frozenset({1}): ('¬x0 ∧ x1', 0),
            frozenset({2}): ('x0 ∧ ¬x1', 0),
            frozenset({3}): ('x0 ∧ x1', 0),
            frozenset({0, 1}): ('¬x0', 1),  # x0=0 (states 0,1)
            frozenset({2, 3}): ('x0', 1),   # x0=1 (states 2,3) *** CORRECTED ***
            frozenset({0, 2}): ('¬x1', 1),  # x1=0 (states 0,2)
            frozenset({1, 3}): ('x1', 1),   # x1=1 (states 1,3)
            frozenset({1, 2}): ('x0 ⊕ x1', 1),  # XOR pattern: states 1,2
            frozenset({0, 3}): ('x0 ⊙ x1', 1),  # XNOR pattern: states 0,3
            frozenset({0, 1, 2}): ('¬(x0 ∧ x1)', 2),  # NAND
            frozenset({1, 2, 3}): ('x0 ∨ x1', 2),     # OR
            frozenset({0, 2, 3}): ('x0 ∨ ¬x1', 2),    
            frozenset({0, 1, 3}): ('¬x0 ∨ x1', 2),
            frozenset({0, 1, 2, 3}): ('TRUE', 3)
        }
        
        pattern_key = frozenset(states_set)
        if pattern_key in patterns:
            expr, reduction = patterns[pattern_key]
            return {'expression': expr, 'method': 'karnaugh_2var', 'complexity_reduction': reduction, 'quality': 'optimal'}
        
        # No optimization found
        return {'expression': self._states_to_dnf(states, 2), 'method': 'karnaugh_2var', 'complexity_reduction': 0, 'quality': 'none'}
    
    def _optimize_3_variable(self, states: Set[int]) -> Dict[str, any]:
        """
        Optimize 3-variable Boolean function.
        
        CORRECTED: Variable ordering with x0 as MSB.
        """
        states_set = set(states)
        
        # 3-variable state mapping with x0 as MSB:
        # State 0 = '000' = x0=0, x1=0, x2=0
        # State 1 = '001' = x0=0, x1=0, x2=1
        # State 2 = '010' = x0=0, x1=1, x2=0
        # State 3 = '011' = x0=0, x1=1, x2=1
        # State 4 = '100' = x0=1, x1=0, x2=0
        # State 5 = '101' = x0=1, x1=0, x2=1
        # State 6 = '110' = x0=1, x1=1, x2=0
        # State 7 = '111' = x0=1, x1=1, x2=1
        
        patterns = {
            frozenset({4, 5, 6, 7}): ('x0', 3),  # All states with x0=1
            frozenset({2, 3, 6, 7}): ('x1', 3),  # All states with x1=1  
            frozenset({1, 3, 5, 7}): ('x2', 3),  # All states with x2=1
            frozenset({5, 6}): ('x0 ∧ (x1 ⊕ x2)', 1),  # States 101, 110 → x0=1 AND (x1 XOR x2)
            frozenset({1, 2}): ('(¬x0 ∧ ¬x1 ∧ x2) ∨ (¬x0 ∧ x1 ∧ ¬x2)', 0),  # States 001, 010
            frozenset({0, 1, 2, 3}): ('¬x0', 3),  # All states with x0=0
            frozenset({0, 1, 4, 5}): ('¬x1', 3),  # All states with x1=0
            frozenset({0, 2, 4, 6}): ('¬x2', 3),  # All states with x2=0
        }
        
        pattern_key = frozenset(states_set)
        if pattern_key in patterns:
            expr, reduction = patterns[pattern_key]
            return {'expression': expr, 'method': 'karnaugh_3var', 'complexity_reduction': reduction, 'quality': 'optimal'}
        
        # Check for partial patterns
        if len(states_set) >= 4:
            # Check if it's mostly one variable
            x0_states = {4, 5, 6, 7}  # x0=1
            x1_states = {2, 3, 6, 7}  # x1=1
            x2_states = {1, 3, 5, 7}  # x2=1
            
            x0_overlap = len(states_set & x0_states) / len(states_set)
            x1_overlap = len(states_set & x1_states) / len(states_set)
            x2_overlap = len(states_set & x2_states) / len(states_set)
            
            if x0_overlap >= 0.75:
                return {'expression': 'x0 (approx)', 'method': 'karnaugh_3var_approx', 'complexity_reduction': 2, 'quality': 'good'}
            elif x1_overlap >= 0.75:
                return {'expression': 'x1 (approx)', 'method': 'karnaugh_3var_approx', 'complexity_reduction': 2, 'quality': 'good'}
            elif x2_overlap >= 0.75:
                return {'expression': 'x2 (approx)', 'method': 'karnaugh_3var_approx', 'complexity_reduction': 2, 'quality': 'good'}
        
        return {'expression': self._states_to_dnf(states, 3), 'method': 'karnaugh_3var', 'complexity_reduction': 0, 'quality': 'none'}
    
    def _optimize_4_variable(self, states: Set[int]) -> Dict[str, any]:
        """Optimize 4-variable Boolean function."""
        # For 4 variables, use simplified pattern recognition
        states_set = set(states)
        n_states = len(states_set)
        
        # Check for single variable patterns (x0 as MSB)
        var_patterns = [
            (set(range(8, 16)), 'x0'),   # x0=1 (MSB)
            (set(range(0, 8)), '¬x0'),   # x0=0
            ({4,5,6,7,12,13,14,15}, 'x1'),  # x1=1
            ({0,1,2,3,8,9,10,11}, '¬x1'),   # x1=0
            ({2,3,6,7,10,11,14,15}, 'x2'),   # x2=1
            ({0,1,4,5,8,9,12,13}, '¬x2'),  # x2=0
            ({1,3,5,7,9,11,13,15}, 'x3'),   # x3=1 (LSB)
            ({0,2,4,6,8,10,12,14}, '¬x3'),  # x3=0
        ]
        
        for pattern_states, expr in var_patterns:
            if states_set == pattern_states:
                return {'expression': expr, 'method': 'karnaugh_4var', 'complexity_reduction': 3, 'quality': 'optimal'}
        
        # Check for approximate matches
        for pattern_states, expr in var_patterns:
            overlap = len(states_set & pattern_states) / max(len(states_set), 1)
            if overlap >= 0.8:
                return {'expression': f'{expr} (approx)', 'method': 'karnaugh_4var_approx', 'complexity_reduction': 2, 'quality': 'good'}
        
        return {'expression': self._states_to_dnf(states, 4), 'method': 'karnaugh_4var', 'complexity_reduction': 0, 'quality': 'none'}
    
    def _optimize_general_karnaugh(self, states: Set[int], n_vars: int) -> Dict[str, any]:
        """General Karnaugh optimization for n > 4 variables."""
        # For large numbers of variables, use heuristic approaches
        states_set = set(states)
        
        # Check if it's mostly determined by a single variable (x0 as MSB)
        for var_idx in range(n_vars):
            var_mask = 1 << (n_vars - 1 - var_idx)  # Bit position for this variable (MSB first)
            
            # States where this variable is 1
            var_true_states = {s for s in range(2**n_vars) if s & var_mask}
            # States where this variable is 0
            var_false_states = {s for s in range(2**n_vars) if not (s & var_mask)}
            
            # Check overlap
            true_overlap = len(states_set & var_true_states) / max(len(states_set), 1)
            false_overlap = len(states_set & var_false_states) / max(len(states_set), 1)
            
            if true_overlap >= 0.85:
                return {'expression': f'x{var_idx}', 'method': f'karnaugh_{n_vars}var_heuristic', 'complexity_reduction': 3, 'quality': 'good'}
            elif false_overlap >= 0.85:
                return {'expression': f'¬x{var_idx}', 'method': f'karnaugh_{n_vars}var_heuristic', 'complexity_reduction': 3, 'quality': 'good'}
        
        # No clear single-variable pattern
        return {'expression': self._states_to_dnf(states, n_vars), 'method': f'karnaugh_{n_vars}var', 'complexity_reduction': 0, 'quality': 'none'}
    
    def _pattern_recognition(self, boolean_func: BooleanFunction) -> Dict[str, any]:
        """Pattern recognition for quantum-specific optimizations."""
        states = boolean_func.satisfying_states
        n_vars = boolean_func.n_variables
        
        # Quantum gate specific patterns (corrected for x0 as MSB)
        quantum_patterns = {
            # XOR patterns (important for quantum)
            frozenset({1, 2}): ('x0 ⊕ x1', 'xor', 2),  # 2-var XOR
            frozenset({0, 3}): ('x0 ⊙ x1', 'xnor', 2), # 2-var XNOR
            
            # Single control patterns
            frozenset({2}): ('x0 ∧ ¬x1', 'single_control', 1),
            frozenset({1}): ('¬x0 ∧ x1', 'single_control', 1),
            frozenset({3}): ('x0 ∧ x1', 'double_control', 1),
            
            # Toffoli-like patterns (3 variables)
            frozenset({7}): ('x0 ∧ x1 ∧ x2', 'toffoli', 1),
            frozenset({6}): ('x0 ∧ x1 ∧ ¬x2', 'toffoli_variant', 1),
            
            # OR patterns (good for quantum synthesis)
            frozenset({1, 2, 3}): ('x0 ∨ x1', 'or_gate', 2),
            frozenset({4, 5, 6, 7}): ('x0', 'projection', 3),
        }
        
        pattern_key = frozenset(states)
        if pattern_key in quantum_patterns:
            expr, pattern_type, reduction = quantum_patterns[pattern_key]
            return {'expression': expr, 'method': f'pattern_recognition_{pattern_type}', 'complexity_reduction': reduction, 'quality': 'optimal'}
        
        # Check for parameterized patterns
        if n_vars == 2:
            if len(states) == 1:
                state = list(states)[0]
                state_bits = format(state, '02b')
                literals = []
                for i, bit in enumerate(state_bits):
                    var_name = f'x{i}'  # MSB first
                    if bit == '1':
                        literals.append(var_name)
                    else:
                        literals.append(f'¬{var_name}')
                expr = ' ∧ '.join(literals)
                return {'expression': expr, 'method': 'pattern_recognition_minterm', 'complexity_reduction': 0, 'quality': 'exact'}
        
        return {'expression': self._states_to_dnf(states, n_vars), 'method': 'pattern_recognition', 'complexity_reduction': 0, 'quality': 'none'}
    
    def _espresso_style_minimization(self, boolean_func: BooleanFunction) -> Dict[str, any]:
        """Simplified Espresso-style Boolean minimization."""
        states = boolean_func.satisfying_states
        n_vars = boolean_func.n_variables
        
        if len(states) <= 1:
            return {'expression': self._states_to_dnf(states, n_vars), 'method': 'espresso_trivial', 'complexity_reduction': 0, 'quality': 'exact'}
        
        # Find prime implicants using a simplified approach
        prime_implicants = self._find_prime_implicants(states, n_vars)
        
        # Select minimal set of prime implicants
        minimal_cover = self._find_minimal_cover(prime_implicants, states)
        
        if len(minimal_cover) < len(states):
            expr = self._implicants_to_expression(minimal_cover)
            reduction = len(states) - len(minimal_cover)
            return {'expression': expr, 'method': 'espresso_style', 'complexity_reduction': reduction, 'quality': 'good'}
        
        return {'expression': self._states_to_dnf(states, n_vars), 'method': 'espresso_style', 'complexity_reduction': 0, 'quality': 'none'}
    
    def _find_prime_implicants(self, states: Set[int], n_vars: int) -> List[Tuple]:
        """Find prime implicants using consensus method."""
        # Convert states to binary representations
        minterms = []
        for state in states:
            minterm = format(state, f'0{n_vars}b')
            minterms.append(minterm)
        
        # This is a simplified version - in practice, you'd use the full Quine-McCluskey algorithm
        return [(minterm, frozenset([state])) for state, minterm in zip(states, minterms)]
    
    def _find_minimal_cover(self, prime_implicants: List[Tuple], states: Set[int]) -> List[Tuple]:
        """Find minimal cover of prime implicants."""
        # Simplified greedy approach
        uncovered = set(states)
        cover = []
        
        while uncovered:
            # Find implicant that covers the most uncovered states
            best_implicant = None
            best_coverage = 0
            
            for implicant in prime_implicants:
                coverage = len(implicant[1] & uncovered)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_implicant = implicant
            
            if best_implicant:
                cover.append(best_implicant)
                uncovered -= best_implicant[1]
            else:
                break
        
        return cover
    
    def _implicants_to_expression(self, implicants: List[Tuple]) -> str:
        """Convert implicants to Boolean expression."""
        terms = []
        for implicant_pattern, _ in implicants:
            terms.append(f"({implicant_pattern})")
        return " ∨ ".join(terms)
    
    def _select_best_optimization(self, karnaugh_result: Dict, pattern_result: Dict, espresso_result: Dict) -> Dict:
        """Select the best optimization result."""
        results = [karnaugh_result, pattern_result, espresso_result]
        
        # Score each result
        def score_result(result):
            quality_scores = {'optimal': 10, 'good': 7, 'exact': 5, 'none': 0}
            quality_score = quality_scores.get(result['quality'], 0)
            complexity_score = result['complexity_reduction'] * 2
            return quality_score + complexity_score
        
        best_result = max(results, key=score_result)
        return best_result
    
    def _determine_quantum_implementation(self, optimization_result: Dict, boolean_func: BooleanFunction) -> Dict[str, any]:
        """Determine optimal quantum circuit implementation strategy."""
        expr = optimization_result['expression']
        n_vars = boolean_func.n_variables
        
        # Quantum implementation strategies
        if expr == 'FALSE':
            return {'type': 'none', 'description': 'No gate needed', 'quantum_cost': 0}
        
        elif expr == 'TRUE':
            return {'type': 'constant', 'description': 'Constant rotation (unconditional)', 'quantum_cost': 1}
        
        elif expr in ['x0', 'x1', 'x2', 'x3']:
            # Single variable like 'x0', 'x1', etc.
            var_num = int(expr[1:])
            return {'type': 'single_control', 'control_qubit': var_num, 'description': f'Single controlled gate on x{var_num}', 'quantum_cost': 1}
        
        elif expr in ['¬x0', '¬x1', '¬x2', '¬x3']:
            # Negated single variable like '¬x0'
            var_num = int(expr[2:])
            return {'type': 'negated_control', 'control_qubit': var_num, 'description': f'X + controlled gate + X on x{var_num}', 'quantum_cost': 3}
        
        elif expr == 'x0 ⊕ x1':
            # XOR between x0 and x1
            return {'type': 'xor_control', 'description': 'XOR logical form (CNOT + CRX + CNOT)', 'quantum_cost': 3}
        
        elif expr == 'x0 ⊙ x1':
            # XNOR between x0 and x1  
            return {'type': 'xnor_control', 'description': 'XNOR logical form', 'quantum_cost': 4}
        
        elif 'x0 ∧ (x1 ⊕ x2)' in expr:
            # Complex overlapping pattern
            return {'type': 'complex_overlap', 'description': 'Complex overlapping form: x0 ∧ (x1 ⊕ x2)', 'quantum_cost': 5}
        
        elif '∧' in expr and '∨' not in expr and '⊕' not in expr:
            # Pure AND expression
            n_controls = expr.count('x')
            return {'type': 'multi_control', 'description': f'{n_controls}-controlled gate', 'quantum_cost': 2**(n_controls-1)}
        
        elif '∨' in expr:
            # OR expression - more complex
            n_terms = expr.count('∨') + 1
            return {'type': 'multi_term', 'description': f'Multi-term OR with {n_terms} terms', 'quantum_cost': n_terms * 3}
        
        else:
            # General case
            n_states = len(boolean_func.satisfying_states)
            return {'type': 'general', 'description': f'General implementation for {n_states} states', 'quantum_cost': n_states * 2}
    
    def _states_to_dnf(self, states: Set[int], n_vars: int) -> str:
        """Convert states to Disjunctive Normal Form."""
        if not states:
            return "FALSE"
        
        terms = []
        for state in sorted(states):
            state_bits = format(state, f'0{n_vars}b')
            term_parts = []
            for i, bit in enumerate(state_bits):
                var_idx = i  # MSB first ordering
                if bit == '1':
                    term_parts.append(f"x{var_idx}")
                else:
                    term_parts.append(f"¬x{var_idx}")
            terms.append("(" + " ∧ ".join(term_parts) + ")")
        
        return " ∨ ".join(terms)
    
    def _create_false_result(self) -> Dict[str, any]:
        """Create result for FALSE function."""
        return {
            'original_function': 'FALSE',
            'simplified_expression': 'FALSE',
            'simplification_method': 'trivial',
            'implementation': {'type': 'none', 'description': 'No gate needed', 'quantum_cost': 0},
            'complexity_reduction': 0,
            'quantum_cost': 0,
            'optimization_quality': 'optimal'
        }


class ControlPatternAnalyzer:
    """
    Analyzes control pattern relationships for optimization decisions with integrated Boolean optimization.
    
    CORRECTED VERSION: Fixed pattern interpretation to match multi_crx convention.
    """
    
    def __init__(self):
        self.boolean_optimizer = BooleanOptimizer()
    
    def classify_pattern_relationships(self, patterns: List[str]) -> Dict[PatternRelationship, List[Tuple[str, str]]]:
        """Classify all pairwise relationships between patterns."""
        relationships = {rel: [] for rel in PatternRelationship}
        
        # Find identical patterns
        pattern_counts = Counter(patterns)
        for pattern, count in pattern_counts.items():
            if count > 1:
                for i in range(count - 1):
                    relationships[PatternRelationship.IDENTICAL].append((pattern, pattern))
        
        # Analyze pairwise relationships for unique patterns
        unique_patterns = list(set(patterns))
        for i in range(len(unique_patterns)):
            for j in range(i + 1, len(unique_patterns)):
                pattern1, pattern2 = unique_patterns[i], unique_patterns[j]
                relationship = self._classify_pair_relationship(pattern1, pattern2)
                relationships[relationship].append((pattern1, pattern2))
        
        return relationships
    
    def _classify_pair_relationship(self, pattern1: str, pattern2: str) -> PatternRelationship:
        """Classify relationship between two patterns."""
        if pattern1 == pattern2:
            return PatternRelationship.IDENTICAL
        
        if len(pattern1) != len(pattern2):
            return PatternRelationship.DISJOINT
        
        # Check if patterns are disjoint (no overlapping states)
        if self._are_disjoint(pattern1, pattern2):
            return PatternRelationship.DISJOINT
        
        # Check for subset relationship  
        if self._is_subset(pattern1, pattern2) or self._is_subset(pattern2, pattern1):
            return PatternRelationship.SUBSET
        
        # Otherwise, they overlap
        return PatternRelationship.OVERLAPPING
    
    def _are_disjoint(self, pattern1: str, pattern2: str) -> bool:
        """Check if two patterns are disjoint (no common satisfying states)."""
        # Two patterns are disjoint if they differ in at least one bit position
        for b1, b2 in zip(pattern1, pattern2):
            if b1 != b2:  # Different requirements for this position
                return True
        return False
    
    def _is_subset(self, pattern1: str, pattern2: str) -> bool:
        """Check if pattern1 is a subset of pattern2."""
        # For exact pattern matching, this is simplified
        # In practice, this could be extended for wildcard patterns
        return False
    
    def analyze_patterns_for_optimization(self, patterns: List[str]) -> Dict[str, any]:
        """
        Analyze patterns and create optimized Boolean function.
        
        CORRECTED VERSION: Fixed pattern interpretation to match TeX file examples.
        
        Key insight: Pattern string interpretation for multi_crx:
        - pattern[i] corresponds to ctrl_qubits[i] which is variable xi
        - x0 is MSB (most significant bit) to match TeX file results
        - Direct conversion: int(pattern, 2) gives correct state
        """
        if not patterns:
            return {
                'total_patterns': 0,
                'boolean_function': None,
                'optimization_result': None,
                'optimization_potential': 'NONE'
            }
        
        # Determine number of control variables
        n_vars = len(patterns[0]) if patterns else 0
        
        # Convert patterns to satisfying states with CORRECT interpretation
        satisfying_states = []
        for pattern in patterns:
            # Direct conversion - no bit reversal needed
            # This matches the TeX file examples where:
            # pattern '11' and '10' -> states {3, 2} -> x0 (correct)
            # pattern '10' and '01' -> states {2, 1} -> x0 ⊕ x1 (correct)
            state_int = int(pattern, 2)
            satisfying_states.append(state_int)
        
        print(f"    Pattern to state conversion (direct):")
        for pattern, state in zip(patterns, satisfying_states):
            print(f"      '{pattern}' -> state {state}")
        
        # Create Boolean function
        boolean_func = BooleanFunction(satisfying_states, n_vars)
        
        # Optimize Boolean function
        optimization_result = self.boolean_optimizer.optimize_function(boolean_func)
        
        # Classify pattern relationships for additional analysis
        relationships = self.classify_pattern_relationships(patterns)
        
        return {
            'total_patterns': len(patterns),
            'unique_patterns': len(set(patterns)),
            'control_variables': n_vars,
            'satisfying_states': sorted(satisfying_states),
            'boolean_function': boolean_func,
            'optimization_result': optimization_result,
            'pattern_relationships': relationships,
            'optimization_potential': self._assess_optimization_potential(optimization_result),
            'quantum_implementation': optimization_result['implementation']
        }
    
    def _assess_optimization_potential(self, optimization_result: Dict) -> str:
        """Assess overall optimization potential based on results."""
        complexity_reduction = optimization_result['complexity_reduction']
        quantum_cost = optimization_result['quantum_cost']
        quality = optimization_result['optimization_quality']
        
        if quality == 'optimal' and complexity_reduction >= 2:
            return 'HIGH'
        elif quality in ['optimal', 'good'] and complexity_reduction >= 1:
            return 'MEDIUM'
        elif complexity_reduction > 0:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def analyze_optimization_potential(self, patterns: List[str]) -> Dict[str, any]:
        """Legacy method for backward compatibility."""
        if not patterns:
            return {
                'total_patterns': 0,
                'optimization_potential': 'NONE',
                'notes': 'No patterns to analyze'
            }
        
        optimization_analysis = self.analyze_patterns_for_optimization(patterns)
        relationships = optimization_analysis['pattern_relationships']
        
        return {
            'total_patterns': len(patterns),
            'unique_patterns': len(set(patterns)),
            'identical_pairs': len(relationships[PatternRelationship.IDENTICAL]),
            'disjoint_pairs': len(relationships[PatternRelationship.DISJOINT]),
            'overlapping_pairs': len(relationships[PatternRelationship.OVERLAPPING]),
            'subset_pairs': len(relationships[PatternRelationship.SUBSET]),
            'optimization_potential': optimization_analysis['optimization_potential'],
            'boolean_optimization': optimization_analysis['optimization_result'],
            'notes': 'Analysis based on pattern relationships and Boolean optimization'
        }