# File: src/pattern_analyzer.py
"""
Control pattern analysis for MCRX optimization using SymPy.
"""

from enum import Enum
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict
import sympy as sp
from sympy.logic import simplify_logic
from sympy.logic.boolalg import And, Or, Not, Xor


class PatternRelationship(Enum):
    """Types of relationships between control patterns."""
    DISJOINT = "disjoint"
    OVERLAPPING = "overlapping"
    SUBSET = "subset"
    IDENTICAL = "identical"


class ControlPatternAnalyzer:
    """
    Analyze control pattern relationships for optimization.
    Uses SymPy for boolean analysis.
    """
    
    def __init__(self):
        self.symbol_cache = {}
    
    def get_symbols(self, n_bits: int) -> List[sp.Symbol]:
        """Get SymPy symbols for n bits."""
        if n_bits not in self.symbol_cache:
            self.symbol_cache[n_bits] = [sp.Symbol(f'x{i}') for i in range(n_bits)]
        return self.symbol_cache[n_bits]
    
    def pattern_to_expr(self, pattern: str, symbols: List[sp.Symbol]) -> sp.Basic:
        """Convert control pattern to SymPy boolean expression."""
        expr_terms = []
        for i, bit in enumerate(pattern):
            if i < len(symbols):
                if bit == '1':
                    expr_terms.append(symbols[i])
                elif bit == '0':
                    expr_terms.append(Not(symbols[i]))
        
        if not expr_terms:
            return sp.true
        elif len(expr_terms) == 1:
            return expr_terms[0]
        else:
            return And(*expr_terms)
    
    def classify_pattern_relationships(self, patterns: List[str]) -> Dict[PatternRelationship, List[Tuple[str, str]]]:
        """
        Classify all pairwise relationships between patterns using SymPy.
        
        Args:
            patterns: List of control pattern strings
            
        Returns:
            Dictionary mapping relationship types to pattern pairs
        """
        relationships = {rel: [] for rel in PatternRelationship}
        
        # Find identical patterns
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        for pattern, count in pattern_counts.items():
            if count > 1:
                for i in range(count - 1):
                    relationships[PatternRelationship.IDENTICAL].append((pattern, pattern))
        
        # Analyze pairwise relationships for unique patterns using SymPy
        unique_patterns = list(set(patterns))
        if unique_patterns and len(unique_patterns[0]) > 0:
            n_bits = len(unique_patterns[0])
            symbols = self.get_symbols(n_bits)
            
            for i in range(len(unique_patterns)):
                for j in range(i + 1, len(unique_patterns)):
                    pattern1, pattern2 = unique_patterns[i], unique_patterns[j]
                    relationship = self._classify_pair_relationship_sympy(pattern1, pattern2, symbols)
                    relationships[relationship].append((pattern1, pattern2))
        
        return relationships
    
    def _classify_pair_relationship_sympy(self, pattern1: str, pattern2: str, 
                                        symbols: List[sp.Symbol]) -> PatternRelationship:
        """
        Classify relationship between two patterns using SymPy boolean algebra.
        """
        if pattern1 == pattern2:
            return PatternRelationship.IDENTICAL
        
        if len(pattern1) != len(pattern2):
            return PatternRelationship.DISJOINT
        
        # Convert patterns to SymPy expressions
        expr1 = self.pattern_to_expr(pattern1, symbols)
        expr2 = self.pattern_to_expr(pattern2, symbols)
        
        # Check if patterns are disjoint using SymPy
        if self._are_disjoint_sympy(expr1, expr2, symbols):
            return PatternRelationship.DISJOINT
        
        # Check for subset relationship using SymPy
        if self._is_subset_sympy(expr1, expr2, symbols):
            return PatternRelationship.SUBSET
        
        # Otherwise, they overlap
        return PatternRelationship.OVERLAPPING
    
    def _are_disjoint_sympy(self, expr1: sp.Basic, expr2: sp.Basic, 
                           symbols: List[sp.Symbol]) -> bool:
        """
        Check if two expressions are disjoint using SymPy.
        Two patterns are disjoint if their AND is always false.
        """
        try:
            and_expr = And(expr1, expr2)
            simplified = simplify_logic(and_expr)
            return simplified == sp.false
        except Exception:
            # Fallback to bit-by-bit comparison
            return self._are_disjoint_bitwise(str(expr1), str(expr2))
    
    def _are_disjoint_bitwise(self, pattern1: str, pattern2: str) -> bool:
        """Fallback bitwise disjoint check."""
        if len(pattern1) != len(pattern2):
            return True
        
        for b1, b2 in zip(pattern1, pattern2):
            if b1 != b2 and b1 in '01' and b2 in '01':
                return True
        return False
    
    def _is_subset_sympy(self, expr1: sp.Basic, expr2: sp.Basic, 
                        symbols: List[sp.Symbol]) -> bool:
        """
        Check if expr1 is a subset of expr2 using SymPy.
        expr1 is subset of expr2 if (expr1 AND NOT expr2) is always false.
        """
        try:
            # expr1 implies expr2 iff (expr1 & ~expr2) is false
            implication_test = And(expr1, Not(expr2))
            simplified = simplify_logic(implication_test)
            is_subset_12 = simplified == sp.false
            
            # Check the other direction
            implication_test = And(expr2, Not(expr1))
            simplified = simplify_logic(implication_test)
            is_subset_21 = simplified == sp.false
            
            return is_subset_12 or is_subset_21
        except Exception:
            return False
    
    def analyze_boolean_complexity(self, patterns: List[str]) -> Dict[str, Any]:
        """
        Analyze boolean complexity of pattern set using SymPy.
        
        Args:
            patterns: List of control pattern strings
            
        Returns:
            Dictionary with complexity analysis
        """
        if not patterns:
            return {'complexity': 'NONE', 'details': 'No patterns'}
        
        n_bits = len(patterns[0])
        unique_patterns = list(set(patterns))
        n_unique = len(unique_patterns)
        total_states = 2 ** n_bits
        
        analysis = {
            'total_patterns': len(patterns),
            'unique_patterns': n_unique,
            'pattern_density': n_unique / total_states,
            'bits': n_bits,
            'total_possible_states': total_states
        }
        
        # Use SymPy to analyze the combined boolean function
        try:
            symbols = self.get_symbols(n_bits)
            
            # Create OR of all unique patterns
            pattern_exprs = [self.pattern_to_expr(p, symbols) for p in unique_patterns]
            if len(pattern_exprs) == 1:
                combined_expr = pattern_exprs[0]
            else:
                combined_expr = Or(*pattern_exprs)
            
            # Simplify using SymPy
            simplified_expr = simplify_logic(combined_expr)
            
            # Analyze simplified expression
            expr_str = str(simplified_expr)
            analysis.update({
                'original_expression': str(combined_expr),
                'simplified_expression': str(simplified_expr),
                'and_operations': expr_str.count('&'),
                'or_operations': expr_str.count('|'),
                'not_operations': expr_str.count('~'),
                'xor_operations': expr_str.count('Xor'),
                'variables_used': len(simplified_expr.free_symbols),
                'expression_length': len(expr_str)
            })
            
            # Determine complexity level
            if simplified_expr == sp.true:
                analysis['complexity'] = 'TRIVIAL_TRUE'
                analysis['optimization_type'] = 'UNCONDITIONAL'
            elif simplified_expr == sp.false:
                analysis['complexity'] = 'TRIVIAL_FALSE'
                analysis['optimization_type'] = 'NO_OPERATION'
            elif simplified_expr.is_Symbol:
                analysis['complexity'] = 'MINIMAL'
                analysis['optimization_type'] = 'SINGLE_CONTROL'
            elif isinstance(simplified_expr, Xor):
                analysis['complexity'] = 'LOW'
                analysis['optimization_type'] = 'XOR_LOGICAL_FORM'
            elif isinstance(simplified_expr, And) and not any(isinstance(arg, Or) for arg in simplified_expr.args):
                analysis['complexity'] = 'LOW'
                analysis['optimization_type'] = 'MULTI_CONTROLLED'
            elif analysis['expression_length'] < len(str(combined_expr)) * 0.5:
                analysis['complexity'] = 'MODERATE'
                analysis['optimization_type'] = 'SIGNIFICANT_SIMPLIFICATION'
            else:
                analysis['complexity'] = 'HIGH'
                analysis['optimization_type'] = 'MINIMAL_SIMPLIFICATION'
                
        except Exception as e:
            analysis.update({
                'complexity': 'UNKNOWN',
                'optimization_type': 'ANALYSIS_FAILED',
                'error': str(e)
            })
        
        return analysis
    
    def get_logical_form_type(self, patterns: List[str]) -> str:
        """
        Determine the logical form type for a set of patterns using SymPy.
        
        Args:
            patterns: List of control pattern strings
            
        Returns:
            String describing the logical form type
        """
        if not patterns:
            return "empty"
        
        if len(patterns) == 1:
            return "single_gate"
        
        unique_patterns = list(set(patterns))
        if len(unique_patterns) < len(patterns):
            return "identical_merge"
        
        try:
            n_bits = len(patterns[0])
            symbols = self.get_symbols(n_bits)
            
            # Create OR expression for all patterns
            pattern_exprs = [self.pattern_to_expr(p, symbols) for p in unique_patterns]
            or_expr = Or(*pattern_exprs) if len(pattern_exprs) > 1 else pattern_exprs[0]
            
            # Simplify and classify
            simplified_expr = simplify_logic(or_expr)
            
            if simplified_expr.is_Symbol:
                return "maximal_simplification"
            elif isinstance(simplified_expr, Xor):
                return "xor_logical_form"
            elif isinstance(simplified_expr, And):
                return "and_logical_form"
            elif isinstance(simplified_expr, Or):
                # Check if it's a simple OR that couldn't be simplified
                if len(simplified_expr.args) < len(pattern_exprs):
                    return "partial_simplification"
                else:
                    return "disjoint_patterns"
            else:
                return "complex_logical_form"
                
        except Exception:
            return "analysis_failed"
    
    def find_optimization_opportunities(self, patterns: List[str]) -> Dict[str, Any]:
        """
        Find optimization opportunities using SymPy analysis.
        
        Args:
            patterns: List of control pattern strings
            
        Returns:
            Dictionary with optimization opportunities
        """
        opportunities = {
            'identical_merging': [],
            'boolean_simplification': None,
            'xor_detection': None,
            'maximal_simplification': None,
            'estimated_reduction': 0
        }
        
        if not patterns:
            return opportunities
        
        # Find identical patterns
        pattern_counts = Counter(patterns)
        for pattern, count in pattern_counts.items():
            if count > 1:
                opportunities['identical_merging'].append({
                    'pattern': pattern,
                    'count': count,
                    'reduction': count - 1
                })
        
        # Analyze boolean simplification potential
        try:
            complexity_analysis = self.analyze_boolean_complexity(patterns)
            opportunities['boolean_simplification'] = complexity_analysis
            
            # Specific pattern detection
            unique_patterns = list(set(patterns))
            n_bits = len(unique_patterns[0]) if unique_patterns else 0
            
            if n_bits > 0:
                symbols = self.get_symbols(n_bits)
                pattern_exprs = [self.pattern_to_expr(p, symbols) for p in unique_patterns]
                
                if len(pattern_exprs) > 1:
                    or_expr = Or(*pattern_exprs)
                    simplified_expr = simplify_logic(or_expr)
                    
                    # XOR detection
                    if isinstance(simplified_expr, Xor):
                        opportunities['xor_detection'] = {
                            'detected': True,
                            'expression': str(simplified_expr),
                            'implementation': 'CNOT + CRX + CNOT'
                        }
                    
                    # Maximal simplification detection
                    if simplified_expr.is_Symbol:
                        opportunities['maximal_simplification'] = {
                            'detected': True,
                            'simplified_to': str(simplified_expr),
                            'original_patterns': len(unique_patterns),
                            'optimized_gates': 1
                        }
                    
                    # Estimate total reduction
                    original_gates = len(patterns)
                    estimated_optimized = self._estimate_optimized_gates(simplified_expr, opportunities)
                    opportunities['estimated_reduction'] = max(0, original_gates - estimated_optimized)
        
        except Exception as e:
            opportunities['error'] = str(e)
        
        return opportunities
    
    def _estimate_optimized_gates(self, simplified_expr: sp.Basic, opportunities: Dict) -> int:
        """Estimate number of gates after optimization."""
        # Base estimate on expression structure
        if simplified_expr.is_Symbol:
            return 1  # Single control
        elif isinstance(simplified_expr, Xor):
            return 3  # CNOT + CRX + CNOT
        elif isinstance(simplified_expr, And):
            return 1  # Multi-controlled gate
        elif isinstance(simplified_expr, Or):
            return len(simplified_expr.args)  # Multiple gates
        else:
            # Conservative estimate
            expr_str = str(simplified_expr)
            return max(1, expr_str.count('&') + expr_str.count('|'))
