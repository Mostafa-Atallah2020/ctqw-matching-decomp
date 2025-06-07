# File: src/pattern_analyzer.py
"""
Control pattern analysis for MCRX optimization.
Analyzes relationships between control patterns to determine optimization strategy.

IMPORTANT: This module provides pattern classification tools. The actual
optimization effectiveness depends on the specific patterns, circuit structure,
and implementation complexity of the resulting Boolean functions.
"""

from enum import Enum
from typing import List, Dict, Tuple
from collections import defaultdict


class PatternRelationship(Enum):
    """
    Types of relationships between control patterns.
    
    Each relationship type enables different optimization strategies
    with varying performance characteristics.
    """
    DISJOINT = "disjoint"
    OVERLAPPING = "overlapping"
    SUBSET = "subset"
    IDENTICAL = "identical"


class ControlPatternAnalyzer:
    """
    Analyzes control pattern relationships for optimization decisions.
    
    This class provides systematic classification of pattern relationships
    but does not guarantee specific performance improvements. Actual
    optimization results depend on:
    - Pattern complexity and structure
    - Number of patterns involved
    - Boolean function optimization potential
    - Circuit size and qubit count
    """
    
    def __init__(self):
        pass
    
    def classify_pattern_relationships(self, patterns: List[str]) -> Dict[PatternRelationship, List[Tuple[str, str]]]:
        """
        Classify all pairwise relationships between patterns.
        
        Args:
            patterns: List of control pattern strings (e.g., ['10', '01', '11'])
            
        Returns:
            Dictionary mapping relationship types to pattern pairs
            
        Note: The presence of specific relationships suggests optimization
        potential but does not guarantee specific performance improvements.
        Results depend on the actual Boolean functions that can be constructed.
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
        
        # Analyze pairwise relationships for unique patterns
        unique_patterns = list(set(patterns))
        for i in range(len(unique_patterns)):
            for j in range(i + 1, len(unique_patterns)):
                pattern1, pattern2 = unique_patterns[i], unique_patterns[j]
                relationship = self._classify_pair_relationship(pattern1, pattern2)
                relationships[relationship].append((pattern1, pattern2))
        
        return relationships
    
    def _classify_pair_relationship(self, pattern1: str, pattern2: str) -> PatternRelationship:
        """
        Classify relationship between two patterns.
        
        Args:
            pattern1, pattern2: Control pattern strings
            
        Returns:
            Relationship type between the patterns
            
        Note: Classification is based on string comparison. More sophisticated
        analysis might be needed for complex pattern structures.
        """
        if pattern1 == pattern2:
            return PatternRelationship.IDENTICAL
        
        if len(pattern1) != len(pattern2):
            return PatternRelationship.DISJOINT
        
        # Check if patterns are disjoint
        if self._are_disjoint(pattern1, pattern2):
            return PatternRelationship.DISJOINT
        
        # Check for subset relationship
        if self._is_subset(pattern1, pattern2) or self._is_subset(pattern2, pattern1):
            return PatternRelationship.SUBSET
        
        # Otherwise, they overlap
        return PatternRelationship.OVERLAPPING
    
    def _are_disjoint(self, pattern1: str, pattern2: str) -> bool:
        """
        Check if two patterns are disjoint (no common satisfying states).
        
        Two patterns are disjoint if they have conflicting requirements
        for at least one bit position, meaning no computational basis
        state can satisfy both patterns simultaneously.
        """
        # Two patterns are disjoint if they have conflicting requirements
        # for at least one bit position
        for b1, b2 in zip(pattern1, pattern2):
            if b1 != b2:  # Different requirements for this position
                return True
        return False
    
    def _is_subset(self, pattern1: str, pattern2: str) -> bool:
        """
        Check if pattern1 is a subset of pattern2.
        
        Currently simplified for basic pattern matching.
        More sophisticated subset detection could be implemented
        for patterns with wildcards or don't-care bits.
        """
        # Pattern1 is subset if it's more specific than pattern2
        # For now, we consider exact matching only
        return False  # Simplified for logical form implementation
    
    def analyze_optimization_potential(self, patterns: List[str]) -> Dict[str, any]:
        """
        Analyze potential for optimization based on pattern relationships.
        
        Args:
            patterns: List of control pattern strings
            
        Returns:
            Dictionary with optimization potential analysis
            
        Important: This provides estimates only. Actual optimization
        results depend on circuit structure and Boolean function complexity.
        The analysis cannot predict exact gate count reductions.
        """
        if not patterns:
            return {
                'total_patterns': 0,
                'optimization_potential': 'NONE',
                'notes': 'No patterns to analyze'
            }
        
        relationships = self.classify_pattern_relationships(patterns)
        
        analysis = {
            'total_patterns': len(patterns),
            'unique_patterns': len(set(patterns)),
            'identical_pairs': len(relationships[PatternRelationship.IDENTICAL]),
            'disjoint_pairs': len(relationships[PatternRelationship.DISJOINT]),
            'overlapping_pairs': len(relationships[PatternRelationship.OVERLAPPING]),
            'subset_pairs': len(relationships[PatternRelationship.SUBSET]),
            'optimization_potential': 'UNKNOWN',
            'notes': 'Analysis based on pattern relationships'
        }

        # Determine optimization potential
        if analysis['identical_pairs'] > 0:
            analysis['optimization_potential'] = 'HIGH'
        elif analysis['overlapping_pairs'] > 0:
            analysis['optimization_potential'] = 'MEDIUM'
        elif analysis['disjoint_pairs'] > 0:
            analysis['optimization_potential'] = 'LOW'
        else:
            analysis['optimization_potential'] = 'NONE'

        return analysis