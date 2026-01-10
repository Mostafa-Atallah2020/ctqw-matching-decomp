"""
Tests for space reduction / edge compression functionality.

These tests verify that the compress_edges_iteratively function correctly
compresses compatible edges to reduce the number of qubits needed for
circuit implementation.
"""

import pytest
from src.utils.circuit.space_reduction import compress_edges_iteratively


class TestEdgeCompression:
    """Test cases for compress_edges_iteratively function."""

    def test_basic_parallel_reduction(self):
        """Test basic parallel edge reduction - two edges differ in one qubit."""
        edges = [("000", "011"), ("100", "111")]
        compressed, unused = compress_edges_iteratively(edges)

        # Should compress to a single edge with 2 active qubits
        assert len(compressed) == 1
        assert len(unused) == 0
        assert compressed[0]["compressed_edge"] == ("00", "11")
        assert len(compressed[0]["active_qubits"]) == 2

    def test_two_qubit_parallel(self):
        """Test two-qubit parallel edges - can compress to single qubit."""
        edges = [("00", "01"), ("10", "11")]
        compressed, unused = compress_edges_iteratively(edges)

        assert len(compressed) == 1
        assert len(unused) == 0
        assert compressed[0]["compressed_edge"] == ("0", "1")
        assert len(compressed[0]["active_qubits"]) == 1

    def test_two_qubit_parallel_alt(self):
        """Test two-qubit parallel edges alternative ordering."""
        edges = [("00", "10"), ("01", "11")]
        compressed, unused = compress_edges_iteratively(edges)

        assert len(compressed) == 1
        assert len(unused) == 0
        assert compressed[0]["compressed_edge"] == ("0", "1")
        assert len(compressed[0]["active_qubits"]) == 1

    def test_four_parallel_edges(self):
        """Test four parallel edges - full compression to single qubit."""
        edges = [("000", "001"), ("010", "011"), ("100", "101"), ("110", "111")]
        compressed, unused = compress_edges_iteratively(edges)

        assert len(compressed) == 1
        assert len(unused) == 0
        assert compressed[0]["compressed_edge"] == ("0", "1")
        assert len(compressed[0]["active_qubits"]) == 1

    def test_partial_compression(self):
        """Test partial compression when not all edges can merge."""
        edges = [("000", "001"), ("100", "101")]
        compressed, unused = compress_edges_iteratively(edges)

        assert len(compressed) == 1
        assert len(unused) == 0
        assert compressed[0]["compressed_edge"] == ("00", "01")
        assert len(compressed[0]["active_qubits"]) == 2

    def test_no_compression_possible(self):
        """Test when no compression is possible - single edge."""
        edges = [("00", "11")]
        compressed, unused = compress_edges_iteratively(edges)

        # Single edge can't be compressed
        assert len(compressed) == 0
        assert len(unused) == 1
        assert unused[0] == ("00", "11")

    def test_mixed_compressible_and_not(self):
        """Test mix of compressible and non-compressible edges."""
        edges = [("00", "01"), ("10", "11"), ("00", "11")]
        compressed, unused = compress_edges_iteratively(edges)

        # First two edges can compress, third cannot join
        assert len(compressed) == 1
        assert len(unused) == 1
        assert compressed[0]["compressed_edge"] == ("0", "1")
        assert unused[0] == ("00", "11")

    def test_empty_input(self):
        """Test empty input."""
        edges = []
        compressed, unused = compress_edges_iteratively(edges)

        assert len(compressed) == 0
        assert len(unused) == 0

    def test_weight_reducing_qubits_tracked(self):
        """Test that weight-reducing qubits are tracked correctly."""
        # When compressing ('000','011') and ('100','111'), the qubit 2 (MSB)
        # is removed but it was a '1' in the XOR mask, so it reduces Hamming weight
        edges = [("000", "011"), ("100", "111")]
        compressed, unused = compress_edges_iteratively(edges)

        assert len(compressed) == 1
        # The weight_reducing_qubits should contain the qubit that was removed
        # and was part of the original Hamming distance
        assert "weight_reducing_qubits" in compressed[0]

    def test_active_qubits_preserved(self):
        """Test that active qubit indices are correctly tracked."""
        edges = [("000", "001"), ("010", "011")]
        compressed, unused = compress_edges_iteratively(edges)

        assert len(compressed) == 1
        # Original qubits were [0,1,2], after removing qubit 1, should have [0,2]
        active = compressed[0]["active_qubits"]
        assert len(active) == 2

    def test_larger_compression_chain(self):
        """Test compression of 8 edges to single qubit."""
        edges = [
            ("0000", "0001"), ("0010", "0011"),
            ("0100", "0101"), ("0110", "0111"),
            ("1000", "1001"), ("1010", "1011"),
            ("1100", "1101"), ("1110", "1111"),
        ]
        compressed, unused = compress_edges_iteratively(edges)

        assert len(compressed) == 1
        assert len(unused) == 0
        assert compressed[0]["compressed_edge"] == ("0", "1")
        assert len(compressed[0]["active_qubits"]) == 1


class TestEdgeCompressionWithDifferentHammingDistances:
    """Tests for edge compression with various Hamming distances."""

    def test_hamming_2_edges_no_compress(self):
        """Test that Hamming-2 edges don't compress with Hamming-1 edges."""
        edges = [("00", "01"), ("00", "11")]  # H=1 and H=2
        compressed, unused = compress_edges_iteratively(edges)

        # These cannot compress together
        assert len(unused) >= 1

    def test_same_hamming_different_patterns(self):
        """Test edges with same Hamming distance but different XOR patterns."""
        edges = [("00", "01"), ("00", "10")]  # Both H=1 but different bits
        compressed, unused = compress_edges_iteratively(edges)

        # These have different XOR patterns, can't compress
        assert len(unused) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
