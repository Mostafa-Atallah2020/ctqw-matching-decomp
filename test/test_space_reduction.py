import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.utils.space_reduction import reduce_graph_space


class TestSpaceReduction:
    """Test cases for reduce_graph_space function"""

    def test_1_basic_parallel_reduction(self):
        """Test 1: Basic parallel edge reduction"""
        edges = {("000", "011"), ("100", "111")}
        active = [0, 1, 2]
        result_edges, result_active, transform_dict, transform_type = reduce_graph_space(
            edges, active, verbose=False
        )

        assert result_edges == {("00", "11")}
        assert result_active == [0, 1]
        assert transform_dict == {}

    def test_2_two_qubit_parallel(self):
        """Test 2: Two-qubit parallel edges"""
        edges = {("00", "01"), ("10", "11")}
        active = [0, 1]
        result_edges, result_active, transform_dict, transform_type = reduce_graph_space(
            edges, active, verbose=False
        )

        assert result_edges == {("0", "1")}
        assert result_active == [0]
        assert transform_dict == {}

    def test_3_two_qubit_parallel_alt(self):
        """Test 3: Two-qubit parallel edges alternative"""
        edges = {("00", "10"), ("01", "11")}
        active = [0, 1]
        result_edges, result_active, transform_dict, transform_type = reduce_graph_space(
            edges, active, verbose=False
        )

        assert result_edges == {("0", "1")}
        assert result_active == [1]
        assert transform_dict == {}

    def test_4_three_qubit_reduction(self):
        """Test 4: Three-qubit reduction"""
        edges = {("000", "001"), ("010", "011")}
        active = [0, 1, 2]
        result_edges, result_active, transform_dict, transform_type = reduce_graph_space(
            edges, active, verbose=False
        )

        assert result_edges == {("00", "01")}
        assert result_active == [0, 2]
        assert transform_dict == {}

    def test_5_three_qubit_reduction_alt(self):
        """Test 5: Three-qubit reduction alternative"""
        edges = {("100", "101"), ("110", "111")}
        active = [0, 1, 2]
        result_edges, result_active, transform_dict, transform_type = reduce_graph_space(
            edges, active, verbose=False
        )

        assert result_edges == {("10", "11")}
        assert result_active == [0, 2]
        assert transform_dict == {}

    def test_6_four_parallel_edges(self):
        """Test 6: Four parallel edges"""
        edges = {("000", "001"), ("010", "011"), ("100", "101"), ("110", "111")}
        active = [0, 1, 2]
        result_edges, result_active, transform_dict, transform_type = reduce_graph_space(
            edges, active, verbose=False
        )

        assert result_edges == {("0", "1")}
        assert result_active == [0]
        assert transform_dict == {}

    def test_7_two_edges_subset(self):
        """Test 7: Two edges subset"""
        edges = {("000", "001"), ("100", "101")}
        active = [0, 1, 2]
        result_edges, result_active, transform_dict, transform_type = reduce_graph_space(
            edges, active, verbose=False
        )

        assert result_edges == {("00", "01")}
        assert result_active == [0, 1]
        assert transform_dict == {}

    def test_8_two_edges_subset_alt(self):
        """Test 8: Two edges subset alternative"""
        edges = {("010", "011"), ("110", "111")}
        active = [0, 1, 2]
        result_edges, result_active, transform_dict, transform_type = reduce_graph_space(
            edges, active, verbose=False
        )

        assert result_edges == {("10", "11")}
        assert result_active == [0, 1]
        assert transform_dict == {}

    def test_9_non_parallel_requiring_transform(self):
        """Test 9: Non-parallel edges requiring transformation"""
        edges = {("110", "111"), ("000", "001")}
        active = [0, 1, 2]
        result_edges, result_active, transform_dict, transform_type = reduce_graph_space(
            edges, active, verbose=False
        )

        expected_edges = {("10", "11")}
        expected_active = [0, 1]
        expected_transform = {
            "000": "010",
            "001": "011",
            "010": "000",
            "011": "001",
            "100": "100",
            "101": "101",
            "110": "110",
            "111": "111",
        }

        assert result_edges == expected_edges
        assert result_active == expected_active
        assert transform_dict == expected_transform

    def test_10_non_parallel_requiring_transform_alt(self):
        """Test 10: Non-parallel edges requiring transformation alternative"""
        edges = {("010", "011"), ("100", "101")}
        active = [0, 1, 2]
        result_edges, result_active, transform_dict, transform_type = reduce_graph_space(
            edges, active, verbose=False
        )

        expected_edges = {("10", "11")}
        expected_active = [0, 2]
        expected_transform = {
            "000": "000",
            "001": "001",
            "010": "110",
            "011": "111",
            "100": "100",
            "101": "101",
            "110": "010",
            "111": "011",
        }
        assert result_edges == expected_edges
        assert result_active == expected_active
        assert transform_dict == expected_transform

    def test_11_four_qubit_transformation(self):
        """Test 11: Four-qubit non-parallel transformation"""
        edges = {("1010", "1011"), ("1100", "1101")}
        active = [0, 1, 2, 3]
        result_edges, result_active, transform_dict, transform_type = reduce_graph_space(
            edges, active, verbose=False
        )

        expected_edges = {("110", "111")}
        expected_active = [0, 2, 3]
        expected_transform = {
            "0000": "0000",
            "0001": "0001",
            "0010": "0010",
            "0011": "0011",
            "0100": "0100",
            "0101": "0101",
            "0110": "0110",
            "0111": "0111",
            "1000": "1000",
            "1001": "1001",
            "1010": "1110",
            "1011": "1111",
            "1100": "1100",
            "1101": "1101",
            "1110": "1010",
            "1111": "1011",
        }

        assert result_edges == expected_edges
        assert result_active == expected_active
        assert transform_dict == expected_transform

    def test_12_four_qubit_transformation_alt(self):
        """Test 12: Four-qubit transformation alternative"""
        edges = {("0000", "0001"), ("1100", "1101")}
        active = [0, 1, 2, 3]
        result_edges, result_active, transform_dict, transform_type = reduce_graph_space(
            edges, active, verbose=False
        )

        expected_edges = {("100", "101")}
        expected_active = [0, 1, 3]
        expected_transform = {
            "0000": "1000",
            "0001": "1001",
            "0010": "0010",
            "0011": "0011",
            "0100": "0100",
            "0101": "0101",
            "0110": "0110",
            "0111": "0111",
            "1000": "0000",
            "1001": "0001",
            "1010": "1010",
            "1011": "1011",
            "1100": "1100",
            "1101": "1101",
            "1110": "1110",
            "1111": "1111",
        }

        assert result_edges == expected_edges
        assert result_active == expected_active
        assert transform_dict == expected_transform

    # def test_13_two_qubit_non_parallel(self):
    #     """Test 13: Two-qubit non-parallel edges with transformation"""
    #     edges = {('01','10'), ('00','11')}
    #     active = [0, 1]
    #     result_edges, result_active, transform_dict, transform_type = \
    #         reduce_graph_space(edges, active, verbose=False)

    #     expected_edges = {('0', '1')}
    #     expected_active = [1]
    #     expected_transform = {'00': '00', '01': '01', '10': '11', '11': '10'}

    #     assert result_edges == expected_edges
    #     assert result_active == expected_active
    #     assert transform_dict == expected_transform

    # def test_14_complex_three_qubit(self):
    #     """Test 14: Complex three-qubit transformation"""
    #     edges = {('000', '111'), ('110', '001')}
    #     active = [0, 1, 2]
    #     result_edges, result_active, transform_dict, transform_type = \
    #         reduce_graph_space(edges, active, verbose=False)

    #     expected_edges = {('10', '11')}
    #     expected_active = [0, 1]
    #     expected_transform = {'000': '010', '001': '001', '010': '000', '011': '111',
    #                          '100': '100', '101': '101', '110': '110', '111': '011'}

    #     assert result_edges == expected_edges
    #     assert result_active == expected_active
    #     assert transform_dict == expected_transform

    # def test_15_complex_three_qubit_alt(self):
    #     """Test 15: Complex three-qubit transformation alternative"""
    #     edges = {('010', '101'), ('100', '011')}
    #     active = [0, 1, 2]
    #     result_edges, result_active, transform_dict, transform_type = \
    #         reduce_graph_space(edges, active, verbose=False)

    #     expected_edges = {('10', '11')}
    #     expected_active = [0, 1]
    #     expected_transform = {'000': '000', '001': '001', '010': '010', '011': '111',
    #                          '100': '110', '101': '101', '110': '100', '111': '011'}

    #     assert result_edges == expected_edges
    #     assert result_active == expected_active
    #     assert transform_dict == expected_transform


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
