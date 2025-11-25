"""
Space reduction utilities for quantum walk graphs

Key fixes:
1. Qubit selection prefers keeping lower-indexed qubits when minimizing index sum
2. Transformation selection considers edge lexicographic ordering correctly
3. When multiple qubits can be dropped, prefer dropping higher-indexed (rightmost) qubits
"""

from itertools import combinations
import math


def is_reducible(edges, active_qubits, bitstring_length):
    """
    Check if edges form a reducible structure.

    Edges are reducible if:
    1. All flip the same set of qubits
    2. They form a proper parallel structure that allows dimensional reduction
    """
    if not edges:
        return True

    edge_list = list(edges)

    # Check if all flip same qubits
    flipping_per_edge = []
    for source, target in edge_list:
        flips = set()
        for q in active_qubits:
            pos = bitstring_length - 1 - q
            if source[pos] != target[pos]:
                flips.add(q)
        flipping_per_edge.append(flips)

    unique_patterns = set(frozenset(s) for s in flipping_per_edge)
    if len(unique_patterns) != 1:
        return False

    if not flipping_per_edge[0]:
        return True

    flipping_qubits = flipping_per_edge[0]
    non_flipping = sorted(set(active_qubits) - flipping_qubits, reverse=True)

    num_edges = len(edge_list)

    if num_edges == 1:
        return True

    is_power_of_2 = (num_edges & (num_edges - 1)) == 0 and num_edges > 0

    if not is_power_of_2:
        return False

    if len(non_flipping) == 0:
        return False

    expected_varying = int(math.log2(num_edges))

    if len(non_flipping) < expected_varying:
        return False

    varying_source = []
    varying_target = []

    for q in non_flipping:
        pos = bitstring_length - 1 - q
        src_vals = set(src[pos] for src, tgt in edge_list)
        tgt_vals = set(tgt[pos] for src, tgt in edge_list)

        if len(src_vals) > 1:
            varying_source.append(q)
        if len(tgt_vals) > 1:
            varying_target.append(q)

    if varying_source == varying_target and len(varying_source) == expected_varying:
        return True

    return False


def count_hamming_distance(v1, v2):
    """Count number of differing bits between two bitstrings."""
    return sum(c1 != c2 for c1, c2 in zip(v1, v2))


def compute_final_qubits_kept(transformed_edges, active_qubits, bitstring_length):
    """
    Compute which qubits would be kept after full reduction.

    Priority (in order):
    1. Minimize number of distinct edges
    2. Maximize number of qubits kept
    3. Maximize the indices of kept qubits (prefer keeping higher-indexed qubits, i.e., drop lower-indexed ones first)
    """
    edge_list = list(transformed_edges)

    flipping_qubits_per_edge = []
    for source, target in edge_list:
        edge_flips = set()
        for qubit_idx in active_qubits:
            pos = bitstring_length - 1 - qubit_idx
            if source[pos] != target[pos]:
                edge_flips.add(qubit_idx)
        flipping_qubits_per_edge.append(edge_flips)

    if not flipping_qubits_per_edge or not flipping_qubits_per_edge[0]:
        return set()

    flipping_qubits = flipping_qubits_per_edge[0]
    non_flipping_qubits = sorted(set(active_qubits) - flipping_qubits, reverse=True)

    num_edges = len(edge_list)
    is_power_of_2 = (num_edges & (num_edges - 1)) == 0 and num_edges > 0

    can_merge = False

    if is_power_of_2 and num_edges >= 2 and len(non_flipping_qubits) > 0:
        expected_varying_bits = int(math.log2(num_edges))

        if len(non_flipping_qubits) >= expected_varying_bits:
            varying_non_flipping_source = []
            varying_non_flipping_target = []

            for qubit_idx in non_flipping_qubits:
                pos = bitstring_length - 1 - qubit_idx
                source_values = set(source[pos] for source, target in edge_list)
                target_values = set(target[pos] for source, target in edge_list)

                if len(source_values) > 1:
                    varying_non_flipping_source.append(qubit_idx)
                if len(target_values) > 1:
                    varying_non_flipping_target.append(qubit_idx)

            if (
                varying_non_flipping_source == varying_non_flipping_target
                and len(varying_non_flipping_source) == expected_varying_bits
            ):
                can_merge = True

    if num_edges == 1 and flipping_qubits:
        can_merge = True

    if can_merge:
        # Priority: (min edges, max qubits, max 1s count, min sum of indices)
        best_score = (float("inf"), float("-inf"), float("-inf"), float("inf"))
        best_qubits = set(flipping_qubits)

        for flip_size in range(len(flipping_qubits), 0, -1):
            for flip_subset in combinations(sorted(flipping_qubits), flip_size):
                for non_flip_size in range(len(non_flipping_qubits) + 1):
                    for non_flip_subset in combinations(sorted(non_flipping_qubits), non_flip_size):
                        test_qubits = sorted(list(flip_subset) + list(non_flip_subset))
                        test_positions = sorted([bitstring_length - 1 - q for q in test_qubits])

                        test_projected = set()
                        valid_projection = True

                        for source, target in edge_list:
                            source_proj = "".join(source[pos] for pos in test_positions)
                            target_proj = "".join(target[pos] for pos in test_positions)

                            if source_proj == target_proj:
                                valid_projection = False
                                break

                            if source_proj <= target_proj:
                                test_projected.add((source_proj, target_proj))
                            else:
                                test_projected.add((target_proj, source_proj))

                        if valid_projection:
                            distinct_count = len(test_projected)
                            num_qubits = len(test_qubits)
                            index_sum = sum(test_qubits)  # Sum of qubit indices

                            # Count total 1s in kept qubits across all edges (clustering near higher basis)
                            ones_count = 0
                            for source, target in edge_list:
                                for pos in test_positions:
                                    ones_count += (source[pos] == "1") + (target[pos] == "1")

                            score = (
                                distinct_count,
                                -num_qubits,
                                -ones_count,
                                index_sum,
                            )  # Maximize 1s, then prefer lower indices

                            if score < best_score:
                                best_score = score
                                best_qubits = set(test_qubits)

        return best_qubits
    else:
        return set(active_qubits)


def generate_greedy_transformations(bitstring_length, edges=None):
    """
    Generate transformations using source/sink separation logic.

    Process source nodes first, then sink nodes:
    - For each node, find Hamming distance 1 candidates
    - Reject if candidate is in opposite set (source rejects sink, sink rejects source)
    - Reject if candidate has value <= current node value
    - Reject if candidate has Hamming distance > 1 from already transformed nodes in same set
    - Accept highest value valid candidate (or stay in place if none)
    """
    n = 2**bitstring_length
    all_vertices = [format(i, f"0{bitstring_length}b") for i in range(n)]

    if not edges:
        yield {v: v for v in all_vertices}
        return

    # Separate edges into source and sink sets based on decimal value parity
    # Sources: even decimal value
    # Sinks: odd decimal value
    edge_list = list(edges)
    sources = set()
    sinks = set()
    original_sources = set()  # Track original positions
    original_sinks = set()

    for source, target in edge_list:
        # Check if decimal value is even or odd
        source_value = int(source, 2)
        target_value = int(target, 2)

        if source_value % 2 == 0:
            sources.add(source)
            original_sources.add(source)
        else:
            sinks.add(source)
            original_sinks.add(source)

        if target_value % 2 == 0:
            sources.add(target)
            original_sources.add(target)
        else:
            sinks.add(target)
            original_sinks.add(target)

    # Sort nodes by value for processing order
    sorted_sources_asc = sorted(sources, key=lambda x: int(x, 2))
    sorted_sinks_asc = sorted(sinks, key=lambda x: int(x, 2))

    # Try both strategies (highest/lowest) and both processing orders (sources-first/sinks-first)
    # Always process nodes in ascending order by decimal value (lowest values first)
    for strategy in ["highest", "lowest"]:
        for process_first in ["sources", "sinks"]:
            sorted_sources = sorted_sources_asc
            sorted_sinks = sorted_sinks_asc

            transform_dict = {v: v for v in all_vertices}

            if process_first == "sources":
                # Transform sources first, then sinks
                transformed_sources = []
                occupied_positions = set(sources) | set(
                    sinks
                )  # Initially all edge vertices are occupied

                for source in sorted_sources:
                    # Find all Hamming distance 1 candidates
                    candidates = []
                    for bit_pos in range(bitstring_length):
                        flipped = list(source)
                        flipped[bit_pos] = "1" if flipped[bit_pos] == "0" else "0"
                        candidate = "".join(flipped)

                        candidate_value = int(candidate, 2)
                        source_value = int(source, 2)

                        # Reject if in original sink set
                        if candidate in original_sinks:
                            continue

                        # Reject if value is not higher
                        if candidate_value <= source_value:
                            continue

                        # Reject if position is already occupied by a transformed node
                        if candidate in transformed_sources:
                            continue

                        # Check Hamming distance with already transformed sources
                        valid = True
                        for transformed in transformed_sources:
                            if count_hamming_distance(candidate, transformed) > 1:
                                valid = False
                                break

                        if valid:
                            candidates.append((candidate_value, candidate))

                    # Select candidate based on strategy
                    if candidates:
                        if strategy == "lowest":
                            candidates.sort()  # Ascending - pick lowest value
                        else:
                            candidates.sort(reverse=True)  # Descending - pick highest value

                        _, best_candidate = candidates[0]
                        transform_dict[source] = best_candidate
                        # If best_candidate was originally a vertex that needs to swap
                        if (
                            best_candidate in all_vertices
                            and transform_dict.get(best_candidate, best_candidate) == best_candidate
                        ):
                            # This creates a swap
                            transform_dict[best_candidate] = source
                        transformed_sources.append(best_candidate)
                        occupied_positions.discard(source)  # Source position is now free
                        occupied_positions.add(best_candidate)  # New position is occupied
                    else:
                        # Stay in place
                        transform_dict[source] = source
                        transformed_sources.append(source)

                # Transform sinks
                transformed_sinks = []

                for sink in sorted_sinks:
                    # Find all Hamming distance 1 candidates
                    candidates = []
                    for bit_pos in range(bitstring_length):
                        flipped = list(sink)
                        flipped[bit_pos] = "1" if flipped[bit_pos] == "0" else "0"
                        candidate = "".join(flipped)

                        candidate_value = int(candidate, 2)
                        sink_value = int(sink, 2)

                        # Reject if in original source set
                        if candidate in original_sources:
                            continue

                        # Reject if value is not higher
                        if candidate_value <= sink_value:
                            continue

                        # Reject if position is already occupied by a transformed node
                        if candidate in transformed_sinks:
                            continue

                        # Check Hamming distance with already transformed sinks
                        valid = True
                        for transformed in transformed_sinks:
                            if count_hamming_distance(candidate, transformed) > 1:
                                valid = False
                                break

                        if valid:
                            candidates.append((candidate_value, candidate))

                    # Select candidate based on strategy
                    if candidates:
                        if strategy == "lowest":
                            candidates.sort()  # Ascending - pick lowest value
                        else:
                            candidates.sort(reverse=True)  # Descending - pick highest value

                        _, best_candidate = candidates[0]
                        transform_dict[sink] = best_candidate
                        # If best_candidate was originally a vertex that needs to swap
                        if (
                            best_candidate in all_vertices
                            and transform_dict.get(best_candidate, best_candidate) == best_candidate
                        ):
                            # This creates a swap
                            transform_dict[best_candidate] = sink
                        transformed_sinks.append(best_candidate)
                        occupied_positions.discard(sink)  # Sink position is now free
                        occupied_positions.add(best_candidate)  # New position is occupied
                    else:
                        # Stay in place
                        transform_dict[sink] = sink
                        transformed_sinks.append(sink)

            else:
                # Transform sinks first, then sources
                transformed_sinks = []
                occupied_positions = set(sources) | set(sinks)

                for sink in sorted_sinks:
                    # Find all Hamming distance 1 candidates
                    candidates = []
                    for bit_pos in range(bitstring_length):
                        flipped = list(sink)
                        flipped[bit_pos] = "1" if flipped[bit_pos] == "0" else "0"
                        candidate = "".join(flipped)

                        candidate_value = int(candidate, 2)
                        sink_value = int(sink, 2)

                        # Reject if in original source set
                        if candidate in original_sources:
                            continue

                        # Reject if value is not higher
                        if candidate_value <= sink_value:
                            continue

                        # Reject if position is already occupied by a transformed node
                        if candidate in transformed_sinks:
                            continue

                        # Check Hamming distance with already transformed sinks
                        valid = True
                        for transformed in transformed_sinks:
                            if count_hamming_distance(candidate, transformed) > 1:
                                valid = False
                                break

                        if valid:
                            candidates.append((candidate_value, candidate))

                    # Select candidate based on strategy
                    if candidates:
                        if strategy == "lowest":
                            candidates.sort()  # Ascending - pick lowest value
                        else:
                            candidates.sort(reverse=True)  # Descending - pick highest value

                        _, best_candidate = candidates[0]
                        transform_dict[sink] = best_candidate
                        # If best_candidate was originally a vertex that needs to swap
                        if (
                            best_candidate in all_vertices
                            and transform_dict.get(best_candidate, best_candidate) == best_candidate
                        ):
                            # This creates a swap
                            transform_dict[best_candidate] = sink
                        transformed_sinks.append(best_candidate)
                        occupied_positions.discard(sink)  # Sink position is now free
                        occupied_positions.add(best_candidate)  # New position is occupied
                    else:
                        # Stay in place
                        transform_dict[sink] = sink
                        transformed_sinks.append(sink)

                # Transform sources
                transformed_sources = []

                for source in sorted_sources:
                    # Find all Hamming distance 1 candidates
                    candidates = []
                    for bit_pos in range(bitstring_length):
                        flipped = list(source)
                        flipped[bit_pos] = "1" if flipped[bit_pos] == "0" else "0"
                        candidate = "".join(flipped)

                        candidate_value = int(candidate, 2)
                        source_value = int(source, 2)

                        # Reject if in original sink set
                        if candidate in original_sinks:
                            continue

                        # Reject if value is not higher
                        if candidate_value <= source_value:
                            continue

                        # Reject if position is already occupied by a transformed node
                        if candidate in transformed_sources:
                            continue

                        # Check Hamming distance with already transformed sources
                        valid = True
                        for transformed in transformed_sources:
                            if count_hamming_distance(candidate, transformed) > 1:
                                valid = False
                                break

                        if valid:
                            candidates.append((candidate_value, candidate))

                    # Select candidate based on strategy
                    if candidates:
                        if strategy == "lowest":
                            candidates.sort()  # Ascending - pick lowest value
                        else:
                            candidates.sort(reverse=True)  # Descending - pick highest value

                        _, best_candidate = candidates[0]
                        transform_dict[source] = best_candidate
                        # If best_candidate was originally a vertex that needs to swap
                        if (
                            best_candidate in all_vertices
                            and transform_dict.get(best_candidate, best_candidate) == best_candidate
                        ):
                            # This creates a swap
                            transform_dict[best_candidate] = source
                        transformed_sources.append(best_candidate)
                        occupied_positions.discard(source)  # Source position is now free
                        occupied_positions.add(best_candidate)  # New position is occupied
                    else:
                        # Stay in place
                        transform_dict[source] = source
                        transformed_sources.append(source)

            yield transform_dict


def find_basis_transformation(edges, active_qubits=None, max_attempts=100000, recursion_depth=0):
    """Find a basis transformation that leads to the best final reduction."""
    if not edges:
        return {}, None

    bitstring_length = len(next(iter(edges))[0])

    if active_qubits is None:
        active_qubits = list(range(bitstring_length))

    edge_list = list(edges)

    if is_reducible(edge_list, active_qubits, bitstring_length):
        return {}, None

    original_flipping_per_edge = []
    for source, target in edge_list:
        flips = set()
        for q in active_qubits:
            pos = bitstring_length - 1 - q
            if source[pos] != target[pos]:
                flips.add(q)
        original_flipping_per_edge.append(flips)

    original_flip_counts = [len(s) for s in original_flipping_per_edge]
    original_max_flipping = max(original_flip_counts) if original_flip_counts else 0

    best_transform_dict = None
    # Score: (num_flipping, num_kept, index_sum, -num_ones, -ones_in_kept, -edge_num_sum, -total_ones_transformed, -lex edges, hamming)
    best_score = (
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
        (),
        float("inf"),
    )

    attempt_count = 0
    for transform_dict in generate_greedy_transformations(bitstring_length, edges=edge_list):
        if attempt_count >= max_attempts:
            break
        attempt_count += 1

        transformed_edges_set = set()
        for source, target in edge_list:
            new_src = transform_dict[source]
            new_tgt = transform_dict[target]
            if new_src <= new_tgt:
                transformed_edges_set.add((new_src, new_tgt))
            else:
                transformed_edges_set.add((new_tgt, new_src))

        transformed_edges_list = list(transformed_edges_set)
        if not is_reducible(transformed_edges_list, active_qubits, bitstring_length):
            continue

        flipping_per_edge = []
        for source, target in transformed_edges_list:
            flips = set()
            for q in active_qubits:
                pos = bitstring_length - 1 - q
                if source[pos] != target[pos]:
                    flips.add(q)
            flipping_per_edge.append(flips)

        if not flipping_per_edge or not flipping_per_edge[0]:
            continue

        unique_patterns = set(frozenset(s) for s in flipping_per_edge)
        if len(unique_patterns) != 1:
            continue

        num_flipping = len(flipping_per_edge[0])

        if num_flipping > original_max_flipping:
            continue

        kept_qubits = compute_final_qubits_kept(
            transformed_edges_list, active_qubits, bitstring_length
        )
        num_kept = len(kept_qubits)
        index_sum = sum(kept_qubits) if kept_qubits else 0

        kept_positions = sorted([bitstring_length - 1 - q for q in kept_qubits])
        reduced_edges = []
        for source, target in transformed_edges_list:
            source_proj = "".join(source[pos] for pos in kept_positions)
            target_proj = "".join(target[pos] for pos in kept_positions)
            if source_proj <= target_proj:
                reduced_edges.append((source_proj, target_proj))
            else:
                reduced_edges.append((target_proj, source_proj))
        edge_tuple = tuple(sorted(reduced_edges))

        # Count number of 1s in final edges (prefer more 1s)
        num_ones = sum(s.count("1") + t.count("1") for s, t in reduced_edges)

        # Count total 1s in kept qubits across all transformed edges (clustering near higher basis)
        ones_count_in_kept = 0
        for source, target in transformed_edges_list:
            for pos in kept_positions:
                ones_count_in_kept += (source[pos] == "1") + (target[pos] == "1")

        # Calculate numerical sum of edges (prefer higher)
        edge_numerical_sum = sum(int(s, 2) + int(t, 2) for s, t in reduced_edges)

        # Count total 1s in ALL transformed edges (maximize to cluster near higher basis)
        total_ones_in_transformed_edges = 0
        for source, target in transformed_edges_list:
            total_ones_in_transformed_edges += source.count("1") + target.count("1")

        hamming_dist = sum(
            count_hamming_distance(orig, trans) for orig, trans in transform_dict.items()
        )

        # Score: minimize flipping, minimize kept, minimize index_sum (prefer lower qubit indices), maximize 1s in reduced edges, maximize 1s in kept qubits, maximize edge_num_sum, maximize total 1s in transformed edges, maximize edges lex, minimize hamming
        score = (
            num_flipping,
            num_kept,
            index_sum,
            -num_ones,
            -ones_count_in_kept,
            -edge_numerical_sum,
            -total_ones_in_transformed_edges,
            tuple(-ord(c) for e in edge_tuple for c in "".join(e)),
            hamming_dist,
        )

        if score < best_score:
            best_score = score
            best_transform_dict = transform_dict.copy()

    if best_transform_dict is None:
        return {}, None

    return best_transform_dict, ("permutation", best_score[0])


def apply_transformation(edges, transform_dict):
    """Apply a basis transformation to edges."""
    transformed_edges = set()
    for source, target in edges:
        new_source = transform_dict.get(source, source)
        new_target = transform_dict.get(target, target)

        if new_source <= new_target:
            transformed_edges.add((new_source, new_target))
        else:
            transformed_edges.add((new_target, new_source))

    return transformed_edges


def reduce_graph_space(edges, active_qubits=None, verbose=True, find_transform_recursion_depth=0):
    """
    Reduce the graph space by keeping only essential qubits.
    Automatically applies basis transformation if needed to make edges parallel.
    """
    if not edges:
        return edges, active_qubits if active_qubits is not None else [], {}, None

    bitstring_length = len(next(iter(edges))[0])

    if active_qubits is None:
        active_qubits = list(range(bitstring_length))

    transform_dict, transform_type = find_basis_transformation(
        edges, active_qubits, recursion_depth=find_transform_recursion_depth
    )

    if transform_dict:
        working_edges = apply_transformation(edges, transform_dict)
        if verbose:
            if transform_type and transform_type[0] == "permutation":
                print(
                    f"Applied basis transformation: permutation reducing to {transform_type[1]} flipping qubits"
                )
            print(f"Transformed edges: {working_edges}")
    else:
        working_edges = edges
        transform_type = None
        if verbose and len(edges) > 1:
            print("No transformation needed - edges already reducible")

    edge_list = list(working_edges)
    num_edges = len(working_edges)

    is_power_of_2 = (num_edges & (num_edges - 1)) == 0 and num_edges > 0

    flipping_qubits_per_edge = []
    for source, target in edge_list:
        edge_flips = set()
        for qubit_idx in active_qubits:
            pos = bitstring_length - 1 - qubit_idx
            if source[pos] != target[pos]:
                edge_flips.add(qubit_idx)
        flipping_qubits_per_edge.append(edge_flips)

    if not flipping_qubits_per_edge:
        return working_edges, active_qubits, transform_dict, transform_type

    flipping_qubits = flipping_qubits_per_edge[0]
    all_same_flips = all(edge_flips == flipping_qubits for edge_flips in flipping_qubits_per_edge)

    if not all_same_flips:
        return working_edges, active_qubits, transform_dict, transform_type

    if not flipping_qubits:
        return working_edges, active_qubits, transform_dict, transform_type

    non_flipping_qubits = sorted(set(active_qubits) - flipping_qubits, reverse=True)

    can_merge = False

    if is_power_of_2 and num_edges >= 2 and len(non_flipping_qubits) > 0:
        expected_varying_bits = int(math.log2(num_edges))

        if len(non_flipping_qubits) >= expected_varying_bits:
            varying_non_flipping_source = []
            varying_non_flipping_target = []

            for qubit_idx in non_flipping_qubits:
                pos = bitstring_length - 1 - qubit_idx
                source_values = set(source[pos] for source, target in edge_list)
                target_values = set(target[pos] for source, target in edge_list)

                if len(source_values) > 1:
                    varying_non_flipping_source.append(qubit_idx)
                if len(target_values) > 1:
                    varying_non_flipping_target.append(qubit_idx)

            if (
                varying_non_flipping_source == varying_non_flipping_target
                and len(varying_non_flipping_source) == expected_varying_bits
            ):
                can_merge = True

    if num_edges == 1 and flipping_qubits:
        can_merge = True

    if can_merge:
        # Priority: min edges, max qubits, max 1s count, min index sum (prefer lower indices)
        best_score = (float("inf"), float("-inf"), float("-inf"), float("inf"))
        best_qubits = list(flipping_qubits)

        for flip_size in range(len(flipping_qubits), 0, -1):
            for flip_subset in combinations(sorted(flipping_qubits), flip_size):
                for non_flip_size in range(len(non_flipping_qubits) + 1):
                    for non_flip_subset in combinations(sorted(non_flipping_qubits), non_flip_size):
                        test_qubits = sorted(list(flip_subset) + list(non_flip_subset))
                        test_positions = sorted([bitstring_length - 1 - q for q in test_qubits])

                        test_projected = set()
                        valid_projection = True

                        for source, target in edge_list:
                            source_proj = "".join(source[pos] for pos in test_positions)
                            target_proj = "".join(target[pos] for pos in test_positions)

                            if source_proj == target_proj:
                                valid_projection = False
                                break

                            if source_proj <= target_proj:
                                test_projected.add((source_proj, target_proj))
                            else:
                                test_projected.add((target_proj, source_proj))

                        if valid_projection:
                            distinct_count = len(test_projected)
                            num_qubits = len(test_qubits)
                            index_sum = sum(test_qubits)

                            # Count total 1s in kept qubits across all edges (clustering near higher basis)
                            ones_count = 0
                            for source, target in edge_list:
                                for pos in test_positions:
                                    ones_count += (source[pos] == "1") + (target[pos] == "1")

                            score = (
                                distinct_count,
                                -num_qubits,
                                -ones_count,
                                index_sum,
                            )  # Maximize 1s, then prefer lower indices

                            if score < best_score:
                                best_score = score
                                best_qubits = test_qubits
    else:
        return working_edges, active_qubits, transform_dict, transform_type

    qubits_to_keep = sorted(best_qubits)
    positions_to_keep = sorted([bitstring_length - 1 - q for q in qubits_to_keep])

    reduced_edges = set()
    for source, target in working_edges:
        source_proj = "".join(source[pos] for pos in positions_to_keep)
        target_proj = "".join(target[pos] for pos in positions_to_keep)

        if source_proj <= target_proj:
            reduced_edges.add((source_proj, target_proj))
        else:
            reduced_edges.add((target_proj, source_proj))

    return reduced_edges, qubits_to_keep, transform_dict, transform_type


def get_inverse_transformation(transform_dict, transform_type, bitstring_length):
    """Get the inverse transformation dictionary."""
    inverse_dict = {}
    for original, transformed in transform_dict.items():
        inverse_dict[transformed] = original
    return inverse_dict
