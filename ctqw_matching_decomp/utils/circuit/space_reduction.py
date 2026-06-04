from typing import List, Tuple, Dict, Set, Any
from qiskit import QuantumCircuit
from ctqw_matching_decomp.utils.circuit.gate_ops import getOpsCirc, getHammingWt

def compress_edges_iteratively(
    edge_list: List[Tuple[str, str]]
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """
    edge_list: list of edges as bitstring pairs, e.g. [("00001","00100"), ...]

    Returns:
        compressed_merges: list of dicts, each with:
            - 'compressed_edge': (a', b') final compressed bitstrings
            - 'active_qubits': list[int] of remaining original qubit indices (0 = rightmost)
            - 'weight_reducing_qubits': list[int] of original indices whose deletion
              reduced the Hamming distance vs. the original edge

        unused_edges: list of original edges (as in edge_list) that were never
          part of any compression chain.
    """

    if not edge_list:
        return [], []

    def bitstring_to_int(b: str) -> int:
        return int(b, 2)

    def int_to_bitstring(x: int, length: int) -> str:
        return format(x, f'0{length}b')

    def delete_bit_int(x: int, k: int) -> int:
        """
        Remove bit at position k (0 = LSB); shift higher bits down by one.
        """
        low = x & ((1 << k) - 1)
        high = x >> (k + 1)
        return low | (high << k)

    class EdgeObj:
        __slots__ = ("u", "v", "bit_len", "active_qubits",
                     "orig_mask", "w_reducing", "members")

        def __init__(self, u, v, bit_len, active_qubits,
                     orig_mask, w_reducing, members):
            self.u = u
            self.v = v
            self.bit_len = bit_len
            self.active_qubits = active_qubits      # list of original indices
            self.orig_mask = orig_mask             # original a1 ^ a2
            self.w_reducing = w_reducing           # list of original indices
            self.members = members                 # set of original edge indices

    # Initialize one EdgeObj per original edge
    edges: List[EdgeObj] = []
    for idx, (a, b) in enumerate(edge_list):
        if len(a) != len(b):
            raise ValueError("All bitstrings in edge_list must have the same length.")
        bit_len = len(a)
        u = bitstring_to_int(a)
        v = bitstring_to_int(b)
        orig_mask = u ^ v
        active_qubits = list(range(bit_len))  # original qubit indices
        edges.append(
            EdgeObj(
                u=u,
                v=v,
                bit_len=bit_len,
                active_qubits=active_qubits,
                orig_mask=orig_mask,
                w_reducing=[],
                members={idx},
            )
        )

    # Iteratively merge as long as possible
    changed = True
    while changed:
        changed = False

        # Group by (active_qubits, weight_reducing_qubits)
        groups: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], List[int]] = {}
        for idx, e in enumerate(edges):
            key = (tuple(e.active_qubits), tuple(e.w_reducing))
            groups.setdefault(key, []).append(idx)

        to_remove: Set[int] = set()
        new_edges: List[EdgeObj] = []

        for key, idxs in groups.items():
            if len(idxs) < 2:
                continue

            # Map unordered vertex pair -> edge index within this group
            pair_to_idx: Dict[Tuple[int, int], int] = {}
            for idx in idxs:
                e = edges[idx]
                keyp = (min(e.u, e.v), max(e.u, e.v))
                pair_to_idx[keyp] = idx

            used_local: Set[int] = set()

            for idx in idxs:
                if idx in used_local or idx in to_remove:
                    continue

                e = edges[idx]
                u, v, bit_len = e.u, e.v, e.bit_len
                mask_curr = u ^ v

                # Try each current bit position
                merged_here = False
                for pos in range(bit_len):
                    s = 1 << pos
                    u_s = u ^ s
                    v_s = v ^ s
                    keyp = (min(u_s, v_s), max(u_s, v_s))
                    j = pair_to_idx.get(keyp)
                    if j is None or j == idx or j in used_local or j in to_remove:
                        continue

                    e2 = edges[j]
                    # Sanity checks to preserve logic
                    if e2.bit_len != bit_len:
                        continue
                    if (e2.u ^ e2.v) != mask_curr:
                        continue
                    if e2.orig_mask != e.orig_mask:
                        continue
                    if not e.members.isdisjoint(e2.members):
                        continue
                    if e.active_qubits != e2.active_qubits or e.w_reducing != e2.w_reducing:
                        continue

                    # Valid merge: compress along position 'pos'
                    new_u = delete_bit_int(u, pos)
                    new_v = delete_bit_int(v, pos)
                    new_bit_len = bit_len - 1

                    # Remove the corresponding original qubit index
                    new_active = e.active_qubits.copy()
                    orig_idx = new_active.pop(pos)

                    # Update weight_reducing_qubits if this bit reduces original Hamming weight
                    new_w_reducing = e.w_reducing.copy()
                    if (e.orig_mask >> orig_idx) & 1:
                        if orig_idx not in new_w_reducing:
                            new_w_reducing.append(orig_idx)

                    new_members = e.members.union(e2.members)

                    new_edges.append(
                        EdgeObj(
                            u=new_u,
                            v=new_v,
                            bit_len=new_bit_len,
                            active_qubits=new_active,
                            orig_mask=e.orig_mask,
                            w_reducing=new_w_reducing,
                            members=new_members,
                        )
                    )

                    used_local.add(idx)
                    used_local.add(j)
                    to_remove.add(idx)
                    to_remove.add(j)
                    changed = True
                    merged_here = True
                    break  # stop trying more bits for this edge

                # done positions for this edge

        if changed:
            # Keep only unmerged edges + newly created merged edges
            remaining_edges = [e for k, e in enumerate(edges) if k not in to_remove]
            edges = remaining_edges + new_edges

    # Build outputs
    used_orig_ids: Set[int] = set()
    compressed_merges = []
    for e in edges:
        if len(e.members) > 1:  # came from at least one merge
            compressed_merges.append({
                "compressed_edge": (
                    int_to_bitstring(e.u, e.bit_len),
                    int_to_bitstring(e.v, e.bit_len),
                ),
                "active_qubits": e.active_qubits,
                "weight_reducing_qubits": sorted(e.w_reducing),
            })
            used_orig_ids.update(e.members)

    unused_edges = [
        edge_list[i] for i in range(len(edge_list)) if i not in used_orig_ids
    ]

    return compressed_merges, unused_edges

def _get_target_qubit(z1, z2):
    z1=z1[::-1]
    z2=z2[::-1]
    if (
            getHammingWt(z2) == 0
            or getHammingWt(z1) == len(z1)
            or getHammingWt(z1) > getHammingWt(z2)
        ):
            # The adjacency matrix is symmetric so we're free to switch z1 and z2.
            # We do this check because we always turn z1 to hamming weight of n-1 and
            # z2 to Hamming weight of n. If z2 starts as all zeros this will not
            # work if we don't switch.
            # Also, if the hamming weight of z1 is n then we have to switch the strings.
            # Otherwise we can't find a position that is zero on z1 and a one on z2.
            temp = z1
            z1 = z2
            z2 = temp
    zeros1 = [idx for idx, elem in enumerate(z1) if elem == "0"]
    zeros2 = [idx for idx, elem in enumerate(z2) if elem == "0"]


    ctrlBit2 = list(set(zeros1) - set(zeros2))[
        0
    ]
    return ctrlBit2

def build_matching_circuit_iteratively(n_qubits, n_steps, delta_t, matchings):
    """
    Build quantum circuit using matching-based decomposition with space reduction. Assumes Qiskit ordering
    of the qubits.
    
    For each Trotter step:
        For each matching:
            1. Reduce the matching space
            2. For each reduced edge, get circuit and append to main circuit
    
    Args:
        n_qubits: Number of qubits
        n_steps: Number of Trotter steps
        delta_t: Time step size
        matchings: List of matchings, where each matching is a list of edges
        
    Returns:
        QuantumCircuit: Complete matching-based circuit with space reduction
    """    
    qc = QuantumCircuit(n_qubits)
    
    # Time per step
    dt = delta_t / n_steps
    
    # for step in range(n_steps):
    for matching in matchings:
        # Convert matching to set of edges with proper string format
        edges_set = set()
        
        for edge in matching:
            node1, node2 = edge
            
            # Ensure they are strings
            if isinstance(node1, int):
                z1 = format(node1, f'0{n_qubits}b')
            else:
                z1 = str(node1).zfill(n_qubits)
                
            if isinstance(node2, int):
                z2 = format(node2, f'0{n_qubits}b')
            else:
                z2 = str(node2).zfill(n_qubits)
            
            edges_set.add((z1, z2))
        
        # Reduce space for this matching
        reduced_edges, not_reduced_edges = compress_edges_iteratively(list(edges_set))
        
        # For each reduced edge, get circuit and append
        for edge_elem in reduced_edges:
            z1_reduced, z2_reduced= edge_elem["compressed_edge"]
            active_qubits=edge_elem["active_qubits"]
            weight_reducing_qubits=edge_elem["weight_reducing_qubits"]

            target_qubit_reduced=_get_target_qubit(z1_reduced, z2_reduced) # this is guaranteed to exist.
            target_qubit_full=active_qubits[target_qubit_reduced]

            # print(f"compressed edge: {edge_elem["compressed_edge"]}")
            # print(f"active qubits: {edge_elem["active_qubits"]}")
            # print(f"weight reducing qubits: {edge_elem["weight_reducing_qubits"]}")
            # print(f"target qubit reduced: {target_qubit_reduced}")
            # print(f"target qubit full: {target_qubit_full}")
            # print(f"z1: {z1_reduced}")
            # print(f"z2: {z2_reduced}")

            # cx gates for the weight reducing qubits.
            for w_qubit in weight_reducing_qubits:
                qc.cx(target_qubit_full, w_qubit)

            # Handle single qubit case: just an RX gate
            if len(z1_reduced) == 1:
                # Single qubit case: ('0', '1') or ('1', '0')
                # This is a simple RX gate on the active qubit
                # active_qubits[0] is the qubit index in the full register
                # target_qubit = active_qubits[0]
                
                # Apply RX gate to the target qubit
                qc.rx(2 * dt, target_qubit_full)  # Factor of 2 matches getOpsCirc convention
                # qc.barrier(label=f'M{matchings.index(matching)+1}')
                
            elif len(z1_reduced) == 0:
                # Edge case: no active qubits (shouldn't happen, but handle it)
                continue
                
            else:
                # Multi-qubit case: use getOpsCirc
                reduced_circ = getOpsCirc(z1_reduced, z2_reduced, 
                                            little_endian=True, param=dt, target_qubit=target_qubit_reduced)
                
                # Compose onto main circuit at active qubit positions
                qc.compose(reduced_circ, qubits=active_qubits, inplace=True)

            # cx gates for the weight reducing qubits.
            for w_qubit in weight_reducing_qubits:
                qc.cx(target_qubit_full, w_qubit)

        # edges that weren't compressed.
        for z1, z2 in not_reduced_edges:
            circ=getOpsCirc(z1, z2, 
                                            little_endian=True, param=dt)
            qc.compose(circ, inplace=True)
        # qc.barrier(label=f'M{matchings.index(matching)+1}')
    # manually repeat. more robust than using decompose
    qc_final=QuantumCircuit(n_qubits)
    for _ in range(n_steps):
        qc_final.compose(qc, inplace=True)
    return qc_final
