from typing import List
import logging
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, PhaseGate

INF_INT = 100000
logger = logging.getLogger(__name__)


def _getCpOps(bitstr: str):
    """
    Helper function for diagonal adjacency matrices with a single nonzero entry. The adjacency
    matrix is given by A=|bitstr><bitstr|. These are all phase gates and possibly bit flip
    gates.

    params:
    bitstr: single bit string identifying the nonzero element of the diagonal adjacency matrix.

    returns:
    list(str): A list of gate operations in strings.
    """
    zeros = [str(idx) for idx, elem in enumerate(bitstr) if elem != "1"]
    xops = ""
    if zeros:
        xops = "x_" + "|".join(zeros)
    ctrls = map(str, list(range(len(bitstr) - 1)))
    cpOp = "cp_" + "|".join(ctrls) + f",{list(range(len(bitstr)))[-1]}"
    return [xops, cpOp, xops]


def getHammingWt(bitstr: str):
    return len([elem for elem in bitstr if elem == "1"])


def _updatez2(z2: str, zeros1: List):
    """
    Helper function for updating z2. zeros1 is a list a of indices where we apply an x gate.

    params:
    z2: the bit string to be updated.
    zeros1: the list of indices where we will apply an x gate. In the protocol this corresponds to
    indices in z1 that have zeros (excluding the control bit that will be used to turn z2 to all ones.)

    returns: str: updated z2
    """
    z2 = list(z2)
    zeros1 = map(int, zeros1)
    for elem in zeros1:
        if z2[elem] == "1":
            z2[elem] = "0"
        else:
            z2[elem] = "1"
    return "".join(z2)


def getGateOps(z1: str, z2: str, little_endian: bool, target_qubit=None):
    """
    Returns the gate operations that implement the quantum walk. When the two strings are equal
    the adjacency matrix is A=|z><z|. When the two bit strings z1 and z2 are not equal, the adjacency
    matrix is A=|z1><z2|+|z2><z1|. Note that the qubit order is assumed to increase from
    left to right, e.g., 0123.

    Params:
    z1, z2: the bit strings that define the adjacency matrix.
    little_endian: keep True if you want the gates to match the qubit order on qiskit.
    In qiskit the top most qubit on the circuit is least significant qubit. It is stored on the
    zero index of the QuantumRegister.
    target_qubit: target qubit for the walk, i.e., it is the qubit that is target qubit of CRX or CP.

    Returns: list(str): A list of gate operations in strings. In the controled gates,
    the target and controls are separated by ','. Ex: crx_0|2|3,1 means controls
    on qubits 0,2,3, and target on qubit 1. Ex: x_0|2|3 means X gates on qubits 0, 2, and 3.
    """
    # print("z1: ", z1)
    # print("z2: ", z2)
    assert len(z1) == len(z2), "the two strings must be equal length."
    if little_endian:
        z1 = z1[::-1]
        z2 = z2[::-1]
    if z1 != z2:
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
        zeros1 = [str(idx) for idx, elem in enumerate(z1) if elem == "0"]
        zeros2 = [str(idx) for idx, elem in enumerate(z2) if elem == "0"]

        if target_qubit:
            ctrlBit2=str(target_qubit)
        else:
            ctrlBit2 = list(set(zeros1) - set(zeros2))[
                0
            ]  # Gets the first nonzero index for z2 that's a zero for z1.

        zeros1.remove(
            ctrlBit2
        )  # we will use this as a control to turn the z2 to all ones without affecting z1.
        z2 = _updatez2(z2, zeros1)  # apply the x gates on z2.
        # the z2 string might have changed so get the zeros from z2 again.
        zeros2 = [str(idx) for idx, elem in enumerate(z2) if elem == "0"]

        # Build the operations.
        cxOps = ["cx_" + ctrlBit2 + "," + elem for elem in zeros2]
        xOps = ""
        if zeros1:
            xOps = "x_" + "|".join(zeros1)
        crxCtrls = [str(elem) for elem in range(len(z1)) if str(elem) != ctrlBit2]
        crxOp = (
            "crx_" + "|".join(crxCtrls) + f",{ctrlBit2}"
        )  # ctrlBit2 has value zero in z1 and is the target of the crx.
        if not cxOps:
            cxOps = [""]
        return [xOps] + cxOps + [crxOp] + list(reversed(cxOps)) + [xOps]
    else:
        ops = _getCpOps(z1)
        return ops


def getOpsCirc(z1: str, z2: str, little_endian: bool, param: float = 0.25, target_qubit=None) -> QuantumCircuit:
    """Wrapper function. Takes a z1 and z2 and returns the corresponding circuit. Works for
    both single-edge and single-self loops.

    Params:
    z1, z2: the bit strings that define the adjacency matrix.
    little_endian: keep True if you want the gates to match the qubit order on qiskit.
    In qiskit the top most qubit on the circuit is least significant qubit. It is stored on the
    zero index of the QuantumRegister.
    param: angle value for crx or cp. Note that the code takes param and manipulates it.
    param is multiplied by 2 for rx and param for cp is multiplied by -1.
    This is so that mathematically rx=exp(-it(|z1><z2|+|z2><z1|). cp=(exp(-it(|z1><z1|))).
    target_qubit: target qubit for the walk, i.e., it is the qubit that is target qubit of CRX or CP.

    Returns:
    Corresponding QuantumCircuit."""
    gateops = getGateOps(z1, z2, little_endian, target_qubit)
    # print(gateops)
    circ = QuantumCircuit(len(z1))
    # implement ops
    for op in gateops:
        if op[0:2] == "x_":
            qubits = op[2:].split("|")
            for q in qubits:
                circ.x(int(q))
        elif op[0:3] == "cx_":
            cntrl_target = op[3:].split(",")
            cntrl_target = [int(elem) for elem in cntrl_target]
            circ.cx(cntrl_target[0], cntrl_target[1])
        elif op[0:3] == "cp_":
            cntrl_target = op[3:].split(",")
            cntrls = cntrl_target[0].split("|")
            target = int(cntrl_target[1])
            cntrls = [int(elem) for elem in cntrls]
            gate = PhaseGate(-param)  # Multiplies param by -1. See function header.
            mc_gate = gate.control(len(cntrls))
            circ.append(mc_gate, cntrls + [target])
        elif op[0:4] == "crx_":
            cntrl_target = op[4:].split(",")
            cntrls = cntrl_target[0].split("|")
            target = int(cntrl_target[1])
            cntrls = [int(elem) for elem in cntrls]
            gate = RXGate(2 * param)  # Multiplies param by 2. See function header.
            mc_gate = gate.control(len(cntrls))
            circ.append(mc_gate, cntrls + [target])
    return circ
