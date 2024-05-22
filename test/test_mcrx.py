import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from sympy import symbols

# from src import MCRX, multi_crx
from src import MCRX, multi_crx


def test_3_qubit_4_parallel_edges_expr():
    """
    Testing `MCRX` for the example in Fig 20 in the overleaf `main.tex` file with
    graph connection: {000-001,010-011,100-101,110-111}
    """
    target = 2
    n_qubits = 3
    rot_angle = np.pi / 2

    x0, x1 = symbols(r"x_0, x_1", Latex=True)
    expr = ~x0 & ~x1 | ~x0 & x1 | x0 & ~x1 | x0 & x1

    mcrx = MCRX(n_qubits, expr, target, rot_angle)
    mcrx_qc_before = mcrx.qc
    mcrx_qc_after = mcrx.simplify().qc

    qc_before = QuantumCircuit(n_qubits)
    gate = multi_crx(rot_angle, "11")
    qc_before.append(gate, range(n_qubits))
    gate = multi_crx(rot_angle, "10")
    qc_before.append(gate, range(n_qubits))
    gate = multi_crx(rot_angle, "01")
    qc_before.append(gate, range(n_qubits))
    gate = multi_crx(rot_angle, "00")
    qc_before.append(gate, range(n_qubits))

    qc_after = QuantumCircuit(n_qubits)
    qc_after.rx(rot_angle, target)

    assert Operator(mcrx_qc_before) == Operator(qc_before)
    assert Operator(mcrx_qc_after) == Operator(qc_after)


def test_3_qubit_3_parallel_edges_expr():
    """
    Testing `MCRX` for the example in Fig 19 in the overleaf `main.tex` file with
    graph connection: {000-001,010-011,100-101}
    """
    target = 2
    n_qubits = 3
    rot_angle = np.pi / 2

    x0, x1 = symbols(r"x_0, x_1", Latex=True)
    expr = ~x0 & ~x1 | ~x0 & x1 | x0 & ~x1

    mcrx = MCRX(n_qubits, expr, target, rot_angle)
    mcrx_qc_before = mcrx.qc
    mcrx_qc_after = mcrx.simplify().qc

    qc_before = QuantumCircuit(n_qubits)
    gate = multi_crx(rot_angle, "01")
    qc_before.append(gate, range(3))
    gate = multi_crx(rot_angle, "10")
    qc_before.append(gate, range(3))
    gate = multi_crx(rot_angle, "00")
    qc_before.append(gate, range(3))

    qc_after = QuantumCircuit(n_qubits)
    gate = multi_crx(rot_angle, "0")
    qc_after.append(gate, [0, 2])
    gate = multi_crx(rot_angle, "10")
    qc_after.append(gate, range(3))

    assert Operator(mcrx_qc_before) == Operator(qc_before)
    assert Operator(mcrx_qc_after) == Operator(qc_after)


def test_3_qubit_2_parallel_edges_expr():
    """
    Testing `MCRX` for the example in Fig 18 in the overleaf `main.tex` file with
    graph connection: {000-001, 100-101}
    """
    target = 2
    n_qubits = 3
    rot_angle = np.pi / 2

    x0, x1 = symbols(r"x_0, x_1", Latex=True)
    expr = ~x0 & ~x1 | x0 & ~x1

    mcrx = MCRX(n_qubits, expr, target, rot_angle)
    mcrx_qc_before = mcrx.qc
    mcrx_qc_after = mcrx.simplify().qc

    qc_before = QuantumCircuit(n_qubits)
    gate = multi_crx(rot_angle, "10")
    qc_before.append(gate, range(3))
    gate = multi_crx(rot_angle, "00")
    qc_before.append(gate, range(3))

    qc_after = QuantumCircuit(n_qubits)
    gate = multi_crx(rot_angle, "0")
    qc_after.append(gate, [1, 2])

    assert Operator(mcrx_qc_before) == Operator(qc_before)
    assert Operator(mcrx_qc_after) == Operator(qc_after)


def test_3_qubit_2_parallel_edges_expr2():
    """
    Testing `MCRX` for the example in Fig 21 a in the overleaf `main.tex` file with
    graph connection: {000-001,010-011}
    """
    target = 2
    n_qubits = 3
    rot_angle = np.pi / 2

    x0, x1 = symbols(r"x_0, x_1", Latex=True)
    expr = ~x0 & ~x1 | ~x0 & x1

    mcrx = MCRX(n_qubits, expr, target, rot_angle)
    mcrx_qc_before = mcrx.qc
    mcrx_qc_after = mcrx.simplify().qc

    qc_before = QuantumCircuit(n_qubits)
    gate = multi_crx(rot_angle, "01")
    qc_before.append(gate, range(3))
    gate = multi_crx(rot_angle, "00")
    qc_before.append(gate, range(3))

    qc_after = QuantumCircuit(n_qubits)
    gate = multi_crx(rot_angle, "0")
    qc_after.append(gate, [0, 2])

    assert Operator(mcrx_qc_before) == Operator(qc_before)
    assert Operator(mcrx_qc_after) == Operator(qc_after)


def test_3_qubit_2_parallel_edges_expr3():
    """
    Testing `MCRX` for the example in Fig 21 b in the overleaf `main.tex` file with
    graph connection: {100-110,101-111}
    """
    target = 1
    n_qubits = 3
    rot_angle = np.pi / 2

    x0, x2 = symbols(r"x_0, x_2", Latex=True)
    expr = x0 & ~x2 | x0 & x2

    mcrx = MCRX(n_qubits, expr, target, rot_angle)
    mcrx_qc_before = mcrx.qc
    mcrx_qc_after = mcrx.simplify().qc

    qc_before = QuantumCircuit(n_qubits)
    gate = multi_crx(rot_angle, "10")
    qc_before.append(gate, [0, 2, 1])
    gate = multi_crx(rot_angle, "11")
    qc_before.append(gate, [0, 2, 1])

    qc_after = QuantumCircuit(n_qubits)
    gate = multi_crx(rot_angle, "1")
    qc_after.append(gate, [0, 1])

    assert Operator(mcrx_qc_before) == Operator(qc_before)
    assert Operator(mcrx_qc_after) == Operator(qc_after)
