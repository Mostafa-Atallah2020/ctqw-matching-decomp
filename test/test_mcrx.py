import numpy as np
from qiskit import QuantumCircuit
from IPython.display import display
from sympy import symbols

from src import MCRX, multi_crx
from src.expression import Expression


def test_2_mcrx_3_qubits():
    x0, x2 = symbols(r"x_0, x_2", Latex=True)
    expr = x0 & ~x2 | x0 & x2
    expr = Expression(expr)
    assert display(expr.simplify().expr) == display(x0)


def test_3_mcrx_5_qubits():
    x1, x2, x3, x4 = symbols(r"x_1, x_2, x_3, x_4", Latex=True)
    expr = ~x1 & ~x2 & ~x3 & ~x4 | ~x1 & x2 & ~x3 & ~x4 | ~x1 & ~x2 & ~x3 & x4
    expr = Expression(expr)
    assert display(expr.simplify().expr) == display(~x1 & ~x2 & ~x3 & ~x4 | ~x1 & ~x3)


def test_5_mcrx_7_qubits():
    x0, x1, x2, x3, x4, x5, x6 = symbols(r"x_0, x_1, x_2, x_3, x_4,x_5, x_6", Latex=True)
    expr = (
        ~x0 & ~x1 & ~x2 & ~x4 & ~x5 & ~x6
        | x0 & ~x1 & ~x2 & ~x4 & ~x5 & ~x6
        | ~x0 & ~x1 & x2 & ~x4 & ~x5 & ~x6
        | ~x0 & ~x1 & ~x2 & x4 & ~x5 & ~x6
        | ~x0 & ~x1 & ~x2 & ~x4 & ~x5 & x6
    )
    expr = Expression(expr)
    assert display(expr.simplify().expr) == display(
        ~x1 & ~x5 & ~x6
        | ~x0 & ~x1 & ~x2 & ~x5
        | ~x0 & ~x1 & ~x2 & ~x4 & ~x5
        | ~x0 & ~x1 & ~x2 & ~x4 & ~x5 & ~x6
    )
