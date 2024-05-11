import re

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.parametervector import ParameterVector
from sympy import Not, Or, simplify

from src.misc import multi_crx


class MCRX:
    def __init__(self, expr: Or, n_qubits, target) -> None:
        self.expr = expr
        self.n_qubits = n_qubits
        self.target = target
        self.t = ParameterVector("t")
        self.ctrls = []
        self.__get_ctrls()
        self.__get_qc()

    def __get_ctrls(self):
        for term in self.expr.args:
            ctrl_state = ""
            oper = []
            for subterm in term.args:
                if isinstance(subterm, Not):
                    ctrl_state += "0"  # append 0 for negation
                    name = subterm.args[0].name
                else:
                    ctrl_state += "1"  # append 1 for non-negated variable
                    name = subterm.name

                number = re.search(r"\d+", name).group()
                oper.append(int(number))

            self.ctrls.append((ctrl_state, oper))

    def __get_qc(self):
        angle = np.pi / 2  # TODO: make it for a general angle.
        self.qc = QuantumCircuit(self.n_qubits)
        for tuple in self.ctrls:
            ctrl_state = tuple[0]
            oper = tuple[1]
            gate = multi_crx(angle, ctrl_state)
            self.qc.append(gate, oper + [self.target])

    def simplify(self):
        if len(self.expr.args) % 2 == 0:
            self.expr = simplify(self.expr)
        else:
            first = self.expr.args[0]
            all_except_first = Or(*self.expr.args[1:])
            simplified_terms = simplify(all_except_first)
            self.expr = simplified_terms | first

        return MCRX(self.expr, self.n_qubits, self.target)
