import re

import numpy as np
from qiskit import QuantumCircuit
from sympy import And, Not, Or, Symbol, simplify

from src import Expression
from src.misc import multi_crx


class MCRX:
    def __init__(self, n_qubits, expr, target, rot_angle) -> None:
        if isinstance(expr, Or):
            self.expr = expr
        elif isinstance(expr, Expression):
            self.expr = expr.expr
        else:
            raise TypeError("The Expression should be of type `Expression` or `sympy.Or`")

        self.n_qubits = n_qubits
        self.target = target
        self.angle = rot_angle

        self.ctrls = self.__extract_controls()
        self.qc = QuantumCircuit(self.n_qubits)
        self.__get_qc()

    def __extract_controls(self):
        ctrls = []

        def process_term(term):
            ctrl_state = ""
            ctrl_qubits = []

            if isinstance(term, And):
                for subterm in term.args:
                    ctrl_state += process_subterm(subterm)
                    ctrl_qubits.append(get_number(subterm))

            else:
                ctrl_state = process_subterm(term)
                ctrl_qubits.append(get_number(term))

            ctrls.append((ctrl_state, ctrl_qubits))

        def process_subterm(subterm):
            if isinstance(subterm, Not):
                return "0"
            else:
                return "1"

        def get_number(term):
            name = str(term) if isinstance(term, Symbol) else str(term.args[0])

            return int(re.search(r"\d+", name).group())

        if isinstance(self.expr, Or):
            for term in self.expr.args:
                process_term(term)

        elif isinstance(self.expr, (And, Not, Symbol)):
            process_term(self.expr)

        else:
            pass

        return ctrls

    def __get_qc(self):
        if self.expr == True:
            self.qc.rx(self.angle, self.target)
        elif self.expr == False:
            pass
        else:
            for ctrl_state, ctrl_qubits in self.ctrls:
                gate = multi_crx(self.angle, ctrl_state)
                self.qc.append(gate, ctrl_qubits + [self.target])

    def simplify(self):
        if len(self.expr.free_symbols) == 1:
            simplified_expr = Expression(self.expr)
        else:
            simplified_expr = Expression(self.expr).simplify()
        return MCRX(self.n_qubits, simplified_expr, self.target, self.angle)
