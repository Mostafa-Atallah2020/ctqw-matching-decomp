import re

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.parametervector import ParameterVector
from sympy import And, Not, Or, Symbol, simplify

from src.misc import multi_crx


class MCRX:
    def __init__(self, n_qubits, expr, target, rot_angle) -> None:
        self.expr = expr
        self.n_qubits = n_qubits
        self.target = target
        self.angle = rot_angle
        self.qc = QuantumCircuit(self.n_qubits)
        self.ctrls = self.__extract_controls()
        self.__get_qc()

    def __extract_controls(self):
        ctrls = []

        def process_term(term):
            ctrl_state = ""
            oper = []

            if isinstance(term, And):
                for subterm in term.args:
                    ctrl_state += process_subterm(subterm)
                    oper.append(get_number(subterm))

            else:
                ctrl_state = process_subterm(term)
                oper.append(get_number(term))

            ctrls.append((ctrl_state, oper))

        def process_subterm(subterm):
            if isinstance(subterm, Not):
                return "0"
            else:
                return "1"

        def get_number(term):
            name = term.name if isinstance(term, Symbol) else term.args[0].name
            return int(re.search(r"\d+", name).group())

        if isinstance(self.expr, Or):
            for term in self.expr.args:
                process_term(term)

        elif isinstance(self.expr, And):
            process_term(self.expr)

        elif isinstance(self.expr, Not):
            process_term(self.expr)

        elif isinstance(self.expr, Symbol):
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
            for tuple in self.ctrls:
                ctrl_state = tuple[0]
                oper = tuple[1]
                gate = multi_crx(self.angle, ctrl_state)
                self.qc.append(gate, oper + [self.target])

    def simplify(self):
        if len(self.expr.args) % 2 == 0:
            self.expr = simplify(self.expr)
        else:
            first = self.expr.args[0]
            all_except_first = Or(*self.expr.args[1:])
            simplified_terms = simplify(all_except_first)
            self.expr = simplified_terms | first

        return MCRX(self.n_qubits, self.expr, self.target, self.angle)
