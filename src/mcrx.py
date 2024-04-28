import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.parametervector import ParameterVector
from sympy import Not, Or

from src.misc import multi_crx


class MCRX:
    def __init__(self, expr: Or, n_qubits, target) -> None:
        self.expr = expr
        self.n_qubits = n_qubits
        self.target = target
        self.t = ParameterVector("t")
        self.__get_ctrls()
        self.__get_mcrx_gates()
        self.__get_qc()

    def __get_ctrls(self):
        self.ctrls = []
        for term in self.expr.args:
            ctrl_state = ""
            for subterm in term.args:
                if isinstance(subterm, Not):
                    ctrl_state += "0"  # append 0 for negation
                else:
                    ctrl_state += "1"  # append 1 for non-negated variable

            if ctrl_state in self.ctrls:
                self.ctrls.append(ctrl_state[::-1])
            else:
                self.ctrls.append(ctrl_state)

    def __get_mcrx_gates(self):
        self.mcrx_gates = []
        angle = np.pi / 2  # TODO: make it for a general angle
        for ctrl_state in self.ctrls:
            gate = multi_crx(angle, ctrl_state)
            self.mcrx_gates.append(gate)

    def __get_qc(self):
        self.qc = QuantumCircuit(self.n_qubits)
        for gate in self.mcrx_gates:
            if gate.num_qubits != self.n_qubits:
                raise ValueError("Enter a valid expression!")

            self.qc.append(gate, range(self.n_qubits))
