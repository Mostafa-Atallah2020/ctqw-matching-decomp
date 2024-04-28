from qiskit.circuit.library import RXGate


def multi_crx(angle, ctrl_state):
    n_ctrls = len(ctrl_state)
    gate = RXGate(angle).control(n_ctrls, ctrl_state=ctrl_state[::-1])
    return gate
