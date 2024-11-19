import re
import numpy as np
from qiskit import QuantumCircuit
from sympy import And, Not, Or, Symbol, simplify
from src import Expression
from src.misc import multi_crx


class MCRX:
    def __init__(self, n_qubits, expr, target, rot_angle) -> None:
        """
        Initialize MCRX with proper expression handling.
        """
        if isinstance(expr, Or):
            self.expr = expr
        elif isinstance(expr, Expression):
            self.expr = expr.expr
        elif isinstance(expr, bool):
            self.expr = expr
        else:
            raise TypeError("The Expression should be of type `Expression` or `sympy.Or`")

        self.n_qubits = n_qubits
        self.target = target
        self.angle = rot_angle

        self.ctrls = [] if isinstance(self.expr, bool) else self._extract_controls()
        self.qc = QuantumCircuit(self.n_qubits)
        self._get_qc()

    def _extract_controls(self):
        """Extract control information from expression."""
        ctrls = []

        def process_term(term):
            ctrl_state = ""
            ctrl_qubits = []
            used_qubits = set()

            if isinstance(term, And):
                for subterm in term.args:
                    qubit_num = get_number(subterm)
                    if qubit_num not in used_qubits and qubit_num != self.target:
                        ctrl_state += process_subterm(subterm)
                        ctrl_qubits.append(qubit_num)
                        used_qubits.add(qubit_num)
            else:
                qubit_num = get_number(term)
                if qubit_num != self.target:
                    ctrl_state = process_subterm(term)
                    ctrl_qubits.append(qubit_num)

            if ctrl_qubits:
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

        return ctrls

    def _get_qc(self):
        """Build quantum circuit from controls."""
        if self.expr == True:
            self.qc.rx(self.angle, self.target)
        elif self.expr == False:
            pass
        else:
            for ctrl_state, ctrl_qubits in self.ctrls:
                gate = multi_crx(self.angle, ctrl_state)
                self.qc.append(gate, ctrl_qubits + [self.target])

    def simplify(self):
        """
        Simplify MCRX by removing cancelled controls.
        """
        # Split into separate MCRXs based on Or terms
        mcrx_gates = []
        if isinstance(self.expr, Or):
            terms = self.expr.args
            for term in terms:
                mcrx = MCRX(self.n_qubits, Expression(term), self.target, self.angle)
                mcrx_gates.append(mcrx)
        else:
            mcrx_gates = [self]

        # Keep simplifying until no more changes
        changed = True
        while changed and len(mcrx_gates) > 1:
            changed = False
            i = 0
            
            while i < len(mcrx_gates) - 1:
                mcrx1 = mcrx_gates[i]
                mcrx2 = mcrx_gates[i+1]
                
                # Find controls with different states
                states1 = {}
                states2 = {}
                
                for state, qubits in mcrx1.ctrls:
                    for q, s in zip(qubits, state):
                        states1[q] = s
                        
                for state, qubits in mcrx2.ctrls:
                    for q, s in zip(qubits, state):
                        states2[q] = s
                
                # Identify cancelling controls
                cancel_qubits = {q for q in states1.keys() & states2.keys() 
                            if states1[q] != states2[q]}
                
                if cancel_qubits:
                    changed = True
                    
                    # Remove cancelled controls from both gates
                    new_gates = []
                    for mcrx in (mcrx1, mcrx2):
                        new_ctrls = []
                        
                        for state, qubits in mcrx.ctrls:
                            new_state = ""
                            new_qubits = []
                            for q, s in zip(qubits, state):
                                if q not in cancel_qubits:
                                    new_state += s
                                    new_qubits.append(q)
                            if new_qubits:
                                new_ctrls.append((new_state, new_qubits))
                        
                        if new_ctrls:
                            # Create expression from remaining controls
                            x = [Symbol(f'x_{q}') for q in range(self.n_qubits)]
                            term_factors = []
                            for state, qubits in new_ctrls:
                                for q, s in zip(qubits, state):
                                    term_factors.append(~x[q] if s == '0' else x[q])
                            new_expr = And(*term_factors) if len(term_factors) > 1 else term_factors[0]
                            
                            # Create new MCRX with updated controls and expression
                            new_mcrx = MCRX(self.n_qubits, Expression(new_expr), self.target, self.angle)
                            new_mcrx.ctrls = new_ctrls
                            new_gates.append(new_mcrx)
                        else:
                            # Create control-free MCRX
                            new_mcrx = MCRX(self.n_qubits, Expression(Symbol(f'x_{self.target}')), self.target, self.angle)
                            new_mcrx.ctrls = []
                            new_gates.append(new_mcrx)
                    
                    # Check if gates are now identical
                    if new_gates[0].ctrls == new_gates[1].ctrls:
                        mcrx_gates[i:i+2] = [new_gates[0]]
                    else:
                        mcrx_gates[i:i+2] = new_gates
                        i += 1
                else:
                    # Check if gates are already identical
                    if mcrx1.ctrls == mcrx2.ctrls:
                        mcrx_gates.pop(i+1)
                        changed = True
                    else:
                        i += 1
        
        # Return single gate or combine gates
        if len(mcrx_gates) == 1:
            return mcrx_gates[0]
        else:
            # Combine expressions from remaining gates
            final_expr = Or(*[m.expr for m in mcrx_gates])
            final_mcrx = MCRX(self.n_qubits, Expression(final_expr), self.target, self.angle)
            final_mcrx.ctrls = []
            for mcrx in mcrx_gates:
                final_mcrx.ctrls.extend(mcrx.ctrls)
            
            # Build circuit
            final_mcrx.qc = QuantumCircuit(self.n_qubits)
            for ctrl_state, ctrl_qubits in final_mcrx.ctrls:
                gate = multi_crx(self.angle, ctrl_state)
                final_mcrx.qc.append(gate, ctrl_qubits + [self.target])
            
            return final_mcrx