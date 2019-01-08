import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

def parityOf(int_type):
    parity = 0
    while (int_type):
        parity = ~parity
        int_type = int_type & (int_type - 1)
    return(parity)

def compute_parity_exp_value(state_vector):

    print(f"parity for vecotr {state_vector}")
    exp = 0.0
    for idx, coeff in enumerate(state_vector):
        exp += coeff * coeff.conj() * (1 if parityOf(idx) == 0 else -1)

    return exp



def apply_hadamard_gate(circ: QuantumCircuit, q_reg: QuantumRegister, qubit_index: int) -> QuantumCircuit:

    circ.h(q_reg[qubit_index])
    return circ

def apply_pauli_x_gate(circ: QuantumCircuit, q_reg: QuantumRegister, qubit_index: int) -> QuantumCircuit:

    circ.x(q_reg[qubit_index])
    return circ

def apply_pauli_y_gate(circ: QuantumCircuit, q_reg: QuantumRegister, qubit_index: int) -> QuantumCircuit:

    circ.y(q_reg[qubit_index])
    return circ

def apply_pauli_z_gate(circ: QuantumCircuit, q_reg: QuantumRegister, qubit_index: int) -> QuantumCircuit:

    circ.z(q_reg[qubit_index])
    return circ

def apply_rx_gate(circ: QuantumCircuit, q_reg: QuantumRegister, qubit_index: int, angle: float) -> QuantumCircuit:

    circ.rx(angle, q_reg[qubit_index])
    return circ

def apply_ry_gate(circ: QuantumCircuit, q_reg: QuantumRegister, qubit_index: int, angle: float) -> QuantumCircuit:

    circ.ry(angle, q_reg[qubit_index])
    return circ

def apply_rz_gate(circ: QuantumCircuit, q_reg: QuantumRegister, qubit_index: int, angle: float) -> QuantumCircuit:

    circ.rz(angle, q_reg[qubit_index])
    return circ

def apply_cnot_gate(circ: QuantumCircuit, q_reg: QuantumRegister, qubit_index_control: int, qubit_index_target: int) \
        -> QuantumCircuit:

    circ.cx(q_reg[qubit_index_control], q_reg[qubit_index_target])
    return circ
