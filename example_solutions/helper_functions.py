import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

def compute_parity_exp_value(state_vector):

    z_op = np.array([[1, 0], [0, -1]])

    for i in range(int(np.log2(len(state_vector))) - 1):
        z_op = np.kron(z_op, z_op)

    return np.dot(state_vector, np.dot(z_op, state_vector.transpose()))

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