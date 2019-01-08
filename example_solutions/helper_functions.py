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
