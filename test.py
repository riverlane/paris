from projectq import MainEngine
from projectq.backends import Simulator
from projectq.ops import Rx, Ry, Rz, CNOT, All, Measure, H

from qiskit import QuantumRegister, QuantumCircuit, execute, BasicAer, ClassicalRegister

import numpy as np
import math

ket_theta = np.array([1/math.sqrt(2), 0, 1/np.sqrt(2), 0], dtype=complex)

# projectq simulator
engine = MainEngine(backend=Simulator(), engine_list=[])

# qiskit StatevectorSimulator from the Aer provider
simulator = BasicAer.get_backend('statevector_simulator')

if __name__ == "__main__":

    print()
    print("ProjectQ test:")
    print()
    qreg = engine.allocate_qureg(2)  # make a new simulator
    engine.backend.set_wavefunction(ket_theta, qreg)  # we've been given this state.

    H | qreg[0]
    CNOT | (qreg[0], qreg[1])

    All(Measure) | qreg
    engine.flush()

    print(engine.backend.cheat()[1])


    print()
    print("Qiskit test:")
    print()

    # Construct quantum circuit without measure
    qr = QuantumRegister(2, 'qr')
    cr = ClassicalRegister(2, 'cr')
    circ = QuantumCircuit(qr, cr)
    circ.h(qr[0])
    circ.cx(qr[0], qr[1])

    circ.measure(qr, cr)

    # Execute and get counts
    result = execute(circ, simulator).result()
    print(result.get_statevector(circ))