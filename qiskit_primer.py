#!/usr/bin/env python3

from qiskit import QuantumCircuit, QuantumRegister, BasicAer, execute


## Setting up:
#
num_qubits = 2
qr = QuantumRegister(num_qubits, "qr")
circ = QuantumCircuit(qr)


## Building the circuit:
#
circ.x(qr[0])
circ.h(qr[1])


## Printing the circuit:
#
print( circ.draw().single_string() )


## Setting up the statevector simulator:
#
simulator = BasicAer.get_backend('statevector_simulator')


## Running the simulator:
#
statevector = execute(circ, simulator).result().get_statevector(circ)
print( statevector )

