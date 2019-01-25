#!/usr/bin/env python3

from math import sqrt
from qiskit import QuantumCircuit, QuantumRegister, BasicAer, execute


## Setting up:
#
num_qubits = 1
qr = QuantumRegister(num_qubits, "qr")
circ = QuantumCircuit(qr)


## Building the circuit:
#
circ.x( qr[0] )     # inverting the first and only qubit


## Printing the circuit:
#
print( circ.draw().single_string() )


## Setting up the statevector simulator:
#
simulator = BasicAer.get_backend('statevector_simulator')


## Running the simulator with the circuit on different input states:
#
input_state_zero    = [1, 0]                        # state |0>     or Z+
input_state_one     = [0, 1]                        # state |1>     or Z-
input_state_plus    = [1/sqrt(2),   1/sqrt(2) ]     # state |+>     or X+
input_state_minus   = [1/sqrt(2),  -1/sqrt(2) ]     # state |->     or X-
input_state_eye     = [1/sqrt(2),  1j/sqrt(2) ]     # state |i>     or Y+
input_state_mye     = [1/sqrt(2), -1j/sqrt(2) ]     # state |-i>    or Y-

for input_statevector in (input_state_zero, input_state_one, input_state_plus, input_state_minus, input_state_eye, input_state_mye):
    print( "Input statevector: {}".format(input_statevector) )
    output_statevector = list( execute(circ, simulator, backend_options={"initial_statevector": input_statevector}).result().get_statevector(circ) )
    print( "Output statevector: {}".format(output_statevector) )
    print("")

