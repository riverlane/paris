#!/usr/bin/env python3

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


## Running the simulator on two different input state:
#
input_state_zero = [1, 0]   # state |0>
input_state_one  = [0, 1]   # state |1>

for input_statevector in (input_state_zero, input_state_one):
    print( "Input statevector: {}".format(input_statevector) )
    output_statevector = execute(circ, simulator, backend_options={"initial_statevector": input_statevector}).result().get_statevector(circ)
    print( "Output statevector: {}".format(output_statevector) )

