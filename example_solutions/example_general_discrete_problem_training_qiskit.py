from .helper_functions import *
from qiskit import QuantumCircuit, QuantumRegister, BasicAer, execute
import itertools

def example_general_discrete_problem_training_qiskit(training_data):
    """The example training function for the users.
    This is for the discrete problems (staring with D), continuous problems
    have a different train function.

    Lots of opportunities exist for speeding this up. We have used this function internally
    to annotate all problems with the expected training time and will report if you beat that
    or not!
    """

    num_qubits = int(np.log2(len(training_data[0][0]))) # the wavefunction has 2**NQ elements.

    # This is not every gate - you may need to extend this.
    allowable_gates = \
        [(apply_hadamard_gate, [i]) for i in range(num_qubits)] + \
        [(apply_pauli_x_gate, [i]) for i in range(num_qubits)] + \
        [(apply_cnot_gate, [i, i + 1]) for i in range(num_qubits - 1)]

    print([(str(g), i) for g, i in allowable_gates])

    max_length = num_qubits * 2 # the total number of gates to consider
    print(f"Maximum gate depth {max_length}")

    # NOTE: some gates are not affected by ordering!
    # for example, 2 gates on 2 different qubits can be exchanged.
    # this is not considered here.
    possible_circuits = itertools.chain(*[
                        itertools.permutations(allowable_gates, r=NG)
                        for NG in range(max_length+1)
                       ])

    possible_circuits = list(possible_circuits)
    print(f"Number of possible circuits to consider: {len(possible_circuits)}")

    print(possible_circuits)

    # if you want to print all the attempts, this can help:
    # for p in possible_circuits:
    #     print([(str(g), i) for g, i in p])

    best_cost = float('Inf')
    best_circuit = None
    for current_circuit in possible_circuits:
        if len(current_circuit) > 0:
            current_cost = 0
            for train_vector, train_label in training_data:

                simulator = BasicAer.get_backend('statevector_simulator')
                qr = QuantumRegister(num_qubits, "qr")
                circ = QuantumCircuit(qr)

                for gate, args in current_circuit:
                    circ = gate(circ, qr, *args)

                opts = {"initial_statevector": train_vector}
                result = execute(circ, simulator, backend_options=opts).result()
                print(result.get_statevector(circ))
                prediction = compute_parity_exp_value(result.get_statevector(circ))

                current_cost += abs(train_label - prediction)
                print(train_label)
                print(prediction)

            print(".", end="", flush=True)
            if current_cost < best_cost:
                best_circuit = current_circuit
                best_cost = current_cost
            # if best_cost == 0.0:
            #     break # done!

    print("done")
    print(f"best circuit: {[(str(g), i) for g, i in best_circuit]} with cost {best_cost}")

    # now we create the inference function. This should take a state and produce a prediction.
    def infer(wavefunction):

        simulator = BasicAer.get_backend('statevector_simulator')
        qr = QuantumRegister(num_qubits, "qr")
        circ = QuantumCircuit(qr)

        for gate, args in best_circuit:
            circ = gate(circ, qr, *args)

        opts = {"initial_statevector": wavefunction}
        result = execute(circ, simulator, backend_options=opts).result()
        result = compute_parity_exp_value(result.get_statevector(circ))

        return result

    return infer