from projectq.ops import *
from projectq import MainEngine
from projectq.backends import Simulator
import itertools
import numpy as np

def qubit_parity(num_qubits):
    return QubitOperator(
                " ".join([f"Z{i}" for i in range(num_qubits)])
           )

def example_general_discrete_problem_training_projectq(training_data):
    """The example training function for the users.
    This is for the discrete problems (staring with D), continuous problems
    have a different train function.

    Lots of opportunities exist for speeding this up. We have used this function internally
    to annotate all problems with the expected training time and will report if you beat that
    or not!
    """

    print(training_data)

    num_qubits = int(np.log2(len(training_data[0][0]))) # the wavefunction has 2**NQ elements.
    measurement = qubit_parity(num_qubits) # we provide this.

    # This is not every gate - you may need to extend this.
    allowable_gates = \
        [(H, i) for i in range(num_qubits)] + \
        [(X, i) for i in range(num_qubits)] + \
        [(CNOT, [i, i + 1]) for i in range(num_qubits - 1)]

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
        current_cost = 0
        for train_vector, train_label in training_data:

            engine = MainEngine(backend=Simulator(), engine_list=[])
            qreg = engine.allocate_qureg(num_qubits)  # make a new simulator
            engine.backend.set_wavefunction(train_vector, qreg)  # we've been given this state.
            engine.flush()

            for gate, qubitidx in current_circuit:
                gate | qreg[qubitidx]

            engine.flush()
            prediction = engine.backend.get_expectation_value(measurement, qreg)
            All(Measure) | qreg;
            del qreg  # clean up.

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

        engine = MainEngine(backend=Simulator(), engine_list=[])
        qreg = engine.allocate_qureg(num_qubits)
        engine.backend.set_wavefunction(wavefunction, qreg)

        for gate, qubitidx in best_circuit:
            gate | qreg[qubitidx]

        engine.flush()
        result = engine.backend.get_expectation_value(measurement, qreg)
        All(Measure) | qreg; engine.flush()

        return result

    return infer