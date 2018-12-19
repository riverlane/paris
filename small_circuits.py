import projectq
from projectq.ops import *
from projectq import MainEngine, cengines
from projectq.backends import Simulator
import itertools
import random
import numpy as np
from sklearn import svm


############# UTILITY DEFS #############################

def N_qubit_parity(N):
    return QubitOperator(
                " ".join([f"Z{i}" for i in range(N)])
           )

class SinglyControlledGate(projectq.ops.ControlledGate):
    """The ProjectQ definition of a controlled gate expects to get a
    list of lists of qubits. This is confusing and inconsistant.
    This version just takes a list of [controlqubit, all, other, qubits].
    """

    def __init__(self, gate):
        super().__init__(gate, n=1)

    def __or__(self, qubits):
        import projectq.meta
        with projectq.meta.Control(qubits[0].engine, [qubits[0]]):
            self._gate | qubits[1:]

CNOT = SinglyControlledGate(X)

####################### PUBLIC #######################
def problemzero_example_train(training_data):
    # we ignore the training data as we will look at it by hand!
    print(training_data)

    def infer(wavefunction):
        engine = MainEngine(backend=Simulator(), engine_list=[])
        qreg = engine.allocate_qureg(1)
        engine.backend.set_wavefunction(wavefunction, qreg)

        H | qreg[0]

        engine.flush()
        result = engine.backend.get_expectation_value(QubitOperator("Z0"), qreg);
        All(Measure) | qreg; engine.flush(); del qreg # cleanup the simulator.
        return result

    return infer


def train_discrete_general_example(training_example_wfns):
    """The example trining function for the users.
    This is for the descrite problems (staring with D), continious problems
    have a diffrent train function.

    Lots of oppertunaties exist for speeding this up. We have used this function internally
    to annoate all problems with thr expected training time and will report if you beat that
    or not!
    """

    num_qubits = int(np.log2(len(training_example_wfns[0][0]))) # the wavefunction has 2**NQ elements.
    print(num_qubits)
    measurement = N_qubit_parity(num_qubits) # we provide this.

    # This is not every gate - you may need to extend this.
    allowable_gates = \
        [(H, i) for i in range(num_qubits)] + \
        [(X, i) for i in range(num_qubits)] + \
        [(CNOT, slice(i, i+2, 1)) for i in range(num_qubits-1)]


    print([(str(g), i) for g, i in allowable_gates])

    max_length = num_qubits * 2 # the total number of gates to consider

    # NOTE: some gates are not affected by ordering!
    # for example, 2 gates on 2 different qubits can be extchanged.
    # this is not considered here.
    possible_circuts = itertools.chain(*[
                        itertools.permutations(allowable_gates, r=NG)
                        for NG in range(max_length+1)
                       ])

    possible_circuts = list(possible_circuts)
    print(f"Number of possible circuits to consider: {len(possible_circuts)}")

    # if yuo want to print all the attempts, this can help:
    # for p in possible_circuts:
    #     print([(str(g), i) for g, i in p])

    best_cost = float('Inf')
    best_circuit = None
    for current_circuit in possible_circuts:
        current_cost = 0
        for train_vector, train_label in training_example_wfns:
            engine = MainEngine(backend=Simulator(), engine_list=[])
            qreg = engine.allocate_qureg(num_qubits) # make a new simulator
            engine.backend.set_wavefunction(train_vector, qreg) # we've been given this state.
            engine.flush()

            for gate, qubitidx in current_circuit:
                gate | qreg[qubitidx]

            engine.flush()
            prediction = engine.backend.get_expectation_value(measurement, qreg)
            All(Measure) | qreg; del qreg # clean up.
            current_cost += abs(train_label-prediction)

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


def train_svm(training_example_wfns):
    """This is a train function for any circuit ignoring all quantum properties.
    This will work given enough examples, but well be very slow!
    """
    clf = svm.SVC()
    vecs, labels = tuple(zip(*training_example_wfns))
    vecs = np.array(vecs); vecs = np.concatenate([vecs.real, vecs.imag], axis=1)

    clf.fit(vecs, labels)

    # now we create the inference function. This should take a state and produce a prediction.
    def infer(wavefunction):
        wavefunction = np.array(wavefunction).reshape(1, -1)
        return clf.predict(np.concatenate([wavefunction.real, wavefunction.imag], axis=1))[0]

    return infer


#### OLD STUFF ##################

def problemzero_exact_train(training_data):
    # we ignore the training data as we will look at it by hand!
    print(training_data)

    def infer(wavefunction):
        engine = MainEngine(backend=Simulator(), engine_list=[])
        qreg = engine.allocate_qureg(2)
        engine.backend.set_wavefunction(wavefunction, qreg)

        H | qreg[0]
        X | qreg[1]

        engine.flush()
        result = engine.backend.get_expectation_value(QubitOperator("Z0 Z1"), qreg);
        All(Measure) | qreg; engine.flush(); del qreg # cleanup the simulator.
        return result

    return infer


def train_2q(training_example_wfns):
    """This is a train function for an unknown circuit of 2 single qubit gates from a given set on each of 2 qubits.
    Using projectq we can encode information about the possible space of problem solutions and
    search over all.

    training_example_wfns: [(vector, label)]
        The |phi> vectors and the labels they should have when untransformed.

    returns: function([vector]) -> label
        the classifier function.
    """
    # this is the information we give people about the structure of the problem
    trial_possible_gates = [H, X, Ry(np.pi/2)]
    measurement = QubitOperator("Z0 Z1") # we provide this.

    # this solution is trying an exhaustive search.
    trial_circuits = list(itertools.product(trial_possible_gates, repeat=2))
    print(f"trying {len(trial_circuits)} possibilties")
    costs = []
    engine = MainEngine(backend=Simulator(), engine_list=[])

    # we try all possible 2 qubit circuits with the given structure.
    for gate0, gate1 in trial_circuits:
        cost = 0.0
        for example_wfn, label in training_example_wfns:
            qreg = engine.allocate_qureg(2) # make a new simulator
            engine.backend.set_wavefunction(example_wfn, qreg) # we've been given this state.
            gate0 | qreg[0] # apply the test gates
            gate1 | qreg[1]

            # get the expected cost if we did real measurements many times
            engine.flush()
            prediction = engine.backend.get_expectation_value(measurement, qreg)
            All(Measure) | qreg # clean up.
            cost += abs(label-prediction)
        costs.append(cost)
        print(".", end="", flush=True)
    print()

    # sort the circuits by the cost to pick the best
    best_circuits = sorted(zip(trial_circuits, costs), key = lambda tple: tple[1])
    best_circuit, best_cost = best_circuits[0]

    print(f"The best solution had cost {best_cost}, and gates {[str(g) for g in best_circuit]}.")

    # now we create the inference function. This should take a state and produce a prediction.
    def infer(wavefunction):
        engine = MainEngine(backend=Simulator(), engine_list=[])
        qreg = engine.allocate_qureg(2)
        engine.backend.set_wavefunction(wavefunction, qreg)
        best_circuit[0] | qreg[0]
        best_circuit[1] | qreg[1]
        engine.flush()
        result = engine.backend.get_expectation_value(measurement, qreg); All(Measure) | qreg; engine.flush()
        return result

    # The only structure we are ENFORCING is that the uers provide us with this
    # function. We will provide quantum examples but people are free to
    # use whatever they want.
    # It only makes sense to run this function in the same python process as
    # it was created in, but that's ok.
    return infer
