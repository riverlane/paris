import projectq
from projectq.ops import *
from projectq import MainEngine, cengines
from projectq.backends import Simulator
import itertools
import random
import numpy as np
from sklearn import svm

####################### PUBLIC #######################

def train(training_example_wfns):
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


def train_svm(training_example_wfns):
    """This is a train function for an unknown circuit of 2 single qubit gates from a given set on each of 2 qubits.
    """
    clf = svm.SVC(gamma='scale')
    vecs, labels = tuple(zip(*training_example_wfns))
    vecs = np.array(vecs); vecs = np.concatenate([vecs.real, vecs.imag], axis=1)
    clf.fit(vecs, labels)

    # now we create the inference function. This should take a state and produce a prediction.
    def infer(wavefunction):
        wavefunction = np.array(wavefunction).reshape(1, -1)
        return clf.predict(np.concatenate([wavefunction.real, wavefunction.imag], axis=1))

    return infer


############## INTERNAL #######################
# generate data for training.
# we measure ZZ - the parity of the string. we can calculate the parity
# of a basis state repr as a int by bin(s).count("1") % 2 (the number of ones)
# and then generate trial states by lin. combinations of these.

NQ = 2
basis_states = [i for i in range(0, 2**NQ)]
evens = [s for s in basis_states if bin(s).count("1") % 2 == 0]
odds = [s for s in basis_states if bin(s).count("1") % 2 == 1]

def int_to_basis_element(i, NQ=NQ):
    wfn = np.zeros((2**NQ,))
    wfn[i] = 1.0
    return wfn

# generate samples.
train_set = []
engine = MainEngine(backend=Simulator(), engine_list=[])

for _ in range(10):

    # generate a coefficent vector in complex space.
    weights_r = np.random.uniform(low=0.0, high=1.0, size=(2**(NQ-1),) )
    weights_theta = np.random.uniform(low=0.0, high=2*np.pi, size=(2**(NQ-1),) )
    weights = weights_r * np.exp(1j*weights_theta)
    weights /= np.linalg.norm(weights) # normalize

    label = random.choices([-1, 1])[0]
    if label == -1: # 1 == odds
        ket_theta = sum( [coeff * int_to_basis_element(i) for coeff, i in zip(weights, odds)] )
    else:
        ket_theta = sum( [coeff * int_to_basis_element(i) for coeff, i in zip(weights, evens)] )

    qreg = engine.allocate_qureg(2) # make a new simulator
    engine.backend.set_wavefunction(ket_theta, qreg) # we've been given this state.
    engine.flush()
    print(f"label {label} exp ZZ { engine.backend.get_expectation_value(QubitOperator('Z0 Z1'), qreg) }")

    H | qreg[0] # apply the test gates
    X | qreg[1]
    engine.flush()
    _, ket_phi = engine.backend.cheat()
    All(Measure) | qreg # clean up.

    train_set.append( (ket_phi, label) )

test = []
for state in basis_states:
    label = 1 if bin(state).count("1") % 2 == 0 else -1
    qreg = engine.allocate_qureg(2) # make a new simulator
    engine.backend.set_wavefunction(int_to_basis_element(state), qreg) # we've been given this state.
    H | qreg[0] # apply the test gates
    X | qreg[1]
    engine.flush()
    _, ket_phi = engine.backend.cheat(); All(Measure) | qreg; engine.flush()
    test.append( (ket_phi, label) )


################# EVALUATION #################
predict = train(train_set)
for testvec, testres in test:
    p = predict(testvec)
    print(testres, p)

print("SVM classifier")
predict_svm = train_svm(train_set)
for testvec, testres in test:
    p = predict_svm(testvec)
    print(testres, p)
