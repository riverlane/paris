from .helper_functions import compute_parity_exp_value
from functools import partial
from scipy.optimize import minimize
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, BasicAer, execute

class HardwareEfficientAnsatzInverse(object):

    def __init__(self, num_qubits, depth, params):

        self._num_qubits = num_qubits
        self._depth = depth
        self._params = params

    def apply(self, circ: QuantumCircuit, q_reg: QuantumRegister) -> QuantumCircuit:

        for iod in range(self._depth):

            #'Undo' CNOTS
            for iq in range(self._num_qubits):
                circ.cx(q_reg[self._num_qubits-iq-2], q_reg[self._num_qubits-iq-1])

            for iq in range(self._num_qubits):
                circ.rz(self._params[iq * (3 * self._depth + 2) + 3 * iod], q_reg[iq])
                circ.rx(self._params[iq * (3 * self._depth + 2) + 3 * iod + 1], q_reg[iq])
                circ.rz(self._params[iq * (3 * self._depth + 2) + 3 * iod + 2], q_reg[iq])


        # Final level
        for iq in range(self._num_qubits):
            circ.cx(q_reg[self._num_qubits-iq-2], q_reg[self._num_qubits-iq-1])

        for iq in range(self._num_qubits):
            circ.rz(self._params[iq * (3 * self._depth + 2) + 3 * self._depth], q_reg[iq])
            circ.rx(self._params[iq * (3 * self._depth + 2) + 3 * self._depth + 1], q_reg[iq])

        return circ


def objective_function(params,n_qubit,depth,data,measurement,engine):

    fun = 0.0
    for train_vector, train_label in data:

        pred = prediction(params, n_qubit, depth, train_vector, measurement, engine)
        fun += ((pred - train_label)**2)

    return fun

def prediction(params, num_qubits, depth, vector, simulator):

    inv_circ = HardwareEfficientAnsatzInverse(num_qubits, depth, params)

    qr = QuantumRegister(num_qubits, "qr")
    circ = QuantumCircuit(qr)

    # build the circuit by appending gates
    circ = inv_circ.apply(circ)

    opts = {"initial_statevector": vector}
    execution = execute(circ, simulator, backend_options=opts)
    result = execution.result()
    pred = compute_parity_exp_value(result.get_statevector(circ))

    return pred

def example_general_continuous_problem_training(training_data):

    simulator = BasicAer.get_backend('statevector_simulator')

    depth = 1
    num_qubits = int(np.log2(len(training_data[0][0])))

    # Hardware efficient parameter setup
    num_params = num_qubits*(3*depth + 2)

    obj_fun = partial(objective_function, num_qubits=num_qubits, depth=depth,
                      data=training_data, simulator=simulator)

    init_params = np.random.uniform(0.0,2.0*np.pi,size=num_params)

    res = minimize(fun=obj_fun, x0=init_params, method='Nelder-Mead',
                   tol=1e-1, options={'disp':True,'maxiter':1000})

    best_params = res.x

    def infer(vector):

        pred = prediction(best_params, num_qubits, depth, vector, simulator)

        if pred < 0.0:
            return -1
        else:
            return 1

    return infer