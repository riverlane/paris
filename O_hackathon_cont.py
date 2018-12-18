from projectq.ops import CNOT, Rx, Ry, Rz, H, QubitOperator, X
from projectq import MainEngine
from projectq.backends import Simulator
from projectq.ops import All, Measure
from QuantumFramework.optimisers import NMMinimiser
from QuantumFramework import QFrameworkLogger
# from QuantumFramework.tools import int_to_state

import numpy as np
from functools import partial
import sys
import logging
import itertools

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
# logger.addHandler(ch)
QFrameworkLogger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logging.getLogger().addHandler(ch)

# Each qubit has same gates (not angles)
# Invert for order Rx Ry Rz


class InverseCircuit(object):

    def __init__(self, params, n_qubit, outer_d):
        self.params = params
        self.n_qubit = n_qubit
        self.outer_depth = outer_d

    def __or__(self, qubit_register):

        inner_d = 1

        depth = self.outer_depth*inner_d

        for iod in range(self.outer_depth):
            #for iq in range(self.n_qubit - 1):
            #    CNOT | (qubit_register[self.n_qubit-iq-2], qubit_register[self.n_qubit-iq-1])

            CNOT | (qubit_register[0], qubit_register[1])

            for iq, qubit in enumerate(qubit_register):
                Rx(self.params[depth*iq + inner_d*iod]) | qubit
                #Ry(self.params[depth*iq + 3*iod + 1]) | qubit
                #Rz(self.params[depth*iq + 3*iod + 2]) | qubit


def objective_function(inv_params,n_qubit,outer_d,parity,data,engine):

    inv_circ = InverseCircuit(inv_params,n_qubit,outer_d)

    q_str = ''
    for iq in range(n_qubit):
        q_str += 'Z' + str(iq) + ' '

    op = QubitOperator(q_str)

    n_data = data.shape[0]

    fun = 0.0
    #print("Allocating qubits")
    qubit_register = engine.allocate_qureg(n_qubit)
    engine.flush()
    for i_data in range(n_data):
        #print(f"Vector {i_data+1} of {n_data}")

        # Prepare data state
        engine.backend.set_wavefunction(data[i_data,:],qubit_register)

        # Act circuit
        inv_circ | qubit_register
        engine.flush()

        # Make measurement
        meas = engine.backend.get_expectation_value(op, qubit_register)
        All(Measure) | qubit_register

        engine.flush()

        # Add misfit to objective function
        fun += (meas - parity[i_data])**2

    All(Measure) | qubit_register
    engine.flush()

    del qubit_register

    return fun

import cProfile
if __name__ == '__main__':

    eng = MainEngine(backend=Simulator())

    data_file = 'hackathon_10_500.out'
    with open(data_file, 'r') as f:
        line = f.readline()
        n_qubit = int(line.split()[0])
        n_data = int(line.split()[1])

        data = np.zeros([n_data,2**n_qubit], dtype=complex)
        parity = np.zeros(n_data)

        for i_data in range(n_data):
            line = f.readline()
            parity[i_data] = int(float(line.split()[0]))
            data[i_data,:] = [complex(li) for li in line.split()[1:]]

    # Tolerance
    tolerance = 1e-10

    # Inverse circuit
    outerd_i = 2
    depth_i = outerd_i
    nparam_i = depth_i*n_qubit

    obj_fun = partial(objective_function, n_qubit=n_qubit, outer_d=outerd_i,
                      parity=parity, data=data, engine=eng)

    init_params = np.random.uniform(0,2*np.pi,size=nparam_i)
    bounds = [{'name': 'ansatzp', 'type': 'continuous', 'domain': (0,2*np.pi)}]*nparam_i

    cProfile.run("""
val, fin_params, n_eval, n_it = (NMMinimiser(tol=tolerance,var_param=False, bounds=bounds)
                                 .optimise(obj_fun,init_params))
""")


    print(f"Value = {val}")
    print(f"Parameters = {fin_params}")
