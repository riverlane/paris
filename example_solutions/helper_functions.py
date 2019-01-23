from qiskit import QuantumCircuit, QuantumRegister, BasicAer, execute
import dis

def parity_of(int_type):
    parity = 0
    while int_type:
        parity = ~parity
        int_type = int_type & (int_type - 1)
    return parity

def compute_parity_exp_value(state_vector):
    exp = 0.0
    for idx, coeff in enumerate(state_vector):
        exp += coeff * coeff.conj() * (1 if parity_of(idx) == 0 else -1)

    # exp = 1 if exp > 0 else -1 # clip to the labels. this may be not what you want for cont. optimisation.
    return exp

def generic_infer(best_circuit, wavefunction):

    simulator = BasicAer.get_backend('statevector_simulator')
    qr = QuantumRegister(num_qubits, "qr")
    circ = QuantumCircuit(qr)

    for gate in best_circuit:
        gate(circ, qr)

    opts = {"initial_statevector": wavefunction}
    result = execute(circ, simulator, backend_options=opts).result()
    result = compute_parity_exp_value(result.get_statevector(circ))

    return result

# alternative: defaultdict and .update() - but this is less self documenting.
def inference_retval(infer_fun = None, infer_circ = None, description = None):
    if infer_circ is not None and infer_fun is None:
        infer_fun = functools.partial(generic_infer, infer_circ)

    return {"infer_fun":infer_fun, "infer_circ":infer_circ, "description":description}

class Mock(object):

    def __init__(self, calllist=None, path=None):
        self.calls = calllist if calllist is not None else []
        self.priors = path if path is not None else []

    def __getattr__(self, name):
        #print(self.priors, ("getattr", name))

        self.calls.append(self.priors + [("getattr", name)])
        return Mock(self.calls, self.priors+[("getattr", name)])

    def __getitem__(self, slice):
        #print(self.priors, ("getitem", slice))

        self.calls.append(self.priors + [("getitem", slice)])
        return Mock(self.calls, self.priors+[("getitem", slice)])

    def __call__(self, *args, **kwargs):
        #print(self.priors, ("call", args, kwargs))

        self.calls.append(self.priors + [("call", args, kwargs)])
        return Mock(self.calls, self.priors+[("call", args, kwargs)])



def gate_repr(f):
    """Extracts a qiskit gate application from a lambda function.

    returns: string repr of the gate.
    """
    circ = Mock()
    qreg = Mock()
    try:
        f(circ, qreg)
    except TypeError:
        # if there are global var lookups in f, we try to concrelty eval
        # it... does not work. so we need the qubit indices to be bound
        return "? unbound qubit indices"
    # print(circ.calls)
    # print(qreg.calls)
    # just works for a single gate.
    gate = circ.calls[-1][0][1] # indexes get the deepest call, root, name.
    # gate_args = [[a for a in args if type(a) != Mock]
    #              for name, args, kwargs in circ.calls[-1][-1]]
    g_name, g_args, g_kwargs = circ.calls[-1][-1]
    g_args = [a for a in g_args if type(a) != Mock]
    qubit_indices = [idxop[0][1] for idxop in qreg.calls]

    #return gate, g_args, qubit_indices
    arg_str = "(" + ", ".join(g_args) + ")" if g_args else ""
    idx_str = ", ".join(map(str, qubit_indices))
    return f"{gate}{arg_str}[{idx_str}]"


def print_circuit(current_circuit, num_qubits):

    if isinstance(current_circuit, tuple):
        qr = QuantumRegister(num_qubits, "qr")
        circ = QuantumCircuit(qr)

        if len(current_circuit) == 0:
            circ.iden(qr)
        else:
            for gate_application_function in current_circuit:
                gate_application_function(circ, qr)
    else:
        circ = current_circuit

    return circ.draw().single_string()
