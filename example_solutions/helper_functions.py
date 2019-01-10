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

    exp = 1 if exp > 0 else -1 # clip to the labels. this may be not what you want for cont. optimisation.
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
def infererance_retval(infer_fun = None, infer_circ = None, discription = None,
                       train_accuracy = None):
    if infer_circ is not None and infer_fun is None:
        infer_fun = functools.partial(generic_infer, infer_circ)

    return {"infer_fun":infer_fun, "infer_circ":infer_circ, "discription":discription,
            "train_best_accuracy":train_accuracy}

def gate_repr(f):
    """Extracts a qiskit gate application from a lambda function.

    returns: sttring repr of the gate.
    """
    gate_arity = None
    gate_name = None
    gate_args = []
    gate_qubits = []

    # get the gate name.
    for i in dis.get_instructions(f):
        if gate_name is None and i.opname == "LOAD_ATTR":
            gate_name = i.argval
            break

    # get qubit args: space bwteen LOAD_FAST qreg and BINARY_SUBSCR is the index.
    state = 0 # outside indexing
    index_expr = None
    for i in dis.get_instructions(f):
        if state == 0 and i.argrepr == "qreg":
            state = 1 # in the index part
            index_expr = []

        if state == 1 and i.opname in ["LOAD_GLOBAL", "LOAD_FAST", "LOAD_CONST"]:
            index_expr.append(i.argrepr)

        if state == 1 and i.opname == "BINARY_ADD": # add the last 2 elements.
            index_expr = index_expr[:-2] + [(index_expr[-2] + "+" + index_expr[-1])]

        if state == 1 and i.opname == "BINARY_SUBSCR":
            # end of index expr.
            gate_qubits.append(" ".join(index_expr))
            state = 0

    return f"{gate_name}({gate_qubits})"

def print_circuit(circ, num_qubits):

    simulator = BasicAer.get_backend('statevector_simulator')
    qr = QuantumRegister(num_qubits, "qr")
    circ = QuantumCircuit(qr)

    if len(circ) == 0:
        circ.iden(qr)
    else:
        for gate_application_function in current_circuit:
            gate_application_function(circ, qr)

    return circ.draw().single_string()
