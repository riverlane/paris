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
