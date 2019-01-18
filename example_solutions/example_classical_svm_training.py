from .helper_functions import infererance_retval
from sklearn import svm
import numpy as np

def example_classical_svm_training(training_example_wfns):
    """This is a train function for any circuit ignoring all quantum properties.
    This will work given enough examples, but well be very slow!
    """
    clf = svm.SVC(gamma='auto', C=8)
    vecs, actual_labels = tuple(zip(*training_example_wfns))
    vecs = np.array(vecs);
    vecs = np.concatenate([vecs.real, vecs.imag], axis=1)

    clf.fit(vecs, actual_labels)

    pred_labels = tuple( clf.predict(vecs) )
    training_error = 0.0
    for actual_label, pred_label in zip(actual_labels, pred_labels):
        training_error += abs(actual_label - pred_label)

    # now we create the inference function. This should take a state and produce a prediction.
    def infer(wavefunction):
        wavefunction = np.array(wavefunction).reshape(1, -1)
        test_vec = np.concatenate([wavefunction.real, wavefunction.imag], axis=1)
        test_prediction = clf.predict( test_vec )[0]

        return test_prediction


    return infererance_retval(
        infer_fun = infer,
        training_error = training_error
    )
