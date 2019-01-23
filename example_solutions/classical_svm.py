from .helper_functions import inference_retval
from sklearn import svm
import numpy as np

def classical_svm(training_example_wfns):
    """This is a train function for any circuit ignoring all quantum properties.
    This will work given enough examples, but well be very slow!
    """
    clf = svm.SVC(gamma='auto', C=8)
    vecs, actual_labels = tuple(zip(*training_example_wfns))
    vecs = np.array(vecs);
    vecs = np.concatenate([vecs.real, vecs.imag], axis=1)

    clf.fit(vecs, actual_labels)

    # now we create the inference function. This should take a state and produce a prediction.
    def infer(wavefunction):
        wavefunction = np.array(wavefunction).reshape(1, -1)
        test_vec = np.concatenate([wavefunction.real, wavefunction.imag], axis=1)
        test_prediction = clf.predict( test_vec )[0]

        return test_prediction


    return inference_retval(
        infer_fun = infer
    )
