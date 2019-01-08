from sklearn import svm
import numpy as np

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