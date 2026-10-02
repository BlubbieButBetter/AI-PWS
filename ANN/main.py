import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        # Sizes is een lijst met het aantal neuronen per laag, [3,4,2] betekent dus 3 neuronen in de invoerlaag, 4 in de verborgen laag en 2 in de uitvoerlaag.
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # Om te beginnen worden alle weights en biases willekeurig ingesteld.

    def feedforward(self, a):
        # Functie die het neurale netwerk activeert en uitvoervector geeft.
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        # Deze code verdeelt de test data in mini-batches (epochs) en traint het netwerk.
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def a_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
    # De afgeleide van de sigmoid functie