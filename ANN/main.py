import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        # Sizes is een lijst met het aantal neuronen per laag, [3,4,2] betekent dus 3 neuronen in de invoerlaag, 4 in de verborgen laag en 2 in de uitvoerlaag.
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # Om te beginnen worden alle weights en biases willekeurig ingesteld.



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def a_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
    # De afgeleide van de sigmoid functie