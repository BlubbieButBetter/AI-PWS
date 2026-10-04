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

    def backprop(self, x, y):
        # Geeft de gradientvector voor de kostenfunctie
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            a_sigmoid(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = a_sigmoid(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y) 
    
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def a_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
    # De afgeleide van de sigmoid functie