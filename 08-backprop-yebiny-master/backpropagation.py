from math import exp
from copy import deepcopy

def sigmoid(x):
    try:
        return 1.0 / (1.0 + exp(-x))
    except OverflowError:
        print("OV", x)
        raise OverflowError

def multi_best(xs):
    best = max(xs)
    return [1 if x == best else 0 for x in xs]

def multi_accuracy(data, target, f):
    tot = 0
    cor = 0
    for d, t in zip(data, target):
        if multi_best(f(d)) == t:
            cor += 1
        tot += 1
    return cor / float(tot)

def initialize_weights(n_nodes, initialize_fn):
    pass

def feedforward_(network, inputs, hidden_neuron=sigmoid_neuron, output_neuron=sigmoid_neuron):
    pass

def feedforward(network, inputs):
    pass

def calculate_deltas(network, activations, y):
    pass

def batch_update_nn(network, activation, deltas, eta):
    pass

# You can use the structure from before (reduce alpha0 every time), or
# you can just run over the data n_epochs times, and then choose a new
# alpha by hand
def sgd_nn(x, y, theta_0, alpha_0: float = 0.01, iterations=20):
    pass

if __name__ == "__main__":
    # Setup for Fisher's iris classification task
    iris = csv.reader(open('Fisher.txt'), delimiter='\t')
    header = iris.__next__()  # change to iris.next() for python2!
    data_ = list(d for d in iris)
    data = list([[float(di) for di in d[1:]] for d in data_])
    target = [[0,0,0] for _ in data_]
    for i, di in enumerate(data_):
        target[i][int(di[0])] = 1
    # Example to setup and use a neural network for Fisher.
    # Further examples in test_backpropagation
    # Network architecture: 4 > 8 > 3
    network_ = initialize_weights([4, 8, 3], lambda: random.gauss(0, 1))
    network, l = sgd_nn(data, target, network_, 1, 10)
    # Test out, show sgd has improved
    print("Before training:")
    print(multi_accuracy(data, target, lambda x: feedforward(network_, x)))
    print("After training:")
    print(multi_accuracy(data, target, lambda x: feedforward(network, x)))
