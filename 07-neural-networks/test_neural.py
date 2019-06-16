from neural import (inner_neuron, sigmoid_neuron, feedforward_, feedforward, xor)
from pytest import approx

def test_inner_neuron():
    assert inner_neuron([2, 0], [0]) == approx(2)
    assert inner_neuron([2, 0], [2]) == approx(2)
    assert inner_neuron([0, 2], [2]) == approx(4)
    assert inner_neuron([2, 2], [2]) == approx(6)
    assert inner_neuron([1, 1, 1], [1, 1]) == approx(3)
    assert inner_neuron([1, 2, 1], [1, 1]) == approx(4)
    assert inner_neuron([0, 2, 1], [1, 1]) == approx(3)

def test_sigmoid_neuron():
    assert sigmoid_neuron([0, 2], [0]) == approx(0.5)
    assert sigmoid_neuron([0, 1], [1]) == approx(0.7310585786300049)
    assert sigmoid_neuron([1, 1], [1]) == approx(0.8807970779778823)
    assert sigmoid_neuron([2, 0], [2]) == approx(0.8807970779778823)
    assert sigmoid_neuron([1, 1, 1], [0, 1]) == approx(0.8807970779778823)
    assert sigmoid_neuron([1, 1, 1], [-2, 1]) == approx(0.5)
    assert sigmoid_neuron([1, 1, 1], [-2, -1]) == approx(0.11920292202211755)

def test_and():
    and_gate = [ [[-30, 20, 20]] ]
    assert (feedforward(and_gate, [0,0], output_neuron=sigmoid_neuron)[0] - 0.0) < 1e-4
    assert (feedforward(and_gate, [1,0], output_neuron=sigmoid_neuron)[0] - 0.0) < 1e-4
    assert (feedforward(and_gate, [0,1], output_neuron=sigmoid_neuron)[0] - 0.0) < 1e-4
    assert (feedforward(and_gate, [1,1], output_neuron=sigmoid_neuron)[0] - 1.0) < 1e-4

def test_nand():
    nand_gate = [ [[30, -20, -20]] ]
    assert (feedforward(nand_gate, [0,0], output_neuron=sigmoid_neuron)[0] - 1.0) < 1e-4
    assert (feedforward(nand_gate, [1,0], output_neuron=sigmoid_neuron)[0] - 1.0) < 1e-4
    assert (feedforward(nand_gate, [0,1], output_neuron=sigmoid_neuron)[0] - 1.0) < 1e-4
    assert (feedforward(nand_gate, [1,1], output_neuron=sigmoid_neuron)[0] - 0.0) < 1e-4

def test_or():
    or_gate = [ [[-10, 20, 20]] ]
    assert (feedforward(or_gate, [0,0], output_neuron=sigmoid_neuron)[0] - 0.0) < 1e-4
    assert (feedforward(or_gate, [1,0], output_neuron=sigmoid_neuron)[0] - 1.0) < 1e-4
    assert (feedforward(or_gate, [0,1], output_neuron=sigmoid_neuron)[0] - 1.0) < 1e-4
    assert (feedforward(or_gate, [1,1], output_neuron=sigmoid_neuron)[0] - 1.0) < 1e-4

def test_xnor():
    xnor = [ [[10, -20, -20], [-30, 20, 20]], [[0, 1, 1]] ]
    assert (feedforward(xnor, [0,1])[0] - 0.0) < 1e-4
    assert (feedforward(xnor, [1,0])[0] - 0.0) < 1e-4
    assert (feedforward(xnor, [0,0])[0] - 1.0) < 1e-4
    assert (feedforward(xnor, [1,1])[0] - 1.0) < 1e-4

def test_xor():
    assert (feedforward(xor, [0,1])[0] - 1.0) < 1e-4
    assert (feedforward(xor, [1,0])[0] - 1.0) < 1e-4
    assert (feedforward(xor, [0,0])[0] - 0.0) < 1e-4
    assert (feedforward(xor, [1,1])[0] - 0.0) < 1e-4
