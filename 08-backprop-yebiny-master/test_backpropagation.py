#!/usr/bin/env python

from backpropagation import (feedforward_, feedforward, calculate_deltas, sigmoid, batch_update_nn, initialize_weights)
from pytest import approx
from random import random

def test_initialize_weights():
    network = initialize_weights([1, 3, 1])
    assert len(network) == 2, "Check network setup has correct sizes (bias?)"
    assert len(network[0]) == 3, "Check network setup has correct sizes (bias?)"
    assert len(network[0][0]) == 2, "Check network setup has correct sizes (bias?)"
    assert len(network[0][1]) == 2, "Check network setup has correct sizes (bias?)"
    assert len(network[0][2]) == 2, "Check network setup has correct sizes (bias?)"
    assert len(network[1]) == 1, "Check network setup has correct sizes (bias?)"
    assert len(network[1][0]) == 4, "Check network setup has correct sizes (bias?)"
    network = initialize_weights([1, 3, 1], lambda: 1)
    assert network == [[[1, 1], [1, 1], [1, 1]], [[1, 1, 1, 1]]], "Check initialize_fn used properly"
    network = initialize_weights([2, 2, 1], lambda: 1)
    assert network == [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1]]]

def test_feedforward():
    network = initialize_weights([1, 2, 1], lambda: 1)
    assert feedforward_(network, [1]) == [[1], [sigmoid(2), sigmoid(2)], [sigmoid(sigmoid(2) + sigmoid(2) + 1)]], "Check outputs are added after sigmoid"
    assert feedforward(network, [1]) == [sigmoid(sigmoid(2) + sigmoid(2) + 1)]

def test_calculate_deltas():
    # 1 > 1 network
    network = [[[1, 1]]]
    # We can make up activations for checking the functionality
    activations = [[1], [.9]]
    y = [1]
    deltas = calculate_deltas(network, activations, y)
    assert len(deltas) == 1
    assert len(deltas[0]) == 1
    assert deltas[0][0] == approx(.9*.1*-.1) # (o (1-o) (o-t)) with o=0.9, t=1
    # 1 > 1 > 1 network
    network = [[[1, 1]], [[1, 1]]]
    # We can make up activations for checking the functionality
    activations = [[1], [.8], [.9]]
    y = [1]
    deltas = calculate_deltas(network, activations, y)
    assert len(deltas) == 2
    assert len(deltas[1]) == 1
    assert len(deltas[0]) == 1
    assert deltas[1][0] == approx(.9*.1*-.1) # (o (1-o) (o-t)) with o=0.9, t=1
    assert deltas[0][0] == approx(.8*.2*.9*.1*-.1) # .8*(1-.8)*[1*delta]
    # 1 > 2 > 1 network
    network = [[[1, 1], [1, 1]], [[1, 1, 1]]]
    # We can make up activations for checking the functionality
    activations = [[1], [.8, .8], [.9]]
    y = [1]
    deltas = calculate_deltas(network, activations, y)
    assert len(deltas) == 2
    assert len(deltas[1]) == 1
    assert len(deltas[0]) == 2
    assert deltas[1][0] == approx(.9*.1*-.1) # (o (1-o) (o-t)) with o=0.9, t=1
    assert deltas[0][0] == approx(.8*.2*.9*.1*-.1) # .8*(1-.8)*[1*delta]
    assert deltas[0][1] == approx(.8*.2*.9*.1*-.1) # .8*(1-.8)*[1*delta]
    # 1 > 2 > 2 network. The two networks should be equivalent
    network_ = [[[1, 1], [1, 1]], [[1, 1, 1], [1, 1, 1]]]
    network = initialize_weights([1, 2, 2], lambda: 1)
    assert network_ == network
    # We can make up activations for checking the functionality
    activations = [[1], [.8, .8], [.9, .9]]
    y = [1, 1]
    deltas = calculate_deltas(network, activations, y)
    assert len(deltas) == 2
    assert len(deltas[1]) == 2
    assert len(deltas[0]) == 2
    assert deltas[1][0] == approx(.9*.1*-.1) # (o (1-o) (o-t)) with o=0.9, t=1
    assert deltas[1][1] == approx(.9*.1*-.1) # (o (1-o) (o-t)) with o=0.9, t=1
    assert deltas[0][0] == approx(.8*.2*(.9*.1*-.1 + .9*.1*-.1)) # .8*(1-.8)*[1*delta_1 + 1*delta_2]
    assert deltas[0][1] == approx(.8*.2*(.9*.1*-.1 + .9*.1*-.1)) # .8*(1-.8)*[1*delta_1 + 1*delta_2]
    # 2 > 2 > 2 network
    network_ = [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]
    network = initialize_weights([2, 2, 2], lambda: 1)
    assert network_ == network
    # We can make up activations for checking the functionality
    activations = [[1, 1], [.8, .8], [.9, .9]]
    y = [1, 1]
    deltas = calculate_deltas(network, activations, y)
    assert len(deltas) == 2
    assert len(deltas[1]) == 2
    assert len(deltas[0]) == 2
    assert deltas[1][0] == approx(.9*.1*-.1) # (o (1-o) (o-t)) with o=0.9, t=1
    assert deltas[1][1] == approx(.9*.1*-.1) # (o (1-o) (o-t)) with o=0.9, t=1
    assert deltas[0][0] == approx(.8*.2*(.9*.1*-.1 + .9*.1*-.1)) # .8*(1-.8)*[1*delta_1 + 1*delta_2]
    assert deltas[0][1] == approx(.8*.2*(.9*.1*-.1 + .9*.1*-.1)) # .8*(1-.8)*[1*delta_1 + 1*delta_2]

def test_batch_update_nn():
    # For testing, we can put whatever we like in activations and deltas
    network = initialize_weights([2, 2, 1], lambda: 1)
    activations = [[1, 1], [1, 1], [1]]
    deltas = [[1, 1], [1]]
    network = batch_update_nn(network, activations, deltas, eta=1)
    assert len(network) == 2 and len(network[0]) == 2 and len(network[1]) == 1 \
        and len(network[0][0]) == 3 and len(network[0][1]) == 3 and len(network[1][0]) == 3, "Check you haven't changed the network shape"
    assert network == [[[0,0,0], [0,0,0]], [[0,0,0]]]
