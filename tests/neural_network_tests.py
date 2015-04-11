from neuralnet.neural_network import NeuralNetwork
from neuralnet.linear_activation_function import LinearActivationFunction

import numpy as np

def test_forward_propagate_zero_layers():
    neuralnet = NeuralNetwork([])
    input = np.zeros(4)
    output = neuralnet.forward_propagate(input)
    np.testing.assert_array_equal(output, input)

def test_forward_propagate_one_layer():
    activation_function = LinearActivationFunction()
    input = np.zeros(7)
    
