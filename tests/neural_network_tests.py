from neuralnet.neural_network import NeuralNetwork
from neuralnet.linear_activation_function import LinearActivationFunction
from neuralnet.layer import Layer
import numpy as np

def test_forward_propagate_zero_layers():
    neuralnet = NeuralNetwork([])
    input = np.zeros(4)
    output = neuralnet.forward_propagate(input)
    np.testing.assert_array_equal(output, input)

def test_forward_propagate_one_layer():
    activation_function = LinearActivationFunction()
    n_input = 3
    n_output = 2
    input = np.array([1, 0, -1])
    matrix = np.array([[2, 1, -1],
                        [3, 1, 2]])
    bias = np.array([-1, -2])
    expected = np.array([2, -1])
    linear = LinearActivationFunction()
    layer = Layer(matrix, bias, linear)
    layers = [layer]
    nn = NeuralNetwork(layers)
    np.testing.assert_array_equal(nn.forward_propagate(input), expected)

def test_forward_propagate_two_layers():
    activation_function = LinearActivationFunction()
    n_input = 3
    n_output = 2
    input = np.array([1, 0, -1])
    weights1 = np.array([[2, 1, -1],
                        [3, 1, 2]])
    bias1 = np.array([-1, -2])
    linear1 = LinearActivationFunction()
    layer1 = Layer(weights1, bias1, linear1)

    weights2 =np.array([[-1, 1],
                        [0, 1],
                        [1, 0]])
    bias2 = np.array([0, 2, 1])
    linear2 = LinearActivationFunction()
    layer2 = Layer(weights2, bias2, linear2)
    layers = [layer1, layer2]
    
    output1 = np.array([2, -1]) #result after first layer
    expected = np.array([-3, 1, 3])
    nn = NeuralNetwork(layers)
    np.testing.assert_array_equal(nn.forward_propagate(input), expected)
