from neuralnet.neural_network import NeuralNetwork
from neuralnet.linear_activation_function import LinearActivationFunction
from neuralnet.layer import Layer
from neuralnet.squared_error_loss import SquaredErrorLoss
from neuralnet.logistic_sigmoid import LogisticSigmoid

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
    
    output0 = np.array([2, -1]) #result after first layer
    expected = np.array([-3, 1, 3])
    nn = NeuralNetwork(layers)
    np.testing.assert_array_equal(nn.forward_propagate(input), expected)

def test_forward_propagate_store():
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
    
    expected0 = np.array([2, -1]) #result after first layer
    expected1 = np.array([-3, 1, 3])
    expected_post = expected1 # linear activation function
    nn = NeuralNetwork(layers)
    (output0, output1), post = nn.forward_propagate_store(input)
    np.testing.assert_array_equal(output0, expected0)
    np.testing.assert_array_equal(output1, expected1)
    np.testing.assert_array_equal(post, expected_post)

def test_backpropagation_one_layer():
    x = np.array([2, 1, 0, -1])
    weights = np.array([[1, 2, 3, -1],
                       [-2, 3, -1, 1]])
    product = np.array([5, -2])
    np.testing.assert_array_equal(np.dot(weights, x), product)
    bias = np.array([-2, 0]) 
    y_hat = product + bias # (3, -2)
    y = np.array([2, 1])
    weight_deriv_hand = np.array([[2, 1, 0, -1],
                                    [-6, -3, 0, 3]])
    bias_deriv_hand = np.array([1, -3])
    linear = LinearActivationFunction()
    layer = Layer(weights, bias, linear)
    squared_error = SquaredErrorLoss()
    nn = NeuralNetwork([layer], squared_error)
    gradients = nn.back_propagate(x, y)
    weight_deriv, bias_deriv = gradients[0]
    np.testing.assert_array_equal(weight_deriv, weight_deriv_hand)
    np.testing.assert_array_equal(bias_deriv, bias_deriv_hand)


def test_backpropagation_three_layers():
    """not really a test right now, need some more work"""
    shape0 = (10,  4)
    shape1 = (13, 10)
    shape2 = (3, 13)
    shapes = [shape0, shape1, shape2]
    layers = []
    for i in range(len(shapes)):
        weights = np.random.randn(shapes[i][0], shapes[i][1])
        bias = np.random.randn(shapes[i][0]) # bias should be for output
        linear = LinearActivationFunction()
        layers.append(Layer(weights, bias, linear))

    # calculate linear transformation
    matrices = []
    degree = len(shapes) 
    for i in range(degree + 1):
        matrices.append(np.eye(shapes[-1][0]))
    for i in range(degree + 1):
        for j in range(degree):
            if j < i:
                weights = layers[-(j+1)].get_weights()
                matrices[i] = np.dot(matrices[i], weights)

    bias_vector = np.zeros(shapes[-1][0])
    # running over all matrices, except highest degree, no bias term there.
    for i in range(degree):
        tmp = np.dot(matrices[i], layers[-(1+i)].get_bias())
        bias_vector += tmp

    weight_matrix = matrices[degree]

    x = np.random.randn(shapes[0][1])
    y = np.random.randn(shapes[-1][0])
    # check that linear transformation is correct
    tmp = x
    for i in range(degree):
        tmp = np.dot(layers[i].get_weights(), tmp) + layers[i].get_bias()

    expected = np.dot(weight_matrix, x) + bias_vector
    np.testing.assert_allclose(tmp, expected, rtol=1e-10, atol=1e-16)
    nn = NeuralNetwork(layers, SquaredErrorLoss())
    gradients = nn.back_propagate(x, y)


def test_linear_backprop_vs_finitedifferences():
    """linear: compare gradient returned by backprop vs finite difference.
    Should have equality in the linear case
    """
    epsilon = 1e-6

    shape0 = (2, 4)
    shape1 = (5, 2)
    shape2 = (3, 5)
    shape_ls = [shape0, shape1, shape2]
    n_layers = len(shape_ls)

    layers = []
    for i in range(n_layers):
        weights = np.random.randn(shape_ls[i][0], shape_ls[i][1])
        bias = np.random.randn(shape_ls[i][0])
        linear = LinearActivationFunction()
        layers.append(Layer(weights, bias, linear))

    loss = SquaredErrorLoss()
    nn = NeuralNetwork(layers, loss)
    x = np.random.randn(shape_ls[0][1])
    y = np.random.randn(shape_ls[-1][0]) # match dimension of output last layer
    gradients = nn.back_propagate(x, y)

    #first do for weights
    finite_diff_weights = []
    for layer_number in range(n_layers):
        shape = shape_ls[layer_number]
        weights_diff = np.zeros(shape)
        weights = nn.layers[layer_number].get_weights()
        for i in range(shape[0]):
            for j in range(shape[1]):
                w0 = weights.copy()
                w1 = weights.copy()
                w0[i, j] = w0[i, j] - epsilon
                w1[i, j] = w1[i, j] + epsilon
                nn.layers[layer_number].set_weights(w0)
                y0 = nn.forward_propagate(x)
                l0 = nn.loss(y, y0)
                nn.layers[layer_number].set_weights(w1)
                y1 = nn.forward_propagate(x)
                l1 = nn.loss(y, y1)
                
                weights_diff[i, j] = (l1 - l0) / (2*epsilon)

        finite_diff_weights.append(weights_diff)
        # have to set weights for the layer back to original
        nn.layers[layer_number].set_weights(weights)
        
    # then for biases
    finite_diff_bias = []
    for layer_number in range(n_layers):
        shape = shape_ls[layer_number]
        bias_diff = np.zeros(shape[0])
        bias = nn.layers[layer_number].get_bias()
        for i in range(shape[0]):
            b0 = bias.copy()
            b1 = bias.copy()
            b0[i] = b0[i] - epsilon
            b1[i] = b1[i] + epsilon
            nn.layers[layer_number].set_bias(b0)
            y0 = nn.forward_propagate(x)
            l0 = nn.loss(y, y0)
            nn.layers[layer_number].set_bias(b1)
            y1 = nn.forward_propagate(x)
            l1 = nn.loss(y, y1)
            bias_diff[i] = (l1 - l0) / (2*epsilon)

        finite_diff_bias.append(bias_diff)
        # set bias back to original
        nn.layers[layer_number].set_bias(bias)


    for layer_number in range(n_layers):
        weights_diff = finite_diff_weights[layer_number]
        bias_diff = finite_diff_bias[layer_number]
        weights_grad, bias_grad = gradients[layer_number]
        np.testing.assert_allclose(weights_diff, weights_grad)
        np.testing.assert_allclose(bias_diff, bias_grad)

def test_sigmoid_backprop_vs_finitedifferences():
    """logistic: compare gradient returned by backprop vs finite difference.
    same as linear, just changed activation_function
    """
    epsilon = 1e-5

    shape0 = (2, 4)
    shape1 = (5, 2)
    shape2 = (3, 5)
    shape_ls = [shape0, shape1, shape2]
    n_layers = len(shape_ls)

    layers = []
    for i in range(n_layers):
        weights = np.random.randn(shape_ls[i][0], shape_ls[i][1])
        bias = np.random.randn(shape_ls[i][0])
        sigmoid = LogisticSigmoid()
        layers.append(Layer(weights, bias, sigmoid))

    loss = SquaredErrorLoss()
    nn = NeuralNetwork(layers, loss)
    x = np.random.randn(shape_ls[0][1])
    y = np.random.randn(shape_ls[-1][0]) # match dimension of output last layer
    gradients = nn.back_propagate(x, y)

    #first do for weights
    finite_diff_weights = []
    for layer_number in range(n_layers):
        shape = shape_ls[layer_number]
        weights_diff = np.zeros(shape)
        weights = nn.layers[layer_number].get_weights()
        for i in range(shape[0]):
            for j in range(shape[1]):
                w0 = weights.copy()
                w1 = weights.copy()
                w0[i, j] = w0[i, j] - epsilon
                w1[i, j] = w1[i, j] + epsilon
                nn.layers[layer_number].set_weights(w0)
                y0 = nn.forward_propagate(x)
                l0 = nn.loss(y, y0)
                nn.layers[layer_number].set_weights(w1)
                y1 = nn.forward_propagate(x)
                l1 = nn.loss(y, y1)
                
                weights_diff[i, j] = (l1 - l0) / (2*epsilon)

        finite_diff_weights.append(weights_diff)
        # have to set weights for the layer back to original
        nn.layers[layer_number].set_weights(weights)
        
    # then for biases
    finite_diff_bias = []
    for layer_number in range(n_layers):
        shape = shape_ls[layer_number]
        bias_diff = np.zeros(shape[0])
        bias = nn.layers[layer_number].get_bias()
        for i in range(shape[0]):
            b0 = bias.copy()
            b1 = bias.copy()
            b0[i] = b0[i] - epsilon
            b1[i] = b1[i] + epsilon
            nn.layers[layer_number].set_bias(b0)
            y0 = nn.forward_propagate(x)
            l0 = nn.loss(y, y0)
            nn.layers[layer_number].set_bias(b1)
            y1 = nn.forward_propagate(x)
            l1 = nn.loss(y, y1)
            bias_diff[i] = (l1 - l0) / (2*epsilon)

        finite_diff_bias.append(bias_diff)
        # set bias back to original
        nn.layers[layer_number].set_bias(bias)


    for layer_number in range(n_layers):
        weights_diff = finite_diff_weights[layer_number]
        bias_diff = finite_diff_bias[layer_number]
        weights_grad, bias_grad = gradients[layer_number]
        np.testing.assert_allclose(weights_diff, weights_grad)
        np.testing.assert_allclose(bias_diff, bias_grad)
