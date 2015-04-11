import numpy as np
from nose.tools import *
from neuralnet.layer import Layer
from neuralnet.logistic_sigmoid import LogisticSigmoid

def test_functionality():
    n_input = 3
    n_output = 4
    shape = (n_output, n_input)
    weights = np.zeros(shape)
    bias = np.zeros(n_output)
    activation_function = LogisticSigmoid()
    layer = Layer(weights, bias, activation_function)

    assert_tuple_equal(layer.get_weights().shape, shape)

    np.random.seed(0)
    random_matrix = np.random.randn(*shape)
    assert_tuple_equal(random_matrix.shape, shape)
    random_vector = np.random.randn(n_output)
    layer.set_weights(random_matrix)
    layer.set_bias(random_vector)
    np.testing.assert_array_equal(layer.get_weights(), random_matrix)
    np.testing.assert_array_equal(layer.get_bias(), random_vector)
    assert_equal(layer.get_activation_function(), activation_function)
