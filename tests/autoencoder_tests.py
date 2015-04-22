import numpy as np
from neuralnet.logistic_sigmoid import LogisticSigmoid
from neuralnet.linear_activation_function import LinearActivationFunction
from neuralnet.squared_error_loss import SquaredErrorLoss
from neuralnet.autoencoder import Autoencoder

def test_back_propagate():
    epsilon = 1e-5
    rtol = 1e-4
    atol = 1e-3
    n_input = 8
    n_hidden = 43

    np.random.seed(0)
    weights = np.random.randn(n_hidden, n_input)
    bias0 = np.random.randn(n_hidden)
    bias1 = np.random.randn(n_input)
    a0 = LogisticSigmoid()
    a1 = LinearActivationFunction()
    loss = SquaredErrorLoss()
    autoencoder = Autoencoder(weights, bias0, bias1, a0, a1, loss)


    x = np.random.randn(n_input)
    y = np.random.randn(n_input)
    derivatives = autoencoder.back_propagate(x, y)
    (gw_0, gb_0), (gw_1, gb_1) = derivatives
    
    # now use finite diff approximation
    weights_diff = np.zeros((n_hidden, n_input))
    bias0_diff = np.zeros(n_hidden)
    bias1_diff = np.zeros(n_input)

    # first weightmatrix, notet that changin the matrix 'weights' should
    # change the weightmatrices in both layers, points to both of them
    for i in range(n_hidden):
        for j in range(n_input):
            weights[i, j] -= epsilon
            y0 = autoencoder.forward_propagate(x)
            l0 = autoencoder.loss(y, y0)
            weights[i, j] += epsilon # adjust back to original
            
            weights[i, j] += epsilon
            y1 = autoencoder.forward_propagate(x)
            l1 = autoencoder.loss(y, y1)
            weights[i, j] -= epsilon # adjust back to original
            weights_diff[i, j] = (l1 - l0) / (2 * epsilon)

    bias_diff_ls = [bias0_diff, bias1_diff]
    bias_ls = [bias0, bias1]
    for k in range(2):
        bias = bias_ls[k]
        bias_diff = bias_diff_ls[k]
        for i in range(len(bias_diff)):
            bias[i] -= epsilon
            y0 = autoencoder.forward_propagate(x)
            l0 = autoencoder.loss(y, y0)
            bias[i] += epsilon

            bias[i] += epsilon
            y1 = autoencoder.forward_propagate(x)
            l1 = autoencoder.loss(y, y1)
            bias[i] -= epsilon

            bias_diff[i] = (l1 - l0) / (2*epsilon)

    
    np.testing.assert_allclose(gw_0, weights_diff, rtol=rtol, atol=atol)
    np.testing.assert_allclose(gw_1, weights_diff.T, rtol=rtol, atol=atol)
    np.testing.assert_allclose(gb_0, bias0_diff, rtol=rtol, atol=atol)
    np.testing.assert_allclose(gb_1, bias1_diff, rtol=rtol, atol=atol)

    
