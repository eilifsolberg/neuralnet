from neuralnet.layer import Layer
from neuralnet.neural_network import NeuralNetwork

class Autoencoder(object):

    def __init__(self, weights, bias0, bias1, a0, a1, loss_function=None):
        """Takes tied weightmatrix, biases and activations functions

        arguments: 
        weights -- weightmatrix n_hidden_nodes x n_hidden_nodes
        bias0 -- real vector of length n_hidden_nodes
        bias1 --real vector of length n_input_nodes == n_output_nodes
        a0 -- activation function

        keyword arguments:
        loss_function -- should implement value(y, y_hat) and deriv(y, y_hat)
        functoins. default=None
        """
        
        first_layer = Layer(weights, bias0, a0)
        # second layer, take copy of weights or keep same?
        second_layer = Layer(weights.T, bias1, a1)
        layers = [first_layer, second_layer]
        self.neural_network = NeuralNetwork(layers, loss_function)

    def forward_propagate(self, x):
        """Propagates the input forward through the layers, returns final
        
        arguments:
        x -- a real vector
        
        output: 
        y_hat -- a real vector
        """
        return self.neural_network.forward_propagate(x)

    def back_propagate(self, x, y):
        """slightly different than before, 2 layers of biases, only one weights

        arguments:
        x -- a real vector, input vector
        y -- a real vector, target vector

        output: 
        a list of length 2 contating a 2-tuple of derivatives with respect to 
        the weights and biases. i.e. output[0] = (weight_deriv0, bias_deriv0).
        Note that since the weights are tight weight_deriv0 and weight_deriv1
        will point to the same object, but have different shapes
        """

        derivatives = self.neural_network.back_propagate(x, y)
        (gw_0, gb_0), (gw_1, gb_1) = derivatives
        weight_deriv0 = gw_0 + gw_1.T
        weight_deriv1 = weight_deriv0.T

        return [(weight_deriv0, gb_0), (weight_deriv1, gb_1)]
    

    def loss(self, y, y_hat):
        """calculates loss where we allow y and y_hat to be vectors
        
        arguments:
        y -- true value
        y_hat -- prediction

        output:
        loss -- take this to be the sum of the loss for each component

        For more details, see NeuralNetwork implementation
        """

        return self.neural_network.loss(y, y_hat)
