import numpy as np

class NeuralNetwork(object):
    """Both a storage class, storing the layers, input and output, but can also
    do some operations like forward propagation, and backward propagation
    """

    def __init__(self, layers):
        """The constructor takes as input a list of Layer objects"""
        self.layers = layers

    def forward_propagate(self, input):
        """Propagates the input forward through the layers, returns result.

        input -- a real vector
        output -- a real vector
        """
        post = input
        for i in range(len(self.layers)):
            weights = self.layers[i].get_weights()
            bias = self.layers[i].get_bias()
            activation_function = self.layers[i].get_activation_function()
            
            pre_activation = np.dot(weights, post) + bias
            # print "i pre_activation: %d, %r" % (i, pre_activation)
            post = activation_function.value(pre_activation)
            # print "i post_activation: %d, %r" % (i, post)
        return post
