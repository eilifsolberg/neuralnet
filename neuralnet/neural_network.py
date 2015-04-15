import numpy as np

class NeuralNetwork(object):
    """Both a storage class, storing the layers, input and output, but can also
    do some operations like forward propagation, and backward propagation
    """

    def __init__(self, layers, loss_function=None):
        """The constructor takes layers and optionally 
        
        arguments: 
        layers -- list of class Layer objectes
        
        keyword argumements:
        loss_function -- should implement value(y, y_hat), and
        deriv(y, y_hat) functions
        """
        self.layers = layers
        self.loss_function = loss_function
        
    def set_loss_function(self, loss_function):
        this.loss_function = loss_function
        
    def forward_propagate(self, x):
        """Propagates the input forward through the layers, returns result.

        input: x --  a real vector
        output post -- a real vector
        """
        
        post = x
        for i in range(len(self.layers)):
            weights = self.layers[i].get_weights()
            bias = self.layers[i].get_bias()
            activation_function = self.layers[i].get_activation_function()
            
            pre_activation = np.dot(weights, post) + bias
            # print "i pre_activation: %d, %r" % (i, pre_activation)
            post = np.array(map(activation_function.value, pre_activation))
            # print "i post_activation: %d, %r" % (i, post)
        return post

    def forward_propagate_store(self, x):
        """Propagated the input forward, stores intermediate results
        
        input: x -- a real vector
        output: pre_activation_ls --
        list of outputs from each layer before activation_function
        is applied, this is useful in the back propagation.
        So e.g. output[i] = output after the linear transformation of 
        layers[i], but before the activation function has been applied.
        post -- output of neural network
        """

        post = x
        pre_activation_list = list()
        for i in range(len(self.layers)):
            weights = self.layers[i].get_weights()
            bias = self.layers[i].get_bias()
            activation_function = self.layers[i].get_activation_function()

            pre_activation = np.dot(weights, post) + bias
            pre_activation_list.append(pre_activation)
            post = np.array(map(activation_function.value, pre_activation))

        return pre_activation_list, post

    def back_propagate(self, x, y):
        """Calculate gradient using backprop algorithm, returns gradient.
        
        input: x and y are real vectors that together form a training
        pair for the algorithm.
        outut: a list containing the derivatives with respect to
        all weight matrices and biases in all layers. Specifically
        output[i] contains a 2-tuple (weight_derivatives, bias_derivatives)
        of the same dimension as the corresponding arrays
        """
        assert self.loss_function != None
        preactivation_ls, y_hat = self.forward_propagate_store(x)
        assert len(preactivation_ls[-1]) == len(y)
        # start from the last layer, and then recursively calculate the
        # needed values
        n_layers = len(self.layers)

        #handle base case
        activation_function = self.layers[-1].get_activation_function()
        deriv_loss = np.array([self.loss_function.deriv(y_sc, y_hat_sc) for
                               y_sc, y_hat_sc in zip(y, y_hat)]) # sc = scalar
        deriv_activation_function = np.array(map(activation_function.deriv,
                                  preactivation_ls[-1]))
        delta_base = deriv_activation_function * deriv_loss
        delta_ls = [delta_base]
        assert deriv_loss.shape == y.shape == delta_ls[0].shape

        # handle the rest n_layers - 1 cases
        for i in reversed(range(n_layers - 1)):
            w_idx = i + 1 # use the weights for the layer ahead
            weights = self.layers[w_idx].get_weights()
            activation_function = self.layers[i].get_activation_function()
            activation_deriv = np.array(map(activation_function.deriv,
                                            preactivation_ls[i]))
            delta_ls.append(activation_deriv * np.dot(weights.T, delta_ls[-1]))
        delta_ls.reverse() # IN-PLACE inversion. Need to reverse the list
        # since we started from the front, instead of the back of the list.
        
        # now calculate the actual gradients
        gradients = []
        # base case, needs special treatment, sine x is not in preactivation_ls
        #print "this is x: %r" % x
        weight_derivatives = np.outer(delta_ls[0], x)
        #print "this is weight_derivs: %r" % weight_derivatives
        bias_derivatives = delta_ls[0].copy()
        gradients.append([weight_derivatives, bias_derivatives])
        # go throught the rest n_layers - 1 cases.
        for i in range(n_layers - 1):
            # product of deltas in the layer ahead of the preactivations
            weight_derivatives = np.outer(delta_ls[i+1], preactivation_ls[i])
            assert weight_derivatives.shape[0] == len(delta_ls[i+1]) 
            assert weight_derivatives.shape[1] == len(preactivation_ls[i])
            bias_derivatives = np.copy(delta_ls[i+1])
            gradients.append((weight_derivatives, bias_derivatives))
            
        return gradients

    def loss(self, y, y_hat):
        """calculates loss where we allow y and y_hat to be vectors. 

        input: y -- true value
        y_hat -- prediction
        output: loss -- take this to be the sum of the loss for each component
        i.e. loss(y, y_hat) = sum_i(loss_function.value(y_i, y_hat_i))
        """

        return sum([self.loss_function.loss(y_i, y_hat_i)
                    for y_i, y_hat_i in zip(y, y_hat)])
