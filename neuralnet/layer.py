class Layer(object):
    """Mainly a storage object, containing the weighmatrix, biasvector and
    activation function
    """

    def __init__(self, weights, bias, activation_function):
        """
        constructor takes three arguments, were weights should be of
        dimension #output_nodes x #input_nodes, and bias should be a vector
        of length #output_nodes
        """
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        
    def get_weights(self):
        """dim = #output x # input"""
        return self.weights

    def get_bias(self):
        """lenght = #output"""
        return self.bias

    def get_activation_function(self):
        return self.activation_function

    def set_weights(self, weights):
        """Should be of dimension #output x #input"""
        self.weights = weights

    def set_bias(self, bias):
        """Should be a vector of length #output"""
        self.bias = bias
