class LinearActivationFunction(object):
    """The identity activation function"""

    def value(self, x):
        return x

    def deriv(self, x):
        return 1
