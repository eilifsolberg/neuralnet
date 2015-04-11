import numpy as np

class LogisticSigmoid(object):
    """Logistic sigmoid activation function"""

    def value(self, x):
        return 1.0/(1 + np.exp(-x))

    def deriv(self, x):
        return self.value(x)*self.value(-x)
