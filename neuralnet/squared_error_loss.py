import numpy as np

class SquaredErrorLoss(object):
    """implements squared error loss"""

    def loss(self, y, y_hat):
        """y and y_hat can be vectors or scalars"""
        return 0.5 * np.sum((y_hat - y)**2)

    def deriv(self, y, y_hat):
        """assume y and y_hat are the output vectors"""
        return  (y_hat - y)
