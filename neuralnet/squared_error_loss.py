import numpy as np

class SquaredErrorLoss(object):
    """implements squared error loss"""

    def loss(self, y, y_hat):
        """calculates the squared error loss
        
        input: y -- scalar, the true value
        y_hat -- scalar, the prediction 
        output: scalar, loss value
        """
        return 0.5 * (y - y_hat)**2

    def deriv(self, y, y_hat):
        """calculates the derivative of the squared error loss with respect
        to y_hat
        
        input: y -- scalar, the true value
        y_hat -- scalar, the prediction 
        output: scalar, derivative of loss 
        """
        return -(y - y_hat)

