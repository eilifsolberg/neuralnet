import numpy as np
from nose.tools import *

from neuralnet.squared_error_loss import SquaredErrorLoss

def test_loss():
    y = np.array([1, 3, 4]).astype(float)
    y_hat = np.array([2, 1, 4]).astype(float)
    loss_function = SquaredErrorLoss()
    assert_equal(loss_function.loss(y, y_hat), 2.5)

def test_deriv():
    y = np.array([1, 3, 4]).astype(float)
    y_hat = np.array([2, 1, 4]).astype(float)
    loss_function = SquaredErrorLoss()
    np.testing.assert_array_equal(loss_function.deriv(y, y_hat),
                                  np.array([1, -2, 0]))
