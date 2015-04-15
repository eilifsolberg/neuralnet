import numpy as np
from nose.tools import *

from neuralnet.squared_error_loss import SquaredErrorLoss

def test_loss():
    y = np.array([1, 3, 4]).astype(float)
    y_hat = np.array([2, 1, 4]).astype(float)
    loss_function = SquaredErrorLoss()
    output = sum([loss_function.loss(y_sc, y_hat_sc)
                  for y_sc, y_hat_sc in zip(y, y_hat)])
    expected = 2.5
    assert_equal(output, expected)

def test_deriv():
    y = np.array([1, 3, 4]).astype(float)
    y_hat = np.array([2, 1, 4]).astype(float)
    loss_function = SquaredErrorLoss()
    output = np.array([loss_function.deriv(y_sc, y_hat_sc)
                       for y_sc, y_hat_sc in zip(y, y_hat)])
    expected = np.array([1, -2, 0]).astype(float)
    np.testing.assert_array_equal(output, expected)
    
