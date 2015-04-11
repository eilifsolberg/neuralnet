from neuralnet.linear_activation_function import LinearActivationFunction
from nose.tools import assert_equal

def test_linear_activation():
    x1 = 0
    x2 = -1
    linear = LinearActivationFunction()
    assert_equal(linear.value(x1), x1)
    assert_equal(linear.deriv(x1), 1)
    assert_equal(linear.value(x2), x2)
    assert_equal(linear.deriv(x2), 1)
