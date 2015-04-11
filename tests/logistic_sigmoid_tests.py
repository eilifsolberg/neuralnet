import numpy as np
from nose.tools import *
from neuralnet.logistic_sigmoid import LogisticSigmoid

def test_logistic_sigmoid():
    sigmoid = LogisticSigmoid()
    x1 = 0
    x2 = np.log(2)
    assert_equal(sigmoid.value(x1), 0.5)
    assert_equal(sigmoid.deriv(x1), 0.25)
    assert_equal(sigmoid.value(x2), 2./3)
    assert_equal(sigmoid.deriv(x2), 2./9)
