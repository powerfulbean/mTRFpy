from pathlib import Path
import numpy as np
from mtrf.model import TRF
root = Path(__file__).parent.absolute()


def test_arithmatic():
    trf1 = TRF()
    trf2 = TRF()
    trf1.weights = np.ones((10, 10))
    trf2.weights = np.ones((10, 10))
    trf1.bias = np.ones(10)
    trf2.bias = np.ones(10)
    trf3 = trf1+trf2
    np.testing.assert_equal(trf3.bias, trf1.bias+trf2.bias)
    np.testing.assert_equal(trf3.weights, trf1.weights+trf2.weights)
    trf4 = sum([trf1, trf2, trf3])
    np.testing.assert_equal(trf4.bias, trf1.bias+trf2.bias+trf3.bias)
    np.testing.assert_equal(
            trf4.weights, trf1.weights+trf2.weights+trf3.weights)
    trf4 /= 4
    np.testing.assert_equal(trf4.weights, trf1.weights)
