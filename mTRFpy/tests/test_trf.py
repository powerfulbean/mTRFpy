from pathlib import Path
import numpy as np
from mTRFpy.Model import CTRF
root = Path(__file__).parent.absolute()


def test_arithmatic():
    trf1 = CTRF()
    trf2 = CTRF()
    trf1.w = np.ones((10, 10))
    trf2.w = np.ones((10, 10))
    trf1.b = np.ones(10)
    trf2.b = np.ones(10)
    trf3 = trf1+trf2
    np.testing.assert_equal(trf3.b, trf1.b+trf2.b)
    np.testing.assert_equal(trf3.w, trf1.w+trf2.w)
