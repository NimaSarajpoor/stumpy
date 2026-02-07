import numpy as np
import pytest
from numpy import testing as npt

from stumpy import sdp


def naive_rolling_window_dot_product(Q, T):
    window = len(Q)
    result = np.zeros(len(T) - window + 1)
    for i in range(len(result)):
        result[i] = np.dot(T[i : i + window], Q)
    return result


test_data = [
    (np.array([-1, 1, 2], dtype=np.float64), np.array(range(5), dtype=np.float64)),
    (
        np.array([9, 8100, -60], dtype=np.float64),
        np.array([584, -11, 23, 79, 1001], dtype=np.float64),
    ),
    (np.random.uniform(-1000, 1000, [8]), np.random.uniform(-1000, 1000, [64])),
]


@pytest.mark.parametrize("Q, T", test_data)
def test_njit_sliding_dot_product(Q, T):
    ref_mp = naive_rolling_window_dot_product(Q, T)
    comp_mp = sdp._njit_sliding_dot_product(Q, T)
    npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("Q, T", test_data)
def test_sliding_dot_product(Q, T):
    ref_mp = naive_rolling_window_dot_product(Q, T)
    comp_mp = sdp._sliding_dot_product(Q, T)
    npt.assert_almost_equal(ref_mp, comp_mp)
