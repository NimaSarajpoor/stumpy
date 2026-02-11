import inspect
import warnings

import naive
import numpy as np
import pytest
from numpy import testing as npt

from stumpy import sdp

test_data = [
    (np.array([-1, 1, 2], dtype=np.float64), np.array(range(5), dtype=np.float64)),
    (
        np.array([9, 8100, -60], dtype=np.float64),
        np.array([584, -11, 23, 79, 1001], dtype=np.float64),
    ),
    (np.random.uniform(-1000, 1000, [8]), np.random.uniform(-1000, 1000, [64])),
]


def get_sdp_function_names():
    out = []
    for func_name, func in inspect.getmembers(sdp, inspect.isfunction):
        if func_name.endswith("sliding_dot_product"):
            out.append(func_name)

    return out


@pytest.mark.parametrize("Q, T", test_data)
def test_sliding_dot_product(Q, T):
    for func_name in get_sdp_function_names():
        func = getattr(sdp, func_name)
        try:
            comp = func(Q, T)
            ref = naive.rolling_window_dot_product(Q, T)
            npt.assert_allclose(comp, ref)
        except Exception as e:  # pragma: no cover
            msg = f"Error in {func_name}, with n_Q={len(Q)} and n_T={len(T)}"
            warnings.warn(msg)
            raise e
