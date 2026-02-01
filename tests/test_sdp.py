import inspect
import warnings
from operator import eq, lt

import numpy as np
import pytest
from numpy import testing as npt
from scipy.fft import next_fast_len

from stumpy import sdp

try:  # pragma: no cover
    import pyfftw

    PYFFTW_IMPORTED = True
except ImportError:  # pragma: no cover
    PYFFTW_IMPORTED = False


# README
# Real FFT algorithm performs more efficiently when the length
# of the input array `arr` is composed of small prime factors.
# The next_fast_len(arr, real=True) function from Scipy returns
# the same length if len(arr) is composed of a subset of
# prime numbers 2, 3, 5. Therefore, these radices are
# considered as the most efficient for the real FFT algorithm.

# To ensure that the tests cover different cases, the following cases
# are considered:
# 1. len(T) is even, and len(T) == next_fast_len(len(T), real=True)
# 2. len(T) is odd, and len(T) == next_fast_len(len(T), real=True)
# 3. len(T) is even, and len(T) < next_fast_len(len(T), real=True)
# 4. len(T) is odd, and len(T) < next_fast_len(len(T), real=True)
# And 5. a special case of 1, where len(T) is power of 2.

# Therefore:
# 1. len(T) is composed of 2 and a subset of {3, 5}
# 2. len(T) is composed of a subset of {3, 5}
# 3. len(T) is composed of a subset of {7, 11, 13, ...} and 2
# 4. len(T) is composed of a subset of {7, 11, 13, ...}
# 5. len(T) is power of 2

# In some cases, the prime factors are raised to a power of
# certain degree to increase the length of array to be around
# 1000-2000. This allows us to test sliding_dot_product for
# wider range of query lengths.

test_inputs = [
    # Input format:
    # (
    #     len(T),
    #     remainder,  #  from `len(T) % 2`
    #     comparator,  # for len(T) comparator next_fast_len(len(T), real=True)
    # )
    (
        2 * (3**2) * (5**3),
        0,
        eq,
    ),  # = 2250, Even `len(T)`, and `len(T) == next_fast_len(len(T), real=True)`
    (
        (3**2) * (5**3),
        1,
        eq,
    ),  # = 1125, Odd `len(T)`, and `len(T) == next_fast_len(len(T), real=True)`.
    (
        2 * 7 * 11 * 13,
        0,
        lt,
    ),  # = 2002, Even `len(T)`, and `len(T) < next_fast_len(len(T), real=True)`
    (
        7 * 11 * 13,
        1,
        lt,
    ),  # = 1001, Odd `len(T)`, and `len(T) < next_fast_len(len(T), real=True)`
]


def naive_sliding_dot_product(Q, T):
    m = len(Q)
    l = T.shape[0] - m + 1
    out = np.empty(l)
    for i in range(l):
        out[i] = np.dot(Q, T[i : i + m])
    return out


def get_sdp_functions():
    out = []
    for func_name, func in inspect.getmembers(sdp, inspect.isfunction):
        if func_name.endswith("sliding_dot_product"):
            out.append((func_name, func))

    return out


@pytest.mark.parametrize("n_T, remainder, comparator", test_inputs)
def test_remainder(n_T, remainder, comparator):
    assert n_T % 2 == remainder


@pytest.mark.parametrize("n_T, remainder, comparator", test_inputs)
def test_comparator(n_T, remainder, comparator):
    shape = next_fast_len(n_T, real=True)
    assert comparator(n_T, shape)


@pytest.mark.parametrize("n_T, remainder, comparator", test_inputs)
def test_sdp(n_T, remainder, comparator):
    # test_sdp for cases 1-4

    n_Q_prime = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]
    n_Q_power2 = [2, 4, 8, 16, 32, 64]
    n_Q_values = n_Q_prime + n_Q_power2 + [n_T]
    n_Q_values = sorted(n_Q for n_Q in set(n_Q_values) if n_Q <= n_T)

    # utils.import_sdp_mods()
    for n_Q in n_Q_values:
        Q = np.random.rand(n_Q)
        T = np.random.rand(n_T)
        ref = naive_sliding_dot_product(Q, T)
        for func_name, func in get_sdp_functions():
            try:
                comp = func(Q, T)
                npt.assert_allclose(comp, ref)
            except Exception as e:  # pragma: no cover
                msg = f"Error in {func_name}, with n_Q={n_Q} and n_T={n_T}"
                warnings.warn(msg)
                raise e

    return


def test_sdp_power2():
    # test for case 5. len(T) is power of 2
    pmin = 3
    pmax = 13

    for func_name, func in get_sdp_functions():
        try:
            for q in range(pmin, pmax + 1):
                n_Q = 2**q
                for p in range(q, pmax + 1):
                    n_T = 2**p
                    Q = np.random.rand(n_Q)
                    T = np.random.rand(n_T)

                    ref = naive_sliding_dot_product(Q, T)
                    comp = func(Q, T)
                    npt.assert_allclose(comp, ref)

        except Exception as e:  # pragma: no cover
            msg = f"Error in {func_name}, with q={q} and p={p}"
            warnings.warn(msg)
            raise e

    return


def test_pyfftw_sdp_max_n():
    if not PYFFTW_IMPORTED:  # pragma: no cover
        pytest.skip("Skipping Test PyFFTW Not Installed")

    # When `len(T)` larger than `max_n` in pyfftw_sdp,
    # the internal preallocated arrays should be resized.
    # This test checks that functionality.
    sliding_dot_product = sdp._PYFFTW_SLIDING_DOT_PRODUCT(max_n=2**10)

    # len(T) > max_n to trigger array resizing
    T = np.random.rand(2**11)
    Q = np.random.rand(2**8)

    comp = sliding_dot_product(Q, T)
    ref = naive_sliding_dot_product(Q, T)

    np.testing.assert_allclose(comp, ref)

    return


def test_pyfftw_sdp_cache():
    if not PYFFTW_IMPORTED:  # pragma: no cover
        pytest.skip("Skipping Test PyFFTW Not Installed")

    # To ensure that the caching mechanism in
    # pyfftw_sdp is working as intended
    sliding_dot_product = sdp._PYFFTW_SLIDING_DOT_PRODUCT(max_n=2**10)
    assert sliding_dot_product.rfft_objects == {}
    assert sliding_dot_product.irfft_objects == {}

    T = np.random.rand(2**5)
    Q = np.random.rand(2**2)

    n_threads = 1
    planning_flag = "FFTW_ESTIMATE"
    sliding_dot_product(Q, T, n_threads=n_threads, planning_flag=planning_flag)

    # Check that the FFTW objects are cached
    expected_key = (pyfftw.next_fast_len(len(T)), n_threads, planning_flag)
    assert expected_key in sliding_dot_product.rfft_objects
    assert expected_key in sliding_dot_product.irfft_objects

    return


def test_pyfftw_sdp_update_arrays():
    if not PYFFTW_IMPORTED:  # pragma: no cover
        pytest.skip("Skipping Test PyFFTW Not Installed")

    # To ensure that the cached FFTW objects
    # can be reused when preallocated arrays
    # are updated.
    sliding_dot_product = sdp._PYFFTW_SLIDING_DOT_PRODUCT(max_n=2**10)

    n_threads = 1
    planning_flag = "FFTW_ESTIMATE"

    T1 = np.random.rand(2**5)
    Q1 = np.random.rand(2**2)
    sliding_dot_product(Q1, T1, n_threads=n_threads, planning_flag=planning_flag)

    # len(T2) > max_n to trigger array resizing
    T2 = np.random.rand(2**11)
    Q2 = np.random.rand(2**3)
    sliding_dot_product(Q2, T2, n_threads=n_threads, planning_flag=planning_flag)

    # Check if the FFTW objects cached for inputs (Q1, T1)
    # can be reused when preallocated arrays are resized
    # after calling with (Q2, T2)
    key1 = (pyfftw.next_fast_len(len(T1)), n_threads, planning_flag)
    rfft_obj_before = sliding_dot_product.rfft_objects[key1]
    irfft_obj_before = sliding_dot_product.irfft_objects[key1]

    comp = sliding_dot_product(Q1, T1, n_threads=n_threads, planning_flag=planning_flag)
    ref = naive_sliding_dot_product(Q1, T1)

    # test for correctness
    np.testing.assert_allclose(comp, ref)

    # Check that the same FFTW objects are reused
    assert sliding_dot_product.rfft_objects[key1] is rfft_obj_before
    assert sliding_dot_product.irfft_objects[key1] is irfft_obj_before
