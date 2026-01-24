import numpy as np
from numba import njit
from scipy.fft import next_fast_len
from scipy.fft._pocketfft.basic import c2r, r2c
from scipy.signal import convolve

from . import config


@njit(fastmath=config.STUMPY_FASTMATH_TRUE)
def _njit_sliding_dot_product(Q, T):
    """
    A Numba JIT-compiled implementation of the sliding dot product.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    Returns
    -------
    out : numpy.ndarray
        Sliding dot product between `Q` and `T`.
    """
    m = len(Q)
    l = T.shape[0] - m + 1
    out = np.empty(l)
    for i in range(l):
        result = 0.0
        for j in range(m):
            result += Q[j] * T[i + j]
        out[i] = result

    return out


def _convolve_sliding_dot_product(Q, T):
    """
    Use FFT or direct convolution to calculate the sliding dot product.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    Returns
    -------
    output : numpy.ndarray
        Sliding dot product between `Q` and `T`.

    Notes
    -----
    Calculate the sliding dot product

    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Table I, Figure 4
    """
    # mode='valid' returns output of convolution where the two
    # sequences fully overlap.

    return convolve(np.flipud(Q), T, mode="valid")


def _pocketfft_sliding_dot_product(Q, T):
    """
    Use scipy.fft._pocketfft to compute
    the sliding dot product.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    Returns
    -------
    output : numpy.ndarray
        Sliding dot product between `Q` and `T`.
    """
    n = len(T)
    m = len(Q)
    next_fast_n = next_fast_len(n, real=True)

    tmp = np.empty((2, next_fast_n))
    tmp[0, :m] = Q[::-1]
    tmp[0, m:] = 0.0
    tmp[1, :n] = T
    tmp[1, n:] = 0.0
    fft_2d = r2c(True, tmp, axis=-1)

    return c2r(False, np.multiply(fft_2d[0], fft_2d[1]), n=next_fast_n)[m - 1 : n]
