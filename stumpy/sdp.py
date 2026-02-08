import numpy as np
from numba import njit
from scipy.signal import convolve

from . import config


@njit(fastmath=config.STUMPY_FASTMATH_TRUE)
def _njit_sliding_dot_product(Q, T):
    """
    A Numba JIT-compiled implementation of the sliding window dot product.

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
    m = Q.shape[0]
    l = T.shape[0] - m + 1
    out = np.empty(l)
    for i in range(l):
        out[i] = np.dot(Q, T[i : i + m])

    return out


def _convolve_sliding_dot_product(Q, T):
    """
    Use (direct or FFT) convolution to calculate the sliding window dot product.

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
    n = T.shape[0]
    m = Q.shape[0]
    Qr = np.flipud(Q)  # Reverse/flip Q
    QT = convolve(Qr, T)

    return QT.real[m - 1 : n]


def _sliding_dot_product(Q, T):
    """
    Compute the sliding dot product between `Q` and `T`

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
    return _convolve_sliding_dot_product(Q, T)
