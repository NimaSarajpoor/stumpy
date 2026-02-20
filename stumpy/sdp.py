import numpy as np
from numba import njit
from scipy.fft import next_fast_len
from scipy.fft._pocketfft.basic import c2r, r2c
from scipy.signal import convolve, oaconvolve

from . import config

try:  # pragma: no cover
    import pyfftw

    PYFFTW_IS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYFFTW_IS_AVAILABLE = False


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
    m = len(Q)
    l = len(T) - m + 1
    out = np.empty(l)
    for i in range(l):
        result = 0.0
        for j in range(m):
            result += Q[j] * T[i + j]
        out[i] = result

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
    return convolve(np.flipud(Q), T, mode="valid")


def _oaconvolve_sliding_dot_product(Q, T):
    """
    Use scipy's oaconvolve to calculate the sliding dot product.

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
    return oaconvolve(np.ascontiguousarray(Q[::-1]), T, mode="valid")


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


class _PYFFTW_SLIDING_DOT_PRODUCT:
    """
    A class to compute the sliding dot product using FFTW via pyfftw.

    This class uses FFTW (via pyfftw) to efficiently compute the sliding dot product
    between a query sequence, Q, and a time series, T. It preallocates arrays and caches
    FFTW objects to optimize repeated computations with similar-sized inputs.

    Parameters
    ----------
    max_n : int, default=2**20
        Maximum length to preallocate arrays for. This will be the size of the
        real-valued array. A complex-valued array of size `1 + (max_n // 2)`
        will also be preallocated. If inputs exceed this size, arrays will be
        reallocated to accommodate larger sizes.

    Attributes
    ----------
    real_arr : pyfftw.empty_aligned
        Preallocated real-valued array for FFTW computations.

    complex_arr : pyfftw.empty_aligned
        Preallocated complex-valued array for FFTW computations.

    rfft_objects : dict
        Cache of FFTW forward transform objects with
        (next_fast_n, n_threads, planning_flag) as lookup keys.

    irfft_objects : dict
        Cache of FFTW inverse transform objects with
        (next_fast_n, n_threads, planning_flag) as lookup keys.

    Methods
    -------
    __call__(Q, T, n_threads=1, planning_flag="FFTW_ESTIMATE")
        Compute the sliding dot product between `Q` and `T` using FFTW
        via pyfftw, and cache FFTW objects if not already cached.

    Notes
    -----
    The class maintains internal caches of FFTW objects to avoid redundant planning
    operations when called multiple times with similar-sized inputs and parameters.
    When `planning_flag=="FFTW_ESTIMATE"`, there will be no planning operation.
    However, caching FFTW objects is still beneficial as the overhead of creating
    those objects can be avoided in subsequent calls.

    Examples
    --------
    >>> sdp_obj = _PYFFTW_SLIDING_DOT_PRODUCT(max_n=1000)
    >>> Q = np.array([1, 2, 3])
    >>> T = np.array([4, 5, 6, 7, 8])
    >>> result = sdp_obj(Q, T)

    References
    ----------
    `FFTW documentation <http://www.fftw.org/>`__

    `pyfftw documentation <https://pyfftw.readthedocs.io/>`__
    """

    def __init__(self, max_n=2**20):
        """
        Initialize the `_PYFFTW_SLIDING_DOT_PRODUCT` object, which can be called
        to compute the sliding dot product using FFTW via pyfftw.

        Parameters
        ----------
        max_n : int, default=2**20
            Maximum length to preallocate arrays for. This will be the size of the
            real-valued array. A complex-valued array of size `1 + (max_n // 2)`
            will also be preallocated.

        Returns
        -------
        None
        """
        # Preallocate arrays
        self.real_arr = pyfftw.empty_aligned(max_n, dtype="float64")
        self.complex_arr = pyfftw.empty_aligned(1 + (max_n // 2), dtype="complex128")

        # Store FFTW objects, keyed by (next_fast_n, n_threads, planning_flag)
        self.rfft_objects = {}
        self.irfft_objects = {}

    def __call__(self, Q, T, n_threads=1, planning_flag="FFTW_ESTIMATE"):
        """
        Compute the sliding dot product between `Q` and `T` using FFTW via pyfftw,
        and cache FFTW objects if not already cached.

        Parameters
        ----------
        Q : numpy.ndarray
            Query array or subsequence.

        T : numpy.ndarray
            Time series or sequence.

        n_threads : int, default=1
            Number of threads to use for FFTW computations.

        planning_flag : str, default="FFTW_ESTIMATE"
            The planning flag that will be used in FFTW for planning.
            See pyfftw documentation for details. Current options, ordered
            ascendingly by the level of aggressiveness in planning, are:
            "FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT", and "FFTW_EXHAUSTIVE".
            The more aggressive the planning, the longer the planning time, but
            the faster the execution time.

        Returns
        -------
        out : numpy.ndarray
            Sliding dot product between `Q` and `T`.

        Notes
        -----
        The planning_flag is defaulted to "FFTW_ESTIMATE" to be aligned with
        MATLAB's FFTW usage (as of version R2025b)
        See: https://www.mathworks.com/help/matlab/ref/fftw.html

        This implementation is inspired by the answer on StackOverflow:
        https://stackoverflow.com/a/30615425/2955541
        """
        m = Q.shape[0]
        n = T.shape[0]
        next_fast_n = pyfftw.next_fast_len(n)

        # Update preallocated arrays if needed
        if next_fast_n > len(self.real_arr):
            self.real_arr = pyfftw.empty_aligned(next_fast_n, dtype="float64")
            self.complex_arr = pyfftw.empty_aligned(
                1 + (next_fast_n // 2), dtype="complex128"
            )

        real_arr = self.real_arr[:next_fast_n]
        complex_arr = self.complex_arr[: 1 + (next_fast_n // 2)]

        # Get or create FFTW objects
        key = (next_fast_n, n_threads, planning_flag)

        rfft_obj = self.rfft_objects.get(key, None)
        irfft_obj = self.irfft_objects.get(key, None)

        if rfft_obj is None or irfft_obj is None:
            rfft_obj = pyfftw.FFTW(
                input_array=real_arr,
                output_array=complex_arr,
                direction="FFTW_FORWARD",
                flags=(planning_flag,),
                threads=n_threads,
            )
            irfft_obj = pyfftw.FFTW(
                input_array=complex_arr,
                output_array=real_arr,
                direction="FFTW_BACKWARD",
                flags=(planning_flag, "FFTW_DESTROY_INPUT"),
                threads=n_threads,
            )
            self.rfft_objects[key] = rfft_obj
            self.irfft_objects[key] = irfft_obj
        else:
            # Update the input and output arrays of the cached FFTW objects
            # in case their original input and output arrays were reallocated
            # in a previous call
            rfft_obj.update_arrays(real_arr, complex_arr)
            irfft_obj.update_arrays(complex_arr, real_arr)

        # Compute RFFT of T
        real_arr[:n] = T
        real_arr[n:] = 0.0
        rfft_obj.execute()
        rfft_of_T = rfft_obj.output_array.copy()
        # output array is stored in complex_arr, so make a copy
        # to avoid losing it when it is overwritten when computing
        # the RFFT of Q

        # Compute RFFT of Q (reversed, scaled, and zero-padded)
        # Note: scaling is needed since the thin wrapper `execute`
        # is called which does not perform normalization
        np.multiply(Q[::-1], 1.0 / next_fast_n, out=real_arr[:m])
        real_arr[m:] = 0.0
        rfft_obj.execute()
        rfft_of_Q = rfft_obj.output_array

        # Convert back to time domain by taking the inverse RFFT
        np.multiply(rfft_of_T, rfft_of_Q, out=irfft_obj.input_array)
        irfft_obj.execute()

        return irfft_obj.output_array[m - 1 : n]


if PYFFTW_IS_AVAILABLE:  # pragma: no cover
    _pyfftw_sliding_dot_product = _PYFFTW_SLIDING_DOT_PRODUCT(max_n=2**20)


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
