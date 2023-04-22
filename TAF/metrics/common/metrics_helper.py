from fractions import Fraction

import numpy as np
from numba import jit
from scipy.signal import firls, upfirdn
from scipy.signal.windows import kaiser


def extract_overlapped_windows(x, nperseg, noverlap, window=None):
    # source: https://github.com/scipy/scipy/blob/v1.2.1/scipy/signal/spectral.py
    step = nperseg - noverlap
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    if window is not None:
        result = window * result
    return result


def resample_matlab_like(x_orig, p, q):
    if len(x_orig.shape) > 2:
        raise ValueError('x must be a vector or 2d matrix')

    if x_orig.shape[0] < x_orig.shape[1]:
        x = x_orig.T
    else:
        x = x_orig
    beta = 5
    N = 10
    frac = Fraction(p, q)
    p = frac.numerator
    q = frac.denominator
    pqmax = max(p, q)
    fc = 1 / 2 / pqmax
    L = 2 * N * pqmax + 1
    h = firls(L, np.array([0, 2 * fc, 2 * fc, 1]), np.array([1, 1, 0, 0])) * kaiser(L, beta)
    h = p * h / sum(h)

    Lhalf = (L - 1) / 2
    Lx = x.shape[0]

    nz = int(np.floor(q - np.mod(Lhalf, q)))
    z = np.zeros((nz,))
    h = np.concatenate((z, h))
    Lhalf = Lhalf + nz
    delay = int(np.floor(np.ceil(Lhalf) / q))
    nz1 = 0
    while np.ceil(((Lx - 1) * p + len(h) + nz1) / q) - delay < np.ceil(Lx * p / q):
        nz1 = nz1 + 1
    h = np.concatenate((h, np.zeros(nz1, )))
    y = upfirdn(h, x, p, q, axis=0)
    Ly = int(np.ceil(Lx * p / q))
    y = y[delay:]
    y = y[:Ly]

    if x_orig.shape[0] < x_orig.shape[1]:
        y = y.T

    return y


@jit
def lpcoeff(speech_frame, model_order):
    eps = np.finfo(np.float64).eps
    # ----------------------------------------------------------
    # (1) Compute Autocorrelation Lags
    # ----------------------------------------------------------
    winlength = max(speech_frame.shape)
    R = np.zeros((model_order + 1,))
    for k in range(model_order + 1):
        if k == 0:
            R[k] = np.sum(speech_frame[0:] * speech_frame[0:])
        else:
            R[k] = np.sum(speech_frame[0:-k] * speech_frame[k:])

    # R=scipy.signal.correlate(speech_frame,speech_frame)
    # R=R[len(speech_frame)-1:len(speech_frame)+model_order]
    # ----------------------------------------------------------
    # (2) Levinson-Durbin
    # ----------------------------------------------------------
    a = np.ones((model_order,))
    a_past = np.ones((model_order,))
    rcoeff = np.zeros((model_order,))
    E = np.zeros((model_order + 1,))

    E[0] = R[0]

    for i in range(0, model_order):
        a_past[0:i] = a[0:i]

        sum_term = np.sum(a_past[0:i] * R[i:0:-1])

        if E[i] == 0.0:  # prevents zero division error, numba doesn't allow try/except statements
            rcoeff[i] = np.inf
        else:
            rcoeff[i] = (R[i + 1] - sum_term) / (E[i])

        a[i] = rcoeff[i]
        # if i==0:
        #    a[0:i] = a_past[0:i] - rcoeff[i]*np.array([])
        # else:
        if i > 0:
            a[0:i] = a_past[0:i] - rcoeff[i] * a_past[i - 1::-1]

        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]

    acorr = R
    refcoeff = rcoeff
    lpparams = np.ones((model_order + 1,))
    lpparams[1:] = -a
    return (lpparams, R)


@jit
def lpc2cep(a):
    #
    # converts prediction to cepstrum coefficients
    #
    # Author: Philipos C. Loizou

    M = len(a)
    cep = np.zeros((M - 1,))

    cep[0] = -a[1]

    for k in range(2, M):
        ix = np.arange(1, k)
        vec1 = cep[ix - 1] * a[k - 1:0:-1] * ix
        cep[k - 1] = -(a[k] + np.sum(vec1) / k)
    return cep
