import numpy as np
from scipy.signal import convolve
from scipy.signal.windows import hann


def mixer(L, bits, lower=0, upper=1, K=256):
    if 2 * K > L:
        K = np.floor(L / 4) - np.mod(np.floor(L / 4), 4)
    else:
        K = K - np.mod(K, 4)
    N = len(bits)
    encbit = np.asarray(bits)

    m_sig = np.transpose(np.ones((L, 1)) * encbit)
    m_sig = m_sig.reshape((N * L))
    c = convolve(m_sig, hann(K))
    wnorm_left = int(K / 2)
    wnorm_right = int(len(c) - K / 2 + 1)
    wnorm = c[wnorm_left:wnorm_right] / np.max(np.abs(c))
    w_sig = wnorm * (upper - lower) + lower
    m_sig = m_sig * (upper - lower) + lower

    return [w_sig, m_sig]
