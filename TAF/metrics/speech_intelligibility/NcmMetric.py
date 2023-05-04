from numbers import Number

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, hilbert

from TAF.models.Metric import Metric
from TAF.metrics.common.metrics_helper import resample_matlab_like


class NcmMetric(Metric):
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        if fs != 8000 and fs != 16000:
            raise ValueError('fs must be either 8 kHz or 16 kHz')

        x = samples_original  # clean signal
        y = samples_processed  # noisy signal
        F_SIGNAL = fs

        F_ENVELOPE = 32  # limits modulations to 0<f<16 Hz
        M_CHANNELS = 20

        #   DEFINE BAND EDGES
        BAND = self.get_band(M_CHANNELS, F_SIGNAL)

        #   Interpolate the ANSI weights in WEIGHT @ fcenter
        fcenter, WEIGHT = self.get_ansis(BAND)

        #   NORMALIZE LENGTHS
        Lx = len(x)
        Ly = len(y)

        if Lx > Ly:
            x = x[0:Ly]
        if Ly > Lx:
            y = y[0:Lx]

        Lx = len(x)
        Ly = len(y)

        X_BANDS = np.zeros((Lx, M_CHANNELS))
        Y_BANDS = np.zeros((Lx, M_CHANNELS))

        #   DESIGN BANDPASS FILTERS
        for a in range(M_CHANNELS):
            B_bp, A_bp = butter(4, np.array([BAND[a], BAND[a + 1]]) * (2 / F_SIGNAL), btype='bandpass')
            X_BANDS[:, a] = lfilter(B_bp, A_bp, x)
            Y_BANDS[:, a] = lfilter(B_bp, A_bp, y)

        gcd = np.gcd(F_SIGNAL, F_ENVELOPE)
        #   CALCULATE HILBERT ENVELOPES, and resample at F_ENVELOPE Hz
        analytic_x = hilbert(X_BANDS, axis=0)
        X = np.abs(analytic_x)
        # X   = resample( X , round(len(x)/F_SIGNAL*F_ENVELOPE))
        X = resample_matlab_like(X, F_ENVELOPE, F_SIGNAL)
        analytic_y = hilbert(Y_BANDS, axis=0)
        Y = np.abs(analytic_y)
        # Y = resample( Y , round(len(x)/F_SIGNAL*F_ENVELOPE))
        Y = resample_matlab_like(Y, F_ENVELOPE, F_SIGNAL)
        ## ---compute weights based on clean signal's rms envelopes-----
        #
        Ldx, pp = X.shape
        p = 3  # power exponent - see Eq. 12

        ro2 = np.zeros((M_CHANNELS,))
        asnr = np.zeros((M_CHANNELS,))
        TI = np.zeros((M_CHANNELS,))

        for k in range(M_CHANNELS):
            x_tmp = X[:, k]
            y_tmp = Y[:, k]
            lambda_x = np.linalg.norm(x_tmp - np.mean(x_tmp)) ** 2
            lambda_y = np.linalg.norm(y_tmp - np.mean(y_tmp)) ** 2
            lambda_xy = np.sum((x_tmp - np.mean(x_tmp)) * (y_tmp - np.mean(y_tmp)))
            ro2[k] = (lambda_xy ** 2) / (lambda_x * lambda_y)
            asnr[k] = 10 * np.log10((ro2[k] + 1e-20) / (1 - ro2[k] + 1e-20))  # Eq.9 in [1]

            if asnr[k] < -15:
                asnr[k] = -15
            elif asnr[k] > 15:
                asnr[k] = 15

            TI[k] = (asnr[k] + 15) / 30  # Eq.10 in [1]

        ncm_val = WEIGHT.dot(TI) / np.sum(WEIGHT)  # Eq.11
        return ncm_val

    def name(self) -> str:
        return "Normalized-covariance measure (NCM)"

    @staticmethod
    def get_band(M, Fs):
        #   This function sets the bandpass filter band edges.
        # It assumes that the sampling frequency is 8000 Hz.
        A = 165
        a = 2.1
        K = 1
        L = 35
        CF = 300
        x_100 = (L / a) * np.log10(CF / A + K)
        CF = Fs / 2 - 600
        x_8000 = (L / a) * np.log10(CF / A + K)
        LX = x_8000 - x_100
        x_step = LX / M
        x = np.arange(x_100, x_8000 + x_step + 1e-20, x_step)
        if len(x) == M:
            np.append(x, x_8000)

        BAND = A * (10 ** (a * x / L) - K)
        return BAND

    @staticmethod
    def get_ansis(BAND):
        fcenter = (BAND[0:-1] + BAND[1:]) / 2

        # Data from Table B.1 in "ANSI (1997). S3.5–1997 Methods for Calculation of the Speech Intelligibility
        # Index. New York: American National Standards Institute."
        f = np.array(
            [150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800,
             7000, 8500])
        BIF = np.array(
            [0.0192, 0.0312, 0.0926, 0.1031, 0.0735, 0.0611, 0.0495, 0.0440, 0.0440, 0.0490, 0.0486, 0.0493, 0.0490,
             0.0547,
             0.0555, 0.0493, 0.0359, 0.0387, 0.0256, 0.0219, 0.0043])
        f_ANSI = interp1d(f, BIF)
        ANSIs = f_ANSI(fcenter)
        return fcenter, ANSIs
