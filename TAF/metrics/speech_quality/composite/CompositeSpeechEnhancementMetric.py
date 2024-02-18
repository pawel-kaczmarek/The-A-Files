from abc import ABC

import numpy as np
from librosa import lpc
from scipy.linalg import toeplitz
from scipy.signal import lfilter

from TAF.models.Metric import Metric


class BaseSpeechEnhancementMetric(Metric, ABC):

    def calculate_internal(self,
                           samples_original: np.ndarray,
                           samples_processed: np.ndarray,
                           fs: int) -> np.ndarray:
        # Compute composite measures
        alpha = 0.95
        # Read audio files
        samples_original = samples_original
        samples_processed = samples_processed

        # Trim or pad to the minimum length
        length = min(len(samples_original), len(samples_processed))
        samples_original = samples_original[:length]
        samples_processed = samples_processed[:length]

        # Compute the WSS measure
        wss_dist_vec = self.wss(samples_original, samples_processed, fs)
        wss_dist_vec.sort()
        wss_dist = np.mean(wss_dist_vec[:round(len(wss_dist_vec) * alpha)])

        # Compute the LLR measure
        llr_dist = self.llr(samples_original, samples_processed, fs)
        llr_dist.sort()
        llr_mean = np.mean(llr_dist[:round(len(llr_dist) * alpha)])

        # Compute the SNRseg
        snr_dist, segsnr_dist = self.snr(samples_original, samples_processed, fs)
        snr_mean = snr_dist
        segSNR = np.mean(segsnr_dist)

        # Placeholder for PESQ, update accordingly
        pesq_mos = 0

        # Compute composite measures
        Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_mos - 0.009 * wss_dist
        Cbak = 1.634 + 0.478 * pesq_mos - 0.007 * wss_dist + 0.063 * segSNR
        Covl = 1.594 + 0.805 * pesq_mos - 0.512 * llr_mean - 0.007 * wss_dist

        return np.asarray((Csig, Cbak, Covl, segSNR))

    def wss(self, clean_speech, processed_speech, sample_rate):
        clean_length = len(clean_speech)
        processed_length = len(processed_speech)

        if clean_length != processed_length:
            print('Error: Files must have the same length.')
            return

        winlength = round(30 * sample_rate / 1000)
        skiprate = winlength // 4
        max_freq = sample_rate / 2
        num_crit = 25
        USE_FFT_SPECTRUM = 1
        n_fft = 2 ** (len(bin(2 * winlength)) - 2)
        n_fftby2 = n_fft // 2
        Kmax = 20
        Klocmax = 1
        bw_min = 70

        cent_freq = [50.0, 120.0, 190.0, 260.0, 330.0, 400.0, 470.0, 540.0, 617.372, 703.378,
                     798.717, 904.128, 1020.38, 1148.30, 1288.72, 1442.54, 1610.70, 1794.16,
                     1993.93, 2211.08, 2446.71, 2701.97, 2978.04, 3276.17, 3597.63]

        bandwidth = [70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 77.3724, 86.0056, 95.3398,
                     105.411, 116.256, 127.914, 140.423, 153.823, 168.154, 183.457, 199.776,
                     217.153, 235.631, 255.255, 276.072, 298.126, 321.465, 346.136]

        min_factor = np.exp(-30.0 / (2.0 * 2.303))  # -30 dB point of filter

        crit_filter = np.zeros((num_crit, n_fftby2))

        for i in range(num_crit):
            f0 = (cent_freq[i] / max_freq) * n_fftby2
            all_f0 = int(f0)
            bw = (bandwidth[i] / max_freq) * n_fftby2
            norm_factor = np.log(bw_min) - np.log(bandwidth[i])
            j = np.arange(n_fftby2)
            crit_filter[i, :] = np.exp(-11 * (((j - all_f0) / bw) ** 2) + norm_factor)
            crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor)

        start = 0
        num_frames = int(clean_length / skiprate - (winlength / skiprate))
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(0, winlength) / winlength))

        distortion = np.zeros(num_frames)
        for frame_count in range(num_frames):
            clean_frame = clean_speech[start:start + winlength]
            processed_frame = processed_speech[start:start + winlength]
            clean_frame = clean_frame * window
            processed_frame = processed_frame * window

            if USE_FFT_SPECTRUM:
                clean_spec = np.abs(np.fft.fft(clean_frame, n_fft)) ** 2
                processed_spec = np.abs(np.fft.fft(processed_frame, n_fft)) ** 2
            else:
                a_vec = np.zeros(n_fft)
                a_vec[:10] = lfilter([1], [1, *lpc(clean_frame, 10)], [1])
                clean_spec = 1.0 / (np.abs(np.fft.fft(a_vec, n_fft)) ** 2)

                a_vec = np.zeros(n_fft)
                a_vec[:10] = lfilter([1], [1, *lpc(processed_frame, 10)], [1])
                processed_spec = 1.0 / (np.abs(np.fft.fft(a_vec, n_fft)) ** 2)

            clean_energy = np.zeros(num_crit)
            processed_energy = np.zeros(num_crit)

            for i in range(num_crit):
                clean_energy[i] = np.sum(clean_spec[:n_fftby2] * crit_filter[i, :])
                processed_energy[i] = np.sum(processed_spec[:n_fftby2] * crit_filter[i, :])

            clean_energy = 10 * np.log10(np.maximum(clean_energy, 1E-10))
            processed_energy = 10 * np.log10(np.maximum(processed_energy, 1E-10))

            clean_slope = clean_energy[1:num_crit] - clean_energy[0:num_crit - 1]
            processed_slope = processed_energy[1:num_crit] - processed_energy[0:num_crit - 1]

            clean_loc_peak = np.zeros(num_crit - 1)
            processed_loc_peak = np.zeros(num_crit - 1)

            for i in range(num_crit - 1):
                if clean_slope[i] > 0:
                    n = i
                    while n < num_crit - 1 and clean_slope[n] > 0:
                        n += 1
                    clean_loc_peak[i] = clean_energy[n - 1]
                else:
                    n = i
                    while n > 0 and clean_slope[n] <= 0:
                        n -= 1
                    clean_loc_peak[i] = clean_energy[n + 1]

                if processed_slope[i] > 0:
                    n = i
                    while n < num_crit - 1 and processed_slope[n] > 0:
                        n += 1
                    processed_loc_peak[i] = processed_energy[n - 1]
                else:
                    n = i
                    while n > 0 and processed_slope[n] <= 0:
                        n -= 1
                    processed_loc_peak[i] = processed_energy[n + 1]

            dBMax_clean = np.max(clean_energy)
            dBMax_processed = np.max(processed_energy)

            Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:num_crit - 1])
            Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - clean_energy[:num_crit - 1])
            W_clean = Wmax_clean * Wlocmax_clean

            Wmax_processed = Kmax / (Kmax + dBMax_processed - processed_energy[:num_crit - 1])
            Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - processed_energy[:num_crit - 1])
            W_processed = Wmax_processed * Wlocmax_processed

            W = (W_clean + W_processed) / 2.0

            distortion[frame_count] = np.sum(W * (clean_slope[:num_crit - 1] - processed_slope[:num_crit - 1]) ** 2)
            distortion[frame_count] = distortion[frame_count] / np.sum(W)

            start += skiprate

        return distortion

    def llr(self, clean_speech, processed_speech, sample_rate):
        clean_length = len(clean_speech)
        processed_length = len(processed_speech)

        if clean_length != processed_length:
            print('Error: Both Speech Files must be same length.')
            return

        winlength = round(30 * sample_rate / 1000)
        skiprate = winlength // 4

        if sample_rate < 10000:
            P = 10
        else:
            P = 16

        frame_len = clean_length // skiprate - (winlength // skiprate)
        start = 0
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(0, winlength) / winlength))

        distortion = np.zeros(frame_len)

        for frame_count in range(frame_len):
            clean_frame = clean_speech[start:start + winlength]
            processed_frame = processed_speech[start:start + winlength]
            clean_frame = clean_frame * window
            processed_frame = processed_frame * window

            R_clean, _, A_clean = self.lpcoeff(clean_frame, P)
            R_processed, _, A_processed = self.lpcoeff(processed_frame, P)

            numerator = A_processed @ toeplitz(R_clean) @ A_processed.T
            denominator = A_clean @ toeplitz(R_clean) @ A_clean.T
            distortion[frame_count] = np.log(numerator / denominator)
            start += skiprate

        return distortion

    def lpcoeff(self, speech_frame, model_order):
        winlength = len(speech_frame)
        R = np.zeros(model_order + 1)

        for k in range(0, model_order + 1):
            R[k] = np.sum(speech_frame[:winlength - k] * speech_frame[k:winlength])

        a = np.ones(model_order)
        E = np.zeros(model_order + 1)
        E[0] = R[0]

        rcoeff = np.zeros(model_order)

        for i in range(model_order):
            a_past = np.copy(a[:i])
            sum_term = np.sum(a_past[:i] * np.flip(R[1:i + 1]))
            # sum_term = np.sum(a_past[:i] * R[i::-1][1:])
            rcoeff[i] = (R[i + 1] - sum_term) / E[i]
            a[i] = rcoeff[i]
            a[:i] = a_past - rcoeff[i] * a_past[::-1]
            E[i + 1] = (1 - rcoeff[i] ** 2) * E[i]

        acorr = R
        refcoeff = rcoeff
        lpparams = np.concatenate(([1], -a))

        return acorr, refcoeff, lpparams

    def snr(self, clean_speech, processed_speech, sample_rate):
        clean_length = len(clean_speech)
        processed_length = len(processed_speech)

        if clean_length != processed_length:
            print('Error: Both Speech Files must be same length.')
            return

        overall_snr = 10 * np.log10(np.sum(clean_speech ** 2) / np.sum((clean_speech - processed_speech) ** 2))

        winlength = round(30 * sample_rate / 1000)
        skiprate = winlength // 4
        MIN_SNR = -10
        MAX_SNR = 35

        frame_len = clean_length // skiprate - (winlength // skiprate)
        start = 0
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(0, winlength) / winlength))

        segmental_snr = np.zeros(frame_len)

        for frame_count in range(frame_len):
            clean_frame = clean_speech[start:start + winlength]
            processed_frame = processed_speech[start:start + winlength]
            clean_frame = clean_frame * window
            processed_frame = processed_frame * window

            signal_energy = np.sum(clean_frame ** 2)
            noise_energy = np.sum((clean_frame - processed_frame) ** 2)
            segmental_snr[frame_count] = 10 * np.log10(signal_energy / noise_energy)
            segmental_snr[frame_count] = max(segmental_snr[frame_count], MIN_SNR)
            segmental_snr[frame_count] = min(segmental_snr[frame_count], MAX_SNR)

            start += skiprate

        return overall_snr, segmental_snr

    def name(self) -> str:
        return "Predicted rating of background distortion"
