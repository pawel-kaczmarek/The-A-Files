import librosa
import numpy as np


def to_frames(audio_data: np.ndarray, fs, frame_length_in_ms):
    samples_per_frame = int(fs * frame_length_in_ms / 1000)
    frames = librosa.util.frame(audio_data, frame_length=samples_per_frame, hop_length=samples_per_frame).T
    tail = len(audio_data) % samples_per_frame
    if tail == 0:
        return frames, None
    last_frame = audio_data[-tail:]
    return frames, last_frame


def to_samples(frame_data: np.ndarray, last_frame: np.array = None):
    if last_frame is not None:
        return np.concatenate((frame_data.flatten(), last_frame))
    return frame_data.flatten()
