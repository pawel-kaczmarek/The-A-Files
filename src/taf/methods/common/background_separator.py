import librosa
import numpy as np


# https://github.com/raaakibul/Background-And-Foreground-sound-separation
def separate_fg_bg_full(y: np.ndarray, sr: int):
    S_full, phase = librosa.magphase(librosa.stft(y))

    S_filter = librosa.decompose.nn_filter(
        S_full,
        aggregate=np.median,
        metric="cosine",
        width=int(librosa.time_to_frames(2, sr=sr))
    )

    S_filter = np.minimum(S_full, S_filter)

    mask_v = librosa.util.softmask(
        S_full - S_filter,
        10 * S_filter,
        power=2
    )

    mask_i = librosa.util.softmask(
        S_filter,
        2 * (S_full - S_filter),
        power=2
    )

    S_fg = mask_v * S_full
    S_bg = mask_i * S_full

    fg_audio = librosa.istft(S_fg * phase, length=len(y))
    bg_audio = librosa.istft(S_bg * phase, length=len(y))

    fg_mask_time = np.abs(fg_audio) >= np.abs(bg_audio)

    return fg_mask_time
