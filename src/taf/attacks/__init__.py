from taf.attacks.attacks import CorruptedWavFile
from taf.models.WavFile import WavFile


def apply_all_attacks(wav_file: WavFile) -> CorruptedWavFile:
    return (
        CorruptedWavFile(wav_file)
        .low_pass_filter()
        .resample()
        .pitch_shift()
        .additive_noise()
        .time_stretch()
        .frequency_filter()
        .flip_random_samples()
        .cut_random_samples()
        .amplitude_scaling()
    )


__all__ = ["CorruptedWavFile", "apply_all_attacks"]
