import glob
import os
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger

from TAF.attacks.attacks import CorruptedWavFile
from TAF.generator.generator import generate_random_message
from TAF.loaders.WavFileLoader import WavFileLoader
from TAF.metrics.factory import MetricFactory
from TAF.models.WavFile import WavFile
from TAF.steganography_methods.factory import SteganographyMethodFactory


def load_files(path: str) -> List[WavFile]:
    loader = WavFileLoader(Path(path))
    return [loader.load(os.path.basename(file_path)) for file_path in glob.iglob(path + '/*.flac')]


def apply_all_attacks(wav_file: WavFile) -> CorruptedWavFile:
    return CorruptedWavFile(wav_file) \
        .low_pass_filter() \
        .resample() \
        .pitch_shift() \
        .additive_noise() \
        .time_stretch() \
        .frequency_filter() \
        .flip_random_samples() \
        .cut_random_samples() \
        .amplitude_scaling()


if __name__ == '__main__':
    root_dir = Path(os.path.abspath(os.curdir)).parent
    vctk = os.path.join(root_dir, 'Dataset', 'VCTK', '16')
    libri_speech = os.path.join(root_dir, 'Dataset', 'LibriSpeech', '142345')

    for file in load_files(vctk) + load_files(libri_speech):
        secret_msg = generate_random_message(length=10)
        logger.info("{} {} - {} {}", "-" * 20, file.path, secret_msg, "-" * 20)
        for method in SteganographyMethodFactory.get_all(file.samplerate):
            try:
                secret_data = method.encode(data=file.samples, message=secret_msg)
                for metric in MetricFactory.get_all():
                    m = metric.calculate(file.samples, secret_data, file.samplerate, 0.03, 0.75)
                    logger.info("{} : {}", metric.name(), m)

                decoded_message = method.decode(secret_data, len(secret_msg))
                logger.info("{} : {}", np.array_equal(secret_msg, decoded_message), method.type())

                file.samples = secret_data

                corrupted_wav_file = apply_all_attacks(file)

                decoded_message_after_attacks = method.decode(corrupted_wav_file.samples, len(secret_msg))
                logger.info("{} : {}", np.array_equal(secret_msg, decoded_message_after_attacks), method.type())

                logger.info("-" * 100)
            except Exception as e:
                logger.opt(exception=e).error("Err : {}", method.type())
