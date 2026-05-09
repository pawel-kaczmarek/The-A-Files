from pathlib import Path
from typing import Iterable, List

import numpy as np
from loguru import logger

from taf.attacks.attacks import CorruptedWavFile
from taf.generator.generator import generate_random_message
from taf.loaders.WavFileLoader import WavFileLoader
from taf.metrics.factory import MetricFactory
from taf.models.WavFile import WavFile
from taf.methods.factory import SteganographyMethodFactory
from taf.resources import packaged_dataset_audio_paths


def load_files(path: str | Path) -> List[WavFile]:
    directory = Path(path)
    loader = WavFileLoader(directory)
    files = sorted(
        file_path
        for file_path in directory.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in {".flac", ".wav"}
    )
    return [loader.load(file_path.name) for file_path in files]


def load_resource_files(paths: Iterable[Path]) -> List[WavFile]:
    return [WavFile.load(path) for path in paths]


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


def main(root_dir: Path | None = None) -> None:
    if root_dir is None:
        with packaged_dataset_audio_paths() as dataset_files:
            run(load_resource_files(dataset_files["vctk"] + dataset_files["librispeech"]))
        return

    dataset_root = root_dir / "src" / "taf" / "resources" / "datasets"
    if not dataset_root.exists():
        dataset_root = root_dir / "Dataset"

    vctk = dataset_root / "VCTK" / "16"
    libri_speech = dataset_root / "LibriSpeech" / "142345"

    run(load_files(vctk) + load_files(libri_speech))


def run(files: List[WavFile]) -> None:
    for file in files:
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


if __name__ == '__main__':
    main()
