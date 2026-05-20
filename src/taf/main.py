import os
import sys
from pathlib import Path

from loguru import logger

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from taf.evaluation.workflow import EvaluationResult, evaluate_files, load_files, load_resource_files
from taf.models.WavFile import WavFile
from taf.resources.paths import packaged_dataset_audio_paths


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


def run(files: list[WavFile]) -> EvaluationResult:
    result = evaluate_files(files)
    for row in result.rows:
        if row.error is not None:
            logger.error(
                "{} {} {} {} failed: {}",
                row.input_path,
                row.method,
                row.message_name,
                row.decode_mode,
                row.error,
            )
            continue

        logger.info(
            "{} {} {} {} {} success={}",
            row.input_path,
            row.method,
            row.message_name,
            row.decode_mode,
            row.format or "",
            row.success,
        )
        for metric_name, metric_value in row.metrics.items():
            logger.info("{} : {}", metric_name, metric_value)

    logger.info("Evaluation success rate: {}", result.success_rate())
    return result


if __name__ == '__main__':
    main()
