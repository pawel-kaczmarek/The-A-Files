import os
import sys
import time
from collections import Counter
from pathlib import Path

from loguru import logger

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from taf import __version__
from taf.evaluation.config import EvaluationConfig
from taf.evaluation.workflow import EvaluationResult, evaluate_files, load_files, load_resource_files
from taf.models.WavFile import WavFile
from taf.resources.paths import packaged_dataset_audio_paths

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "examples" / "config.yaml"


def _load_config(config_path: str | Path | None) -> EvaluationConfig:
    """Resolve the EvaluationConfig: explicit path > examples/config.yaml > dataclass defaults."""
    if config_path is not None:
        path = Path(config_path)
        logger.info("Loading evaluation config from {}", path)
        return EvaluationConfig.from_yaml(path)
    if DEFAULT_CONFIG_PATH.is_file():
        logger.info("Loading evaluation config from default example {}", DEFAULT_CONFIG_PATH)
        return EvaluationConfig.from_yaml(DEFAULT_CONFIG_PATH)
    logger.info("No config file found — using EvaluationConfig() defaults")
    return EvaluationConfig()


def main(root_dir: Path | None = None, config_path: str | Path | None = None) -> None:
    logger.info("the-a-files v{} — starting evaluation", __version__)
    config = _load_config(config_path)
    logger.info(
        "Config: target={} formats={} max_workers={} failure_policy={} output_dir={}",
        config.target.value,
        [getattr(f, "value", str(f)) for f in config.formats],
        config.max_workers,
        config.failure_policy.value,
        config.output_dir,
    )

    if root_dir is None:
        logger.info("Using packaged datasets (VCTK + LibriSpeech)")
        with packaged_dataset_audio_paths() as dataset_files:
            vctk_paths = dataset_files["vctk"]
            libri_paths = dataset_files["librispeech"]
            logger.info(
                "Resolved {} VCTK file(s) and {} LibriSpeech file(s)",
                len(vctk_paths),
                len(libri_paths),
            )
            files = load_resource_files(vctk_paths + libri_paths)
            logger.info("Loaded {} audio file(s) into memory", len(files))
            run(files, config)
        return

    logger.info("Using local dataset root: {}", root_dir)
    dataset_root = root_dir / "src" / "taf" / "resources" / "datasets"
    if not dataset_root.exists():
        dataset_root = root_dir / "Dataset"
    logger.info("Resolved dataset root: {}", dataset_root)

    vctk = dataset_root / "VCTK" / "16"
    libri_speech = dataset_root / "LibriSpeech" / "142345"

    vctk_files = load_files(vctk)[:2]
    libri_files = load_files(libri_speech)[:2]
    logger.info(
        "Loaded {} file(s) from VCTK and {} file(s) from LibriSpeech",
        len(vctk_files),
        len(libri_files),
    )
    run(vctk_files + libri_files, config)


def run(files: list[WavFile], config: EvaluationConfig | None = None) -> EvaluationResult:
    logger.info("Starting evaluation over {} input file(s)", len(files))
    for index, wav in enumerate(files, start=1):
        logger.debug(
            "  [{}/{}] {} (samplerate={} Hz, samples={})",
            index,
            len(files),
            wav.path,
            wav.samplerate,
            len(wav.samples),
        )

    start = time.perf_counter()
    result = evaluate_files(files, config)
    elapsed = time.perf_counter() - start
    logger.info("Evaluation finished in {:.2f}s ({} row(s) produced)", elapsed, len(result.rows))

    success_count = 0
    error_count = 0
    metric_error_total = 0

    for row in result.rows:
        if row.error is not None:
            error_count += 1
            logger.error(
                "FAIL  {} | method={} | msg={} | mode={} | err={}",
                row.input_path,
                row.method,
                row.message_name,
                row.decode_mode,
                row.error,
            )
            continue

        if row.success:
            success_count += 1

        logger.info(
            "OK    {} | method={} | msg={} | mode={} | fmt={} | success={}",
            row.input_path,
            row.method,
            row.message_name,
            row.decode_mode,
            row.format or "-",
            row.success,
        )
        for metric_name, metric_value in row.metrics.items():
            logger.info("    metric  {}: {}", metric_name, metric_value)
        if row.metric_errors:
            metric_error_total += len(row.metric_errors)
            for metric_name, metric_error in row.metric_errors.items():
                logger.warning("    metric  {} skipped: {}", metric_name, metric_error)

    by_method = result.by_method()
    if by_method:
        logger.info("Per-method success summary:")
        for method_name in sorted(by_method):
            rows = by_method[method_name]
            successes = sum(1 for r in rows if r.success)
            logger.info(
                "  {:<40s} {:>3d}/{:<3d} success  ({} error(s))",
                method_name,
                successes,
                len(rows),
                sum(1 for r in rows if r.error is not None),
            )

    formats = Counter(row.format or "direct" for row in result.rows)
    if formats:
        logger.info("Row breakdown by format: {}", dict(formats))

    logger.info(
        "Totals: success={} errored={} metric_errors={} success_rate={:.2%}",
        success_count,
        error_count,
        metric_error_total,
        result.success_rate(),
    )
    return result


if __name__ == '__main__':
    cli_config = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path=cli_config)
