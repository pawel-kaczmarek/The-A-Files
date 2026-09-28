"""Small reproducible research workflow on both bundled speech corpora."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger
from taf.experiments import ExperimentConfig, run_experiment


def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    for dataset in ("vctk", "librispeech"):
        config = ExperimentConfig(
            name=f"Research smoke: {dataset}", experiment_type="dataset_benchmark", dataset_id=dataset,
            methods=["LSB_METHOD", "QIM_METHOD"], metrics=["SNR_METRIC", "ESTOI_METRIC", "LSD_METRIC", "MRSC_METRIC"],
            attacks=["awgn:snr_db=30"], payload_rates_bps=[8, 16], repetitions=1, random_seed=42,
            subset_seed=7, file_limit=2, max_workers=1,
        )
        run = run_experiment(config)
        assert run.status == "completed", run.error
        clean = [row for row in run.rows if row.attack is None]
        assert len(run.rows) == 16 and len(clean) == 8
        assert all(row.decode_success and not row.metric_errors for row in clean)
        assert all(row.audio_sha256 and row.payload_sha256 for row in run.rows)
        assert all(row.bit_depth and row.source_channels for row in run.rows)
        print(f"{dataset}: {len(run.rows)} rows; {sum(row.decode_success for row in clean)}/{len(clean)} exact baseline decodes")
        example = clean[0]
        print(f"  payload={example.payload_length} bits, rate={example.payload_rate_bps:.6f} bps, category={example.audio_category}")
        print(f"  cover/stego metrics={example.metrics}")
        for method in sorted({row.method for row in run.rows}):
            attacked = [row for row in run.rows if row.method == method and row.attack]
            print(f"  {method}: attacked BER={[row.ber for row in attacked]}, exact goodput={[row.exact_goodput_bps for row in attacked]}")


if __name__ == "__main__":
    main()
