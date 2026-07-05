"""Reusable experiment services for The A-Files research platform.

The package is UI-agnostic: everything here can be driven from scripts and
notebooks as well as from the FastAPI layer.

    from taf.experiments import ExperimentConfig, ExperimentType, run_experiment

    config = ExperimentConfig(
        experiment_type=ExperimentType.DATASET_BENCHMARK,
        name="lsb-vs-fsvc",
        dataset_id="vctk",
        file_limit=4,
        methods=["LSB_METHOD", "FSVC_METHOD"],
        metrics=["SNR_METRIC"],
        payload_lengths=[16, 32],
    )
    run = run_experiment(config)
    run.to_csv("results.csv")
"""

from taf.experiments.csv_export import export_detailed_csv, export_summary_csv, make_export_filename
from taf.experiments.results import ExperimentResultRow, bit_accuracy, bit_error_rate, normalize_row
from taf.experiments.runner import ExperimentRun, preview_experiment, run_experiment, run_experiment_async
from taf.experiments.schema import ExperimentConfig, ExperimentPlan, ExperimentType

__all__ = [
    "ExperimentConfig",
    "ExperimentPlan",
    "ExperimentResultRow",
    "ExperimentRun",
    "ExperimentType",
    "bit_accuracy",
    "bit_error_rate",
    "export_detailed_csv",
    "export_summary_csv",
    "make_export_filename",
    "normalize_row",
    "preview_experiment",
    "run_experiment",
    "run_experiment_async",
]
