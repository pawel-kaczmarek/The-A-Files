"""Detectability experiments: can a steganalyser tell stego from cover?

The fourth property of an embedding method, next to capacity, transparency
and robustness, and the one that separates steganography from watermarking.
For every method and payload length a detector is trained on windows of the
dataset (``taf.steganalysis.measure_detectability``) and scored on held-out
windows. The result is one entry per (method, payload), not one row per
trial, so it goes into the summary.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from loguru import logger

from taf.evaluation.seeding import derive_seed

DEFAULT_WINDOW_LENGTH = 32000
DEFAULT_TEST_FRACTION = 0.3


def detectability_settings(config: Any) -> dict[str, Any]:
    options = config.advanced_options or {}
    return {
        "window_length": int(options.get("window_length", DEFAULT_WINDOW_LENGTH)),
        "test_fraction": float(options.get("test_fraction", DEFAULT_TEST_FRACTION)),
    }


def run_detectability(config: Any, files: Sequence[Any]) -> dict[str, Any]:
    """Steganalysis of every configured method at every payload length."""
    from taf.plugins import create_method
    from taf.steganalysis import measure_detectability

    settings = detectability_settings(config)
    rates = sorted({wav_file.samplerate for wav_file in files})
    if len(rates) > 1:
        raise ValueError(
            f"detectability needs one sampling rate across the dataset; found {rates} Hz"
        )
    sample_rate = rates[0] if rates else 16000
    covers = [wav_file.samples for wav_file in files]

    entries: list[dict[str, Any]] = []
    for method_name in config.resolved_methods():
        for payload in config.payload_lengths:
            seed = derive_seed(config.random_seed, "steganalysis", method_name, payload)
            entry: dict[str, Any] = {"method_name": method_name, "payload_length": payload, "seed": seed}
            try:
                result = measure_detectability(
                    create_method(method_name, sample_rate),
                    covers,
                    message_length=payload,
                    window_length=settings["window_length"],
                    test_fraction=settings["test_fraction"],
                    seed=seed,
                )
            except Exception as error:  # noqa: BLE001 - recorded per entry, like a failed row
                logger.warning("Detectability of {} failed: {}", method_name, error)
                entry.update({"status": "error", "error": str(error)})
            else:
                entry.update(asdict(result))
                entry.update(
                    {
                        "status": "ok",
                        "undetectable": result.undetectable,
                        "significantly_detectable": result.significantly_detectable,
                    }
                )
            entries.append(entry)

    scored = [entry for entry in entries if entry["status"] == "ok"]
    return {
        "detectability": entries,
        "settings": {**settings, "sample_rate": sample_rate, "files": len(files)},
        "most_detectable": max(scored, key=lambda e: e["accuracy"])["method"] if scored else None,
        "least_detectable": min(scored, key=lambda e: e["accuracy"])["method"] if scored else None,
        "significantly_detectable_methods": sorted(
            {entry["method"] for entry in scored if entry["significantly_detectable"]}
        ),
    }


__all__ = ["DEFAULT_TEST_FRACTION", "DEFAULT_WINDOW_LENGTH", "detectability_settings", "run_detectability"]
