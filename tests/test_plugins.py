"""Third-party components through entry points, and package layering."""

from __future__ import annotations

import ast
from importlib.metadata import EntryPoint
from pathlib import Path

import numpy as np
import pytest

import taf.plugins as plugins

_ENTRY_POINTS = {
    plugins.METHOD_GROUP: [
        # A packaged class registered again under a new name stands in for a
        # third-party method; its constructor takes the sampling rate.
        EntryPoint("TOY_FSVC", "taf.methods.FsvcMethod:FsvcMethod", plugins.METHOD_GROUP),
        # Packaged names cannot be replaced by a plugin.
        EntryPoint("LSB_METHOD", "taf.methods.QimMethod:QimMethod", plugins.METHOD_GROUP),
        EntryPoint("BROKEN", "no_such_module:Thing", plugins.METHOD_GROUP),
    ],
    plugins.METRIC_GROUP: [
        EntryPoint("TOY_SNR", "taf.metrics.speech_quality.SnrMetric:SnrMetric", plugins.METRIC_GROUP),
    ],
    plugins.ATTACK_GROUP: [
        EntryPoint("toy_gain", "taf.attacks.amplitude:GainChange", plugins.ATTACK_GROUP),
    ],
}


@pytest.fixture
def installed_plugins(monkeypatch):
    monkeypatch.setattr(plugins, "entry_points", lambda group: _ENTRY_POINTS.get(group, []))
    plugins.refresh()
    yield
    plugins.refresh()


def test_plugins_join_the_catalogue(installed_plugins):
    assert "TOY_FSVC" in plugins.method_names()
    assert "TOY_SNR" in plugins.metric_names()
    assert "BROKEN" not in plugins.method_names()  # skipped, not fatal
    assert type(plugins.create_method("LSB_METHOD", 16000)).__name__ == "LsbMethod"
    assert type(plugins.create_method("TOY_FSVC", 16000)).__name__ == "FsvcMethod"


def test_plugin_attack_builds_from_a_specification(installed_plugins):
    from taf.attacks.registry import available_attacks, build

    assert "toy_gain" in available_attacks()
    result = build("toy_gain:gain_db=-6").apply(np.ones(1000) * 0.5, 16000)
    assert np.allclose(result.audio, 0.5 * 10 ** (-6 / 20))


def test_plugin_method_runs_in_an_experiment(installed_plugins):
    pytest.importorskip("pandas")
    from taf.experiments import ExperimentConfig, ExperimentType, run_experiment
    from taf.experiments.registry import list_methods

    catalogue = {entry["name"]: entry for entry in list_methods()}
    assert catalogue["TOY_FSVC"]["packaged"] is False
    assert catalogue["LSB_METHOD"]["packaged"] is True

    run = run_experiment(
        ExperimentConfig(
            experiment_type=ExperimentType.DATASET_BENCHMARK,
            name="plugin",
            dataset_id="vctk",
            file_limit=1,
            methods=["TOY_FSVC"],
            metrics=["TOY_SNR"],
            payload_lengths=[8],
            random_seed=1,
        )
    )
    assert run.status == "completed"
    (row,) = run.rows
    assert row.status == "ok" and row.decode_success
    assert "Signal-to-Noise Ratio (SNR)" in row.metrics


# ----------------------------------------------------------------- layers

SRC = Path(__file__).resolve().parents[1] / "src" / "taf"

#: Package -> packages it must never import. The building blocks (methods,
#: metrics, attacks, ...) know nothing about how they are orchestrated, and
#: the engine knows nothing about the HTTP layer, so each can be used, tested
#: and cited on its own.
FORBIDDEN = {
    "models": ("taf.evaluation", "taf.experiments", "taf.persistence", "taf.api"),
    "methods": ("taf.evaluation", "taf.experiments", "taf.persistence", "taf.api"),
    "metrics": ("taf.evaluation", "taf.experiments", "taf.persistence", "taf.api"),
    "attacks": ("taf.evaluation", "taf.experiments", "taf.persistence", "taf.api"),
    "audio": ("taf.evaluation", "taf.experiments", "taf.persistence", "taf.api"),
    "steganalysis": ("taf.evaluation", "taf.experiments", "taf.persistence", "taf.api"),
    "generator": ("taf.evaluation", "taf.experiments", "taf.persistence", "taf.api"),
    "corpora": ("taf.evaluation", "taf.experiments", "taf.persistence", "taf.api"),
    "evaluation": ("taf.experiments", "taf.persistence", "taf.api"),
    # The engine runs without a database; the platform resolves library
    # datasets for it through a registered resolver.
    "experiments": ("taf.persistence", "taf.api"),
    "persistence": ("taf.api",),
}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_layering(package: str):
    violations = [
        f"{path.relative_to(SRC)} imports {name}"
        for path in (SRC / package).rglob("*.py")
        for name in _imports(path)
        if any(name == banned or name.startswith(banned + ".") for banned in FORBIDDEN[package])
    ]
    assert not violations, "\n".join(violations)


def test_packaged_methods_have_distinct_short_names():
    """Figures label methods by short name, so two methods must never share one."""
    from taf.methods.catalog import method_abbreviation
    from taf.models.types import MethodType

    names = {method.name: method_abbreviation(method.name) for method in MethodType}
    assert len(set(names.values())) == len(names), names
    assert method_abbreviation("MY_METHOD", "My method (MM) for tests") == "MM"
    assert method_abbreviation("MY_METHOD") == "MY"
