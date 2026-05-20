"""Built-in evaluation config scenarios shipped with the package.

Use :func:`config_path` to obtain a real filesystem path for one of the
packaged YAML scenarios, then pass it to ``EvaluationConfig.from_yaml(...)``.
The shipped scenarios are listed in :data:`SCENARIO_NAMES`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

CONFIG_PACKAGE = "taf.configs"
CONFIG_PREFIX = "config-"

#: Scenarios shipped with the package. Each maps to ``config-<name>.yaml``.
SCENARIO_NAMES: tuple[str, ...] = (
    "full",
    "stego-only",
    "direct-no-metrics",
    "metrics-only",
)


def is_scenario(name: str) -> bool:
    """Return True if ``name`` is one of the built-in scenarios."""
    return name in SCENARIO_NAMES


@contextmanager
def config_path(name: str) -> Iterator[Path]:
    """Yield a real filesystem path to a packaged scenario YAML.

    ``name`` may be either a bare scenario name (``"full"``) or the full file
    name (``"config-full.yaml"``). Raises ``FileNotFoundError`` if the
    resource is not packaged with this install.
    """
    filename = name if name.endswith(".yaml") else f"{CONFIG_PREFIX}{name}.yaml"
    resource = files(CONFIG_PACKAGE).joinpath(filename)
    if not resource.is_file():
        raise FileNotFoundError(
            f"Built-in config '{name}' not found in {CONFIG_PACKAGE}; "
            f"known scenarios: {SCENARIO_NAMES}"
        )
    with as_file(resource) as path:
        yield path


__all__ = ["CONFIG_PACKAGE", "CONFIG_PREFIX", "SCENARIO_NAMES", "config_path", "is_scenario"]
