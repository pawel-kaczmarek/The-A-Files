"""Catalogue of steganography methods and metrics, packaged and third-party.

The packaged components are listed in the factories and keyed by the names of
``MethodType`` and ``MetricType``. Other distributions can add their own
through entry points, without editing this package::

    # pyproject.toml of the distribution that provides the method
    [project.entry-points."taf.methods"]
    MY_METHOD = "my_package.method:MyMethod"

    [project.entry-points."taf.metrics"]
    MY_METRIC = "my_package.metric:MyMetric"

    [project.entry-points."taf.attacks"]
    my_attack = "my_package.attack:MyAttack"

A method entry point resolves to a callable taking the sampling rate and
returning a ``SteganographyMethod`` (a class whose constructor takes ``sr``
qualifies); a metric entry point to a callable taking no arguments and
returning a ``Metric``; an attack entry point to an ``Attack`` subclass (see
``taf.attacks.registry``). The entry-point name is the name used in
experiment configurations. Packaged names take precedence: a plugin cannot
replace a packaged component, so a published result that names a packaged
method always refers to the packaged implementation.

Once registered, a plugin method is held to the same contract as the
packaged ones and appears in the API catalogue and in experiments. What the
catalogue, the UI and the documentation say about a component comes from the
``card`` attribute of its class (``taf.models.card``); a plugin that declares
none is described by its class docstring.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import entry_points
from typing import Any, Callable

from loguru import logger

METHOD_GROUP = "taf.methods"
METRIC_GROUP = "taf.metrics"
ATTACK_GROUP = "taf.attacks"


@lru_cache(maxsize=None)
def load_entry_points(group: str) -> dict[str, Any]:
    """Objects registered under ``group`` by installed distributions.

    A plugin that fails to import is skipped with a warning rather than
    breaking every experiment.
    """
    loaded: dict[str, Any] = {}
    for entry_point in entry_points(group=group):
        try:
            loaded[entry_point.name] = entry_point.load()
        except Exception as error:  # noqa: BLE001 - a broken plugin must not break the package
            logger.warning(
                "Skipping plugin {} from group {}: {}", entry_point.value, group, error
            )
    return loaded


def refresh() -> None:
    """Forget discovered plugins, e.g. after installing one in a notebook."""
    load_entry_points.cache_clear()


def _merge(builtin: dict[str, Any], group: str) -> dict[str, Any]:
    merged = dict(builtin)
    for name, factory in load_entry_points(group).items():
        if name in merged:
            logger.warning("Plugin {} in group {} shadows a packaged name; ignored", name, group)
            continue
        merged[name] = factory
    return merged


def method_sources() -> dict[str, Callable[..., Any]]:
    """Method name -> class (packaged) or factory (plugin).

    Packaged entries are classes, so that parameters can be passed to their
    constructor; a plugin entry is whatever the entry point resolves to.
    """
    from taf.methods.factory import BUILTIN_METHOD_CLASSES

    return _merge({method.name: cls for method, cls in BUILTIN_METHOD_CLASSES.items()}, METHOD_GROUP)


def method_factories() -> dict[str, Callable[[int], Any]]:
    """Method name -> constructor taking the sampling rate (default parameters)."""
    return {
        name: (lambda sample_rate, name=name: create_method(name, sample_rate))
        for name in method_sources()
    }


def metric_factories() -> dict[str, Callable[[], Any]]:
    """Metric name -> constructor taking no arguments."""
    from taf.metrics.factory import BUILTIN_METRICS

    return _merge({metric.name: create for metric, create in BUILTIN_METRICS.items()}, METRIC_GROUP)


def method_names() -> list[str]:
    return sorted(method_sources())


def metric_names() -> list[str]:
    return sorted(metric_factories())


def is_packaged_method(name: str) -> bool:
    from taf.models.types import MethodType

    return name in MethodType.__members__


def parse_method_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Split ``"QIM_METHOD:step_scale=0.2,key=7"`` into name and parameters.

    Values are parsed with Python literal rules, so numbers, booleans and
    ``None`` arrive typed; bare words are strings. A name without ``:`` has
    no parameters, which is how every configuration written before method
    parameters existed reads.
    """
    import ast

    text = str(spec).strip()
    name, _, parameter_text = text.partition(":")
    parameters: dict[str, Any] = {}
    for chunk in parameter_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"parameter {chunk!r} in {spec!r} must be written as key=value")
        key, _, value = chunk.partition("=")
        try:
            parameters[key.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            parameters[key.strip()] = value.strip()
    return name.strip(), parameters


def format_method_spec(name: str, parameters: dict[str, Any] | None = None) -> str:
    """Inverse of ``parse_method_spec``."""
    if not parameters:
        return name
    rendered = ",".join(
        f"{key}={value!r}" if isinstance(value, str) else f"{key}={value}"
        for key, value in parameters.items()
    )
    return f"{name}:{rendered}"


def method_parameters(name: str) -> list[dict[str, Any]]:
    """Tunable parameters of a method, from its constructor."""
    from taf.methods.catalog import constructor_parameters

    source = method_sources().get(name)
    if source is None:
        raise KeyError(f"Unknown steganography method {name!r}")
    return constructor_parameters(source) if isinstance(source, type) else []


def method_spec_problems(spec: str) -> list[str]:
    """Why a method specification cannot be built; empty when it can."""
    try:
        name, parameters = parse_method_spec(spec)
    except ValueError as error:
        return [str(error)]
    sources = method_sources()
    if name not in sources:
        return [f"unknown method {name!r}"]
    if parameters and isinstance(sources[name], type):
        known = {entry["name"] for entry in method_parameters(name)}
        unknown = sorted(set(parameters) - known)
        if unknown:
            return [f"{name} has no parameter(s) {unknown}; known: {sorted(known)}"]
    return []


def create_method(spec: str, sample_rate: int, **parameters: Any):
    """Build a method from a name or a specification, e.g. ``"QIM_METHOD:step_scale=0.2"``."""
    from taf.methods.factory import build_method

    name, parsed = parse_method_spec(spec)
    parsed.update(parameters)
    sources = method_sources()
    if name not in sources:
        raise KeyError(f"Unknown steganography method {name!r}; known: {sorted(sources)}")
    source = sources[name]
    if isinstance(source, type):
        return build_method(source, sample_rate, **parsed)
    # A plugin factory takes the sampling rate; parameters, if any, as keywords.
    return source(sample_rate, **parsed) if parsed else source(sample_rate)


def _card_of(source: Any, build: Callable[[], Any], kind: type) -> Any:
    """The card declared by ``source``, or by the class a factory builds."""
    card = getattr(source, "card", None)
    if isinstance(card, kind):
        return card
    if isinstance(source, type):
        return None
    try:
        card = getattr(type(build()), "card", None)
    except Exception as error:  # noqa: BLE001 - a plugin that cannot be built is still listed
        logger.debug("Cannot build {} to read its card: {}", source, error)
        return None
    return card if isinstance(card, kind) else None


def method_card(name: str):
    """What the method ``name`` is (``MethodCard``), from its class.

    A method without a card is described by its registry name and docstring.
    """
    from taf.models.card import MethodCard, fallback_card

    source = method_sources().get(name)
    if source is None:
        raise KeyError(f"Unknown steganography method {name!r}")
    card = _card_of(source, lambda: create_method(name, 16000), MethodCard)
    return card or fallback_card(source, name, MethodCard)


def metric_card(name: str):
    """What the metric ``name`` measures (``MetricCard``), from its class."""
    from taf.models.card import MetricCard, fallback_card

    source = metric_factories().get(name)
    if source is None:
        raise KeyError(f"Unknown metric {name!r}")
    card = _card_of(source, source, MetricCard)
    return card or fallback_card(source, name, MetricCard)


def create_metric(name: str):
    factories = metric_factories()
    if name not in factories:
        raise KeyError(f"Unknown metric {name!r}; known: {sorted(factories)}")
    return factories[name]()


def metric_directions() -> dict[str, bool | None]:
    """Direction of every metric value label a run can produce.

    Keys are the labels found in result rows (``Metric.labelled_values``);
    ``None`` means the value is reported but never ranked. A metric that
    cannot be constructed here (a missing optional dependency) is left out,
    and its values are then treated as having no direction.
    """
    directions: dict[str, bool | None] = {}
    for name, create in metric_factories().items():
        try:
            metric = create()
            directions.update(metric.directions())
        except Exception as error:  # noqa: BLE001 - optional dependencies
            logger.debug("Cannot inspect metric {}: {}", name, error)
    return directions


__all__ = [
    "ATTACK_GROUP",
    "METHOD_GROUP",
    "METRIC_GROUP",
    "create_method",
    "create_metric",
    "format_method_spec",
    "method_parameters",
    "method_sources",
    "method_spec_problems",
    "parse_method_spec",
    "is_packaged_method",
    "load_entry_points",
    "method_card",
    "method_factories",
    "method_names",
    "metric_card",
    "metric_directions",
    "metric_factories",
    "metric_names",
    "refresh",
]
