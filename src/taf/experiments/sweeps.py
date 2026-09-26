"""Parameter sweeps: one factor varied over an ordered list of settings.

A sweep is how the two classical curves of the watermarking literature are
measured:

* a **robustness curve** varies one parameter of one attack (SNR, bitrate,
  cutoff frequency ...) and plots the bit error rate against it;
* a **trade-off curve** varies one parameter of one method (usually its
  embedding strength) and plots imperceptibility against robustness.

Both expand into ordinary specifications, so they run through the same
factorial engine as every other design and their rows carry the swept value
in their recorded parameters. The order of the values is kept: it is the
order in which the curve is read and in which a breakdown point is looked
for, so values should go from the mildest to the harshest setting.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ParameterSweep(BaseModel):
    """``target`` with ``parameter`` set to each of ``values`` in turn.

    ``target`` is an attack or method specification whose other parameters
    stay fixed, e.g. ``"mp3"`` or ``"QIM_METHOD:key=7"``.
    """

    target: str = Field(min_length=1)
    parameter: str = Field(min_length=1)
    values: list[Any] = Field(min_length=2, max_length=30)

    @field_validator("values")
    @classmethod
    def _distinct(cls, values: list[Any]) -> list[Any]:
        if len({repr(value) for value in values}) != len(values):
            raise ValueError("Sweep values must be distinct.")
        return values


def _format(name: str, parameters: dict[str, Any]) -> str:
    from taf.plugins import format_method_spec

    return format_method_spec(name, parameters)


def attack_sweep_specs(sweep: ParameterSweep | None) -> list[tuple[str, Any]]:
    """``(attack specification, swept value)`` for every value, in order."""
    if sweep is None:
        return []
    from taf.attacks.registry import parse_spec

    name, fixed, _ = parse_spec(sweep.target)
    return [(_format(name, {**fixed, sweep.parameter: value}), value) for value in sweep.values]


def method_sweep_specs(sweep: ParameterSweep | None) -> list[tuple[str, Any]]:
    """``(method specification, swept value)`` for every value, in order."""
    if sweep is None:
        return []
    from taf.plugins import parse_method_spec

    name, fixed = parse_method_spec(sweep.target)
    return [(_format(name, {**fixed, sweep.parameter: value}), value) for value in sweep.values]


def attack_sweep_problems(sweep: ParameterSweep) -> list[str]:
    from dataclasses import fields

    from taf.attacks.base import AttackError
    from taf.attacks.registry import attack_class, parse_spec, resolve_name

    try:
        name, _, severity = parse_spec(sweep.target)
        cls = attack_class(resolve_name(name))
    except AttackError as error:
        return [str(error)]
    if severity is not None:
        return ["a swept attack cannot also carry a severity level"]
    known = {field.name for field in fields(cls)}
    if sweep.parameter not in known:
        return [f"attack {name!r} has no parameter {sweep.parameter!r}; known: {sorted(known)}"]
    return []


def method_sweep_problems(sweep: ParameterSweep) -> list[str]:
    from taf.plugins import method_parameters, method_spec_problems, parse_method_spec

    problems = method_spec_problems(sweep.target)
    if problems:
        return problems
    name, _ = parse_method_spec(sweep.target)
    known = {entry["name"] for entry in method_parameters(name)}
    if sweep.parameter not in known:
        return [f"method {name!r} has no parameter {sweep.parameter!r}; known: {sorted(known)}"]
    return []


def threshold_crossing(points: list[tuple[Any, float | None]], threshold: float) -> dict[str, Any]:
    """Where a curve, read in sweep order, first rises above ``threshold``.

    Returns ``status`` ``"never"`` (every setting stayed usable),
    ``"always"`` (already above at the first setting) or ``"crossed"`` with
    the value linearly interpolated between the last usable and the first
    unusable setting when both are numeric.
    """
    previous: tuple[Any, float] | None = None
    for value, level in points:
        if level is None:
            continue
        if level > threshold:
            if previous is None:
                return {"status": "always", "value": value}
            last_value, last_level = previous
            if isinstance(value, (int, float)) and isinstance(last_value, (int, float)) and level != last_level:
                fraction = (threshold - last_level) / (level - last_level)
                return {"status": "crossed", "value": last_value + fraction * (value - last_value)}
            return {"status": "crossed", "value": value}
        previous = (value, level)
    return {"status": "never", "value": None}


__all__ = [
    "ParameterSweep",
    "attack_sweep_problems",
    "attack_sweep_specs",
    "method_sweep_problems",
    "method_sweep_specs",
    "threshold_crossing",
]
