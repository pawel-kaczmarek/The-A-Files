"""Catalogue helpers for methods: short names and constructor parameters.

What a method *is* - family, purpose, reference, strength parameter and its
descriptions - is declared on the class itself as ``card``
(``taf.models.card.MethodCard``). This module derives what the catalogue
needs from the class: a short name for figures and the tunable constructor
parameters.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from taf.models.card import METHOD_FAMILIES as FAMILIES

#: Constructor arguments that are not experimental parameters.
_NOT_PARAMETERS = {"self", "sr", "args", "kwargs"}
#: Parameters that act as a secret key rather than as a tuning knob.
KEY_PARAMETERS = {"key", "seed", "hhat_seed"}


def method_abbreviation(name: str, description: str = "") -> str:
    """A short label for figures: the packaged method's card, the acronym in
    the description's parentheses, or the registry name without its suffix."""
    from taf.methods.factory import BUILTIN_METHOD_CLASSES
    from taf.models.types import MethodType

    if name in MethodType.__members__:
        card = BUILTIN_METHOD_CLASSES[MethodType[name]].card
        if card is not None:
            return card.abbreviation
    match = re.search(r"\(([^()]{2,12})\)", description)
    if match:
        return match.group(1)
    for suffix in ("_METHOD", "Method"):
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", "-")


def constructor_parameters(cls: type) -> list[dict[str, Any]]:
    """Tunable constructor parameters of a method class, with their defaults.

    The sampling rate is supplied by the engine and ``*args``/``**kwargs``
    are not parameters. Parameters without a default are reported with
    ``required``.
    """
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return []
    parameters: list[dict[str, Any]] = []
    for name, parameter in signature.parameters.items():
        if name in _NOT_PARAMETERS or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        default = None if parameter.default is inspect.Parameter.empty else parameter.default
        parameters.append(
            {
                "name": name,
                "default": default,
                "type": type(default).__name__ if default is not None else None,
                "required": parameter.default is inspect.Parameter.empty,
                "is_key": name in KEY_PARAMETERS,
            }
        )
    return parameters


__all__ = [
    "FAMILIES",
    "KEY_PARAMETERS",
    "constructor_parameters",
    "method_abbreviation",
]
