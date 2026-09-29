"""Descriptive cards of methods, metrics and attacks.

A card says what a component is: its name for people, what it does in one
sentence, how it works and how to read its results, where it comes from, and
what it needs to run. It is declared once, as the ``card`` class attribute of
the implementation, and everything that describes the component reads it: the
catalogue (``taf.experiments.registry``), the HTTP API and research UI, and the
generated documentation (``python -m taf.catalogue_docs``). A description therefore
lives next to the code it describes and cannot drift into a second copy.

Texts are bilingual. English is required; Polish is optional and clients fall
back to English when it is missing. A plain string is accepted wherever a
``Text`` is, so a plugin author who writes only English writes only strings::

    class MyMethod(SteganographyMethod):
        card = MethodCard(
            title="My method",
            summary="Hides bits in ...",
            details="Each frame ... Larger alpha ...",
            family="transform",
            purpose="watermarking",
            strength_parameter="alpha",
            references=(Reference("Doe & Roe", 2024, doi="10.0000/example"),),
        )

A component without a card is still listed: its title is its registry name
and its summary the first paragraph of its class docstring.
"""

from __future__ import annotations

import importlib.util
import inspect
from dataclasses import dataclass, fields
from typing import Any

LOCALES = ("en", "pl")


@dataclass(frozen=True)
class Text:
    """A text in English and, optionally, Polish."""

    en: str
    pl: str = ""

    def get(self, locale: str = "en") -> str:
        return (self.pl if locale == "pl" else self.en) or self.en

    def as_dict(self) -> dict[str, str]:
        """Every locale filled in, falling back to English."""
        return {locale: self.get(locale) for locale in LOCALES}

    @classmethod
    def of(cls, value: str | Text | None) -> Text:
        if isinstance(value, Text):
            return value
        return cls(str(value or "").strip())


# Groups in which the catalogue presents components, in presentation order,
# with their names. Clients take both the order and the names from here.

#: Families of embedding techniques.
METHOD_FAMILIES: dict[str, Text] = {
    "lsb": Text("Sample-domain LSB", "LSB w dziedzinie próbek"),
    "transform": Text("Transform domain", "Dziedzina transformaty"),
    "spread_spectrum": Text("Spread spectrum", "Rozpraszanie widma"),
    "echo": Text("Echo hiding", "Ukrywanie w echu"),
    "phase": Text("Phase coding", "Kodowanie fazy"),
    "quantization": Text("Quantisation (QIM)", "Kwantyzacja (QIM)"),
    "statistical": Text("Statistical / patchwork", "Statystyczne / patchwork"),
    "adaptive": Text("Adaptive coding", "Kodowanie adaptacyjne"),
    "reversible": Text("Reversible (lossless)", "Odwracalne (bezstratne)"),
    "learned": Text("Learned embedding", "Osadzanie uczone"),
    "neural": Text("Neural network", "Sieć neuronowa"),
}

#: Why a method was designed: covert communication or robust marking.
METHOD_PURPOSES: dict[str, Text] = {
    "steganography": Text("steganography", "steganografia"),
    "watermarking": Text("watermarking", "znakowanie wodne"),
}

#: Metric groups.
METRIC_CATEGORIES: dict[str, Text] = {
    "speech_quality": Text("Speech quality", "Jakość mowy"),
    "speech_intelligibility": Text("Speech intelligibility", "Zrozumiałość mowy"),
    "speech_reverberation": Text("Reverberation", "Pogłos"),
    "ai_based": Text("Learned (AI-based)", "Uczone (AI)"),
}

#: Attack families, keyed by ``taf.attacks.base.AttackCategory`` values.
ATTACK_FAMILIES: dict[str, Text] = {
    "noise": Text("Additive noise", "Szum addytywny"),
    "codec": Text("Lossy codecs", "Kodeki stratne"),
    "filtering": Text("Filtering", "Filtracja"),
    "resampling": Text("Resampling & clock", "Resampling i zegar"),
    "quantization": Text("Quantisation", "Kwantyzacja"),
    "amplitude": Text("Amplitude & dynamics", "Amplituda i dynamika"),
    "temporal": Text("Temporal & desynchronisation", "Czas i desynchronizacja"),
    "acoustic": Text("Acoustic channel", "Kanał akustyczny"),
    "pipeline": Text("Channel pipelines", "Potoki kanałowe"),
}

#: Group of a component that declares none (a plugin without a card).
UNGROUPED = Text("Plugin", "Wtyczka")


def group_label(groups: dict[str, Text], key: str | None) -> Text:
    """Name of a group; a group a plugin introduced is named after its key."""
    if key is None:
        return UNGROUPED
    return groups.get(key) or Text(key.replace("_", " ").capitalize())


def group_order(groups: dict[str, Text], key: str | None) -> tuple[int, str]:
    """Sort key: known groups in presentation order, then others, then none."""
    keys = list(groups)
    if key in groups:
        return keys.index(key), ""
    return len(keys), key or "\uffff"


@dataclass(frozen=True)
class Reference:
    """A publication the component implements or is derived from."""

    #: Short citation as it appears in tables: "Chen & Wornell".
    citation: str
    year: int | None = None
    doi: str | None = None
    url: str | None = None

    @property
    def link(self) -> str | None:
        return f"https://doi.org/{self.doi}" if self.doi else self.url

    def label(self) -> str:
        return f"{self.citation} ({self.year})" if self.year else self.citation

    def as_dict(self) -> dict[str, Any]:
        return {
            "citation": self.citation,
            "year": self.year,
            "doi": self.doi,
            "url": self.url,
            "link": self.link,
        }


@dataclass(frozen=True, kw_only=True)
class Card:
    """What every component card states."""

    #: Name for people: "QIM / ST-DM", "White Gaussian noise".
    title: str | Text
    #: One sentence: what the method hides, the attack models, the metric measures.
    summary: str | Text
    #: How it works, what its parameters do, how to interpret results, and how
    #: the implementation departs from the publication.
    details: str | Text = ""
    #: Short label for figures and tables; the title when empty.
    abbreviation: str = ""
    references: tuple[Reference, ...] = ()
    #: Importable modules needed at run time, beyond the core dependencies.
    requires: tuple[str, ...] = ()
    #: Optional-dependency group that installs them ("ai", "neural").
    extra: str | None = None

    def __post_init__(self) -> None:
        # Strings are accepted for convenience; the stored form is always Text.
        for name in ("title", "summary", "details"):
            object.__setattr__(self, name, Text.of(getattr(self, name)))
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "requires", tuple(self.requires))
        if not self.abbreviation:
            object.__setattr__(self, "abbreviation", self.title.en)

    def missing_requirements(self) -> list[str]:
        """Modules in ``requires`` that are not installed here."""
        return [module for module in self.requires if importlib.util.find_spec(module) is None]

    def requirement_message(self, name: str) -> str | None:
        """Why ``name`` cannot run here, or None when it can."""
        missing = self.missing_requirements()
        if not missing:
            return None
        hint = f" (install the '{self.extra}' extra)" if self.extra else ""
        return f"{name} requires {', '.join(missing)}{hint}"

    def problems(self) -> list[str]:
        """Why the card is incomplete; empty when it is complete."""
        issues = []
        for name in ("title", "summary", "details"):
            text = getattr(self, name)
            if not text.en.strip():
                issues.append(f"{name} has no English text")
        return issues

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form, texts expanded to every locale."""
        result: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Text):
                value = value.as_dict()
            elif item.name == "references":
                value = [reference.as_dict() for reference in value]
            elif isinstance(value, tuple):
                value = list(value)
            result[item.name] = value
        return result


@dataclass(frozen=True, kw_only=True)
class MethodCard(Card):
    #: One of ``METHOD_FAMILIES``; plugins may introduce their own.
    family: str | None = None
    #: One of ``METHOD_PURPOSES``.
    purpose: str | None = None
    #: Constructor parameter that scales the embedding strength - the knob a
    #: trade-off curve sweeps.
    strength_parameter: str | None = None
    #: The method needs several seconds of audio to hold a short message.
    needs_long_input: bool = False

    def problems(self) -> list[str]:
        issues = super().problems()
        if not self.family:
            issues.append("family is missing")
        if self.purpose not in METHOD_PURPOSES:
            issues.append(f"purpose {self.purpose!r} is not one of {list(METHOD_PURPOSES)}")
        return issues


@dataclass(frozen=True, kw_only=True)
class MetricCard(Card):
    #: Unit or range of the value: "dB", "MOS-LQO 1–4.64".
    scale: str | None = None
    #: One of ``METRIC_CATEGORIES``; plugins may introduce their own.
    category: str | None = None
    #: Compares the processed signal with the original (True) or rates it alone.
    intrusive: bool = True
    #: Material the metric was designed for: "speech" or "audio".
    domain: str = "speech"

    def problems(self) -> list[str]:
        issues = super().problems()
        if not self.scale:
            issues.append("scale is missing")
        if not self.category:
            issues.append("category is missing")
        return issues


@dataclass(frozen=True, kw_only=True)
class AttackCard(Card):
    """Attacks declare their family (``category``) and parameters on the class."""


def fallback_card(cls: Any, name: str, kind: type[Card] = Card, **values: Any) -> Card:
    """A card for a component that declares none: its name and docstring.

    ``values`` fills the fields the kind requires (a method's family, a
    metric's scale); they are what the catalogue can say without the author.
    """
    # The class's own docstring only: an inherited one describes the base
    # class, and a dataclass without one gets its signature as ``__doc__``.
    doc = inspect.cleandoc(getattr(cls, "__doc__", None) or "")
    if doc.startswith(f"{getattr(cls, '__name__', '')}("):
        doc = ""
    paragraphs = [" ".join(part.split()) for part in doc.split("\n\n") if part.strip()]
    summary = paragraphs[0] if paragraphs else ""
    details = " ".join(paragraphs[1:3]) if len(paragraphs) > 1 else ""
    return kind(title=name, summary=summary, details=details, **values)


__all__ = [
    "ATTACK_FAMILIES",
    "AttackCard",
    "Card",
    "LOCALES",
    "METHOD_FAMILIES",
    "METHOD_PURPOSES",
    "METRIC_CATEGORIES",
    "MethodCard",
    "MetricCard",
    "Reference",
    "Text",
    "UNGROUPED",
    "fallback_card",
    "group_label",
    "group_order",
]
