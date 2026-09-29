"""Every packaged method, metric and attack is described, and described once.

A component's card (``taf.models.card``) feeds the API, the research UI and
the generated documentation. These tests make a new component fail loudly
until its card is complete, keep the texts consistent with the code they
describe, and keep the generated pages in step with the cards.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from taf.attacks.registry import ATTACK_CLASSES, ATTACK_FACTORIES, attack_card, available_attacks
from taf.methods.catalog import constructor_parameters
from taf.methods.factory import BUILTIN_METHOD_CLASSES
from taf.metrics.factory import BUILTIN_METRICS
from taf.models.card import (
    ATTACK_FAMILIES,
    METHOD_FAMILIES,
    METRIC_CATEGORIES,
    AttackCard,
    MethodCard,
    MetricCard,
    Reference,
    Text,
    fallback_card,
    group_label,
)
from taf.models.types import MethodType, MetricType

ROOT = Path(__file__).resolve().parents[1]

#: A compound identifier in running text: ``snr_db``, ``step_scale``.
_IDENTIFIER = re.compile(r"(?<![\w.])[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?![\w(])")


def _assert_bilingual(card) -> None:
    assert not card.problems(), card.problems()
    for name in ("summary", "details"):
        text = getattr(card, name)
        assert text.pl.strip(), f"{name} has no Polish text"
        assert text.en.strip() != text.pl.strip(), f"{name} is not translated"


@pytest.mark.parametrize("method_type", list(MethodType), ids=lambda m: m.name)
def test_every_method_has_a_complete_card(method_type: MethodType):
    cls = BUILTIN_METHOD_CLASSES[method_type]
    card = cls.card
    assert isinstance(card, MethodCard), f"{cls.__name__} declares no MethodCard"
    _assert_bilingual(card)
    assert card.family in METHOD_FAMILIES
    assert card.references, "a packaged method cites what it implements"

    parameters = {entry["name"] for entry in constructor_parameters(cls)}
    if card.strength_parameter is not None:
        assert card.strength_parameter in parameters
    # A parameter named in the description must exist: descriptions are
    # written against the code and must not outlive a renamed argument.
    mentioned = set(_IDENTIFIER.findall(card.details.en))
    assert mentioned <= parameters, f"details mention {sorted(mentioned - parameters)}; parameters are {sorted(parameters)}"


@pytest.mark.parametrize("metric_type", list(MetricType), ids=lambda m: m.name)
def test_every_metric_has_a_complete_card(metric_type: MetricType):
    cls = BUILTIN_METRICS[metric_type]
    card = cls.card
    assert isinstance(card, MetricCard), f"{cls.__name__} declares no MetricCard"
    _assert_bilingual(card)
    assert card.category in METRIC_CATEGORIES
    # The direction is declared by the class, never inferred from the card.
    assert cls.higher_is_better is not None or cls.components


@pytest.mark.parametrize("name", available_attacks())
def test_every_attack_has_a_complete_card(name: str):
    card = attack_card(name)
    assert isinstance(card, AttackCard)
    _assert_bilingual(card)
    if name in ATTACK_FACTORIES:
        return
    cls = ATTACK_CLASSES[name]
    assert cls.card is card, f"{cls.__name__} declares no AttackCard"
    assert cls.category.value in ATTACK_FAMILIES
    parameters = {item.name for item in fields(cls)}
    mentioned = set(_IDENTIFIER.findall(card.details.en))
    assert mentioned <= parameters, f"details mention {sorted(mentioned - parameters)}; fields are {sorted(parameters)}"


def test_abbreviations_are_distinct_within_each_kind():
    """Figures label components by short name, so two must never share one."""
    methods = [cls.card.abbreviation for cls in BUILTIN_METHOD_CLASSES.values()]
    metrics = [cls.card.abbreviation for cls in BUILTIN_METRICS.values()]
    assert len(set(methods)) == len(methods)
    assert len(set(metrics)) == len(metrics)


def test_catalogue_rows_carry_the_cards():
    from taf.experiments.registry import list_attacks, list_methods, list_metrics

    methods = {row["name"]: row for row in list_methods()}
    qim = methods["QIM_METHOD"]
    assert qim["title"]["en"] == "QIM / ST-DM" and qim["summary"]["pl"]
    assert qim["references"][0]["link"] == "https://doi.org/10.1109/18.923725"
    assert qim["family_label"] == METHOD_FAMILIES["quantization"].as_dict()
    # Rows arrive in presentation order, so clients group without a list of groups.
    order = list(METHOD_FAMILIES)
    positions = [order.index(row["family"]) for row in methods.values() if row["family"] in order]
    assert positions == sorted(positions)

    metrics = {row["name"]: row for row in list_metrics()}
    assert metrics["PESQ_METRIC"]["category_label"]["pl"] == "Jakość mowy"
    assert metrics["AI_MOSNET_METRIC"]["requires"] == ["tensorflow"]

    attacks = {spec.name: spec for spec in list_attacks()}
    assert attacks["mp3"].card["title"]["en"] == "MP3 round trip"
    assert attacks["awgn"].description == attacks["awgn"].card["summary"]["en"]


def test_text_falls_back_to_english():
    assert Text("gain").as_dict() == {"en": "gain", "pl": "gain"}
    assert Text("gain", "wzmocnienie").get("pl") == "wzmocnienie"
    card = MethodCard(title="Toy", summary="Hides bits.", family="toy", purpose="watermarking")
    assert card.title == Text("Toy") and card.abbreviation == "Toy"
    assert group_label(METHOD_FAMILIES, "toy").en == "Toy"


def test_a_component_without_a_card_is_described_by_its_docstring():
    class ToyMethod:
        """Hides bits in a toy carrier.

        Longer explanation of the mechanism.
        """

    card = fallback_card(ToyMethod, "TOY_METHOD", MethodCard)
    assert card.title.en == "TOY_METHOD"
    assert card.summary.en == "Hides bits in a toy carrier."
    assert card.details.en == "Longer explanation of the mechanism."
    assert card.family is None and card.problems()

    class Undocumented:
        pass

    assert fallback_card(Undocumented, "BARE").summary.en == ""


def test_missing_requirements_are_reported_with_their_extra():
    card = MethodCard(
        title="Toy",
        summary="Hides bits.",
        family="neural",
        purpose="watermarking",
        requires=("module_that_does_not_exist_anywhere",),
        extra="neural",
        references=(Reference("Doe", 2024),),
    )
    message = card.requirement_message("TOY")
    assert message == "TOY requires module_that_does_not_exist_anywhere (install the 'neural' extra)"


def test_generated_documentation_matches_the_cards():
    if not (ROOT / "docs" / "methods.md").is_file():
        pytest.skip("documentation sources are not available")
    from taf.catalogue_docs import stale

    outdated = [str(path.relative_to(ROOT)) for path in stale(ROOT)]
    assert not outdated, f"{outdated} are out of date; run: python -m taf.catalogue_docs"
