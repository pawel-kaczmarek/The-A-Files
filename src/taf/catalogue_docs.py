"""Catalogue pages of the documentation, generated from the component cards.

The method, attack and metric pages of the documentation site and the short
catalogue in the README are built from the same cards (``taf.models.card``)
that the API serves to the research UI, so the three never describe a
component differently::

    python -m taf.catalogue_docs            # rewrite the generated regions
    python -m taf.catalogue_docs --check    # exit 1 when they are out of date

Only regions delimited by ``<!-- catalogue:NAME -->`` and
``<!-- /catalogue:NAME -->`` are rewritten; the prose around them is written
by hand. Only packaged components are documented: an installed plugin never
changes the generated pages. The documentation is English; the UI also shows
the Polish texts of the cards.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

REPOSITORY = "https://github.com/pawel-kaczmarek/The-A-Files"
BRANCH = "master"
SITE = "https://pawel-kaczmarek.github.io/The-A-Files"

#: Sampling rate at which severity levels and sweep ladders are resolved.
DOCS_SAMPLE_RATE = 16000

_REGION = re.compile(r"(<!-- catalogue:(?P<name>[\w-]+) -->\n)(?P<body>.*?)(<!-- /catalogue:(?P=name) -->)", re.S)


# --------------------------------------------------------------------------- helpers


def _anchor(name: str) -> str:
    return name.lower().replace("_", "-")


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _value(value: Any) -> str:
    return f"`{value!r}`"


def _code_names(text: str, names: Iterable[str]) -> str:
    """Set parameter names in code font where the text mentions them.

    Only compound names (``snr_db``) are marked: single words such as
    ``rate`` or ``order`` are also ordinary English.
    """
    for name in sorted(set(names), key=len, reverse=True):
        if "_" in name:
            text = re.sub(rf"(?<![`\w]){re.escape(name)}(?![`\w])", f"`{name}`", text)
    return text


def _source_url(cls: Any) -> str | None:
    try:
        path = Path(inspect.getsourcefile(cls) or "").resolve()
    except TypeError:
        return None
    package = Path(__file__).resolve().parent
    try:
        relative = path.relative_to(package.parent)
    except ValueError:
        return None
    return f"{REPOSITORY}/blob/{BRANCH}/src/{relative.as_posix()}"


def _reference_index(references_page: Path) -> dict[str, int]:
    """DOI, arXiv identifier or URL -> number in the bibliography page."""
    index: dict[str, int] = {}
    if not references_page.is_file():
        return index
    for match in re.finditer(r"\*\*\[(\d+)\]\*\* (.*)", references_page.read_text(encoding="utf-8")):
        number, entry = int(match.group(1)), match.group(2)
        for doi in re.findall(r"doi\.org/([^)\s]+)\)", entry):
            index.setdefault(doi.lower(), number)
        for arxiv in re.findall(r"arxiv\.org/abs/([\d.]+)", entry, re.I):
            index.setdefault(f"10.48550/arxiv.{arxiv}".lower(), number)
        for url in re.findall(r"\]\((https?://[^)]+)\)", entry):
            index.setdefault(url.lower(), number)
    return index


def _citation(reference: dict[str, Any], index: dict[str, int], short: bool = False) -> str:
    number = index.get((reference.get("doi") or "").lower()) or index.get((reference.get("url") or "").lower())
    label = f"{reference['citation']} ({reference['year']})" if reference.get("year") else reference["citation"]
    if number is not None:
        return f"[{number}](references.md#ref-{number})" if short else f"{label} [[{number}](references.md#ref-{number})]"
    if reference.get("link"):
        return f"[{label}]({reference['link']})"
    return label


def _citations(references: list[dict[str, Any]], index: dict[str, int], short: bool = False) -> str:
    return ", ".join(_citation(reference, index, short) for reference in references) or "—"


def _requirements(row: dict[str, Any]) -> str | None:
    if not row["requires"]:
        return None
    modules = ", ".join(f"`{module}`" for module in row["requires"])
    extra = f' — install with `pip install "the-a-files[{row["extra"]}]"`' if row.get("extra") else ""
    return f"**Requires:** {modules}{extra}"


def _parameter_table(parameters: list[dict[str, Any]], roles: dict[str, str]) -> list[str]:
    if not parameters:
        return ["No tunable parameters."]
    lines = ["| Parameter | Default | Role |", "| --- | --- | --- |"]
    for parameter in parameters:
        default = "required" if parameter.get("required") else _value(parameter["default"])
        lines.append(f"| `{parameter['name']}` | {default} | {roles.get(parameter['name'], '')} |")
    return lines


def _grouped(rows: list[Any], key: Callable[[Any], str]) -> list[tuple[str, list[Any]]]:
    """Consecutive runs of rows sharing a group, in the catalogue's order."""
    groups: list[tuple[str, list[Any]]] = []
    for row in rows:
        if groups and groups[-1][0] == key(row):
            groups[-1][1].append(row)
        else:
            groups.append((key(row), [row]))
    return groups


# --------------------------------------------------------------------------- catalogue


class Catalogue:
    """Packaged components as the catalogue lists them."""

    def __init__(self, root: Path) -> None:
        from taf.experiments import registry
        from taf.plugins import metric_factories, method_sources

        self.root = root
        self.references = _reference_index(root / "docs" / "references.md")
        self.methods = [row for row in registry.list_methods() if row["packaged"]]
        self.metrics = [row for row in registry.list_metrics() if row["packaged"]]
        self.method_classes = method_sources()
        self.metric_classes = metric_factories()
        from taf.attacks.registry import ATTACK_CLASSES, ATTACK_FACTORIES

        self.attack_classes = ATTACK_CLASSES
        self.shortcuts = ATTACK_FACTORIES
        self.attacks = [spec for spec in registry.list_attacks() if spec.name in ATTACK_CLASSES or spec.name in ATTACK_FACTORIES]

    # ------------------------------------------------------------------ methods

    def methods_table(self) -> str:
        lines = [
            f"**{len(self.methods)} methods** are registered. Select a name for its mechanism, parameters and limits.",
            "",
            "| Method / registry identifier | What it does | Family · purpose | Reference |",
            "| --- | --- | --- | --- |",
        ]
        for row in self.methods:
            lines.append(
                f"| [**{_cell(row['title']['en'])}**](#{_anchor(row['name'])})<br>`{row['name']}` "
                f"| {_cell(row['summary']['en'])} "
                f"| {row['family_label']['en']} · {row['purpose'] or '—'} "
                f"| {_citations(row['references'], self.references, short=True)} |"
            )
        return "\n".join(lines)

    def methods_details(self) -> str:
        blocks: list[str] = []
        for family, rows in _grouped(self.methods, lambda row: row["family_label"]["en"]):
            blocks.append(f"### {family}")
            for row in rows:
                names = [parameter["name"] for parameter in row["parameters"]]
                roles = {name: "secret key" for name in names if any(p["name"] == name and p["is_key"] for p in row["parameters"])}
                if row["strength_parameter"]:
                    roles[row["strength_parameter"]] = "embedding strength"
                facts = [f"`{row['name']}`", row["family_label"]["en"], row["purpose"] or "—"]
                source = _source_url(self.method_classes[row["name"]])
                if source:
                    facts.append(f"[source code]({source})")
                notes = [f"**Reference:** {_citations(row['references'], self.references)}"]
                if _requirements(row):
                    notes.append(_requirements(row))
                if row["needs_long_input"]:
                    notes.append("**Needs long input:** short files produce failed trials.")
                blocks.append(
                    "\n".join(
                        [
                            f"#### {row['title']['en']} {{ #{_anchor(row['name'])} }}",
                            "",
                            " · ".join(facts),
                            "",
                            f"*{row['summary']['en']}*",
                            "",
                            _code_names(row["details"]["en"], names),
                            "",
                            *_parameter_table(row["parameters"], roles),
                            "",
                            " · ".join(notes),
                        ]
                    )
                )
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------ attacks

    def attacks_table(self) -> str:
        classes = sum(1 for spec in self.attacks if spec.name in self.attack_classes)
        shortcuts = ", ".join(f"`{name}`" for name in self.shortcuts)
        lines = [
            f"The registry contains **{classes} attack classes** and **{len(self.shortcuts)} codec shortcuts** "
            f"({shortcuts}); the shortcuts fix the `codec` parameter and are not separate algorithms.",
            "",
            "| Attack | Transformation | Parameters |",
            "| --- | --- | --- |",
        ]
        for spec in self.attacks:
            target = "codec" if spec.name in self.shortcuts else spec.name
            parameters = ", ".join(f"`{p.name}`" for p in spec.parameters if p.name != "seed") or "—"
            lines.append(f"| [`{spec.name}`](#{_anchor(target)}) | {_cell(spec.card['summary']['en'])} | {parameters} |")
        return "\n".join(lines)

    def attacks_details(self) -> str:
        from taf.attacks.base import Severity
        from taf.attacks.presets import severity_parameters

        blocks: list[str] = []
        for family, specs in _grouped(self.attacks, lambda spec: spec.family_label["en"]):
            specs = [spec for spec in specs if spec.name not in self.shortcuts]
            if not specs:
                continue
            blocks.append(f"### {family}")
            for spec in specs:
                names = [parameter.name for parameter in spec.parameters]
                facts = [f"`{spec.name}`", family]
                if spec.stochastic:
                    facts.append("stochastic (seeded)")
                if spec.changes_length_or_rate:
                    facts.append("changes length or sample rate")
                source = _source_url(self.attack_classes[spec.name])
                if source:
                    facts.append(f"[source code]({source})")
                parameters = [
                    {"name": parameter.name, "default": parameter.default, "required": False}
                    for parameter in spec.parameters
                ]
                roles = {"seed": "random seed; replaced per trial in experiments"}
                if spec.sweep:
                    roles[spec.sweep["parameter"]] = "swept in robustness curves"
                lines = [
                    f"#### {spec.card['title']['en']} {{ #{_anchor(spec.name)} }}",
                    "",
                    " · ".join(facts),
                    "",
                    f"*{spec.card['summary']['en']}*",
                    "",
                    _code_names(spec.card["details"]["en"], names),
                    "",
                    *_parameter_table(parameters, roles),
                ]
                if spec.name == "codec":
                    lines += ["", "Shortcuts: " + ", ".join(f"`{name}`" for name in self.shortcuts) + " select the codec."]
                if spec.has_severity:
                    levels = []
                    for severity in Severity:
                        values = severity_parameters(spec.name, severity, DOCS_SAMPLE_RATE)
                        levels.append(f"{severity.value} " + ", ".join(f"`{k}={v!r}`" for k, v in values.items()))
                    lines += ["", f"**Severity at {DOCS_SAMPLE_RATE // 1000} kHz:** " + " · ".join(levels)]
                if spec.sweep:
                    values = ", ".join(str(value) for value in spec.sweep["values"])
                    lines += ["", f"**Sweep ladder:** `{spec.sweep['parameter']}` = {values} ({spec.sweep['unit']})"]
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def attacks_severity(self) -> str:
        from taf.attacks.base import Severity
        from taf.attacks.presets import severity_parameters

        header = " | ".join(level.value.upper() for level in Severity)
        lines = [f"| Attack | {header} |", "| --- |" + " --- |" * len(Severity)]
        for spec in self.attacks:
            if not spec.has_severity:
                continue
            cells = []
            for severity in Severity:
                values = severity_parameters(spec.name, severity, DOCS_SAMPLE_RATE)
                cells.append(", ".join(f"{k}={v!r}" for k, v in values.items()))
            lines.append(f"| `{spec.name}` | " + " | ".join(_cell(cell) for cell in cells) + " |")
        return "\n".join(lines)

    # ------------------------------------------------------------------ metrics

    @staticmethod
    def _direction(row: dict[str, Any]) -> str:
        return {True: "↑ higher is better", False: "↓ lower is better"}.get(row["higher_is_better"], "not ranked")

    def metrics_table(self) -> str:
        lines = [
            f"**{len(self.metrics)} metrics** are registered. ↑ indicates higher is better; ↓ indicates lower is better.",
            "",
            "| Metric / registry identifier | Direction · scale | What it measures | Reference |",
            "| --- | --- | --- | --- |",
        ]
        for row in self.metrics:
            arrow = {True: "↑", False: "↓"}.get(row["higher_is_better"], "–")
            lines.append(
                f"| [**{_cell(row['abbreviation'])}**](#{_anchor(row['name'])})<br>`{row['name']}` "
                f"| {arrow} {_cell(row['scale'] or '')} "
                f"| {_cell(row['summary']['en'])} "
                f"| {_citations(row['references'], self.references, short=True)} |"
            )
        return "\n".join(lines)

    def metrics_details(self) -> str:
        blocks: list[str] = []
        for category, rows in _grouped(self.metrics, lambda row: row["category_label"]["en"]):
            blocks.append(f"### {category}")
            for row in rows:
                facts = [
                    f"`{row['name']}`",
                    self._direction(row),
                    row["scale"] or "—",
                    "intrusive (compares with the original)" if row["intrusive"] else "non-intrusive (rates the signal alone)",
                    f"{row['domain']} material",
                ]
                source = _source_url(self.metric_classes[row["name"]])
                if source:
                    facts.append(f"[source code]({source})")
                notes = [f"**Reference:** {_citations(row['references'], self.references)}"]
                if row["components"]:
                    notes.insert(0, "**Components:** " + ", ".join(f"`{c}`" for c in row["components"]) + " (reported separately)")
                if _requirements(row):
                    notes.append(_requirements(row))
                blocks.append(
                    "\n".join(
                        [
                            f"#### {row['title']['en']} ({row['abbreviation']}) {{ #{_anchor(row['name'])} }}",
                            "",
                            " · ".join(facts),
                            "",
                            f"*{row['summary']['en']}*",
                            "",
                            row["details"]["en"],
                            "",
                            " · ".join(notes),
                        ]
                    )
                )
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------ README

    def readme_methods(self) -> str:
        lines = [f"## Methods · {len(self.methods)}", ""]
        for family, rows in _grouped(self.methods, lambda row: row["family_label"]["en"]):
            lines.append(f"- **{family}:** " + ", ".join(sorted((row["abbreviation"] for row in rows), key=str.lower)) + ".")
        lines += ["", f"[Mechanisms, parameters, implementation limits and paper references]({SITE}/methods/)."]
        return "\n".join(lines)

    def readme_attacks(self) -> str:
        from taf.attacks.presets import PIPELINES

        classes = sum(1 for spec in self.attacks if spec.name in self.attack_classes)
        lines = [f"## Attacks · {classes}", ""]
        for family, specs in _grouped(self.attacks, lambda spec: spec.family_label["en"]):
            names = [f"`{spec.name}`" for spec in specs if spec.name not in self.shortcuts]
            shortcuts = [f"`{spec.name}`" for spec in specs if spec.name in self.shortcuts]
            text = ", ".join(names) + (f", with {', '.join(shortcuts)} shortcuts" if shortcuts else "")
            lines.append(f"- **{family}:** {text}.")
        lines.append("- **Pipelines:** " + ", ".join(f"`{name}`" for name in PIPELINES) + ".")
        lines += ["", f"[Individual descriptions, parameters, severity levels and scientific context]({SITE}/attacks/)."]
        return "\n".join(lines)

    def readme_metrics(self) -> str:
        lines = [f"## Metrics · {len(self.metrics)}", ""]
        for category, rows in _grouped(self.metrics, lambda row: row["category_label"]["en"]):
            lines.append(f"- **{category}:** " + ", ".join(sorted((row["abbreviation"] for row in rows), key=str.lower)) + ".")
        lines += ["", f"[Definitions, score directions and references]({SITE}/metrics/)."]
        return "\n".join(lines)


#: File -> regions it contains -> the catalogue method that renders each.
REGIONS: dict[str, dict[str, str]] = {
    "docs/methods.md": {"methods-table": "methods_table", "methods-details": "methods_details"},
    "docs/attacks.md": {
        "attacks-table": "attacks_table",
        "attacks-severity": "attacks_severity",
        "attacks-details": "attacks_details",
    },
    "docs/metrics.md": {"metrics-table": "metrics_table", "metrics-details": "metrics_details"},
    "README.md": {
        "readme-methods": "readme_methods",
        "readme-attacks": "readme_attacks",
        "readme-metrics": "readme_metrics",
    },
}


def render(root: Path) -> dict[Path, str]:
    """Every generated file with its regions rewritten, keyed by path."""
    catalogue = Catalogue(root)
    output: dict[Path, str] = {}
    for relative, regions in REGIONS.items():
        path = root / relative
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        found = {match.group("name") for match in _REGION.finditer(text)}
        missing = sorted(set(regions) - found)
        if missing:
            raise ValueError(f"{relative} lacks the generated region(s) {missing}")

        def replace(match: re.Match[str]) -> str:
            name = match.group("name")
            if name not in regions:
                return match.group(0)
            body = getattr(catalogue, regions[name])()
            return f"{match.group(1)}{body}\n{match.group(4)}"

        output[path] = _REGION.sub(replace, text)
    return output


def stale(root: Path) -> list[Path]:
    """Generated files whose content differs from the cards."""
    return [
        path
        for path, text in render(root).items()
        if path.read_text(encoding="utf-8").replace("\r\n", "\n") != text
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root (default: current directory)")
    parser.add_argument("--check", action="store_true", help="only report out-of-date files")
    arguments = parser.parse_args(argv)

    if arguments.check:
        outdated = stale(arguments.root)
        for path in outdated:
            print(f"out of date: {path.relative_to(arguments.root)}")
        if outdated:
            print("run: python -m taf.catalogue_docs")
        return 1 if outdated else 0

    for path, text in render(arguments.root).items():
        if path.read_text(encoding="utf-8").replace("\r\n", "\n") != text:
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"updated {path.relative_to(arguments.root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
