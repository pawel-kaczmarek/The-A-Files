"""Publication-ready reports of a run: LaTeX (booktabs) and Markdown.

A report has two parts. The *experimental setup* paragraph is generated from
the resolved configuration and the run manifest - dataset, methods, payloads,
repetitions, attacks, metrics, seed, statistical procedure and software
versions - so the methods section of a paper states exactly what was run.
The *results* are the tables the design produces, each value with its 95%
cluster-bootstrap interval where one exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from taf.experiments.analysis import BOOTSTRAP_RESAMPLES, CONFIDENCE
from taf.experiments.results import USABLE_BER_THRESHOLD
from taf.experiments.schema import ExperimentConfig


@dataclass
class Table:
    caption: str
    label: str
    columns: list[str]
    rows: list[list[str]] = field(default_factory=list)
    note: str | None = None


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def _number(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _interval(estimate: dict[str, Any] | None, digits: int = 3) -> str:
    if not estimate or estimate.get("estimate") is None:
        return "–"
    text = _number(estimate["estimate"], digits)
    if estimate.get("ci95_low") is not None:
        text += f" [{_number(estimate['ci95_low'], digits)}, {_number(estimate['ci95_high'], digits)}]"
    return text


def _p(value: float | None) -> str:
    if value is None:
        return "–"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def _list(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------
# Setup paragraph
# --------------------------------------------------------------------------


def setup_paragraph(config: ExperimentConfig, manifest: dict[str, Any], design: dict[str, Any] | None = None) -> str:
    inputs = manifest.get("inputs") or []
    total = sum(item.get("duration_seconds") or 0 for item in inputs)
    rates = sorted({item.get("sample_rate") for item in inputs if item.get("sample_rate")})
    methods = config.resolved_methods()
    sentences = [
        f"We evaluated {len(methods)} method configuration{'s' if len(methods) != 1 else ''} "
        f"({_list(methods)}) on {len(inputs)} audio file{'s' if len(inputs) != 1 else ''} "
        f"from the {config.dataset_id or 'local'} dataset "
        f"({total / 60:.1f} min in total, sampled at {_list([f'{rate / 1000:g} kHz' for rate in rates])})."
    ]
    if config.experiment_type.value != "detectability":
        if config.payload.kind != "random":
            payload_description = f"Exact {config.payload.kind} payloads of {len(config.payload.bits())} bits (identical content across repetitions)"
        elif config.payload_rates_bps:
            payload_description = f"Seeded random payloads at requested rates of {_list([str(rate) for rate in config.payload_rates_bps])} bits/s, with each file receiving floor(rate × duration) bits"
        else:
            payload_description = f"Random binary payloads of {_list([str(length) for length in config.payload_lengths])} bits"
        sentences.append(
            f"{payload_description} were embedded, with {config.repetitions} repetition(s) per file and payload setting. "
            "Exact goodput counts only fully recovered messages per second of cover audio; offered payload rate is not maximum capacity."
        )
    if config.subset_seed is not None:
        sentences.append(f"File selection used independent subset seed {config.subset_seed} before applying the file limit.")
    categories = sorted({item.get("category") for item in inputs if item.get("category")})
    if categories:
        sentences.append(f"Audio categories were {_list(categories)}; source PCM subtypes and channels are recorded in the manifest. Multichannel policy: {config.channel_policy}.")
    attacks = manifest.get("resolved_attacks") or []
    if config.attack_sweep is not None:
        sweep = config.attack_sweep
        sentences.append(
            f"Robustness was measured against {sweep.target} with {sweep.parameter} swept over "
            f"{_list([str(v) for v in sweep.values])}, next to an unattacked baseline."
        )
    elif attacks:
        sentences.append(
            f"Each stego signal was decoded without attack and after each of {len(attacks)} attack"
            f"{'s' if len(attacks) != 1 else ''} ({_list(attacks[:8])}{', …' if len(attacks) > 8 else ''})."
        )
    if config.method_sweep is not None:
        sweep = config.method_sweep
        sentences.append(
            f"The {sweep.parameter} parameter of {sweep.target} was varied over {_list([str(v) for v in sweep.values])}."
        )
    if config.metrics:
        sentences.append(
            f"Imperceptibility was quantified between cover and stego signals with {_list(config.metrics)}."
        )
    sentences.append(
        f"All random quantities were derived from seed {config.random_seed}; attack realisations were shared "
        "across methods within each file and repetition (common random numbers), and decoding used a fresh "
        "decoder instance."
    )
    sentences.append(
        f"Files were the unit of replication: {int(CONFIDENCE * 100)}% confidence intervals were obtained by a "
        f"cluster bootstrap over files ({BOOTSTRAP_RESAMPLES} resamples), and methods were compared on per-file "
        "means with the Friedman test followed by Holm-corrected Wilcoxon signed-rank tests (Wilcoxon alone for "
        "two methods). Failed extractions were counted separately and scored at chance level (BER = 0.5) in "
        f"rankings; a BER of at most {USABLE_BER_THRESHOLD} was considered usable."
    )
    packages = manifest.get("packages") or {}
    source = manifest.get("source") or {}
    versions = [f"Python {manifest.get('python', '?')}"] + [
        f"{name} {version}" for name, version in packages.items() if version and name in ("numpy", "scipy", "librosa")
    ]
    software = f"The A-Files {manifest.get('taf_version', '?')}"
    if source.get("commit"):
        software += f" (commit {source['commit'][:10]}{', with local changes' if source.get('dirty') else ''})"
    if manifest.get("ffmpeg"):
        versions.append(manifest["ffmpeg"].split(" Copyright")[0])
    sentences.append(f"Experiments were run with {software}, {_list(versions)}.")
    return " ".join(sentences)


# --------------------------------------------------------------------------
# Result tables
# --------------------------------------------------------------------------


def _method_table(entries: list[dict[str, Any]], caption: str, label: str) -> Table:
    metric_names = sorted({name for entry in entries for name in (entry.get("avg_metrics") or {})})[:4]
    table = Table(
        caption=caption,
        label=label,
        columns=["Method", "BER (95% CI)", "Completed", "Exact"] + metric_names,
    )
    for entry in entries:
        table.rows.append(
            [
                entry["method"],
                _interval(entry.get("ber_imputed")),
                _number(entry.get("completion_rate"), 2),
                _number(entry.get("perfect_extraction_rate"), 2),
            ]
            + [_number((entry.get("avg_metrics") or {}).get(name), 2) for name in metric_names]
        )
    table.note = "BER includes failed extractions at chance level; Exact = share of completed trials decoded without error."
    return table


def _comparison_table(name: str, comparison: dict[str, Any], label: str) -> Table | None:
    if not comparison.get("available"):
        return None
    omnibus = comparison.get("omnibus") or {}
    table = Table(
        caption=(
            f"Paired comparison ({name}): {omnibus.get('test', '').replace('_', ' ')} "
            f"p = {_p(omnibus.get('p_value'))}, {comparison.get('blocks')} files, "
            f"critical difference {comparison.get('critical_difference', 0):.2f} ranks."
        ),
        label=label,
        columns=["Pair", "Better", "Median difference", "r (rank-biserial)", "p (Holm)"],
    )
    for pair in comparison.get("pairwise") or []:
        table.rows.append(
            [
                f"{pair['a']} vs {pair['b']}",
                pair.get("better") or "–",
                _number(pair.get("median_difference")),
                _number(pair.get("rank_biserial"), 2),
                _p(pair.get("p_holm")) + (" *" if pair.get("significant") else ""),
            ]
        )
    table.note = "* significant at alpha = 0.05 after Holm correction."
    return table


def result_tables(summary: dict[str, Any]) -> list[Table]:
    tables: list[Table] = []
    if summary.get("curves"):
        values = summary["sweep"]["values"]
        table = Table(
            caption=f"BER (95% CI) of each method as {summary['sweep']['target']} {summary['sweep']['parameter']} varies.",
            label="tab:robustness-curve",
            columns=["Method"] + [str(value) for value in values] + ["Breakdown"],
        )
        for curve in summary["curves"]:
            breakdown = curve["breakdown"]
            table.rows.append(
                [curve["method"]]
                + [_interval(point["ber"], 3) for point in curve["points"]]
                + [
                    {"never": "none", "always": "first setting"}.get(breakdown["status"], _number(breakdown["value"], 2))
                ]
            )
        tables.append(table)
    if summary.get("points"):
        metrics = list(summary.get("ranked_metrics") or {})
        table = Table(
            caption=f"Trade-off of {summary['sweep']['target']} as {summary['sweep']['parameter']} varies.",
            label="tab:tradeoff",
            columns=[summary["sweep"]["parameter"], "BER", "BER under attack"] + metrics,
        )
        for point in summary["points"]:
            table.rows.append(
                [str(point["value"]), _interval(point["ber_baseline"]), _interval(point.get("ber_attacked"))]
                + [_interval(point["metrics"].get(name), 2) for name in metrics]
            )
        tables.append(table)
    if summary.get("capacity_by_method"):
        table = Table(
            caption="Embedding capacity per method, determined per file.",
            label="tab:capacity",
            columns=["Method", "Median bits", "Median bit/s", "Mean bit/s (95% CI)", "Over-capacity trials"],
        )
        for entry in summary["capacity_by_method"]:
            table.rows.append(
                [
                    entry["method"],
                    _number(entry.get("capacity_bits_median"), 0),
                    _number(entry.get("capacity_bps_median"), 1),
                    _interval(
                        {
                            "estimate": entry.get("capacity_bps_mean"),
                            "ci95_low": entry.get("capacity_bps_ci95_low"),
                            "ci95_high": entry.get("capacity_bps_ci95_high"),
                        },
                        1,
                    ),
                    _number(entry.get("over_capacity_rows")),
                ]
            )
        tables.append(table)
    if summary.get("detectability"):
        table = Table(
            caption="Detectability: held-out accuracy of the ensemble steganalyser (0.5 = chance).",
            label="tab:detectability",
            columns=["Method", "Payload", "Accuracy (95% CI)", "FPR", "FNR", "p (vs chance)"],
        )
        for entry in summary["detectability"]:
            if entry.get("status") != "ok":
                continue
            table.rows.append(
                [
                    entry["method"],
                    str(entry["payload_length"]),
                    _interval(
                        {
                            "estimate": entry["accuracy"],
                            "ci95_low": entry.get("accuracy_ci95_low"),
                            "ci95_high": entry.get("accuracy_ci95_high"),
                        }
                    ),
                    _number(entry.get("false_positive_rate")),
                    _number(entry.get("false_negative_rate")),
                    _p(entry.get("p_value")),
                ]
            )
        tables.append(table)
    if summary.get("by_method") and not summary.get("curves"):
        tables.append(_method_table(summary["by_method"], "Results per method (all conditions).", "tab:methods"))
    if summary.get("robustness_ranking") and not summary.get("curves"):
        tables.append(
            _method_table(summary["robustness_ranking"], "Robustness per method (attacked trials only).", "tab:robustness")
        )
    for name, comparison in (summary.get("statistics") or {}).items():
        if isinstance(comparison, dict) and "available" in comparison:
            table = _comparison_table(name.replace("_", " "), comparison, f"tab:stats-{name}")
            if table:
                tables.append(table)
    return tables


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------

_LATEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def _latex(text: str) -> str:
    return (
        "".join(_LATEX_SPECIAL.get(ch, ch) for ch in str(text))
        .replace("–", "--")
        .replace("≤", r"$\leq$")
        .replace("ρ", r"$\rho$")
        .replace("α", r"$\alpha$")
    )


def render_latex(title: str, setup: str, tables: list[Table], findings: Sequence[str] = ()) -> str:
    parts = [
        "% Generated by The A-Files. Requires \\usepackage{booktabs}.",
        f"\\section*{{{_latex(title)}}}",
        "\\subsection*{Experimental setup}",
        _latex(setup),
        "",
    ]
    if findings:
        parts += ["\\subsection*{Findings}", "\\begin{itemize}"]
        parts += [f"\\item {_latex(finding)}" for finding in findings]
        parts += ["\\end{itemize}", ""]
    for table in tables:
        spec = "l" + "r" * (len(table.columns) - 1)
        parts += [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            f"\\caption{{{_latex(table.caption)}}}",
            f"\\label{{{table.label}}}",
            f"\\begin{{tabular}}{{{spec}}}",
            "\\toprule",
            " & ".join(_latex(column) for column in table.columns) + " \\\\",
            "\\midrule",
        ]
        parts += [" & ".join(_latex(cell) for cell in row) + " \\\\" for row in table.rows]
        parts += ["\\bottomrule", "\\end{tabular}"]
        if table.note:
            parts.append(f"\\\\[2pt]\\footnotesize {_latex(table.note)}")
        parts += ["\\end{table}", ""]
    return "\n".join(parts)


def render_markdown(title: str, setup: str, tables: list[Table], findings: Sequence[str] = ()) -> str:
    parts = [f"# {title}", "", "## Experimental setup", "", setup, ""]
    if findings:
        parts += ["## Findings", ""] + [f"- {finding}" for finding in findings] + [""]
    for table in tables:
        parts += [f"**{table.caption}**", ""]
        parts.append("| " + " | ".join(table.columns) + " |")
        parts.append("| " + " | ".join(["---"] + ["---:"] * (len(table.columns) - 1)) + " |")
        parts += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in table.rows]
        if table.note:
            parts += ["", f"_{table.note}_"]
        parts.append("")
    return "\n".join(parts)


def build_report(
    title: str,
    config: ExperimentConfig,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    format: str = "markdown",
) -> str:
    setup = setup_paragraph(config, manifest)
    tables = result_tables(summary)
    # Each finding is a statement computed from the results (see
    # taf.experiments.scenarios.evaluation), not an interpretation.
    findings = [fact["text"] for fact in (summary.get("evaluation") or {}).get("facts", [])]
    if format == "latex":
        return render_latex(title, setup, tables, findings)
    return render_markdown(title, setup, tables, findings)


__all__ = ["Table", "build_report", "render_latex", "render_markdown", "result_tables", "setup_paragraph"]
