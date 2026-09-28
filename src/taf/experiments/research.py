"""Filtered, descriptive comparisons with explicit channel/reference conditions."""

from taf.experiments.analysis import file_of
from taf.experiments.results import distribution_stats, group_by


FILTER_FIELDS = ("method", "audio_category", "audio_source", "payload_kind", "payload_length", "requested_payload_rate_bps", "sample_rate", "source_channels", "bit_depth")
MEASURES = ("ber", "payload_rate_bps", "payload_bits_per_sample", "exact_goodput_bps", "encode_rtf", "decode_rtf")


def research_comparison(rows, *, filters=None, attack="", reference="embedding", group="method", timing_reliable=True):
    if group not in FILTER_FIELDS:
        raise ValueError("Unsupported grouping field.")
    if reference not in ("embedding", "attack"):
        raise ValueError("Reference must be embedding or attack.")
    filters = filters or {}
    if set(filters) - set(FILTER_FIELDS):
        raise ValueError("Unsupported filter field.")
    # A single attack condition, including the baseline, prevents pseudo-
    # replication of cover–stego metrics across attack variants.
    selected = [row for row in rows if (row.attack or "") == attack and all(
        str(getattr(row, name)) == str(value) for name, value in filters.items() if value is not None
    )]
    facets = {name: sorted({str(getattr(row, name)) for row in rows if getattr(row, name) is not None})
              for name in FILTER_FIELDS}
    facets["attack"] = sorted({row.attack or "" for row in rows})
    metric_attr = "metrics" if reference == "embedding" else "attack_metrics"
    names = sorted({name for row in selected for name in getattr(row, metric_attr)})
    groups = []
    for label, members in sorted(group_by(selected, lambda row: str(getattr(row, group) or "unknown")).items()):
        def stats(getter):
            pairs = [(getter(row), file_of(row)) for row in members if getter(row) is not None]
            return distribution_stats([p[0] for p in pairs], [p[1] for p in pairs])

        values = {name: stats(lambda row, n=name: getattr(row, n)) for name in MEASURES
                  if timing_reliable or name not in ("encode_rtf", "decode_rtf")}
        values.update({f"quality:{name}": stats(lambda row, n=name: getattr(row, metric_attr).get(n)) for name in names})
        groups.append({"label": label, "rows": len(members), "completed": sum(row.status == "ok" for row in members),
                       "exact": sum(row.decode_success for row in members), "measures": values})
    return {"reference": reference, "attack": attack or None, "group": group, "facets": facets,
            "rows": len(selected), "groups": groups, "timing_reliable": timing_reliable,
            "interpretation": "Descriptive row-weighted means; 95% CIs resample files. BER excludes failures; exact goodput includes failed deliveries as zero. Payload bps is offered load, not maximum capacity. Comparisons across mixed payloads or audio types may be confounded."}
