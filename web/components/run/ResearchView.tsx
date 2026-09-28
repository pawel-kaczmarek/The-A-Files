"use client";

import { useState } from "react";
import { DataTable } from "@/components/charts/base";
import { IntervalChart } from "@/components/charts/IntervalChart";
import { ErrorNotice, LoadingLine, Section } from "@/components/common";
import { Select } from "@/components/ui/select";
import { api } from "@/lib/api";
import { useAsync } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import { researchLabels } from "@/lib/research";

const filters = ["method", "audio_category", "audio_source", "payload_kind", "payload_length", "requested_payload_rate_bps", "sample_rate", "source_channels", "bit_depth"];

export function ResearchView({ runId }: { runId: string }) {
  const { number } = useI18n();
  const [selection, setSelection] = useState<Record<string, string>>({ attack: "", reference: "embedding", group: "method" });
  const [measure, setMeasure] = useState("exact_goodput_bps");
  const result = useAsync(() => api.research(runId, selection), [runId, JSON.stringify(selection)]);
  const update = (key: string, value: string) => setSelection((old) => {
    const next = { ...old, [key]: value };
    if (filters.includes(key) && !value) delete next[key];
    return next;
  });
  const data = result.data;
  const measures = [...new Set(data?.groups.flatMap((g) => Object.keys(g.measures)) ?? [])];
  const chosen = measures.includes(measure) ? measure : measures[0] ?? "";
  const label = (name: string) => researchLabels[name] ?? name.replace(/^quality:/, "");
  return <Section title="Research comparisons" className="mt-8">
    <p className="mb-4 text-sm text-muted-foreground">Select one channel condition and signal reference. Filter by audio characteristics and exact payload size before comparing methods.</p>
    <div className="mb-4 grid gap-3 sm:grid-cols-3">
      <label className="space-y-1 text-xs">Channel condition
        <Select value={selection.attack} onChange={(e) => update("attack", e.target.value)}>
          <option value="">No attack (baseline)</option>
          {data?.facets.attack.filter(Boolean).map((v) => <option key={v}>{v}</option>)}
        </Select>
      </label>
      <label className="space-y-1 text-xs">Quality reference
        <Select value={selection.reference} onChange={(e) => update("reference", e.target.value)}>
          <option value="embedding">Cover ↔ stego: embedding distortion</option>
          <option value="attack">Stego ↔ attacked: attack damage</option>
        </Select>
      </label>
      <label className="space-y-1 text-xs">Group by
        <Select value={selection.group} onChange={(e) => update("group", e.target.value)}>
          {filters.map((name) => <option key={name} value={name}>{label(name)}</option>)}
        </Select>
      </label>
      {filters.map((name) => <label key={name} className="space-y-1 text-xs">{label(name)}
        <Select value={selection[name] ?? ""} onChange={(e) => update(name, e.target.value)}>
          <option value="">All</option>{data?.facets[name]?.map((v) => <option key={v}>{v}</option>)}
        </Select>
      </label>)}
    </div>
    {result.error && <ErrorNotice error={result.error} onRetry={result.reload} />}
    {result.loading ? <LoadingLine /> : data && <>
      <p className="my-3 text-xs text-muted-foreground">{data.rows} trials. {data.interpretation}</p>
      {!data.timing_reliable && <p className="my-3 text-xs text-muted-foreground">Runtime comparisons are omitted because this run used concurrent workers. Individual row timings remain in the CSV.</p>}
      {data.groups.length ? <>
        <Select aria-label="Comparison measure" className="mb-4 max-w-xl" value={chosen} onChange={(e) => setMeasure(e.target.value)}>
          {measures.map((name) => <option key={name} value={name}>{label(name)}</option>)}
        </Select>
        <IntervalChart axisTitle={`${label(chosen)} · 95% file-bootstrap CI`} rows={data.groups.map((g) => ({
          label: g.label, estimate: g.measures[chosen]?.mean ?? null,
          lo: g.measures[chosen]?.ci95_low, hi: g.measures[chosen]?.ci95_high,
          note: `${g.completed}/${g.rows} completed; ${g.exact} exact`,
        }))} />
        <DataTable table={{ columns: ["Group", "Trials", "Completed", "Exact", "Mean", "95% CI", "Finite scores", "Files"], rows: data.groups.map((g) => {
          const m = g.measures[chosen];
          return [g.label, g.rows, g.completed, g.exact, number(m?.mean, 4), `${number(m?.ci95_low, 4)} – ${number(m?.ci95_high, 4)}`, m?.count ?? 0, m?.clusters ?? 0];
        }) }} />
      </> : <p className="text-sm">No trials match these filters.</p>}
    </>}
  </Section>;
}
