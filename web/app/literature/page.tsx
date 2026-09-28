"use client";

import Link from "next/link";
import { useState } from "react";
import { DataTable } from "@/components/charts/base";
import { ErrorNotice, LoadingLine, PageHeader, Section } from "@/components/common";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { api } from "@/lib/api";
import { useAsync } from "@/lib/hooks";

export default function LiteraturePage() {
  const data = useAsync(api.literature);
  const [search, setSearch] = useState("");
  const [purpose, setPurpose] = useState("");
  const [year, setYear] = useState("");
  const [family, setFamily] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  if (data.error) return <ErrorNotice error={data.error} onRetry={data.reload} />;
  if (!data.data) return <LoadingLine />;
  const papers = data.data.filter((p) => (!purpose || p.purpose === purpose) && (!year || String(p.year) === year)
    && (!family || p.family === family) && JSON.stringify(p).toLowerCase().includes(search.toLowerCase()));
  const compare = data.data.filter((p) => selected.includes(p.id));
  return <>
    <PageHeader title="Literature evidence" subtitle="Primary-source experimental evidence. Paper-reported values are not measurements from The A-Files." />
    <p className="mb-5 text-sm text-muted-foreground">These papers extend the <Link href="/methods" className="text-primary underline">implemented method catalog</Link>. They are reference studies; their models are not installed by adding a citation. Protocols, payload definitions and datasets differ, so values are not ranked across papers.</p>
    <div className="mb-5 flex flex-wrap gap-3">
      <Input className="max-w-md" aria-label="Search literature" placeholder="Search authors, methods, metrics, attacks or datasets" value={search} onChange={(e) => setSearch(e.target.value)} />
      <Select className="w-auto" aria-label="Purpose" value={purpose} onChange={(e) => setPurpose(e.target.value)}><option value="">All purposes</option><option>steganography</option><option>watermarking</option></Select>
      <Select className="w-auto" aria-label="Publication year" value={year} onChange={(e) => setYear(e.target.value)}><option value="">All years</option>{[...new Set(data.data.map((p) => p.year))].sort().map((y) => <option key={y}>{y}</option>)}</Select>
      <Select className="w-auto" aria-label="Method family" value={family} onChange={(e) => setFamily(e.target.value)}><option value="">All families</option>{[...new Set(data.data.map((p) => p.family))].map((f) => <option key={f}>{f}</option>)}</Select>
    </div>
    {compare.length > 0 && <Section title="Selected study protocols" className="mb-6"><DataTable table={{ columns: ["Study", "Year", "Purpose", "Datasets", "Payload / capacity", "Measures"], rows: compare.map((p) => [p.title, p.year, p.purpose, p.datasets.join(", "), p.payload, p.metrics.join(", ")]) }} /></Section>}
    <div className="space-y-6">
      {papers.map((p) => <Section key={p.id} title={p.title}>
        <div className="space-y-3 text-sm">
          <label className="flex items-center gap-2"><input type="checkbox" checked={selected.includes(p.id)} onChange={(e) => setSelected((old) => e.target.checked ? [...old, p.id] : old.filter((id) => id !== p.id))} />Compare protocol</label>
          <p>{p.authors.join(", ")} · {p.venue} · {p.family} · {p.purpose}</p>
          <p><a href={p.url} className="text-primary underline">Publication</a> · <a href={p.source_url} className="text-primary underline">Evidence source</a>{p.doi && <> · DOI: <a href={`https://doi.org/${p.doi}`} className="text-primary underline">{p.doi}</a></>}{p.arxiv && ` · arXiv:${p.arxiv}`}</p>
          <p><strong>Datasets:</strong> {p.datasets.join(", ")}</p>
          <p><strong>Reported measures:</strong> {p.metrics.join(", ")}</p>
          <p><strong>Attacks:</strong> {p.attacks.join(", ")}</p>
          <p><strong>Payload:</strong> {p.payload}</p>
          <DataTable table={{ columns: ["Paper-reported result", "Value", "Condition", "Location"], rows: p.results.map((r) => [r.metric, `${r.value} ${r.unit}`, r.condition, r.locator]) }} />
          <p className="text-muted-foreground">{p.limitations}</p>
          <p className="text-xs text-muted-foreground">Verified {p.verified_on}. Unlisted results have not been transcribed; no values are inferred.</p>
        </div>
      </Section>)}
    </div>
  </>;
}
