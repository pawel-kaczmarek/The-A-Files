"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { ExperimentConfig } from "@/lib/types";

export function PayloadEditor({ config, onChange }: { config: ExperimentConfig; onChange: (patch: Partial<ExperimentConfig>) => void }) {
  const kind = config.payload?.kind ?? "random";
  const value = config.payload?.value ?? "";
  const [rates, setRates] = useState((config.payload_rates_bps ?? []).join(", "));
  const [error, setError] = useState("");
  const size = kind === "text" ? new TextEncoder().encode(value).length * 8 : kind === "binary" ? value.replace(/\s/g, "").length * 4 : value.length;
  return <div className="space-y-3 rounded-lg border p-4">
    <Label htmlFor="payload-kind">Hidden message</Label>
    <Select id="payload-kind" value={kind} onChange={(event) => {
      const next = event.target.value as NonNullable<ExperimentConfig["payload"]>["kind"];
      onChange({ payload: { kind: next, value: next === "random" ? null : "" }, payload_rates_bps: [] });
      setRates(""); setError("");
    }}>
      <option value="random">Seeded random bits</option><option value="text">Text (UTF-8)</option>
      <option value="binary">Binary (hexadecimal)</option><option value="bits">Explicit bits</option>
    </Select>
    {kind === "random" ? <>
      <Label htmlFor="payload-rates">Optional embedding rates (bits/s, comma separated)</Label>
      <Input id="payload-rates" value={rates} placeholder="8, 16, 32" onChange={(event) => {
        setRates(event.target.value);
        const values = event.target.value.trim() ? event.target.value.split(",").map(Number) : [];
        const invalid = values.some((v) => !Number.isFinite(v) || v <= 0);
        setError(invalid ? "Enter positive finite rates." : "");
        onChange({ payload_rates_bps: invalid ? [0] : values });
      }} />
      <p className="text-xs text-muted-foreground">Leave rates empty to use fixed bit lengths. Rates replace the length list: each file receives floor(rate × duration) bits. The supported rate-derived payload range is 1–8192 bits.</p>
    </> : <>
      <Textarea aria-label="Payload content" value={value} onChange={(event) => onChange({ payload: { kind, value: event.target.value } })} />
      {kind === "binary" && <Input aria-label="Read binary payload file" type="file" onChange={async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;
        if (file.size > 1024) { setError("Maximum file size is 1024 bytes (8192 bits)."); return; }
        try {
          const bytes = new Uint8Array(await file.arrayBuffer());
          onChange({ payload: { kind: "binary", value: Array.from(bytes, (v) => v.toString(16).padStart(2, "0")).join("") } });
          setError("");
        } catch { setError("Could not read the payload file."); }
      }} />}
      <p className="text-xs text-muted-foreground">{size} bits ({size / 8} byte-equivalents). Exact content is reused for repetitions. UTF-8 bytes and binary bytes use most-significant bit first. No padding or error correction is added. The exported configuration contains this content.</p>
    </>}
    {error && <p role="alert" className="text-xs text-destructive">{error}</p>}
  </div>;
}
