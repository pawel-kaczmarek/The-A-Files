"use client";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { ExperimentConfig } from "@/lib/types";

export function AudioOptions({ config, onChange }: { config: ExperimentConfig; onChange: (patch: Partial<ExperimentConfig>) => void }) {
  return <div className="mt-5 grid gap-4 sm:grid-cols-2">
    <label className="space-y-1 text-sm">Subset seed (optional)
      <Input type="number" min={0} value={config.subset_seed ?? ""} onChange={(e) => onChange({ subset_seed: e.target.value === "" ? null : Number(e.target.value) })} />
      <p className="text-xs text-muted-foreground">Seeded selection before the file limit; independent of the payload seed.</p>
    </label>
    <label className="space-y-1 text-sm">Audio category
      <Select value={config.audio_category ?? ""} onChange={(e) => onChange({ audio_category: e.target.value || null })}>
        <option value="">From dataset metadata</option>
        {["speech", "music", "environmental", "synthetic_speech", "synthetic_signal", "mixed"].map((v) => <option key={v}>{v}</option>)}
      </Select>
    </label>
    <label className="space-y-1 text-sm">Multichannel input
      <Select value={config.channel_policy ?? "mono"} onChange={(e) => onChange({ channel_policy: e.target.value as "mono" | "reject" })}>
        <option value="mono">Arithmetic mean downmix to mono</option><option value="reject">Require mono files</option>
      </Select>
      <p className="text-xs text-muted-foreground">Results retain source channels and record the evaluated mono signal.</p>
    </label>
    <label className="space-y-1 text-sm">Source description (optional)
      <Input value={config.audio_source ?? ""} onChange={(e) => onChange({ audio_source: e.target.value || null })} />
    </label>
    <div className="space-y-1 sm:col-span-2">
      <Label htmlFor="selected-audio">Exact subset: relative file paths, one per line (optional)</Label>
      <Textarea id="selected-audio" value={(config.selected_files ?? []).join("\n")} onChange={(e) => onChange({ selected_files: e.target.value.split("\n") })} onBlur={() => onChange({ selected_files: (config.selected_files ?? []).map((v) => v.trim()).filter(Boolean) })} />
    </div>
  </div>;
}
