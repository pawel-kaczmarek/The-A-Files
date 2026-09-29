"use client";

import Link from "next/link";
import { useState } from "react";
import { Plus, X } from "lucide-react";

import { Chip, Spec } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { groupsInOrder, localized } from "@/lib/catalogue";
import { useI18n } from "@/lib/i18n";
import type { AttackInfo, AttackPresets, CatalogDataset, ParameterSweep } from "@/lib/types";
import { cn } from "@/lib/utils";

import { formatSpec, parseValues } from "./draft";

const SEVERITIES = ["mild", "moderate", "strong", "extreme"];

function AttackRow({ attack, onAdd }: { attack: AttackInfo; onAdd: (spec: string) => void }) {
  const { t, locale } = useI18n();
  const [mode, setMode] = useState(attack.has_severity ? "moderate" : "custom");
  const [parameters, setParameters] = useState<Record<string, string>>({});
  const editable = attack.parameters.filter((parameter) => !["seed", "codec"].includes(parameter.name));

  function add() {
    if (mode !== "custom") {
      onAdd(`${attack.name}@${mode}`);
      return;
    }
    const values = Object.fromEntries(
      Object.entries(parameters)
        .filter(([, value]) => value !== "")
        .map(([key, value]) => [key, /^-?\d+(\.\d+)?$/.test(value) ? Number(value) : value === "true" ? true : value === "false" ? false : value])
    );
    onAdd(formatSpec(attack.name, values));
  }

  return (
    <div className="px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-sm">
            <span className="font-mono">{attack.name}</span>
            {attack.stochastic && <Chip>{t("catalogue.stochastic")}</Chip>}
            {attack.changes_length_or_rate && <Chip>{t("catalogue.changesLength")}</Chip>}
          </div>
          <div className="text-xs text-muted-foreground">{localized(attack.summary, locale) || attack.description}</div>
        </div>
        <Select value={mode} onChange={(event) => setMode(event.target.value)} className="h-8 w-36 text-xs">
          {attack.has_severity &&
            SEVERITIES.map((severity) => (
              <option key={severity} value={severity}>
                {t("editor.conditions.severity")}: {severity}
              </option>
            ))}
          <option value="custom">{t("editor.conditions.customParameters")}</option>
        </Select>
        <Button type="button" size="sm" variant="outline" onClick={add}>
          <Plus className="h-3.5 w-3.5" /> {t("editor.conditions.add")}
        </Button>
      </div>
      {mode === "custom" && editable.length > 0 && (
        <div className="mt-2 grid gap-2 sm:grid-cols-3 lg:grid-cols-4">
          {editable.map((parameter) => (
            <div key={parameter.name} className="space-y-0.5">
              <Label className="font-mono text-[11px]">{parameter.name}</Label>
              <Input
                className="h-7 font-mono text-xs"
                placeholder={String(parameter.default)}
                value={parameters[parameter.name] ?? ""}
                onChange={(event) => setParameters({ ...parameters, [parameter.name]: event.target.value })}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function AttackPicker({
  attacks,
  presets,
  value,
  preset,
  onChange,
}: {
  attacks: AttackInfo[];
  presets?: AttackPresets;
  value: string[];
  preset: string | null | undefined;
  onChange: (attacks: string[], preset: string | null) => void;
}) {
  const { t, locale } = useI18n();
  const [mode, setMode] = useState<"none" | "battery" | "suite">(preset ? "suite" : value.length ? "battery" : "none");
  const add = (spec: string) => !value.includes(spec) && onChange([...value, spec], preset ?? null);

  const modes = [
    { id: "none" as const, title: t("editor.conditions.none"), hint: t("editor.conditions.noneHint") },
    { id: "battery" as const, title: t("editor.conditions.battery"), hint: t("editor.conditions.batteryHint") },
    { id: "suite" as const, title: t("editor.conditions.suite"), hint: t("editor.conditions.suiteHint") },
  ];

  return (
    <div className="space-y-5">
      <div className="grid gap-3 md:grid-cols-3">
        {modes.map((entry) => (
          <button
            key={entry.id}
            type="button"
            onClick={() => {
              setMode(entry.id);
              if (entry.id === "none") onChange([], null);
              if (entry.id === "battery") onChange(value, null);
              if (entry.id === "suite") onChange([], preset ?? "quick");
            }}
            className={cn("rounded-md border p-3 text-left", mode === entry.id ? "border-primary bg-accent/50" : "hover:bg-accent/30")}
          >
            <div className="text-sm font-medium">{entry.title}</div>
            <div className="mt-0.5 text-xs text-muted-foreground">{entry.hint}</div>
          </button>
        ))}
      </div>

      {mode === "suite" && (
        <div className="space-y-2">
          <Select value={preset ?? "quick"} onChange={(event) => onChange([], event.target.value)} className="w-60">
            {Object.entries(presets?.suites ?? { quick: [], standard: [], full: [] }).map(([name, specs]) => (
              <option key={name} value={name}>
                {name} ({specs.length})
              </option>
            ))}
          </Select>
          <div className="flex flex-wrap gap-1.5">
            {(presets?.suites[preset ?? "quick"] ?? []).map((spec) => (
              <Spec key={spec}>{spec}</Spec>
            ))}
          </div>
        </div>
      )}

      {mode === "battery" && (
        <>
          <div>
            <div className="mb-1.5 text-xs font-medium">{t("editor.conditions.selectedAttacks")}</div>
            {value.length ? (
              <div className="flex flex-wrap gap-1.5">
                {value.map((spec) => (
                  <span key={spec} className="inline-flex items-center gap-1 rounded bg-muted px-1.5 py-0.5">
                    <span className="spec">{spec}</span>
                    <button type="button" onClick={() => onChange(value.filter((entry) => entry !== spec), null)} aria-label={t("common.delete")}>
                      <X className="h-3 w-3 text-muted-foreground" />
                    </button>
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">{t("common.none")}</p>
            )}
            <p className="mt-1.5 text-[11px] text-muted-foreground">{t("editor.conditions.baselineNote")}</p>
          </div>
          {groupsInOrder(attacks.filter((attack) => attack.name !== "codec"), (attack) => attack.family, (attack) => attack.family_label).map((family) => (
            <div key={family.key}>
              <div className="eyebrow mb-1.5">{localized(family.label, locale)}</div>
              <div className="divide-y rounded-md border">
                {family.items
                  .map((attack) => (
                    <AttackRow key={attack.name} attack={attack} onAdd={add} />
                  ))}
              </div>
            </div>
          ))}
          <div>
            <div className="eyebrow mb-1.5">{t("editor.conditions.pipelines")}</div>
            <div className="divide-y rounded-md border">
              {Object.entries(presets?.pipelines ?? {}).map(([name, stages]) => (
                <div key={name} className="flex flex-wrap items-center gap-3 px-3 py-2.5">
                  <div className="min-w-0 flex-1">
                    <div className="font-mono text-sm">{name}</div>
                    <div className="text-xs text-muted-foreground">{stages.join(" → ")}</div>
                  </div>
                  <Button type="button" size="sm" variant="outline" onClick={() => add(`pipeline:name=${name}`)}>
                    <Plus className="h-3.5 w-3.5" /> {t("editor.conditions.add")}
                  </Button>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export function AttackSweepEditor({
  attacks,
  presets,
  value,
  onChange,
}: {
  attacks: AttackInfo[];
  presets?: AttackPresets;
  value: ParameterSweep | null | undefined;
  onChange: (sweep: ParameterSweep) => void;
}) {
  const { t, locale } = useI18n();
  const current = value ?? { target: "awgn", parameter: "snr_db", values: [] };
  const attack = attacks.find((entry) => entry.name === current.target);
  const [text, setText] = useState(current.values.join(", "));
  const ladder = presets?.sweeps[current.target];

  function apply(sweep: ParameterSweep) {
    setText(sweep.values.join(", "));
    onChange(sweep);
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">{t("editor.conditions.sweepHint")}</p>
      <div className="grid gap-4 md:grid-cols-3">
        <div className="space-y-1">
          <Label>{t("editor.conditions.attack")}</Label>
          <Select
            value={current.target}
            onChange={(event) => {
              const name = event.target.value;
              const preset = presets?.sweeps[name];
              const fallback = attacks.find((entry) => entry.name === name)?.parameters.find((p) => typeof p.default === "number")?.name ?? "";
              apply({ target: name, parameter: preset?.parameter ?? fallback, values: preset?.values ?? [] });
            }}
          >
            {groupsInOrder(attacks.filter((entry) => entry.name !== "codec"), (entry) => entry.family, (entry) => entry.family_label).map((family) => (
              <optgroup key={family.key} label={localized(family.label, locale)}>
                {family.items
                  .map((entry) => (
                    <option key={entry.name} value={entry.name}>
                      {entry.name}
                    </option>
                  ))}
              </optgroup>
            ))}
          </Select>
        </div>
        <div className="space-y-1">
          <Label>{t("editor.conditions.parameter")}</Label>
          <Select value={current.parameter} onChange={(event) => onChange({ ...current, parameter: event.target.value })}>
            {(attack?.parameters ?? [])
              .filter((parameter) => !["seed", "codec"].includes(parameter.name))
              .map((parameter) => (
                <option key={parameter.name} value={parameter.name}>
                  {parameter.name} — {t("common.default")} {String(parameter.default)}
                </option>
              ))}
          </Select>
        </div>
        <div className="space-y-1">
          <Label>
            {t("editor.conditions.values")}
            {ladder && ladder.parameter === current.parameter && <span className="ml-1 text-muted-foreground">({ladder.unit})</span>}
          </Label>
          <Input
            className="font-mono text-xs"
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              onChange({ ...current, values: parseValues(event.target.value) });
            }}
          />
          <div className="flex items-center justify-between">
            <p className="text-[11px] text-muted-foreground">{t("editor.conditions.valuesHint")}</p>
            {ladder && (
              <button type="button" className="text-[11px] text-primary underline" onClick={() => apply({ ...current, parameter: ladder.parameter, values: ladder.values })}>
                {t("editor.conditions.usePreset")}
              </button>
            )}
          </div>
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {current.values.map((entry) => (
          <Spec key={String(entry)}>{formatSpec(current.target, { [current.parameter]: entry })}</Spec>
        ))}
      </div>
    </div>
  );
}

export function DatasetPicker({
  datasets,
  value,
  fileLimit,
  onChange,
}: {
  datasets: CatalogDataset[];
  value: string | null | undefined;
  fileLimit: number | null | undefined;
  onChange: (datasetId: string, fileLimit: number | null) => void;
}) {
  const { t, tOr } = useI18n();
  const selected = datasets.find((dataset) => dataset.id === value);
  const library = datasets.filter((dataset) => dataset.kind !== "packaged");
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="space-y-1">
          <Label>{t("editor.data.dataset")}</Label>
          <Select value={value ?? ""} onChange={(event) => onChange(event.target.value, fileLimit ?? null)}>
            <optgroup label={t("datasetKinds.packaged")}>
              {datasets
                .filter((dataset) => dataset.kind === "packaged")
                .map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.label} ({dataset.file_count} {t("common.files")})
                  </option>
                ))}
            </optgroup>
            {library.length > 0 && (
              <optgroup label={t("datasets.library")}>
                {library.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.label} ({dataset.file_count} {t("common.files")})
                  </option>
                ))}
              </optgroup>
            )}
          </Select>
        </div>
        <div className="space-y-1">
          <Label>{t("editor.data.fileLimit")}</Label>
          <Input
            type="number"
            min={1}
            placeholder={selected ? String(selected.file_count) : ""}
            value={fileLimit ?? ""}
            onChange={(event) => onChange(value ?? "", event.target.value ? Number(event.target.value) : null)}
          />
        </div>
      </div>
      <p className="text-xs text-muted-foreground">{t("editor.data.fileLimitHint")}</p>
      {selected && (
        <div className="flex flex-wrap gap-1.5">
          <Chip>{tOr(`datasetKinds.${selected.kind}`, selected.kind)}</Chip>
          {selected.domain && <Chip>{tOr(`domains.${selected.domain}`, selected.domain)}</Chip>}
          {selected.sample_rate && <Chip>{selected.sample_rate / 1000} kHz</Chip>}
          {selected.total_duration_seconds && <Chip>{(selected.total_duration_seconds / 60).toFixed(1)} min</Chip>}
        </div>
      )}
      <p className="text-xs text-muted-foreground">
        {!library.length && <>{t("editor.data.noLibrary")} </>}
        <Link href="/datasets" className="text-primary underline underline-offset-2">
          {t("editor.data.manage")}
        </Link>
      </p>
    </div>
  );
}
