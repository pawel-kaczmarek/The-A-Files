"use client";

import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight, Plus, X } from "lucide-react";

import { Chip, Spec } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { useI18n } from "@/lib/i18n";
import type { MethodInfo, MetricInfo, ParameterSweep } from "@/lib/types";
import { cn } from "@/lib/utils";

import { formatSpec, parseSpec, parseValues, strengthLadder } from "./draft";

const FAMILY_ORDER = ["lsb", "transform", "spread_spectrum", "echo", "phase", "quantization", "statistical", "adaptive", "learned", "neural"];

function familyOf(method: MethodInfo): string {
  return method.family ?? "plugin";
}

function ParameterFields({
  method,
  parameters,
  onChange,
}: {
  method: MethodInfo;
  parameters: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
}) {
  const { t } = useI18n();
  if (!method.parameters.length) return <p className="text-xs text-muted-foreground">{t("editor.methods.noParameters")}</p>;
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {method.parameters.map((parameter) => (
        <div key={parameter.name} className="space-y-1">
          <Label className="flex items-center gap-1.5 text-xs">
            <span className="font-mono">{parameter.name}</span>
            {parameter.name === method.strength_parameter && <Chip>{t("catalogue.strength")}</Chip>}
            {parameter.is_key && <Chip>{t("catalogue.key")}</Chip>}
          </Label>
          <Input
            className="h-8 font-mono text-xs"
            placeholder={`${t("common.default")}: ${String(parameter.default)}`}
            value={parameters[parameter.name] === undefined ? "" : String(parameters[parameter.name])}
            onChange={(event) => {
              const raw = event.target.value;
              const next = { ...parameters };
              if (raw === "") delete next[parameter.name];
              else next[parameter.name] = /^-?\d+(\.\d+)?(e-?\d+)?$/i.test(raw) ? Number(raw) : raw === "true" ? true : raw === "false" ? false : raw;
              onChange(next);
            }}
          />
        </div>
      ))}
    </div>
  );
}

export function MethodPicker({
  methods,
  value,
  onChange,
}: {
  methods: MethodInfo[];
  value: string[];
  onChange: (specs: string[]) => void;
}) {
  const { t } = useI18n();
  const [search, setSearch] = useState("");
  const [family, setFamily] = useState("");
  const [purpose, setPurpose] = useState("");
  const [open, setOpen] = useState<string | null>(null);

  const byName = useMemo(() => new Map(methods.map((method) => [method.name, method])), [methods]);
  const settings = (name: string) => value.map((spec, index) => ({ spec, index })).filter(({ spec }) => parseSpec(spec).name === name);

  const filtered = methods.filter(
    (method) =>
      (!family || familyOf(method) === family) &&
      (!purpose || method.purpose === purpose) &&
      (!search || `${method.name} ${method.description} ${method.reference ?? ""}`.toLowerCase().includes(search.toLowerCase()))
  );
  const families = [...FAMILY_ORDER, "plugin"].filter((entry) => filtered.some((method) => familyOf(method) === entry));

  function toggle(name: string) {
    if (settings(name).length) onChange(value.filter((spec) => parseSpec(spec).name !== name));
    else onChange([...value, name]);
  }

  function updateSetting(index: number, name: string, parameters: Record<string, unknown>) {
    const next = [...value];
    next[index] = formatSpec(name, parameters);
    onChange(next);
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Input placeholder={t("common.search")} value={search} onChange={(event) => setSearch(event.target.value)} className="w-56" />
        <Select value={family} onChange={(event) => setFamily(event.target.value)} className="w-52">
          <option value="">{t("editor.methods.filterFamily")}: {t("common.all")}</option>
          {[...FAMILY_ORDER, "plugin"].map((entry) => (
            <option key={entry} value={entry}>
              {t(`families.${entry}`)}
            </option>
          ))}
        </Select>
        <Select value={purpose} onChange={(event) => setPurpose(event.target.value)} className="w-48">
          <option value="">{t("editor.methods.filterPurpose")}: {t("common.all")}</option>
          <option value="steganography">{t("purposes.steganography")}</option>
          <option value="watermarking">{t("purposes.watermarking")}</option>
        </Select>
        <span className="text-xs text-muted-foreground">{t("common.selected", { count: value.length })}</span>
        {value.length > 0 && (
          <button type="button" className="text-xs text-muted-foreground underline" onClick={() => onChange([])}>
            {t("common.clear")}
          </button>
        )}
      </div>

      <div className="space-y-5">
        {families.map((entry) => (
          <div key={entry}>
            <div className="eyebrow mb-1.5">{t(`families.${entry}`)}</div>
            <div className="divide-y rounded-md border">
              {filtered
                .filter((method) => familyOf(method) === entry)
                .map((method) => {
                  const own = settings(method.name);
                  const selected = own.length > 0;
                  const expanded = open === method.name;
                  return (
                    <div key={method.name} className={cn(selected && "bg-accent/40")}>
                      <div className="flex items-center gap-3 px-3 py-2">
                        <input type="checkbox" checked={selected} onChange={() => toggle(method.name)} aria-label={method.name} />
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm">{method.description || method.name}</div>
                          <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
                            <span className="font-mono">{method.name}</span>
                            {method.purpose && <span>· {t(`purposes.${method.purpose}`)}</span>}
                            {method.reference && (
                              <span>
                                · {method.reference} ({method.year})
                              </span>
                            )}
                            {method.requires_tensorflow && <Chip>{t("catalogue.tensorflow")}</Chip>}
                            {method.needs_long_input && <Chip>{t("catalogue.longInput")}</Chip>}
                            {!method.packaged && <Chip>{t("catalogue.plugin")}</Chip>}
                          </div>
                        </div>
                        {selected && (
                          <button
                            type="button"
                            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                            onClick={() => setOpen(expanded ? null : method.name)}
                          >
                            {expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                            {t("editor.methods.configure")}
                            {own.length > 1 && <Chip>{own.length}</Chip>}
                          </button>
                        )}
                      </div>
                      {selected && expanded && (
                        <div className="space-y-3 border-t bg-background px-3 py-3">
                          {own.map(({ spec, index }) => (
                            <div key={index} className="rounded-md border p-3">
                              <div className="mb-2 flex items-center justify-between gap-2">
                                <Spec>{spec}</Spec>
                                {own.length > 1 && (
                                  <button type="button" onClick={() => onChange(value.filter((_, i) => i !== index))} aria-label={t("common.delete")}>
                                    <X className="h-3.5 w-3.5 text-muted-foreground" />
                                  </button>
                                )}
                              </div>
                              <ParameterFields
                                method={method}
                                parameters={parseSpec(spec).parameters}
                                onChange={(parameters) => updateSetting(index, method.name, parameters)}
                              />
                            </div>
                          ))}
                          {method.parameters.length > 0 && (
                            <Button type="button" size="sm" variant="outline" onClick={() => onChange([...value, method.name])}>
                              <Plus className="h-3.5 w-3.5" /> {t("editor.methods.addSetting")}
                            </Button>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          </div>
        ))}
      </div>
      {value.some((spec) => !byName.has(parseSpec(spec).name)) && (
        <p className="text-xs text-destructive">{value.filter((spec) => !byName.has(parseSpec(spec).name)).join(", ")}</p>
      )}
    </div>
  );
}

export function MethodSweepEditor({
  methods,
  value,
  onChange,
}: {
  methods: MethodInfo[];
  value: ParameterSweep | null | undefined;
  onChange: (sweep: ParameterSweep) => void;
}) {
  const { t } = useI18n();
  const tunable = methods.filter((method) => method.parameters.some((parameter) => !parameter.is_key));
  const current = value ?? { target: "", parameter: "", values: [] };
  const target = parseSpec(current.target).name;
  const method = tunable.find((entry) => entry.name === target);
  const [text, setText] = useState(current.values.join(", "));

  function chooseMethod(name: string) {
    const next = tunable.find((entry) => entry.name === name);
    const parameter = next?.strength_parameter ?? next?.parameters.find((p) => !p.is_key)?.name ?? "";
    const defaultValue = next?.parameters.find((p) => p.name === parameter)?.default;
    const values = strengthLadder(defaultValue);
    setText(values.join(", "));
    onChange({ target: name, parameter, values });
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">{t("editor.methods.sweepHint")}</p>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-1 md:col-span-2">
          <Label>{t("editor.methods.sweepTitle")}</Label>
          <Select value={target} onChange={(event) => chooseMethod(event.target.value)}>
            <option value="" disabled>
              —
            </option>
            {tunable.map((entry) => (
              <option key={entry.name} value={entry.name}>
                {entry.description || entry.name}
              </option>
            ))}
          </Select>
        </div>
        <div className="space-y-1">
          <Label>{t("editor.conditions.parameter")}</Label>
          <Select
            value={current.parameter}
            onChange={(event) => {
              const defaultValue = method?.parameters.find((p) => p.name === event.target.value)?.default;
              const values = strengthLadder(defaultValue);
              setText(values.join(", "));
              onChange({ ...current, parameter: event.target.value, values });
            }}
          >
            {(method?.parameters ?? []).map((parameter) => (
              <option key={parameter.name} value={parameter.name}>
                {parameter.name}
                {parameter.name === method?.strength_parameter ? ` (${t("catalogue.strength")})` : ""} — {t("common.default")} {String(parameter.default)}
              </option>
            ))}
          </Select>
        </div>
        <div className="space-y-1">
          <Label>{t("editor.conditions.values")}</Label>
          <Input
            className="font-mono text-xs"
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              onChange({ ...current, values: parseValues(event.target.value) });
            }}
          />
          <p className="text-[11px] text-muted-foreground">{t("editor.conditions.valuesHint")}</p>
        </div>
      </div>
      {current.target && current.parameter && (
        <div className="flex flex-wrap gap-1.5">
          {current.values.map((entry) => (
            <Spec key={String(entry)}>{formatSpec(target, { ...parseSpec(current.target).parameters, [current.parameter]: entry })}</Spec>
          ))}
        </div>
      )}
    </div>
  );
}

const CATEGORY_ORDER = ["speech_quality", "speech_intelligibility", "speech_reverberation", "ai_based", "unknown"];

export function MetricPicker({ metrics, value, onChange }: { metrics: MetricInfo[]; value: string[]; onChange: (names: string[]) => void }) {
  const { t } = useI18n();
  const toggle = (name: string) => onChange(value.includes(name) ? value.filter((entry) => entry !== name) : [...value, name]);
  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3 text-xs text-muted-foreground">
        <span>{t("common.selected", { count: value.length })}</span>
        {value.length > 0 && (
          <button type="button" className="underline" onClick={() => onChange([])}>
            {t("common.clear")}
          </button>
        )}
      </div>
      {CATEGORY_ORDER.filter((category) => metrics.some((metric) => metric.category === category)).map((category) => (
        <div key={category}>
          <div className="eyebrow mb-1.5">{t(`metricCategories.${category}`)}</div>
          <div className="grid gap-2 md:grid-cols-2">
            {metrics
              .filter((metric) => metric.category === category)
              .map((metric) => {
                const selected = value.includes(metric.name);
                return (
                  <label
                    key={metric.name}
                    className={cn("flex cursor-pointer items-start gap-3 rounded-md border px-3 py-2", selected && "border-primary/50 bg-accent/40")}
                  >
                    <input type="checkbox" className="mt-1" checked={selected} onChange={() => toggle(metric.name)} />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="text-sm font-medium">{metric.abbreviation}</span>
                        <span className="text-[11px] text-muted-foreground">{metric.scale}</span>
                      </div>
                      <div className="truncate text-xs text-muted-foreground" title={metric.label}>
                        {metric.label}
                      </div>
                      <div className="mt-1 flex flex-wrap gap-1">
                        <Chip>
                          {metric.higher_is_better === null
                            ? t("common.notRanked")
                            : metric.higher_is_better
                              ? t("common.higherIsBetter")
                              : t("common.lowerIsBetter")}
                        </Chip>
                        <Chip>{metric.intrusive ? t("catalogue.intrusive") : t("catalogue.nonIntrusive")}</Chip>
                        {metric.components.length > 0 && <Chip>{metric.components.join(" · ")}</Chip>}
                        {metric.reference && (
                          <Chip>
                            {metric.reference} {metric.year}
                          </Chip>
                        )}
                      </div>
                    </div>
                  </label>
                );
              })}
          </div>
        </div>
      ))}
    </div>
  );
}
