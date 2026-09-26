"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Check, Circle, Dices, Loader2, Play, RefreshCw, Save } from "lucide-react";

import { Chip, ErrorNotice, PropertyTag, Section } from "@/components/common";
import { NumberTicker } from "@/components/magicui/number-ticker";
import { ShimmerButton } from "@/components/magicui/shimmer-button";
import { ShineBorder } from "@/components/magicui/shine-border";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/api";
import { useAsync, type Catalog } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import type { ExperimentInput, ExperimentPlan } from "@/lib/types";
import { cn } from "@/lib/utils";

import { AttackPicker, AttackSweepEditor, DatasetPicker } from "./ConditionsPicker";
import { newSeed, stepProblems, stepsFor, type StepId } from "./draft";
import { MethodPicker, MethodSweepEditor, MetricPicker } from "./MethodPicker";

const PAYLOAD_PRESETS = [4, 8, 16, 32, 64, 128, 256, 512, 1024];

export function ExperimentEditor({
  catalog,
  initial,
  experimentId,
  version,
  hasRuns,
}: {
  catalog: Catalog;
  initial: ExperimentInput;
  experimentId?: string;
  version?: number;
  hasRuns?: boolean;
}) {
  const { t } = useI18n();
  const router = useRouter();
  const [draft, setDraft] = useState<ExperimentInput>(initial);
  const [step, setStep] = useState<StepId>("protocol");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plan, setPlan] = useState<ExperimentPlan | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [planning, setPlanning] = useState(false);

  const design = catalog.designs.find((entry) => entry.type === draft.experiment_type);
  const steps = stepsFor(draft.experiment_type);
  const datasets = useAsync(() => api.catalogDatasets(), []);
  const sampleRate = datasets.data?.find((entry) => entry.id === draft.config.dataset_id)?.sample_rate ?? 16000;
  const presets = useAsync(() => api.presets(sampleRate), [sampleRate]);
  const configChanged = JSON.stringify(draft.config) !== JSON.stringify(initial.config);

  const patch = (update: Partial<ExperimentInput>) => setDraft((previous) => ({ ...previous, ...update }));
  const patchConfig = (update: Partial<ExperimentInput["config"]>) =>
    setDraft((previous) => ({ ...previous, config: { ...previous.config, ...update } }));
  const advanced = (draft.config.advanced_options ?? {}) as Record<string, unknown>;
  const setAdvanced = (key: string, value: unknown) =>
    patchConfig({ advanced_options: { ...advanced, [key]: value === "" || value === undefined ? undefined : value } });

  const problems = useMemo(
    () => Object.fromEntries(steps.map((id) => [id, stepProblems(id, draft, design)])),
    [steps, draft, design]
  );

  async function refreshPlan() {
    setPlanning(true);
    setPlanError(null);
    try {
      setPlan(await api.preview(draft.name, draft.experiment_type, draft.config));
    } catch (reason) {
      setPlanError((reason as Error).message);
      setPlan(null);
    } finally {
      setPlanning(false);
    }
  }

  // A link may open a given step (?step=conditions).
  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get("step") as StepId | null;
    if (requested && steps.includes(requested)) setStep(requested);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (step === "review") void refreshPlan();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

  async function save(andRun: boolean) {
    setSaving(true);
    setError(null);
    try {
      const body: ExperimentInput = {
        ...draft,
        tags: draft.tags.map((tag) => tag.trim()).filter(Boolean),
        config: {
          ...draft.config,
          advanced_options: Object.fromEntries(Object.entries(advanced).filter(([, value]) => value !== undefined)),
        },
      };
      const saved = experimentId ? await api.updateExperiment(experimentId, body) : await api.createExperiment(body);
      if (andRun) {
        const run = await api.startRun(saved.id);
        router.push(`/runs/${run.id}`);
      } else {
        router.push(`/experiments/${saved.id}`);
      }
    } catch (reason) {
      setError((reason as Error).message);
      setSaving(false);
    }
  }

  const index = steps.indexOf(step);
  const blocking = Object.values(problems).some((list) => list.length > 0);

  return (
    <div className="grid gap-8 lg:grid-cols-[220px_minmax(0,1fr)]">
      <aside className="lg:sticky lg:top-20 lg:self-start">
        {design && (
          <div className="mb-5 space-y-1">
            <PropertyTag property={design.property} />
            <div className="text-sm font-semibold">{t(`designs.${design.type}.title`)}</div>
            <div className="text-xs italic text-muted-foreground">{t(`designs.${design.type}.question`)}</div>
          </div>
        )}
        <ol className="space-y-0.5">
          {steps.map((id, position) => {
            const done = problems[id]?.length === 0;
            return (
              <li key={id}>
                <button
                  type="button"
                  onClick={() => setStep(id)}
                  className={cn(
                    "flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-sm",
                    step === id ? "bg-accent font-medium" : "text-muted-foreground hover:bg-accent/50"
                  )}
                >
                  <span className="num w-4 text-xs text-muted-foreground">{position + 1}</span>
                  <span className="flex-1">{t(`editor.steps.${id}`)}</span>
                  {id !== "review" &&
                    (done ? (
                      <Check className="h-3.5 w-3.5" style={{ color: "var(--status-good)" }} />
                    ) : (
                      <Circle className="h-3 w-3 text-muted-foreground" />
                    ))}
                </button>
              </li>
            );
          })}
        </ol>
      </aside>

      <div className="min-w-0 space-y-5">
        {experimentId && hasRuns && configChanged && (
          <Alert variant="warning">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{t("editor.editWarning", { version: (version ?? 1) + 1 })}</AlertDescription>
          </Alert>
        )}

        {step === "protocol" && (
          <Section title={t("editor.steps.protocol")} hint={t("editor.protocol.preregistration")}>
            <div className="space-y-4">
              <div className="space-y-1">
                <Label>{t("editor.protocol.name")}</Label>
                <Input value={draft.name} placeholder={t("editor.protocol.namePlaceholder")} onChange={(event) => patch({ name: event.target.value })} />
              </div>
              <div className="space-y-1">
                <Label>{t("editor.protocol.question")}</Label>
                <Textarea
                  rows={2}
                  value={draft.research_question ?? ""}
                  placeholder={t("editor.protocol.questionPlaceholder")}
                  onChange={(event) => patch({ research_question: event.target.value })}
                />
              </div>
              <div className="space-y-1">
                <Label>{t("editor.protocol.hypothesis")}</Label>
                <Textarea
                  rows={2}
                  value={draft.hypothesis ?? ""}
                  placeholder={t("editor.protocol.hypothesisPlaceholder")}
                  onChange={(event) => patch({ hypothesis: event.target.value })}
                />
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-1">
                  <Label>{t("editor.protocol.description")}</Label>
                  <Textarea rows={3} value={draft.description ?? ""} onChange={(event) => patch({ description: event.target.value })} />
                </div>
                <div className="space-y-1">
                  <Label>{t("editor.protocol.tags")}</Label>
                  <Input
                    value={draft.tags.join(", ")}
                    placeholder={t("editor.protocol.tagsPlaceholder")}
                    onChange={(event) => patch({ tags: event.target.value.split(",").map((tag) => tag.trimStart()) })}
                  />
                </div>
              </div>
            </div>
          </Section>
        )}

        {step === "data" && (
          <Section title={t("editor.steps.data")}>
            {datasets.error && <ErrorNotice error={datasets.error} onRetry={datasets.reload} />}
            <DatasetPicker
              datasets={datasets.data ?? []}
              value={draft.config.dataset_id}
              fileLimit={draft.config.file_limit}
              onChange={(datasetId, fileLimit) => patchConfig({ dataset_id: datasetId, file_limit: fileLimit })}
            />
          </Section>
        )}

        {step === "methods" && (
          <>
            {design?.requires_sweep === "method" ? (
              <>
                <Section title={t("editor.methods.sweepTitle")}>
                  <MethodSweepEditor
                    methods={catalog.methods}
                    value={draft.config.method_sweep}
                    onChange={(sweep) => patchConfig({ method_sweep: sweep })}
                  />
                </Section>
                <Section title={t("editor.methods.references")} hint={t("editor.methods.referencesHint")}>
                  <MethodPicker methods={catalog.methods} value={draft.config.methods} onChange={(methods) => patchConfig({ methods })} />
                </Section>
              </>
            ) : (
              <Section title={t("editor.steps.methods")} hint={t("editor.methods.hint")}>
                <MethodPicker methods={catalog.methods} value={draft.config.methods} onChange={(methods) => patchConfig({ methods })} />
              </Section>
            )}
          </>
        )}

        {step === "conditions" && (
          <Section title={t("editor.steps.conditions")}>
            {design?.requires_sweep === "attack" ? (
              <AttackSweepEditor
                attacks={catalog.attacks}
                presets={presets.data}
                value={draft.config.attack_sweep}
                onChange={(sweep) => patchConfig({ attack_sweep: sweep })}
              />
            ) : (
              <AttackPicker
                attacks={catalog.attacks}
                presets={presets.data}
                value={draft.config.attacks}
                preset={draft.config.attack_preset}
                onChange={(attacks, preset) => patchConfig({ attacks, attack_preset: preset })}
              />
            )}
          </Section>
        )}

        {step === "measures" && (
          <Section title={t("editor.steps.measures")} hint={t("editor.measures.hint")}>
            {design?.requires_metrics && !draft.config.metrics.length && (
              <p className="mb-3 text-xs text-destructive">{t("editor.measures.required")}</p>
            )}
            <MetricPicker metrics={catalog.metrics} value={draft.config.metrics} onChange={(metrics) => patchConfig({ metrics })} />
            <p className="mt-4 text-xs text-muted-foreground">{t("editor.measures.slow")}</p>
          </Section>
        )}

        {step === "design" && (
          <Section title={t("editor.steps.design")}>
            <div className="space-y-6">
              <div className="space-y-2">
                <Label>{t("editor.design.payloads")}</Label>
                <div className="flex flex-wrap gap-1.5">
                  {[...new Set([...PAYLOAD_PRESETS, ...draft.config.payload_lengths])]
                    .sort((a, b) => a - b)
                    .map((length) => {
                      const on = draft.config.payload_lengths.includes(length);
                      return (
                        <button
                          key={length}
                          type="button"
                          onClick={() =>
                            patchConfig({
                              payload_lengths: on
                                ? draft.config.payload_lengths.filter((entry) => entry !== length)
                                : [...draft.config.payload_lengths, length].sort((a, b) => a - b),
                            })
                          }
                          className={cn(
                            "num rounded-md border px-2.5 py-1 text-xs",
                            on ? "border-primary bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-accent"
                          )}
                        >
                          {length}
                        </button>
                      );
                    })}
                  <Input
                    type="number"
                    min={4}
                    max={8192}
                    placeholder="+"
                    className="h-7 w-20 text-xs"
                    onKeyDown={(event) => {
                      const value = Number((event.target as HTMLInputElement).value);
                      if (event.key === "Enter" && value >= 4 && value <= 8192 && !draft.config.payload_lengths.includes(value)) {
                        patchConfig({ payload_lengths: [...draft.config.payload_lengths, value].sort((a, b) => a - b) });
                        (event.target as HTMLInputElement).value = "";
                      }
                    }}
                  />
                </div>
                <p className="text-xs text-muted-foreground">{t("editor.design.payloadsHint")}</p>
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-1">
                  <Label>{t("editor.design.repetitions")}</Label>
                  <Input
                    type="number"
                    min={1}
                    max={50}
                    value={draft.config.repetitions}
                    onChange={(event) => patchConfig({ repetitions: Math.max(1, Number(event.target.value) || 1) })}
                  />
                  <p className="text-[11px] text-muted-foreground">{t("editor.design.repetitionsHint")}</p>
                </div>
                <div className="space-y-1">
                  <Label>{t("editor.design.seed")}</Label>
                  <div className="flex gap-2">
                    <Input
                      type="number"
                      value={draft.config.random_seed ?? ""}
                      onChange={(event) => patchConfig({ random_seed: event.target.value === "" ? null : Number(event.target.value) })}
                    />
                    <Button type="button" variant="outline" size="icon" onClick={() => patchConfig({ random_seed: newSeed() })} title="Random">
                      <Dices className="h-4 w-4" />
                    </Button>
                  </div>
                  <p className="text-[11px] text-muted-foreground">{t("editor.design.seedHint")}</p>
                </div>
                <div className="space-y-1">
                  <Label>{t("editor.design.workers")}</Label>
                  <Input
                    type="number"
                    min={1}
                    max={16}
                    value={draft.config.max_workers ?? 2}
                    onChange={(event) => patchConfig({ max_workers: Math.max(1, Number(event.target.value) || 1) })}
                  />
                  <p className="text-[11px] text-muted-foreground">{t("editor.design.workersHint")}</p>
                </div>
              </div>

              {draft.experiment_type === "embedding_capacity" && (
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="space-y-1">
                    <Label>{t("editor.design.minAccuracy")}</Label>
                    <Input type="number" step={0.01} min={0} max={1} placeholder="0.95" value={String(advanced.min_bit_accuracy ?? "")} onChange={(event) => setAdvanced("min_bit_accuracy", event.target.value === "" ? undefined : Number(event.target.value))} />
                  </div>
                  <div className="space-y-1">
                    <Label>{t("editor.design.maxBer")}</Label>
                    <Input type="number" step={0.01} min={0} max={1} placeholder="0.05" value={String(advanced.max_ber ?? "")} onChange={(event) => setAdvanced("max_ber", event.target.value === "" ? undefined : Number(event.target.value))} />
                  </div>
                </div>
              )}

              {draft.experiment_type === "detectability" && (
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="space-y-1">
                    <Label>{t("editor.design.windowLength")}</Label>
                    <Input type="number" min={1024} placeholder="32000" value={String(advanced.window_length ?? "")} onChange={(event) => setAdvanced("window_length", event.target.value === "" ? undefined : Number(event.target.value))} />
                  </div>
                  <div className="space-y-1">
                    <Label>{t("editor.design.testFraction")}</Label>
                    <Input type="number" step={0.05} min={0.1} max={0.5} placeholder="0.3" value={String(advanced.test_fraction ?? "")} onChange={(event) => setAdvanced("test_fraction", event.target.value === "" ? undefined : Number(event.target.value))} />
                  </div>
                </div>
              )}

              {draft.experiment_type === "method_comparison" && (
                <div className="space-y-2">
                  <Label>{t("editor.design.weights")}</Label>
                  <div className="grid gap-3 md:grid-cols-4">
                    {["accuracy", "robustness", "quality", "speed"].map((key) => {
                      const weights = (advanced.weights ?? {}) as Record<string, number>;
                      return (
                        <div key={key} className="space-y-0.5">
                          <Label className="text-xs text-muted-foreground">{t(`editor.design.weightKeys.${key}`)}</Label>
                          <Input
                            type="number"
                            step={0.1}
                            min={0}
                            value={weights[key] ?? ""}
                            onChange={(event) => {
                              const next = { ...weights };
                              if (event.target.value === "") delete next[key];
                              else next[key] = Number(event.target.value);
                              setAdvanced("weights", Object.keys(next).length ? next : undefined);
                            }}
                          />
                        </div>
                      );
                    })}
                  </div>
                  <p className="text-xs text-muted-foreground">{t("editor.design.weightsHint")}</p>
                </div>
              )}
            </div>
          </Section>
        )}

        {step === "review" && (
          <>
            <Section
              className="relative"
              title={t("editor.review.plan")}
              actions={
                <Button type="button" variant="outline" size="sm" onClick={refreshPlan} disabled={planning}>
                  {planning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
                  {t("editor.review.refresh")}
                </Button>
              }
            >
              {planError && <ErrorNotice error={planError} />}
              {plan && !blocking && <ShineBorder duration={12} />}
              {plan && (
                <div className="space-y-4">
                  <div className="grid gap-4 sm:grid-cols-3">
                    <div>
                      <div className="text-xs text-muted-foreground">{t("editor.review.trials")}</div>
                      <div className="text-2xl font-semibold">
                        <NumberTicker value={plan.estimated_result_rows} />
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">{t("editor.review.encodes")}</div>
                      <div className="text-2xl font-semibold">
                        <NumberTicker value={plan.encode_operations} />
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">{t("editor.review.metricEvaluations")}</div>
                      <div className="text-2xl font-semibold">
                        <NumberTicker value={plan.estimated_metric_calculations} />
                      </div>
                    </div>
                  </div>
                  <p className="num text-xs text-muted-foreground">
                    {t("editor.review.factorial", {
                      files: plan.file_count,
                      methods: plan.method_count,
                      payloads: plan.payload_length_count,
                      reps: plan.repetitions,
                      variants: plan.attack_variant_count,
                    })}
                  </p>
                  {plan.warnings.length > 0 ? (
                    <ul className="space-y-1.5">
                      {plan.warnings.map((warning, position) => (
                        <li key={position} className="flex items-start gap-2 text-sm">
                          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" style={{ color: warning.code === "validation" ? "var(--status-critical)" : "var(--status-serious)" }} />
                          <span>
                            <Chip className="mr-1.5">{warning.code}</Chip>
                            {warning.message}
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm" style={{ color: "hsl(var(--success))" }}>
                      ✓ {t("editor.review.ok")}
                    </p>
                  )}
                </div>
              )}
            </Section>
            <Section title={t("editor.review.config")}>
              <pre className="spec max-h-96 overflow-auto rounded-md bg-muted p-3 leading-relaxed">
                {JSON.stringify({ ...draft.config, experiment_type: draft.experiment_type }, null, 2)}
              </pre>
            </Section>
          </>
        )}

        {error && <ErrorNotice error={error} />}

        <div className="flex flex-wrap items-center justify-between gap-3 border-t pt-4">
          <Button type="button" variant="ghost" disabled={index === 0} onClick={() => setStep(steps[index - 1])}>
            {t("common.back")}
          </Button>
          <div className="flex flex-wrap items-center gap-2">
            {step !== "review" && (
              <Button type="button" variant="outline" onClick={() => setStep(steps[index + 1])}>
                {t("common.next")}
              </Button>
            )}
            <Button type="button" variant={step === "review" ? "outline" : "ghost"} disabled={saving || !draft.name.trim()} onClick={() => save(false)}>
              <Save className="h-4 w-4" /> {t("common.save")}
            </Button>
            {step === "review" && (
              <ShimmerButton type="button" disabled={saving || blocking} onClick={() => save(true)}>
                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                {t("common.saveAndRun")}
              </ShimmerButton>
            )}
          </div>
        </div>
        {step === "review" && blocking && (
          <p className="text-right text-xs text-muted-foreground">
            {Object.entries(problems)
              .filter(([, list]) => list.length)
              .map(([id]) => t(`editor.steps.${id}`))
              .join(" · ")}
          </p>
        )}
      </div>
    </div>
  );
}
