"use client";

import { useMemo, useState, type ReactNode } from "react";
import { AlertTriangle } from "lucide-react";

import { BoxPlot } from "@/components/charts/BoxPlot";
import { CurveChart } from "@/components/charts/CurveChart";
import { IntervalChart } from "@/components/charts/IntervalChart";
import { RateChart } from "@/components/charts/RateChart";
import { ScatterChart } from "@/components/charts/ScatterChart";
import { ChartFrame, DataTable } from "@/components/charts/base";
import { KeyValues, Section } from "@/components/common";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Select } from "@/components/ui/select";
import { useMethodAbbreviation } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import type { BerBlock, Correlation, Distribution, Estimate, Evaluation, ExperimentType, Fact, RecoveryKey } from "@/lib/types";

const RECOVERY_KEYS: RecoveryKey[] = ["exact", "le_1pct", "le_5pct"];

// ------------------------------------------------------------------ helpers

function Picker({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: [string, string][] }) {
  if (options.length < 2) return null;
  return (
    <label className="flex items-center gap-2 text-xs text-muted-foreground">
      {label}
      <Select className="h-7 w-auto min-w-[9rem] text-xs" value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map(([key, text]) => (
          <option key={key} value={key}>
            {text}
          </option>
        ))}
      </Select>
    </label>
  );
}

function useFormat() {
  const { number, percent } = useI18n();
  return {
    number,
    percent,
    /** "mean [low, high]" in the page's number format. */
    interval: (value: Estimate | Distribution | null | undefined, digits = 3) => {
      if (!value) return "–";
      const centre = "estimate" in value ? value.estimate : value.mean;
      if (centre === null || centre === undefined) return "–";
      return value.ci95_low === null || value.ci95_low === undefined
        ? number(centre, digits)
        : `${number(centre, digits)} [${number(value.ci95_low, digits)}, ${number(value.ci95_high, digits)}]`;
    },
  };
}

function p(value: number | undefined | null, number: (v: number | null | undefined, d?: number) => string) {
  if (value === undefined || value === null) return "–";
  return value < 0.001 ? `< ${number(0.001, 3)}` : `= ${number(value, 3)}`;
}

/** Method order and colour slot come from the backend's sorted list. */
function useMethods(evaluation: Evaluation) {
  const abbreviate = useMethodAbbreviation();
  return evaluation.methods.map((entry, slot) => ({ name: entry.method, short: abbreviate(entry.method), slot }));
}

function ResolutionNote({ evaluation }: { evaluation: Evaluation }) {
  const { t, number } = useI18n();
  const resolution = evaluation.resolution;
  if (!resolution.le_1pct_equals_exact || !resolution.min_payload_bits) return null;
  return <p className="mt-3 text-xs text-muted-foreground">{t("evaluation.resolutionNote", { bits: resolution.min_payload_bits, step: number(resolution.ber_step, 4) })}</p>;
}

// ------------------------------------------------------- scientific summary

function factText(fact: Fact, t: (key: string, vars?: Record<string, string | number>) => string, number: (v: number | null | undefined, d?: number) => string, abbreviate: (s: string) => string, alpha: number): string {
  const f = fact as Record<string, unknown>;
  const pv = (value: unknown) => (typeof value === "number" ? p(value, number) : "–");
  switch (fact.kind) {
    case "largest_ber_increase":
      return t("evaluation.facts.largest_ber_increase", {
        method: abbreviate(String(f.method)),
        attack: String(f.attack),
        delta: number(f.delta as number, 3),
        lo: number(f.ci95_low as number, 3),
        hi: number(f.ci95_high as number, 3),
        files: String(f.files),
      });
    case "no_confirmed_increase":
      return t("evaluation.facts.no_confirmed_increase", { method: abbreviate(String(f.method)), attack: String(f.attack), delta: number(f.delta as number, 3) });
    case "stable_under_attack":
      return t("evaluation.facts.stable_under_attack", {
        attack: String(f.attack),
        threshold: number(f.threshold as number, 2),
        methods: (f.methods as string[]).map((method, index) => `${abbreviate(method)} (${number((f.bounds as number[])[index], 3)})`).join(", "),
      });
    case "failures":
      return t("evaluation.facts.failures", {
        method: abbreviate(String(f.method)),
        failed: String(f.failed),
        trials: String(f.trials),
        kinds: Object.entries(f.by_kind as Record<string, number>).map(([kind, count]) => `${kind}: ${count}`).join(", "),
      });
    case "variability":
      return t("evaluation.facts.variability", {
        highest: abbreviate(String(f.highest)),
        highestSd: number(f.highest_sd as number, 3),
        lowest: abbreviate(String(f.lowest)),
        lowestSd: number(f.lowest_sd as number, 3),
      });
    case "correlation": {
      const direction = t(`evaluation.facts.${(f.rho as number) > 0 ? "increased" : "decreased"}`);
      const common = { method: abbreviate(String(f.method)), direction, rho: number(f.rho as number, 2), p: pv(f.p_holm), files: String(f.n_files) };
      if (f.x === "attack_strength") return t("evaluation.facts.correlationStrength", { ...common, group: String(f.group) });
      const y = String(f.y);
      const label = y.startsWith("metric: ") ? y.slice(8) : t(`evaluation.variables.${y}`);
      return t("evaluation.facts.correlationPayload", { ...common, y: label });
    }
    case "correlations_tested":
      return t("evaluation.facts.correlations_tested", { significant: String(f.significant), tested: String(f.tested), alpha: number(alpha, 2) });
    case "method_difference": {
      const comparisonKey = String(f.comparison);
      const labels: Record<string, string> = { ber_baseline: "run.berClean", ber_attacked: "run.berAttacked", ber_sweep: "run.curve", runtime_total: "evaluation.total" };
      let text = t("evaluation.facts.method_difference", {
        comparison: labels[comparisonKey] ? t(labels[comparisonKey]) : comparisonKey.replaceAll("_", " "),
        test: f.test === "friedman" ? "Friedman" : "Wilcoxon",
        outcome: t(f.significant ? "evaluation.facts.detected" : "evaluation.facts.notDetected"),
        methods: String(f.methods),
        p: pv(f.p_value),
        files: String(f.files),
        pairsSignificant: String(f.pairs_significant),
        pairs: String(f.pairs),
      });
      const effect = f.largest_effect as { better: string; worse: string; directional: boolean; r: number } | null;
      if (effect)
        text +=
          " " +
          t(effect.directional ? "evaluation.facts.largestEffect" : "evaluation.facts.largestEffectPair", {
            better: abbreviate(effect.better),
            worse: abbreviate(effect.worse),
            r: number(effect.r, 2),
          });
      if (f.underpowered) text += " " + t("evaluation.facts.underpowered", { files: String(f.files), alpha: number(alpha, 2), min: String(f.min_files) });
      return text;
    }
    default:
      return fact.text;
  }
}

export function ScientificSummary({ evaluation }: { evaluation: Evaluation }) {
  const { t, number } = useI18n();
  const abbreviate = useMethodAbbreviation();
  return (
    <Section title={t("evaluation.summaryTitle")} hint={t("evaluation.summaryHint")}>
      {evaluation.facts.length === 0 ? (
        <p className="text-sm text-muted-foreground">{t("evaluation.noFacts")}</p>
      ) : (
        <ul className="space-y-2 text-sm leading-relaxed">
          {evaluation.facts.map((fact, index) => (
            <li key={index} className="flex gap-2.5">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-muted-foreground/60" aria-hidden />
              <span title={fact.text}>{factText(fact, t, number, abbreviate, evaluation.settings.alpha)}</span>
            </li>
          ))}
        </ul>
      )}
    </Section>
  );
}

// ----------------------------------------------------------- per method

function conditionOptions(evaluation: Evaluation, t: (key: string) => string): [string, string][] {
  const options: [string, string][] = [];
  if (evaluation.methods.some((entry) => entry.attacked)) options.push(["attacked", t("evaluation.attacked")]);
  if (evaluation.methods.some((entry) => entry.clean)) options.push(["clean", t("evaluation.clean")]);
  return options;
}

function MethodStatistics({ evaluation }: { evaluation: Evaluation }) {
  const { t } = useI18n();
  const { number, percent, interval } = useFormat();
  const methods = useMethods(evaluation);
  const options = conditionOptions(evaluation, t);
  const [condition, setCondition] = useState(options[0]?.[0] ?? "clean");
  const blockOf = (index: number) => evaluation.methods[index][condition as "clean" | "attacked"] as BerBlock | null;
  const rate = (block: BerBlock | null, key: RecoveryKey) => {
    const value = block?.recovery[key];
    if (!value || value.estimate === null) return "–";
    return (
      <span title={`95% CI [${percent(value.ci95_low, 1)}, ${percent(value.ci95_high, 1)}]`} className="cursor-help underline decoration-dotted underline-offset-2">
        {percent(value.estimate, 1)}
      </span>
    );
  };

  return (
    <>
      <Section
        title={t("evaluation.methodStats")}
        hint={t("evaluation.methodStatsHint")}
        actions={<Picker label={t("evaluation.condition")} value={condition} onChange={setCondition} options={options} />}
      >
        <DataTable
          table={{
            columns: [
              t("run.filterMethod"),
              `${t("evaluation.trials")} / ${t("evaluation.files")}`,
              t("evaluation.mean"),
              t("evaluation.median"),
              t("evaluation.sd"),
              t("evaluation.quartiles"),
              t("evaluation.iqr"),
              t("evaluation.range"),
              t("evaluation.exact"),
              t("evaluation.le1"),
              t("evaluation.le5"),
              t("run.completion"),
            ],
            rows: methods.map((method) => {
              const block = blockOf(method.slot);
              const ber = block?.ber;
              return [
                <span key="m" title={method.name} className="whitespace-nowrap">
                  {method.short}
                </span>,
                block ? `${block.trials} / ${block.files}` : "–",
                interval(ber),
                number(ber?.median, 3),
                number(ber?.std, 3),
                ber ? `${number(ber.q1, 3)} – ${number(ber.q3, 3)}` : "–",
                number(ber?.iqr, 3),
                ber ? `${number(ber.min, 3)} – ${number(ber.max, 3)}` : "–",
                rate(block, "exact"),
                rate(block, "le_1pct"),
                rate(block, "le_5pct"),
                percent(block?.completion_rate, 1),
              ];
            }),
          }}
        />
        <ResolutionNote evaluation={evaluation} />
      </Section>

      <ChartFrame
        title={t("evaluation.distribution")}
        hint={t("evaluation.distributionHint")}
        actions={<Picker label={t("evaluation.condition")} value={condition} onChange={setCondition} options={options} />}
        table={{
          columns: [t("run.filterMethod"), "min", "Q1", t("evaluation.median"), "Q3", "max", t("run.mean"), t("evaluation.outliers"), t("evaluation.n")],
          rows: methods.map((method) => {
            const box = blockOf(method.slot)?.box;
            return [method.short, number(box?.min, 3), number(box?.q1, 3), number(box?.median, 3), number(box?.q3, 3), number(box?.max, 3), number(box?.mean, 3), box?.outliers ?? "–", box?.n ?? "–"];
          }),
        }}
      >
        <BoxPlot
          rows={methods.map((method) => ({ label: method.short, fullLabel: method.name, box: blockOf(method.slot)?.box ?? null }))}
          axisTitle="BER"
          domain={[0, Math.max(0.5, ...methods.map((method) => blockOf(method.slot)?.box?.max ?? 0))]}
          format={(value) => number(value, 3)}
          labels={{
            median: t("evaluation.median"),
            quartiles: t("evaluation.quartiles"),
            whiskers: t("evaluation.whiskers"),
            mean: t("run.mean"),
            outliers: t("evaluation.outliers"),
            n: t("evaluation.trials"),
          }}
        />
      </ChartFrame>
    </>
  );
}

// -------------------------------------------------------------- attacks

function AttackAnalysis({ evaluation }: { evaluation: Evaluation }) {
  const { t } = useI18n();
  const { number, percent, interval } = useFormat();
  const methods = useMethods(evaluation);
  const [method, setMethod] = useState(methods[0]?.name ?? "");
  const [attack, setAttack] = useState(evaluation.attacks[0]?.attack ?? "");
  const methodOptions = methods.map((entry): [string, string] => [entry.name, entry.short]);
  const short = (name: string) => methods.find((entry) => entry.name === name)?.short ?? name;

  const destructive = evaluation.most_destructive[method] ?? [];
  const current = evaluation.attacks.find((entry) => entry.attack === attack);
  const resistant = current ? [...current.per_method].sort((a, b) => (a.ber_imputed.estimate ?? 1) - (b.ber_imputed.estimate ?? 1)) : [];
  const methodEntry = evaluation.methods.find((entry) => entry.method === method);
  const recoveryRows = [
    ...(methodEntry?.clean ? [{ label: t("run.baseline"), values: RECOVERY_KEYS.map((key) => methodEntry.clean!.recovery[key]) }] : []),
    ...evaluation.attacks.flatMap((entry) => {
      const own = entry.per_method.find((item) => item.method === method);
      return own ? [{ label: entry.attack, values: RECOVERY_KEYS.map((key) => own.recovery[key]) }] : [];
    }),
  ];

  return (
    <>
      <div className="grid gap-6 xl:grid-cols-2">
        <ChartFrame
          title={t("evaluation.destructive")}
          hint={t("evaluation.destructiveHint")}
          actions={<Picker label={t("run.filterMethod")} value={method} onChange={setMethod} options={methodOptions} />}
          table={{
            columns: [t("run.filterMethod"), ...evaluation.attacks.map((entry) => entry.attack)],
            rows: methods.map((entry) => [
              entry.short,
              ...evaluation.attacks.map((attackEntry) => interval(attackEntry.per_method.find((item) => item.method === entry.name)?.delta_ber)),
            ]),
          }}
        >
          <IntervalChart
            rows={destructive.map((entry) => ({
              label: entry.attack,
              estimate: entry.delta_ber.estimate,
              lo: entry.delta_ber.ci95_low,
              hi: entry.delta_ber.ci95_high,
              note: `${t("evaluation.files")}: ${entry.delta_ber.clusters}`,
            }))}
            axisTitle={`${t("evaluation.delta")} (${short(method)})`}
            reference={0}
            format={(value) => number(value, 3)}
            color="var(--series-2)"
          />
        </ChartFrame>
        <ChartFrame
          title={t("evaluation.resistant")}
          hint={t("evaluation.resistantHint")}
          actions={
            <Picker label={t("evaluation.attack")} value={attack} onChange={setAttack} options={evaluation.attacks.map((entry): [string, string] => [entry.attack, entry.attack])} />
          }
          table={{
            columns: [t("run.filterMethod"), "BER", t("evaluation.exact"), t("run.completion")],
            rows: resistant.map((entry) => [short(entry.method), interval(entry.ber_imputed), percent(entry.recovery.exact.estimate, 1), percent(entry.completion_rate, 1)]),
          }}
        >
          <IntervalChart
            rows={resistant.map((entry) => ({
              label: short(entry.method),
              fullLabel: entry.method,
              estimate: entry.ber_imputed.estimate,
              lo: entry.ber_imputed.ci95_low,
              hi: entry.ber_imputed.ci95_high,
            }))}
            axisTitle={`BER — ${attack}`}
            domain={[0, 0.5]}
            format={(value) => number(value, 3)}
          />
        </ChartFrame>
      </div>

      <ChartFrame
        title={t("evaluation.recovery")}
        hint={t("evaluation.recoveryHint")}
        actions={<Picker label={t("run.filterMethod")} value={method} onChange={setMethod} options={methodOptions} />}
        table={{
          columns: [t("run.filterAttack"), t("evaluation.exact"), t("evaluation.le1"), t("evaluation.le5")],
          rows: recoveryRows.map((row) => [row.label, ...row.values.map((value) => interval(value, 3))]),
        }}
      >
        <RateChart
          rows={recoveryRows}
          series={[t("evaluation.exact"), t("evaluation.le1"), t("evaluation.le5")]}
          axisTitle={short(method)}
          percent={(value) => percent(value, 0)}
        />
        <ResolutionNote evaluation={evaluation} />
      </ChartFrame>
    </>
  );
}

// ------------------------------------------------------------- severity

function CorrelationTable({ tests, showMethod = true }: { tests: Correlation[]; showMethod?: boolean }) {
  const { t, tOr } = useI18n();
  const { number } = useFormat();
  const abbreviate = useMethodAbbreviation();
  const variable = (name: string) => (name.startsWith("metric: ") ? name.slice(8) : tOr(`evaluation.variables.${name}`, name));
  return (
    <DataTable
      table={{
        columns: [
          ...(showMethod ? [t("run.filterMethod")] : []),
          t("evaluation.x"),
          t("evaluation.y"),
          t("evaluation.group"),
          t("evaluation.rho"),
          t("evaluation.pRaw"),
          t("evaluation.pHolm"),
          t("evaluation.files"),
          t("evaluation.points"),
          t("evaluation.levels"),
        ],
        rows: tests.map((test) => [
          ...(showMethod ? [<span key="m" title={test.method}>{abbreviate(test.method)}</span>] : []),
          variable(test.x),
          variable(test.y),
          test.group ?? "–",
          test.available ? `${number(test.rho, 2)} [${number(test.rho_ci95_low, 2)}, ${number(test.rho_ci95_high, 2)}]` : <span key="r" className="text-muted-foreground">{test.reason}</span>,
          test.available ? p(test.p_value, number) : "–",
          test.available ? (
            <span key="h" className={test.significant ? "font-semibold" : "text-muted-foreground"}>
              {p(test.p_holm, number)}
              {test.significant ? " *" : ""}
            </span>
          ) : (
            "–"
          ),
          test.n_files ?? "–",
          test.n_points ?? "–",
          test.levels ?? "–",
        ]),
      }}
    />
  );
}

function Severity({ evaluation, showCurves }: { evaluation: Evaluation; showCurves: boolean }) {
  const { t } = useI18n();
  const { number, interval } = useFormat();
  const methods = useMethods(evaluation);
  const [family, setFamily] = useState(evaluation.severity[0]?.family ?? "");
  const current = evaluation.severity.find((entry) => entry.family === family) ?? evaluation.severity[0];
  if (!current) return null;
  const levels = current.levels.map((level) => (level === "clean" ? t("run.baseline") : level));
  const picker = (
    <Picker label={t("evaluation.attack")} value={current.family} onChange={setFamily} options={evaluation.severity.map((entry): [string, string] => [entry.family, entry.family])} />
  );
  return (
    <>
      {showCurves && (
        <ChartFrame
          title={`${t("evaluation.severity")}: ${current.family}`}
          hint={t("evaluation.severityHint")}
          actions={picker}
          table={{
            columns: [t("run.filterMethod"), ...levels],
            rows: current.curves.map((curve) => [methods.find((m) => m.name === curve.method)?.short ?? curve.method, ...curve.points.map((point) => interval(point.ber))]),
          }}
        >
          <CurveChart
            xLabels={levels}
            xTitle={`${current.family} · ${current.parameter ?? ""}`}
            yTitle="BER"
            format={(value) => number(value, 3)}
            series={current.curves.map((curve) => ({
              label: methods.find((m) => m.name === curve.method)?.short ?? curve.method,
              points: curve.points.map((point) => ({ y: point.ber.estimate, lo: point.ber.ci95_low, hi: point.ber.ci95_high })),
            }))}
          />
        </ChartFrame>
      )}
      <Section title={`${t("evaluation.trend")}: ${current.family}`} hint={t("evaluation.trendHint")} actions={showCurves ? undefined : picker}>
        <CorrelationTable tests={current.curves.map((curve) => ({ ...curve.trend, method: curve.method, x: "attack_strength", y: "ber", group: current.family }))} />
      </Section>
    </>
  );
}

// -------------------------------------------------------------- quality

function QualityAnalysis({ evaluation }: { evaluation: Evaluation }) {
  const { t } = useI18n();
  const { number, interval } = useFormat();
  const methods = useMethods(evaluation);
  const metrics = evaluation.quality.metrics;
  const [metricName, setMetricName] = useState(metrics[0]?.name ?? "");
  const metric = metrics.find((entry) => entry.name === metricName) ?? metrics[0];
  const damage = evaluation.quality.attack_damage;
  const [damageName, setDamageName] = useState(damage[0]?.name ?? "");
  const damageMetric = damage.find((entry) => entry.name === damageName) ?? damage[0];
  const [damageAttack, setDamageAttack] = useState(damageMetric?.per_attack[0]?.attack ?? "");
  const damageEntry = damageMetric?.per_attack.find((entry) => entry.attack === damageAttack) ?? damageMetric?.per_attack[0];
  const attacked = evaluation.methods.some((entry) => entry.attacked);
  const short = (name: string) => methods.find((entry) => entry.name === name)?.short ?? name;
  const metricOptions = metrics.map((entry): [string, string] => [entry.name, entry.name]);
  const direction = (higher: boolean | null) => (higher === null ? null : higher ? "up" : "down");

  const distributionRows = (entries: (Distribution & { method: string })[]) =>
    entries.map((entry) => [
      <span key="m" title={entry.method}>
        {short(entry.method)}
      </span>,
      interval(entry, 2),
      number(entry.median, 2),
      number(entry.std, 2),
      entry.q1 === null ? "–" : `${number(entry.q1, 2)} – ${number(entry.q3, 2)}`,
      entry.count,
      entry.clusters,
    ]);
  const distributionColumns = [t("run.filterMethod"), t("evaluation.mean"), t("evaluation.median"), t("evaluation.sd"), t("evaluation.quartiles"), t("evaluation.trials"), t("evaluation.files")];

  return (
    <>
      {metric && (
        <div className="grid gap-6 xl:grid-cols-2">
          <ChartFrame
            title={t("evaluation.quality")}
            hint={t("evaluation.qualityHint")}
            actions={<Picker label={t("evaluation.metric")} value={metric.name} onChange={setMetricName} options={metricOptions} />}
            table={{
              columns: [t("run.filterMethod"), metric.name, attacked ? t("run.berAttacked") : t("run.berClean")],
              rows: evaluation.methods.map((entry, index) => [
                methods[index].short,
                interval(metric.per_method.find((item) => item.method === entry.method), 2),
                interval((attacked ? entry.attacked : entry.clean)?.ber_imputed),
              ]),
            }}
          >
            <ScatterChart
              points={evaluation.methods.map((entry, index) => {
                const quality = metric.per_method.find((item) => item.method === entry.method);
                const ber = (attacked ? entry.attacked : entry.clean)?.ber_imputed;
                return {
                  slot: index,
                  label: methods[index].short,
                  fullLabel: entry.method,
                  x: quality?.mean ?? null,
                  xLo: quality?.ci95_low,
                  xHi: quality?.ci95_high,
                  y: ber?.estimate ?? null,
                  yLo: ber?.ci95_low,
                  yHi: ber?.ci95_high,
                };
              })}
              xTitle={metric.name}
              yTitle={attacked ? t("run.berAttacked") : t("run.berClean")}
              xBetter={direction(metric.higher_is_better)}
              yBetter="down"
              betterWord={t("common.better")}
              formatX={(value) => number(value, 2)}
              formatY={(value) => number(value, 3)}
            />
          </ChartFrame>
          <Section
            title={t("evaluation.qualityTable")}
            hint={t("evaluation.qualityTableHint")}
            actions={<Picker label={t("evaluation.metric")} value={metric.name} onChange={setMetricName} options={metricOptions} />}
          >
            <DataTable table={{ columns: distributionColumns, rows: distributionRows(metric.per_method) }} />
            {!evaluation.quality.psnr_available && <p className="mt-3 text-xs text-muted-foreground">{t("evaluation.noPsnr")}</p>}
          </Section>
        </div>
      )}
      {damageMetric && damageEntry && (
        <Section
          title={t("evaluation.damageTable")}
          hint={t("evaluation.damageTableHint")}
          actions={
            <div className="flex flex-wrap items-center gap-3">
              <Picker label={t("evaluation.metric")} value={damageMetric.name} onChange={setDamageName} options={damage.map((entry): [string, string] => [entry.name, entry.name])} />
              <Picker
                label={t("evaluation.attack")}
                value={damageEntry.attack}
                onChange={setDamageAttack}
                options={damageMetric.per_attack.map((entry): [string, string] => [entry.attack, entry.attack])}
              />
            </div>
          }
        >
          <DataTable table={{ columns: distributionColumns, rows: distributionRows(damageEntry.per_method) }} />
        </Section>
      )}
    </>
  );
}

// -------------------------------------------------------------- payload

function PayloadAnalysis({ evaluation }: { evaluation: Evaluation }) {
  const { t } = useI18n();
  const { number, percent, interval } = useFormat();
  const methods = useMethods(evaluation);
  const payload = evaluation.payload;
  const [metricName, setMetricName] = useState(payload.metrics[0] ?? "");
  const xLabels = payload.levels.map(String);
  const pointAt = (method: string, bits: number) => payload.per_method.find((entry) => entry.method === method)?.points.find((point) => point.payload_bits === bits);
  const rateOf = (bits: number) => {
    const rates = payload.per_method.flatMap((entry) => entry.points.filter((point) => point.payload_bits === bits).map((point) => point.payload_bps.mean ?? 0));
    return rates.length ? rates.reduce((a, b) => a + b, 0) / rates.length : null;
  };
  return (
    <div className={payload.metrics.length > 0 ? "grid gap-6 xl:grid-cols-2" : "grid gap-6"}>
      <ChartFrame
        title={t("evaluation.payloadBer")}
        hint={t("evaluation.payloadBerHint")}
        table={{
          columns: [t("run.filterMethod"), ...payload.levels.map((bits) => `${bits} ${t("evaluation.bits")} (${number(rateOf(bits), 1)} ${t("evaluation.bps")})`)],
          rows: methods.map((method) => [
            method.short,
            ...payload.levels.map((bits) => {
              const point = pointAt(method.name, bits);
              return point ? `${interval(point.ber)} · ${percent(point.completion_rate, 0)}` : "–";
            }),
          ]),
        }}
      >
        <CurveChart
          xLabels={xLabels}
          xTitle={`${t("evaluation.payload")} (${t("evaluation.bits")})`}
          yTitle="BER"
          format={(value) => number(value, 3)}
          series={methods.map((method) => ({
            label: method.short,
            points: payload.levels.map((bits) => {
              const ber = pointAt(method.name, bits)?.ber;
              return { y: ber?.estimate ?? null, lo: ber?.ci95_low, hi: ber?.ci95_high };
            }),
          }))}
        />
      </ChartFrame>
      {payload.metrics.length > 0 && (
        <ChartFrame
          title={t("evaluation.payloadQuality")}
          hint={t("evaluation.payloadQualityHint")}
          actions={<Picker label={t("evaluation.metric")} value={metricName} onChange={setMetricName} options={payload.metrics.map((name): [string, string] => [name, name])} />}
          table={{
            columns: [t("run.filterMethod"), ...payload.levels.map((bits) => `${bits} ${t("evaluation.bits")}`)],
            rows: methods.map((method) => [method.short, ...payload.levels.map((bits) => interval(pointAt(method.name, bits)?.quality[metricName], 2))]),
          }}
        >
          <CurveChart
            xLabels={xLabels}
            xTitle={`${t("evaluation.payload")} (${t("evaluation.bits")})`}
            yTitle={metricName}
            format={(value) => number(value, 2)}
            series={methods.map((method) => ({
              label: method.short,
              points: payload.levels.map((bits) => {
                const value = pointAt(method.name, bits)?.quality[metricName];
                return { y: value?.estimate ?? null, lo: value?.ci95_low, hi: value?.ci95_high };
              }),
            }))}
          />
        </ChartFrame>
      )}
    </div>
  );
}

// -------------------------------------------------------------- runtime

type RuntimeKey = "encode_seconds" | "decode_seconds" | "total_seconds" | "real_time_factor";

function RuntimeAnalysis({ evaluation }: { evaluation: Evaluation }) {
  const { t } = useI18n();
  const { number } = useFormat();
  const methods = useMethods(evaluation);
  const [measure, setMeasure] = useState<RuntimeKey>("total_seconds");
  const labels: Record<RuntimeKey, string> = {
    encode_seconds: t("evaluation.encode"),
    decode_seconds: t("evaluation.decode"),
    total_seconds: t("evaluation.total"),
    real_time_factor: t("evaluation.rtf"),
  };
  const runtime = evaluation.runtime;
  const digits = (value: number | null | undefined) => (value !== null && value !== undefined && Math.abs(value) < 0.01 ? 4 : 3);
  return (
    <ChartFrame
      title={t("evaluation.runtime")}
      hint={t("evaluation.runtimeHint")}
      actions={<Picker label={t("evaluation.measure")} value={measure} onChange={(value) => setMeasure(value as RuntimeKey)} options={Object.entries(labels) as [string, string][]} />}
      table={{
        columns: [t("run.filterMethod"), ...Object.values(labels).map((label) => `${label}: ${t("run.mean")} [95% CI]`), `${labels.real_time_factor}: ${t("evaluation.median")}`],
        rows: runtime.per_method.map((entry) => [
          methods.find((m) => m.name === entry.method)?.short ?? entry.method,
          ...(Object.keys(labels) as RuntimeKey[]).map((key) => {
            const value = entry[key];
            return value.mean === null ? "–" : `${number(value.mean, digits(value.mean))} [${number(value.ci95_low, digits(value.mean))}, ${number(value.ci95_high, digits(value.mean))}]`;
          }),
          number(entry.real_time_factor.median, 4),
        ]),
      }}
    >
      {!runtime.comparable && (
        <Alert className="mb-4">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription className="text-xs">{t("evaluation.runtimeWarning", { workers: String(runtime.max_workers ?? "?") })}</AlertDescription>
        </Alert>
      )}
      <IntervalChart
        rows={runtime.per_method.map((entry) => ({
          label: methods.find((m) => m.name === entry.method)?.short ?? entry.method,
          fullLabel: entry.method,
          estimate: entry[measure].mean,
          lo: entry[measure].ci95_low,
          hi: entry[measure].ci95_high,
          note: `${t("evaluation.median")}: ${number(entry[measure].median, 4)}`,
        }))}
        axisTitle={labels[measure]}
        reference={measure === "real_time_factor" ? 1 : undefined}
        referenceLabel={measure === "real_time_factor" ? t("evaluation.realTime") : undefined}
        format={(value) => number(value, digits(value))}
      />
    </ChartFrame>
  );
}

// ------------------------------------------------------------ assembly

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="space-y-6">
      <h2 className="eyebrow border-b pb-2 pt-4">{title}</h2>
      {children}
    </div>
  );
}

/** The evaluation block's figures, after the design's own results. */
export function EvaluationSections({ evaluation, type }: { evaluation: Evaluation; type: ExperimentType }) {
  const { t } = useI18n();
  const hasAttacks = evaluation.attacks.length > 0;
  const hasQuality = evaluation.quality.metrics.length > 0 || evaluation.quality.attack_damage.length > 0;
  return (
    <div className="space-y-6">
      <Group title={t("evaluation.groupDescriptive")}>
        <MethodStatistics evaluation={evaluation} />
      </Group>
      {hasAttacks && (
        <Group title={t("evaluation.groupAttacks")}>
          <AttackAnalysis evaluation={evaluation} />
          {evaluation.severity.length > 0 && <Severity evaluation={evaluation} showCurves={type !== "robustness_curve"} />}
        </Group>
      )}
      {hasQuality && (
        <Group title={t("run.quality")}>
          <QualityAnalysis evaluation={evaluation} />
        </Group>
      )}
      {evaluation.payload.available && (
        <Group title={t("evaluation.payload")}>
          <PayloadAnalysis evaluation={evaluation} />
        </Group>
      )}
      <Group title={t("evaluation.runtime")}>
        <RuntimeAnalysis evaluation={evaluation} />
      </Group>
    </div>
  );
}

/** Correlations and the settings that make the figures reproducible. */
export function EvaluationStatistics({ evaluation }: { evaluation: Evaluation }) {
  const { t, tOr } = useI18n();
  const settings = useMemo(() => {
    // Stored as JSONB, which does not keep key order: the order is set here.
    const order = [
      "unit_of_replication", "aggregation", "descriptive_statistics", "confidence", "interval", "bootstrap_resamples",
      "bootstrap_seed", "failure_policy", "recovery_thresholds", "stable_ber", "paired_test_two", "paired_test_many",
      "correction", "effect_size", "correlation", "correlation_test", "permutations", "permutation_seed",
      "correlation_correction", "alpha",
    ];
    const keys = [...order.filter((key) => key in evaluation.settings), ...Object.keys(evaluation.settings).filter((key) => !order.includes(key))];
    return keys.map((key): [string, string] => {
      const value = evaluation.settings[key];
      return [
        tOr(`evaluation.settings.${key}`, key.replaceAll("_", " ")),
        typeof value === "object" && value !== null
          ? Object.entries(value as Record<string, unknown>).map(([name, entry]) => `${name}: ${entry}`).join(" · ")
          : String(value),
      ];
    });
  }, [evaluation.settings, tOr]);
  return (
    <div className="space-y-6">
      <Section title={t("evaluation.correlations")} hint={t("evaluation.correlationsHint")}>
        {evaluation.correlations.length ? <CorrelationTable tests={evaluation.correlations} /> : <p className="text-sm text-muted-foreground">{t("evaluation.noCorrelations")}</p>}
      </Section>
      <Section title={t("evaluation.reproducibility")} hint={t("evaluation.reproducibilityHint")}>
        <KeyValues items={[...settings, [t("evaluation.settings.version"), String(evaluation.version)]]} />
      </Section>
    </div>
  );
}
