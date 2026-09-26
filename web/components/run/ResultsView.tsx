"use client";

import { useState } from "react";

import { CurveChart } from "@/components/charts/CurveChart";
import { Heatmap } from "@/components/charts/Heatmap";
import { IntervalChart } from "@/components/charts/IntervalChart";
import { TradeoffChart } from "@/components/charts/TradeoffChart";
import { ChartFrame, DataTable, seriesColor, shorten } from "@/components/charts/base";
import { Chip, EstimateText, Section, StatTile } from "@/components/common";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { useCatalog, useMethodAbbreviation } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import type { Estimate, ExperimentType, GroupStats, ParetoResult, Summary } from "@/lib/types";

type MethodStats = GroupStats & { method: string };
type Cell = GroupStats & { method: string; attack: string | null };

// ---------------------------------------------------------------- overview

function Overview({ summary }: { summary: Summary }) {
  const { t, percent, tOr } = useI18n();
  const overall = summary.overall;
  if (!overall) return null;
  const failures = Object.entries(overall.failures ?? {});
  const exact = (overall as unknown as { decode_success?: Estimate }).decode_success;
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatTile label={t("run.completion")} value={percent(overall.completion_rate, 1)} hint={`${overall.rows} ${t("common.rows")}`} />
      <StatTile label={`${t("run.ber")} (${t("common.ci")})`} value={<EstimateText value={overall.ber_imputed} />} />
      <StatTile label={t("run.exact")} value={<EstimateText value={exact} percent />} />
      <StatTile
        label={t("run.failures")}
        value={overall.error_rows}
        hint={failures.length ? failures.map(([kind, count]) => `${tOr(`failureKinds.${kind}`, kind)}: ${count}`).join(" · ") : "–"}
      />
    </div>
  );
}

// ---------------------------------------------------------------- figures

function RobustnessMatrix({ cells }: { cells: Cell[] }) {
  const { t, number } = useI18n();
  const methods = [...new Set(cells.map((cell) => cell.method))];
  const attacks = [...new Set(cells.map((cell) => cell.attack ?? ""))].sort((a, b) => (a === "" ? -1 : b === "" ? 1 : 0));
  const lookup = new Map(cells.map((cell) => [`${cell.method}|${cell.attack ?? ""}`, cell]));
  const label = (attack: string) => (attack === "" ? t("run.baseline") : attack);
  return (
    <ChartFrame
      title={t("run.robustnessMatrix")}
      hint={t("run.robustnessMatrixHint")}
      table={{
        columns: [t("run.filterMethod"), ...attacks.map(label)],
        rows: methods.map((method) => [method, ...attacks.map((attack) => number(lookup.get(`${method}|${attack}`)?.avg_ber_imputed ?? null, 3))]),
      }}
    >
      <Heatmap
        rows={methods}
        columns={attacks.map(label)}
        value={(method, column) => {
          const attack = column === t("run.baseline") ? "" : column;
          return lookup.get(`${method}|${attack}`)?.avg_ber_imputed ?? null;
        }}
        max={0.5}
        format={(value) => number(value, 2)}
        scaleLabel="BER"
        detail={(method, column) => {
          const cell = lookup.get(`${method}|${column === t("run.baseline") ? "" : column}`);
          if (!cell) return [];
          return [
            { label: "BER", value: `${number(cell.ber_imputed.estimate, 3)} [${number(cell.ber_imputed.ci95_low, 3)}, ${number(cell.ber_imputed.ci95_high, 3)}]` },
            { label: t("run.completion"), value: `${Math.round((cell.completion_rate ?? 0) * 100)}%` },
            { label: t("common.rows"), value: String(cell.rows) },
          ];
        }}
      />
    </ChartFrame>
  );
}

interface CurveSummary {
  method: string;
  baseline: Estimate;
  points: { value: number | string; ber: Estimate }[];
  breakdown: { status: "never" | "always" | "crossed"; value: number | string | null };
  mean_ber: Estimate;
}

function Curves({ summary }: { summary: Summary }) {
  const { t, number } = useI18n();
  const abbreviate = useMethodAbbreviation();
  const curves = summary.curves as CurveSummary[];
  const sweep = summary.sweep as { target: string; parameter: string; values: (number | string)[]; usable_ber_threshold: number };
  const xLabels = [t("run.baseline"), ...sweep.values.map(String)];
  const breakdown = (curve: CurveSummary) =>
    curve.breakdown.status === "never"
      ? t("run.breakdownNever")
      : curve.breakdown.status === "always"
        ? t("run.breakdownAlways")
        : typeof curve.breakdown.value === "number"
          ? number(curve.breakdown.value, 2)
          : String(curve.breakdown.value);
  return (
    <>
      <ChartFrame
        title={`${t("run.curve")}: ${sweep.target} · ${sweep.parameter}`}
        hint={t("run.curveHint", { threshold: number(sweep.usable_ber_threshold, 2) })}
        table={{
          columns: [t("run.filterMethod"), ...xLabels, t("run.breakdown")],
          rows: curves.map((curve) => [
            curve.method,
            number(curve.baseline.estimate, 3),
            ...curve.points.map((point) => number(point.ber.estimate, 3)),
            breakdown(curve),
          ]),
        }}
      >
        <CurveChart
          xLabels={xLabels}
          xTitle={`${sweep.target} ${sweep.parameter}`}
          yTitle="BER"
          threshold={sweep.usable_ber_threshold}
          format={(value) => number(value, 2)}
          series={curves.map((curve) => ({
            label: abbreviate(curve.method),
            points: [
              { y: curve.baseline.estimate, lo: curve.baseline.ci95_low, hi: curve.baseline.ci95_high },
              ...curve.points.map((point) => ({ y: point.ber.estimate, lo: point.ber.ci95_low, hi: point.ber.ci95_high })),
            ],
          }))}
        />
      </ChartFrame>
      <Section title={t("run.breakdown")}>
        <DataTable
          table={{
            columns: [t("run.filterMethod"), `${t("run.ber")} (${t("common.ci")})`, `${t("run.breakdown")} (${sweep.parameter})`],
            rows: curves.map((curve, index) => [
              <span key="m" className="flex items-center gap-2">
                <span className="h-0.5 w-4 shrink-0 rounded" style={{ background: seriesColor(index) }} />
                {curve.method}
              </span>,
              <EstimateText key="e" value={curve.mean_ber} />,
              breakdown(curve),
            ]),
          }}
        />
      </Section>
    </>
  );
}

interface TradeoffPointSummary {
  value: number | string;
  label: string;
  spec: string;
  ber_baseline: Estimate;
  ber_attacked?: Estimate;
  metrics: Record<string, Estimate>;
}

/** "Signal-to-Noise Ratio (SNR) [x]" -> "SNR [x]", from the catalogue's abbreviations. */
function useMetricAbbreviation() {
  const catalog = useCatalog();
  return (label: string) => {
    const match = label.match(/^(.*?)( \[[^\]]+\])?$/);
    const base = match?.[1] ?? label;
    const metric = catalog.data?.metrics.find((entry) => entry.label === base);
    return metric ? `${metric.abbreviation}${match?.[2] ?? ""}` : label;
  };
}

function Tradeoff({ summary }: { summary: Summary }) {
  const { t, number } = useI18n();
  const abbreviate = useMetricAbbreviation();
  const points = summary.points as TradeoffPointSummary[];
  const references = (summary.references ?? []) as TradeoffPointSummary[];
  const metrics = summary.ranked_metrics as Record<string, boolean>;
  const sweep = summary.sweep as { target: string; parameter: string };
  const pareto = summary.pareto as ParetoResult;
  const metricNames = Object.keys(metrics);
  const [metric, setMetric] = useState(metricNames[0] ?? "");
  const hasAttack = points.some((point) => point.ber_attacked);
  const [measure, setMeasure] = useState<"ber_attacked" | "ber_baseline">(hasAttack ? "ber_attacked" : "ber_baseline");
  const toPoint = (point: TradeoffPointSummary, label: string) => {
    const x = point.metrics[metric];
    const y = point[measure] ?? point.ber_baseline;
    return { label, x: x?.estimate ?? null, xLo: x?.ci95_low, xHi: x?.ci95_high, y: y?.estimate ?? null, yLo: y?.ci95_low, yHi: y?.ci95_high };
  };
  return (
    <ChartFrame
      title={`${t("run.tradeoff")}: ${sweep.target} · ${sweep.parameter}`}
      hint={t("run.tradeoffHint")}
      actions={
        <div className="flex items-center gap-2">
          <Label className="text-xs text-muted-foreground">{t("run.xMetric")}</Label>
          <Select value={metric} onChange={(event) => setMetric(event.target.value)} className="h-7 w-44 text-xs">
            {metricNames.map((name) => (
              <option key={name} value={name}>
                {abbreviate(name)}
              </option>
            ))}
          </Select>
          {hasAttack && (
            <Select value={measure} onChange={(event) => setMeasure(event.target.value as typeof measure)} className="h-7 w-40 text-xs">
              <option value="ber_attacked">{t("run.berAttacked")}</option>
              <option value="ber_baseline">{t("run.berClean")}</option>
            </Select>
          )}
        </div>
      }
      table={{
        columns: [sweep.parameter, t("run.berClean"), ...(hasAttack ? [t("run.berAttacked")] : []), ...metricNames.map(abbreviate), t("run.pareto")],
        rows: points.map((point) => [
          String(point.value),
          <EstimateText key="b" value={point.ber_baseline} />,
          ...(hasAttack ? [<EstimateText key="a" value={point.ber_attacked} />] : []),
          ...metricNames.map((name) => <EstimateText key={name} value={point.metrics[name]} digits={2} />),
          pareto.front.includes(String(point.value)) ? "✓" : "",
        ]),
      }}
    >
      <TradeoffChart
        points={points.map((point) => ({ ...toPoint(point, `${sweep.parameter} = ${point.value}`), optimal: pareto.front.includes(String(point.value)) }))}
        references={references.map((point) => toPoint(point, point.spec))}
        xTitle={abbreviate(metric)}
        yTitle={measure === "ber_attacked" ? t("run.berAttacked") : t("run.berClean")}
        xHigherIsBetter={metrics[metric] ?? null}
        seriesLabel={sweep.target}
        optimalLabel={t("run.paretoOptimal")}
        dominatedLabel={t("run.dominated")}
        outsideLabel={t("run.outsideRange")}
        betterWord={t("common.better")}
        formatX={(value) => number(value, 1)}
        formatY={(value) => number(value, 2)}
      />
    </ChartFrame>
  );
}

interface CapacityEntry {
  method: string;
  capacity_bits_median: number | null;
  capacity_bps_median: number | null;
  capacity_bps_mean: number | null;
  capacity_bps_ci95_low: number | null;
  capacity_bps_ci95_high: number | null;
  max_passing_payload: number | null;
  censored_files: number;
  over_capacity_rows: number;
  files: number;
}

function Capacity({ summary }: { summary: Summary }) {
  const { t, number } = useI18n();
  const abbreviate = useMethodAbbreviation();
  const entries = summary.capacity_by_method as CapacityEntry[];
  return (
    <ChartFrame
      title={t("run.capacity")}
      hint={t("run.capacityHint")}
      table={{
        columns: [
          t("run.filterMethod"),
          `${t("run.median")} (${t("units.bits")})`,
          `${t("run.median")} (${t("units.bps")})`,
          `${t("run.mean")} (${t("units.bps")}, ${t("common.ci")})`,
          t("run.censored"),
          t("run.overCapacity"),
        ],
        rows: entries.map((entry) => [
          entry.method,
          number(entry.capacity_bits_median, 0),
          number(entry.capacity_bps_median, 1),
          `${number(entry.capacity_bps_mean, 1)} [${number(entry.capacity_bps_ci95_low, 1)}, ${number(entry.capacity_bps_ci95_high, 1)}]`,
          `${entry.censored_files}/${entry.files}`,
          entry.over_capacity_rows,
        ]),
      }}
    >
      <IntervalChart
        rows={entries.map((entry) => ({
          label: abbreviate(entry.method),
          fullLabel: entry.method,
          estimate: entry.capacity_bps_mean,
          lo: entry.capacity_bps_ci95_low,
          hi: entry.capacity_bps_ci95_high,
          note: entry.censored_files ? t("run.censoredNote", { count: entry.censored_files, total: entry.files }) : undefined,
          censored: entry.censored_files > 0,
        }))}
        axisTitle={t("units.bps")}
        format={(value) => number(value, 0)}
      />
    </ChartFrame>
  );
}

interface DetectabilityEntry {
  method: string;
  method_name: string;
  payload_length: number;
  status: string;
  error?: string;
  accuracy: number;
  accuracy_ci95_low: number;
  accuracy_ci95_high: number;
  p_value: number;
  significantly_detectable: boolean;
  false_positive_rate: number;
  false_negative_rate: number;
  test_size: number;
}

function Detectability({ summary }: { summary: Summary }) {
  const { t, number } = useI18n();
  const abbreviate = useMethodAbbreviation();
  const entries = (summary.detectability as DetectabilityEntry[]).filter((entry) => entry.status === "ok");
  const failed = (summary.detectability as DetectabilityEntry[]).filter((entry) => entry.status !== "ok");
  return (
    <ChartFrame
      title={t("run.detectability")}
      hint={t("run.detectabilityHint")}
      table={{
        columns: [t("run.filterMethod"), t("run.bits"), t("run.accuracy"), "FPR", "FNR", "p", "n"],
        rows: entries.map((entry) => [
          entry.method,
          entry.payload_length,
          `${number(entry.accuracy, 3)} [${number(entry.accuracy_ci95_low, 3)}, ${number(entry.accuracy_ci95_high, 3)}]`,
          number(entry.false_positive_rate, 3),
          number(entry.false_negative_rate, 3),
          entry.p_value < 0.001 ? "< 0.001" : number(entry.p_value, 3),
          entry.test_size,
        ]),
      }}
    >
      <IntervalChart
        rows={entries.map((entry) => ({
          label: shorten(abbreviate(entry.method), 44, ` · ${entry.payload_length} ${t("units.bits")}`),
          fullLabel: `${entry.method} · ${entry.payload_length} ${t("units.bits")}`,
          estimate: entry.accuracy,
          lo: entry.accuracy_ci95_low,
          hi: entry.accuracy_ci95_high,
          note: `p = ${entry.p_value < 0.001 ? "< 0.001" : number(entry.p_value, 3)} · ${entry.significantly_detectable ? t("run.significant") : t("run.notSignificant")}`,
        }))}
        axisTitle={t("run.accuracy")}
        domain={[0.3, 1]}
        reference={0.5}
        referenceLabel={t("run.chance")}
        format={(value) => number(value, 2)}
      />
      {failed.length > 0 && (
        <ul className="mt-3 space-y-1 text-xs text-destructive">
          {failed.map((entry) => (
            <li key={`${entry.method_name}-${entry.payload_length}`}>
              {entry.method_name} · {entry.payload_length}: {entry.error}
            </li>
          ))}
        </ul>
      )}
    </ChartFrame>
  );
}

function Quality({ summary }: { summary: Summary }) {
  const { t, number } = useI18n();
  const abbreviate = useMethodAbbreviation();
  const ranking = summary.quality_ranking as { method: string; mean_rank: number | null; metrics_ranked: number; avg_metrics: Record<string, number> }[];
  const metricNames = [...new Set(ranking.flatMap((entry) => Object.keys(entry.avg_metrics)))];
  return (
    <ChartFrame
      title={t("run.quality")}
      hint={t("run.qualityHint")}
      table={{
        columns: [t("run.filterMethod"), t("run.meanRanks"), ...metricNames],
        rows: ranking.map((entry) => [entry.method, number(entry.mean_rank, 2), ...metricNames.map((name) => number(entry.avg_metrics[name], 2))]),
      }}
    >
      <IntervalChart
        rows={ranking.map((entry) => ({ label: abbreviate(entry.method), fullLabel: entry.method, estimate: entry.mean_rank }))}
        axisTitle={`${t("run.meanRanks")} (${t("run.bestIsOne")})`}
        domain={[1, Math.max(2, ranking.length)]}
        format={(value) => number(value, 1)}
      />
    </ChartFrame>
  );
}

function MethodTable({ entries }: { entries: MethodStats[] }) {
  const { t, percent } = useI18n();
  const abbreviate = useMethodAbbreviation();
  const metricNames = [...new Set(entries.flatMap((entry) => Object.keys(entry.avg_metrics ?? {})))].slice(0, 4);
  return (
    <ChartFrame
      title={t("run.methods")}
      hint={`BER: ${t("common.ci")}`}
      table={{
        columns: [t("run.filterMethod"), "BER", t("run.completion"), t("run.exact"), ...metricNames],
        rows: entries.map((entry) => [
          entry.method,
          <EstimateText key="b" value={entry.ber_imputed} />,
          percent(entry.completion_rate, 1),
          percent(entry.perfect_extraction_rate, 1),
          ...metricNames.map((name) => (entry.avg_metrics[name] ?? null)?.toFixed?.(2) ?? "–"),
        ]),
      }}
    >
      <IntervalChart
        rows={entries.map((entry) => ({
          label: abbreviate(entry.method),
          fullLabel: entry.method,
          estimate: entry.ber_imputed?.estimate ?? null,
          lo: entry.ber_imputed?.ci95_low,
          hi: entry.ber_imputed?.ci95_high,
          note: `${t("run.completion")}: ${percent(entry.completion_rate, 1)}`,
        }))}
        axisTitle="BER"
        domain={[0, 0.5]}
        format={(value) => value.toFixed(2)}
      />
    </ChartFrame>
  );
}

function Pareto({ pareto, parameter }: { pareto: ParetoResult; parameter?: string }) {
  const { t } = useI18n();
  const dominated = Object.entries(pareto.dominated_by ?? {});
  return (
    <Section title={t("run.pareto")} hint={t("run.paretoHint")}>
      <div className="space-y-3 text-sm">
        <div className="flex flex-wrap gap-1.5">
          {pareto.front.map((method) => (
            <Chip key={method} className="border-primary/40 text-foreground">
              ✓ {parameter ? `${parameter} = ${method}` : method}
            </Chip>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          {t("run.objectives")}: {pareto.objectives.join(" · ") || "–"}
        </p>
        {pareto.excluded_objectives.length > 0 && (
          <p className="text-xs text-muted-foreground">
            {t("run.notCompared")}: {pareto.excluded_objectives.join(" · ")}
          </p>
        )}
        {dominated.length > 0 && (
          <ul className="space-y-1 text-xs text-muted-foreground">
            {dominated.map(([method, by]) => (
              <li key={method}>
                <span className="text-foreground">{parameter ? `${parameter} = ${method}` : method}</span> —{" "}
                {t("run.dominatedBy", { methods: by.map((entry) => (parameter ? `${parameter} = ${entry}` : entry)).join(", ") })}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------- view

export function ResultsView({ summary, type }: { summary: Summary; type: ExperimentType }) {
  const matrix = (summary.matrix ?? summary.by_method_attack) as Cell[] | undefined;
  const byMethod = (summary.robustness_ranking && type === "attack_robustness"
    ? summary.robustness_ranking
    : summary.by_method) as MethodStats[] | undefined;
  return (
    <div className="space-y-6">
      <Overview summary={summary} />
      {type === "robustness_curve" && summary.curves ? <Curves summary={summary} /> : null}
      {type === "tradeoff_curve" && summary.points ? <Tradeoff summary={summary} /> : null}
      {summary.capacity_by_method ? <Capacity summary={summary} /> : null}
      {summary.detectability ? <Detectability summary={summary} /> : null}
      {summary.quality_ranking ? <Quality summary={summary} /> : null}
      {matrix && matrix.length > 0 && type !== "robustness_curve" && new Set(matrix.map((cell) => cell.attack)).size > 1 ? <RobustnessMatrix cells={matrix} /> : null}
      {byMethod && byMethod.length > 0 && type !== "robustness_curve" ? <MethodTable entries={byMethod} /> : null}
      {summary.pareto && summary.pareto.front ? (
        <Pareto
          pareto={summary.pareto}
          parameter={type === "tradeoff_curve" ? (summary.sweep as { parameter: string }).parameter : undefined}
        />
      ) : null}
    </div>
  );
}
