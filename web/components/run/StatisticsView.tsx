"use client";

import { CDDiagram } from "@/components/charts/CDDiagram";
import { ChartFrame, DataTable } from "@/components/charts/base";
import { EmptyState } from "@/components/common";
import { useMethodAbbreviation } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import type { Comparison, Summary } from "@/lib/types";

const TITLES: Record<string, string> = {
  ber_baseline: "run.berClean",
  ber_attacked: "run.berAttacked",
  ber_sweep: "run.curve",
};

function isComparison(value: unknown): value is Comparison {
  return typeof value === "object" && value !== null && "available" in value;
}

/** Flatten ``summary.statistics`` into titled comparisons. */
// PostgreSQL's JSONB does not preserve key order, so the order in which
// comparisons are shown is decided here: the headline comparisons first, then
// groups, sweep values in sweep order.
const KEY_ORDER = ["ber_baseline", "ber_attacked", "ber_sweep", "metrics", "per_value", "per_attack"];

export function comparisonsOf(summary: Summary, t: (key: string) => string): { key: string; title: string; comparison: Comparison }[] {
  const entries: { key: string; title: string; comparison: Comparison }[] = [];
  const statistics = Object.entries(summary.statistics ?? {}).sort(
    ([a], [b]) => (KEY_ORDER.indexOf(a) + 1 || 99) - (KEY_ORDER.indexOf(b) + 1 || 99)
  );
  const sweepValues = ((summary.sweep as { values?: unknown[] } | undefined)?.values ?? []).map(String);
  for (const [key, value] of statistics) {
    if (isComparison(value)) {
      entries.push({ key, title: TITLES[key] ? t(TITLES[key]) : key.replaceAll("_", " "), comparison: value });
    } else if (value && typeof value === "object") {
      const nestedEntries = Object.entries(value).sort(([a], [b]) =>
        key === "per_value" ? sweepValues.indexOf(a) - sweepValues.indexOf(b) : a.localeCompare(b)
      );
      for (const [name, nested] of nestedEntries) {
        if (isComparison(nested)) {
          const group = key === "metrics" ? t("editor.steps.measures") : key === "per_value" ? t("editor.conditions.parameter") : key === "per_attack" ? t("editor.conditions.attack") : key;
          entries.push({ key: `${key}.${name}`, title: `${group}: ${name}`, comparison: nested });
        }
      }
    }
  }
  return entries;
}

function pValue(value: number | undefined, locale: string): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return "–";
  if (value < 0.001) return "< 0.001";
  return value.toLocaleString(locale === "pl" ? "pl-PL" : "en-US", { maximumFractionDigits: 3, minimumFractionDigits: 3 });
}

/** "= 0.012" or "< 0.001", so the relation reads correctly after "p". */
function pRelation(value: number | undefined, locale: string): string {
  const text = pValue(value, locale);
  return text.startsWith("<") || text === "–" ? text : `= ${text}`;
}

export function ComparisonBlock({ title, comparison }: { title: string; comparison: Comparison }) {
  const { t, number, locale } = useI18n();
  const abbreviate = useMethodAbbreviation();
  if (!comparison.available) {
    return (
      <div className="rounded-lg border bg-card px-5 py-4">
        <div className="text-sm font-semibold">{title}</div>
        <p className="mt-1 text-xs text-muted-foreground">{t("run.notAvailable", { reason: comparison.reason ?? "" })}</p>
      </div>
    );
  }
  const omnibus = comparison.omnibus!;
  const ranks = Object.fromEntries(Object.entries(comparison.mean_ranks ?? {}).sort((a, b) => a[1] - b[1]));
  const test = omnibus.test === "friedman" ? "Friedman χ²" : "Wilcoxon signed-rank";
  return (
    <ChartFrame
      title={title}
      hint={
        <span className="num">
          {t("run.omnibus", { test, p: pRelation(omnibus.p_value, locale) })} ·{" "}
          <span style={{ color: omnibus.significant ? "hsl(var(--success))" : undefined }}>
            {omnibus.significant ? t("run.significant") : t("run.notSignificant")}
          </span>{" "}
          · {t("run.blocks", { count: comparison.blocks ?? 0 })} · {t("run.comparisonHint")}
        </span>
      }
      table={{
        columns: [t("run.meanRanks"), "rank"],
        rows: Object.entries(ranks).map(([name, rank]) => [name, number(rank, 2)]),
      }}
    >
      <div className="space-y-6">
        {Object.keys(ranks).length > 2 && comparison.critical_difference !== undefined ? (
          <div>
            <div className="mb-1 text-xs font-medium">{t("run.cdDiagram")}</div>
            <p className="mb-2 text-xs text-muted-foreground">{t("run.cdHint", { cd: number(comparison.critical_difference, 2) })}</p>
            <CDDiagram ranks={Object.fromEntries(Object.entries(ranks).map(([name, rank]) => [abbreviate(name), rank]))} cd={comparison.critical_difference} cdLabel={`CD = ${number(comparison.critical_difference, 2)}`} />
          </div>
        ) : (
          <DataTable table={{ columns: [t("run.meanRanks"), "rank"], rows: Object.entries(ranks).map(([name, rank]) => [name, number(rank, 2)]) }} />
        )}
        {(comparison.pairwise ?? []).length > 0 && (
          <div>
            <div className="mb-1.5 text-xs font-medium">{t("run.pairwise")}</div>
            <DataTable
              table={{
                columns: [t("run.pair"), t("run.better"), t("run.medianDiff"), t("run.effect"), t("run.pHolm")],
                rows: (comparison.pairwise ?? []).map((pair) => [
                  <span key="p" className="text-xs">
                    <span title={pair.a}>{abbreviate(pair.a)}</span> <span className="text-muted-foreground">vs</span>{" "}
                    <span title={pair.b}>{abbreviate(pair.b)}</span>
                  </span>,
                  <span key="b" className="text-xs" title={pair.better ?? undefined}>
                    {pair.better ? abbreviate(pair.better) : "–"}
                  </span>,
                  number(pair.median_difference, 4),
                  number(pair.rank_biserial, 2),
                  <span key="h" className={pair.significant ? "font-semibold" : "text-muted-foreground"}>
                    {pValue(pair.p_holm, locale)}
                    {pair.significant ? " *" : ""}
                  </span>,
                ]),
              }}
            />
          </div>
        )}
      </div>
    </ChartFrame>
  );
}

export function StatisticsView({ summary }: { summary: Summary }) {
  const { t } = useI18n();
  const comparisons = comparisonsOf(summary, t);
  if (!comparisons.length) return <EmptyState>{t("run.noSummary")}</EmptyState>;
  return (
    <div className="space-y-6">
      {comparisons.map((entry) => (
        <ComparisonBlock key={entry.key} title={entry.title} comparison={entry.comparison} />
      ))}
    </div>
  );
}
