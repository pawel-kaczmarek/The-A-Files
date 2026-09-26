"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowRight, Database, FlaskConical } from "lucide-react";

import {
  EmptyState,
  ErrorNotice,
  LoadingLine,
  PROPERTY_ORDER,
  PROPERTY_SLOT,
  RunStatusBadge,
  Section,
  StatTile,
  formatDuration,
  runDuration,
} from "@/components/common";
import { EvaluationModel } from "@/components/evaluation-model";
import { AnimatedShinyText } from "@/components/magicui/animated-shiny-text";
import { BlurFade } from "@/components/magicui/blur-fade";
import { BorderBeam } from "@/components/magicui/border-beam";
import { CircularProgress } from "@/components/magicui/circular-progress";
import { DotPattern } from "@/components/magicui/dot-pattern";
import { NumberTicker } from "@/components/magicui/number-ticker";
import { ShimmerButton } from "@/components/magicui/shimmer-button";
import { WordRotate } from "@/components/magicui/word-rotate";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useAsync, useCatalog, useInterval } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import type { Property } from "@/lib/types";

export default function DashboardPage() {
  const { t, date } = useI18n();
  const router = useRouter();
  const stats = useAsync(() => api.stats(), []);
  const experiments = useAsync(() => api.experiments(true), []);
  const runs = useAsync(() => api.runs(12), []);
  const catalog = useCatalog();
  const active = runs.data?.some((run) => run.status === "running" || run.status === "queued") ?? false;
  useInterval(() => {
    runs.reload();
    stats.reload();
  }, 3000, active);

  const propertyOf = (type: string): Property =>
    catalog.data?.designs.find((design) => design.type === type)?.property ?? "multi_criteria";

  const coverage = PROPERTY_ORDER.map((property) => {
    const ofProperty = (experiments.data ?? []).filter((experiment) => propertyOf(experiment.experiment_type) === property);
    return {
      property,
      experiments: ofProperty.length,
      runs: ofProperty.reduce((sum, experiment) => sum + (experiment.latest_run?.status === "completed" ? 1 : 0), 0),
      completed: ofProperty.filter((experiment) => experiment.latest_run?.status === "completed").length,
    };
  });
  const maxCoverage = Math.max(1, ...coverage.map((entry) => entry.experiments));
  const activeRuns = (runs.data ?? []).filter((run) => run.status === "running" || run.status === "queued");
  const measured = PROPERTY_ORDER.filter((property) => property !== "multi_criteria");
  const ticker = (value: number | undefined, delay = 0) =>
    value === undefined ? "–" : <NumberTicker value={value} delay={delay} />;

  return (
    <>
      <BlurFade>
        <section className="relative mb-6 overflow-hidden rounded-xl border bg-card">
          <DotPattern className="opacity-70 [mask-image:radial-gradient(ellipse_at_top_right,black,transparent_65%)]" />
          <div className="relative flex flex-wrap items-end justify-between gap-8 px-8 py-9">
            <div className="max-w-2xl">
              <div className="inline-flex items-center rounded-full border bg-background/80 px-3 py-1 text-xs backdrop-blur">
                <AnimatedShinyText className="num">
                  {t("dashboard.heroPill", {
                    designs: catalog.data?.designs.length ?? "–",
                    methods: stats.data?.methods ?? "–",
                    attacks: stats.data?.attacks ?? "–",
                    metrics: stats.data?.metrics ?? "–",
                  })}
                </AnimatedShinyText>
              </div>
              <h1 className="mt-4 text-3xl font-semibold leading-tight tracking-tight sm:text-4xl">
                {t("dashboard.heroLead")}{" "}
                <WordRotate
                  words={measured.map((property) => ({
                    key: property,
                    content: (
                      <span
                        className="underline decoration-[3px] underline-offset-[7px]"
                        style={{ textDecorationColor: PROPERTY_SLOT[property] }}
                      >
                        {t(`properties.${property}.name`).toLowerCase()}
                      </span>
                    ),
                  }))}
                />
                <br />
                <span className="text-muted-foreground">{t("dashboard.heroTail")}</span>
              </h1>
              <p className="mt-4 max-w-xl text-sm leading-relaxed text-muted-foreground">{t("dashboard.subtitle")}</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="outline" asChild className="bg-background/80">
                <Link href="/datasets">
                  <Database className="h-4 w-4" /> {t("dashboard.prepareData")}
                </Link>
              </Button>
              <ShimmerButton onClick={() => router.push("/experiments/new")}>
                <FlaskConical className="h-4 w-4" /> {t("dashboard.newExperiment")}
              </ShimmerButton>
            </div>
          </div>
        </section>
      </BlurFade>

      {stats.error && <ErrorNotice error={stats.error} onRetry={stats.reload} />}

      <BlurFade delay={0.08} className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label={t("dashboard.experiments")} value={ticker(stats.data?.experiments)} />
        <StatTile label={t("dashboard.completedRuns")} value={ticker(stats.data?.runs.completed, 0.05)} />
        <StatTile label={t("dashboard.datasets")} value={ticker(stats.data?.datasets, 0.1)} />
        <StatTile
          label={t("dashboard.components")}
          value={
            stats.data ? (
              <span className="inline-flex items-baseline gap-1.5">
                {ticker(stats.data.methods, 0.15)}
                <span className="text-muted-foreground">·</span>
                {ticker(stats.data.attacks, 0.2)}
                <span className="text-muted-foreground">·</span>
                {ticker(stats.data.metrics, 0.25)}
              </span>
            ) : (
              "–"
            )
          }
        />
      </BlurFade>

      {activeRuns.length > 0 && (
        <BlurFade className="mt-6">
          <section className="relative rounded-lg border bg-card">
            <BorderBeam duration={7} size={110} />
            <div className="border-b px-5 py-3.5">
              <h2 className="text-sm font-semibold">{t("dashboard.live")}</h2>
              <p className="mt-0.5 text-xs text-muted-foreground">{t("dashboard.liveHint")}</p>
            </div>
            <ul className="divide-y">
              {activeRuns.map((run) => (
                <li key={run.id}>
                  <Link href={`/runs/${run.id}`} className="flex items-center gap-4 px-5 py-3 hover:bg-accent/40">
                    <CircularProgress
                      className="h-11 w-11 shrink-0"
                      value={run.total_rows ? (run.completed_rows / run.total_rows) * 100 : 0}
                      label={run.experiment_name}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium">{run.experiment_name}</div>
                      <div className="text-xs text-muted-foreground">
                        {t(`designs.${run.experiment_type}.title`)} · #{run.number}
                      </div>
                    </div>
                    <RunStatusBadge status={run.status} />
                    <ArrowRight className="h-4 w-4 text-muted-foreground" />
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        </BlurFade>
      )}

      <BlurFade delay={0.14} className="mt-6">
        <Section title={t("model.title")} hint={t("model.hint")}>
          <EvaluationModel />
        </Section>
      </BlurFade>

      <BlurFade inView className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        <Section title={t("dashboard.coverage")} hint={t("dashboard.coverageHint")}>
          <ul className="space-y-4">
            {coverage.map((entry) => (
              <li key={entry.property}>
                <div className="flex items-baseline justify-between gap-3 text-sm">
                  <span className="flex items-center gap-2 font-medium">
                    <span className="h-2 w-2 rounded-full" style={{ background: PROPERTY_SLOT[entry.property] }} />
                    {t(`properties.${entry.property}.name`)}
                  </span>
                  <span className="num text-xs text-muted-foreground">
                    {t("dashboard.experimentsCount", { count: entry.experiments })} · {entry.completed} ✓
                  </span>
                </div>
                <div className="mt-1.5 h-1.5 w-full rounded-full bg-muted">
                  <div
                    className="h-1.5 rounded-full"
                    style={{ width: `${(entry.experiments / maxCoverage) * 100}%`, background: PROPERTY_SLOT[entry.property] }}
                  />
                </div>
                <p className="mt-1 text-xs text-muted-foreground">{t(`properties.${entry.property}.short`)}</p>
              </li>
            ))}
          </ul>
        </Section>

        <Section
          title={t("dashboard.recentRuns")}
          actions={
            <Link href="/runs" className="text-xs text-muted-foreground underline-offset-2 hover:underline">
              {t("nav.runs")} →
            </Link>
          }
        >
          {runs.loading && !runs.data ? (
            <LoadingLine />
          ) : !runs.data?.length ? (
            <EmptyState>{t("dashboard.noRuns")}</EmptyState>
          ) : (
            <table className="w-full text-sm">
              <tbody>
                {runs.data.map((run) => (
                  <tr key={run.id} className="border-b last:border-0">
                    <td className="py-2 pr-3">
                      <Link href={`/runs/${run.id}`} className="font-medium hover:underline">
                        {run.experiment_name}
                      </Link>
                      <div className="text-xs text-muted-foreground">
                        {t(`designs.${run.experiment_type}.title`)} · #{run.number}
                      </div>
                    </td>
                    <td className="py-2 pr-3">
                      <RunStatusBadge status={run.status} />
                    </td>
                    <td className="num py-2 pr-3 text-right text-xs text-muted-foreground">
                      {run.total_rows || run.completed_rows ? `${run.completed_rows}/${run.total_rows || "?"}` : "–"}
                    </td>
                    <td className="num py-2 text-right text-xs text-muted-foreground">
                      {run.finished_at ? formatDuration(runDuration(run.started_at, run.finished_at)) : date(run.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>
      </BlurFade>
    </>
  );
}
