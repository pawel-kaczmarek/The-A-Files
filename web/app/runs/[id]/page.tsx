"use client";
import { ResearchView } from "@/components/run/ResearchView";

import Link from "next/link";
import { use, useEffect, useState } from "react";
import { Square } from "lucide-react";

import {
  EmptyState,
  ErrorNotice,
  LoadingLine,
  PageHeader,
  PropertyTag,
  RunStatusBadge,
  formatDuration,
  runDuration,
} from "@/components/common";
import { ProvenanceView, ReportView } from "@/components/run/ProvenanceView";
import { ResultsView } from "@/components/run/ResultsView";
import { StatisticsView } from "@/components/run/StatisticsView";
import { TrialsView } from "@/components/run/TrialsView";
import { BorderBeam } from "@/components/magicui/border-beam";
import { CircularProgress } from "@/components/magicui/circular-progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/lib/api";
import { useAsync, useCatalog, useInterval } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

export default function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { t, date } = useI18n();
  const catalog = useCatalog();
  const run = useAsync(() => api.run(id), [id]);
  const active = run.data?.status === "running" || run.data?.status === "queued";
  const [tick, setTick] = useState(0);
  // The open tab is part of the URL (?tab=statistics), so a view can be linked.
  const [tab, setTab] = useState("results");
  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get("tab");
    if (requested) setTab(requested);
  }, []);
  function changeTab(next: string) {
    setTab(next);
    const url = new URL(window.location.href);
    url.searchParams.set("tab", next);
    window.history.replaceState(null, "", url);
  }
  const summary = useAsync(() => api.summary(id), [id, run.data?.status]);

  useInterval(() => {
    run.reload();
    setTick((value) => value + 1);
  }, 1500, active);

  if (run.error) return <ErrorNotice error={run.error} onRetry={run.reload} />;
  if (!run.data) return <LoadingLine />;
  const current = run.data;
  const design = catalog.data?.designs.find((entry) => entry.type === current.experiment_type);
  const progress = current.total_rows ? (current.completed_rows / current.total_rows) * 100 : 0;
  const hasSummary = summary.data && Object.keys(summary.data.summary ?? {}).length > 0;

  return (
    <>
      <PageHeader
        eyebrow={
          <span className="flex flex-wrap items-center gap-3">
            <Link href={`/experiments/${current.experiment_id}`} className="hover:underline">
              {current.experiment_name}
            </Link>
            <span>{t(`designs.${current.experiment_type}.title`)}</span>
            {design && <PropertyTag property={design.property} />}
            <span className="num normal-case tracking-normal">{t("experiment.version", { version: current.experiment_version })}</span>
          </span>
        }
        title={t("run.title", { number: current.number })}
        subtitle={
          <span className="flex flex-wrap items-center gap-3">
            <RunStatusBadge status={current.status} />
            <span>{date(current.started_at ?? current.created_at)}</span>
            <span className="num">{formatDuration(runDuration(current.started_at, current.finished_at))}</span>
            {current.config.random_seed !== undefined && current.config.random_seed !== null && (
              <span className="num">
                {t("run.seed")} {current.config.random_seed}
              </span>
            )}
          </span>
        }
        actions={
          active ? (
            <Button variant="outline" size="sm" onClick={() => api.cancelRun(id).then(run.reload)}>
              <Square className="h-3.5 w-3.5" /> {t("run.cancel")}
            </Button>
          ) : null
        }
      >
        {active ? (
          <div className="relative mt-5 flex max-w-xl items-center gap-4 rounded-lg border bg-card px-4 py-3">
            <BorderBeam duration={5} size={90} />
            <CircularProgress value={progress} label={t("run.tabs.results")} />
            <div className="min-w-0 flex-1 space-y-1.5">
              <Progress value={progress} className="h-1.5" />
              <div className="num text-xs text-muted-foreground">
                {t("run.progress", { done: current.completed_rows, total: current.total_rows || "?" })}
              </div>
            </div>
          </div>
        ) : (
          current.total_rows > 0 && (
            <div className="mt-5 max-w-xl space-y-1.5">
              <Progress value={progress} className="h-1.5" />
              <div className="num text-xs text-muted-foreground">
                {t("run.progress", { done: current.completed_rows, total: current.total_rows || "?" })}
              </div>
            </div>
          )
        )}
      </PageHeader>

      {current.error && (
        <Alert variant="destructive" className="mb-6">
          <AlertDescription>
            <span className="font-medium">{t("run.failed")}: </span>
            {current.error}
          </AlertDescription>
        </Alert>
      )}

      <Tabs value={tab} onValueChange={changeTab}>
        <TabsList>
          <TabsTrigger value="results">{t("run.tabs.results")}</TabsTrigger>
          <TabsTrigger value="statistics">{t("run.tabs.statistics")}</TabsTrigger>
          <TabsTrigger value="trials">{t("run.tabs.trials")}</TabsTrigger>
          <TabsTrigger value="provenance">{t("run.tabs.provenance")}</TabsTrigger>
          <TabsTrigger value="report">{t("run.tabs.report")}</TabsTrigger>
        </TabsList>
        <TabsContent value="results" className="mt-6">
          {hasSummary ? <ResultsView summary={summary.data!.summary} type={current.experiment_type} /> : <EmptyState>{t("run.noSummary")}</EmptyState>}
        </TabsContent>
        <TabsContent value="statistics" className="mt-6">
          {hasSummary && <ResearchView runId={id} />}
          {hasSummary ? <StatisticsView summary={summary.data!.summary} /> : <EmptyState>{t("run.noSummary")}</EmptyState>}
        </TabsContent>
        <TabsContent value="trials" className="mt-6">
          <TrialsView runId={id} refreshKey={active ? tick : 0} />
        </TabsContent>
        <TabsContent value="provenance" className="mt-6">
          {current.status === "queued" ? <EmptyState>{t("run.noSummary")}</EmptyState> : <ProvenanceView run={current} />}
        </TabsContent>
        <TabsContent value="report" className="mt-6">
          <ReportView run={current} />
        </TabsContent>
      </Tabs>
    </>
  );
}
