"use client";

import Link from "next/link";

import { EmptyState, ErrorNotice, LoadingLine, PageHeader, RunStatusBadge, formatDuration, runDuration } from "@/components/common";
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";
import { useAsync, useInterval } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

export default function RunsPage() {
  const { t, date } = useI18n();
  const runs = useAsync(() => api.runs(300), []);
  const active = runs.data?.some((run) => run.status === "running" || run.status === "queued") ?? false;
  useInterval(runs.reload, 2000, active);

  return (
    <>
      <PageHeader title={t("runs.title")} subtitle={t("runs.subtitle")} />
      {runs.error && <ErrorNotice error={runs.error} onRetry={runs.reload} />}
      {!runs.data ? (
        <LoadingLine />
      ) : !runs.data.length ? (
        <EmptyState>{t("dashboard.noRuns")}</EmptyState>
      ) : (
        <div className="overflow-x-auto rounded-lg border bg-card">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-xs text-muted-foreground">
                <th className="px-4 py-2.5 font-medium">{t("runs.experiment")}</th>
                <th className="px-4 py-2.5 font-medium">#</th>
                <th className="px-4 py-2.5 font-medium">{t("common.status")}</th>
                <th className="px-4 py-2.5 font-medium">{t("experiment.trials")}</th>
                <th className="px-4 py-2.5 font-medium">{t("experiment.started")}</th>
                <th className="px-4 py-2.5 text-right font-medium">{t("experiment.duration")}</th>
              </tr>
            </thead>
            <tbody>
              {runs.data.map((run) => (
                <tr key={run.id} className="border-b last:border-0 hover:bg-accent/40">
                  <td className="px-4 py-2.5">
                    <Link href={`/runs/${run.id}`} className="font-medium hover:underline">
                      {run.experiment_name}
                    </Link>
                    <div className="text-xs text-muted-foreground">
                      {t(`designs.${run.experiment_type}.title`)} · v{run.experiment_version}
                    </div>
                  </td>
                  <td className="num px-4 py-2.5">{run.number}</td>
                  <td className="px-4 py-2.5">
                    <RunStatusBadge status={run.status} />
                  </td>
                  <td className="w-40 px-4 py-2.5">
                    <div className="num text-xs">
                      {run.total_rows || run.completed_rows ? `${run.completed_rows}/${run.total_rows || "?"}` : "–"}
                    </div>
                    {run.status === "running" && run.total_rows > 0 && (
                      <Progress value={(run.completed_rows / run.total_rows) * 100} className="mt-1 h-1" />
                    )}
                  </td>
                  <td className="px-4 py-2.5 text-xs text-muted-foreground">{date(run.started_at ?? run.created_at)}</td>
                  <td className="num px-4 py-2.5 text-right text-xs text-muted-foreground">
                    {formatDuration(runDuration(run.started_at, run.finished_at))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
