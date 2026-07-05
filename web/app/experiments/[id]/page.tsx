"use client";

import { use, useEffect, useRef, useState } from "react";
import { FileJson } from "lucide-react";

import { PageHeader } from "@/components/layout/PageHeader";
import { CsvExportButton } from "@/components/experiments/CsvExportButton";
import { ExperimentResultsTable } from "@/components/experiments/ExperimentResultsTable";
import { ExperimentSummaryCards } from "@/components/experiments/ExperimentSummaryCards";
import { RobustnessMatrix, SummarySectionTable } from "@/components/experiments/SummaryTables";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { experimentLabel } from "@/lib/experimentLabels";
import type { JobSummary, ResultRow, ScenarioSummary } from "@/lib/types";

const SECTIONS: { key: string; title: string }[] = [
  { key: "comparison", title: "Method ranking" },
  { key: "robustness_ranking", title: "Robustness ranking (attacked rows only)" },
  { key: "per_attack", title: "Per attack" },
  { key: "quality_ranking", title: "Quality ranking" },
  { key: "capacity_by_method", title: "Capacity by method" },
  { key: "by_method", title: "By method" },
  { key: "by_method_payload", title: "By method and payload length" },
];

export default function ExperimentDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const [job, setJob] = useState<JobSummary | null>(null);
  const [rows, setRows] = useState<ResultRow[]>([]);
  const [summary, setSummary] = useState<ScenarioSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    let cancelled = false;

    function finish() {
      api.summary(id).then((r) => !cancelled && setSummary(r.summary)).catch(() => {});
      api.experiment(id).then((j) => !cancelled && setJob(j)).catch(() => {});
      sourceRef.current?.close();
    }

    api
      .experiment(id)
      .then((summaryRow) => {
        if (cancelled) return;
        setJob(summaryRow);
        api.results(id).then((r) => !cancelled && setRows(r)).catch(() => {});
        if (summaryRow.status === "completed" || summaryRow.status === "failed") {
          api.summary(id).then((r) => !cancelled && setSummary(r.summary)).catch(() => {});
          return;
        }
        const source = new EventSource(api.eventsUrl(id));
        sourceRef.current = source;
        source.addEventListener("row", (event) => {
          setRows((previous) => [...previous, JSON.parse((event as MessageEvent).data)]);
        });
        source.addEventListener("status", (event) => {
          setJob(JSON.parse((event as MessageEvent).data));
        });
        source.addEventListener("done", () => finish());
        source.onerror = () => finish();
      })
      .catch((err: Error) => setError(err.message));

    return () => {
      cancelled = true;
      sourceRef.current?.close();
    };
  }, [id]);

  if (error) return <p className="text-sm text-destructive">{error}</p>;
  if (!job) return <Skeleton className="h-64 w-full" />;

  const progress = job.total_tasks > 0 ? Math.min(100, (rows.length / job.total_tasks) * 100) : 0;

  return (
    <div className="space-y-6">
      <PageHeader
        title={job.name}
        description={
          <>
            {experimentLabel(job.experiment_type)} · {job.dataset} · {job.methods.length} method(s)
            · payloads [{job.payload_lengths.join(", ")}] · {job.repetitions} repetition(s)
            {job.attacks.length > 0 && <> · attacks: {job.attacks.join(", ")}</>}
          </>
        }
        actions={
          <div className="flex items-center gap-2">
            <Badge
              variant={
                job.status === "completed"
                  ? "success"
                  : job.status === "failed"
                    ? "destructive"
                    : "default"
              }
            >
              {job.status}
            </Badge>
            <CsvExportButton experimentId={job.experiment_id} disabled={rows.length === 0} />
            <Button asChild variant="outline" size="sm">
              <a href={api.exportConfigUrl(job.experiment_id)} download>
                <FileJson className="h-4 w-4" /> Config
              </a>
            </Button>
          </div>
        }
      />

      {job.error && (
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="text-destructive">Run failed</CardTitle>
            <CardDescription>{job.error}</CardDescription>
          </CardHeader>
        </Card>
      )}

      {job.status === "running" && (
        <Card>
          <CardHeader>
            <CardTitle>
              Progress — {rows.length}/{job.total_tasks || "?"} rows
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={progress} />
          </CardContent>
        </Card>
      )}

      {summary && <ExperimentSummaryCards summary={summary} />}

      {summary && Array.isArray(summary.matrix) && (
        <Card>
          <CardHeader>
            <CardTitle>Robustness matrix</CardTitle>
          </CardHeader>
          <CardContent>
            <RobustnessMatrix cells={summary.matrix as never} />
          </CardContent>
        </Card>
      )}

      {summary &&
        SECTIONS.filter(
          ({ key }) => Array.isArray(summary[key]) && (summary[key] as unknown[]).length > 0
        ).map(({ key, title }) => (
          <Card key={key}>
            <CardHeader>
              <CardTitle>{title}</CardTitle>
            </CardHeader>
            <CardContent>
              <SummarySectionTable rows={summary[key] as Record<string, unknown>[]} />
            </CardContent>
          </Card>
        ))}

      <Card>
        <CardHeader>
          <CardTitle>Results</CardTitle>
          <CardDescription>
            One row per file × method × payload × repetition × attack variant.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ExperimentResultsTable rows={rows} />
        </CardContent>
      </Card>
    </div>
  );
}
