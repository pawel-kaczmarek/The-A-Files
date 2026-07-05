"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Eye, FlaskConical, Loader2, ShieldAlert } from "lucide-react";

import { PageHeader } from "@/components/layout/PageHeader";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { api, API_BASE } from "@/lib/api";
import { experimentMeta } from "@/lib/experimentLabels";
import type {
  ExperimentConfig,
  ExperimentPlan,
  ExperimentType,
  JobSummary,
  ResultRow,
  ScenarioSummary,
} from "@/lib/types";
import { validateConfig } from "@/lib/validators";

import { AdvancedSettings } from "./AdvancedSettings";
import { AttackSelector } from "./AttackSelector";
import { CsvExportButton } from "./CsvExportButton";
import { DatasetSelector } from "./DatasetSelector";
import { ExperimentPreview } from "./ExperimentPreview";
import { ExperimentResultsTable } from "./ExperimentResultsTable";
import { ExperimentSummaryCards } from "./ExperimentSummaryCards";
import { MethodSelector } from "./MethodSelector";
import { MetricSelector } from "./MetricSelector";
import { PayloadLengthSelector } from "./PayloadLengthSelector";
import { RobustnessMatrix, SummarySectionTable } from "./SummaryTables";
import { useCatalog } from "./useCatalog";

export interface ExperimentLayoutProps {
  type: ExperimentType;
  requireMetrics?: boolean;
  requireAttacks?: boolean;
  showMetrics?: boolean;
  showAttacks?: boolean;
  showNotes?: boolean;
  showConfigPreview?: boolean;
  defaultPayloads?: number[];
  /** Scenario-specific option cards writing into config.advanced_options. */
  extraCards?: (context: {
    advanced: Record<string, unknown>;
    setAdvanced: (patch: Record<string, unknown>) => void;
  }) => ReactNode;
}

const SUMMARY_SECTIONS: { key: string; title: string }[] = [
  { key: "comparison", title: "Method ranking" },
  { key: "robustness_ranking", title: "Robustness ranking (attacked rows only)" },
  { key: "per_attack", title: "Per attack" },
  { key: "quality_ranking", title: "Quality ranking" },
  { key: "capacity_by_method", title: "Capacity by method" },
  { key: "by_method", title: "By method" },
  { key: "by_method_payload", title: "By method and payload length" },
];

export function ExperimentLayout(props: ExperimentLayoutProps) {
  const meta = experimentMeta(props.type);
  const showMetrics = props.showMetrics ?? true;
  const showAttacks = props.showAttacks ?? true;

  const { catalog, error: catalogError, refresh: refreshCatalog } = useCatalog();

  const [config, setConfig] = useState<ExperimentConfig>({
    experiment_type: props.type,
    name: "",
    dataset_id: "example",
    file_limit: 2,
    methods: [],
    metrics: [],
    attacks: [],
    payload_lengths: props.defaultPayloads ?? [16],
    repetitions: 1,
    random_seed: 42,
    max_workers: 2,
    advanced_options: {},
  });
  const patch = useCallback(
    (update: Partial<ExperimentConfig>) => setConfig((previous) => ({ ...previous, ...update })),
    []
  );

  const [plan, setPlan] = useState<ExperimentPlan | null>(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [problems, setProblems] = useState<string[]>([]);
  const [runError, setRunError] = useState<string | null>(null);
  const [job, setJob] = useState<JobSummary | null>(null);
  const [rows, setRows] = useState<ResultRow[]>([]);
  const [summary, setSummary] = useState<ScenarioSummary | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => () => sourceRef.current?.close(), []);

  const clientProblems = useMemo(
    () =>
      validateConfig(config, {
        requireMetrics: props.requireMetrics,
        requireAttacks: props.requireAttacks,
      }),
    [config, props.requireMetrics, props.requireAttacks]
  );

  async function preview() {
    setProblems(clientProblems);
    if (clientProblems.length > 0) return;
    setPlanLoading(true);
    setRunError(null);
    try {
      setPlan(await api.previewExperiment({ ...config, name: config.name || "preview" }));
    } catch (err) {
      setRunError((err as Error).message);
    } finally {
      setPlanLoading(false);
    }
  }

  async function run() {
    setProblems(clientProblems);
    if (clientProblems.length > 0) return;
    setRunError(null);
    setRows([]);
    setSummary(null);
    try {
      const submitted = await api.runExperiment({
        ...config,
        name: config.name || `${meta.label.toLowerCase().replaceAll(" ", "-")}`,
      });
      setJob(submitted);
      subscribe(submitted.experiment_id);
    } catch (err) {
      setRunError((err as Error).message);
    }
  }

  function subscribe(id: string) {
    sourceRef.current?.close();
    const source = new EventSource(api.eventsUrl(id));
    sourceRef.current = source;
    const finish = () => {
      api.summary(id).then((response) => setSummary(response.summary)).catch(() => {});
      api.experiment(id).then(setJob).catch(() => {});
      source.close();
    };
    source.addEventListener("snapshot", (event) => {
      const detail = JSON.parse((event as MessageEvent).data);
      setJob(detail);
      setRows(detail.rows ?? []);
      if (detail.status === "completed" || detail.status === "failed") finish();
    });
    source.addEventListener("row", (event) => {
      const row = JSON.parse((event as MessageEvent).data) as ResultRow;
      setRows((previous) => [...previous, row]);
    });
    source.addEventListener("status", (event) => {
      setJob(JSON.parse((event as MessageEvent).data) as JobSummary);
    });
    source.addEventListener("done", (event) => {
      setJob(JSON.parse((event as MessageEvent).data) as JobSummary);
      finish();
    });
    source.onerror = () => finish();
  }

  const running = job?.status === "running" || job?.status === "pending";
  const progress =
    job && job.total_tasks > 0 ? Math.min(100, (rows.length / job.total_tasks) * 100) : 0;

  if (catalogError) {
    return (
      <div className="space-y-6">
        <PageHeader title={meta.label} description={meta.description} />
        <Alert variant="destructive">
          <ShieldAlert className="h-4 w-4" />
          <AlertTitle>API unreachable</AlertTitle>
          <AlertDescription>
            Could not reach the taf API at <code>{API_BASE}</code>. Start it with{" "}
            <code>taf-api</code> or <code>uvicorn taf.api.main:app</code>. ({catalogError})
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title={meta.label}
        description={
          <>
            <span className="font-medium text-foreground">{meta.question}</span> {meta.description}
          </>
        }
      />

      {!catalog ? (
        <div className="space-y-4">
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>General</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
              <div className="space-y-1">
                <Label htmlFor="name">Experiment name</Label>
                <Input
                  id="name"
                  placeholder={`e.g. ${meta.label.toLowerCase().replaceAll(" ", "-")}-vctk`}
                  value={config.name}
                  onChange={(event) => patch({ name: event.target.value })}
                />
              </div>
              {props.showNotes && (
                <div className="space-y-1">
                  <Label htmlFor="description">Description</Label>
                  <Input
                    id="description"
                    placeholder="optional short description"
                    value={config.description ?? ""}
                    onChange={(event) => patch({ description: event.target.value || null })}
                  />
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Dataset</CardTitle>
              <CardDescription>
                Packaged corpora, your uploaded sounds, or (Research Experiment) a local directory
                path on the backend machine.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <DatasetSelector
                datasets={catalog.datasets}
                value={config.dataset_id ?? null}
                onChange={(datasetId) => patch({ dataset_id: datasetId, dataset_path: null })}
                fileLimit={config.file_limit ?? null}
                onFileLimitChange={(limit) => patch({ file_limit: limit })}
                onDatasetsChanged={refreshCatalog}
              />
              {props.showConfigPreview && (
                <div className="mt-4 space-y-1">
                  <Label htmlFor="datasetPath">Local dataset path (optional, overrides dataset)</Label>
                  <Input
                    id="datasetPath"
                    placeholder="e.g. C:\\data\\my-corpus (WAV/FLAC files)"
                    value={config.dataset_path ?? ""}
                    onChange={(event) => patch({ dataset_path: event.target.value || null })}
                  />
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Methods ({config.methods.length} selected)</CardTitle>
              <CardDescription>Steganography algorithms to evaluate.</CardDescription>
            </CardHeader>
            <CardContent>
              <MethodSelector
                methods={catalog.methods}
                selected={config.methods}
                onChange={(methods) => patch({ methods })}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Payload lengths ({config.payload_lengths.length} selected)</CardTitle>
              <CardDescription>Message sizes in bits embedded into each file.</CardDescription>
            </CardHeader>
            <CardContent>
              <PayloadLengthSelector
                value={config.payload_lengths}
                onChange={(payload_lengths) => patch({ payload_lengths })}
              />
            </CardContent>
          </Card>

          {showMetrics && (
            <Card>
              <CardHeader>
                <CardTitle>
                  Metrics ({(config.metrics ?? []).length} selected
                  {props.requireMetrics ? ", required" : ", optional"})
                </CardTitle>
                <CardDescription>
                  Signal comparison measures between the original and processed audio.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <MetricSelector
                  metrics={catalog.metrics}
                  selected={config.metrics ?? []}
                  onChange={(metrics) => patch({ metrics })}
                />
              </CardContent>
            </Card>
          )}

          {showAttacks && (
            <Card>
              <CardHeader>
                <CardTitle>
                  Attacks ({(config.attacks ?? []).length} selected
                  {props.requireAttacks ? ", required" : ", optional"})
                </CardTitle>
                <CardDescription>
                  Each attack is evaluated separately, next to a no-attack baseline.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <AttackSelector
                  attacks={catalog.attacks}
                  selected={config.attacks ?? []}
                  onChange={(attacks) => patch({ attacks })}
                />
              </CardContent>
            </Card>
          )}

          {props.extraCards?.({
            advanced: config.advanced_options ?? {},
            setAdvanced: (update) =>
              patch({ advanced_options: { ...(config.advanced_options ?? {}), ...update } }),
          })}

          <AdvancedSettings config={config} onChange={patch} />

          {props.showNotes && (
            <Card>
              <CardHeader>
                <CardTitle>Notes</CardTitle>
              </CardHeader>
              <CardContent>
                <Textarea
                  placeholder="Free-form notes stored with the experiment config."
                  value={config.notes ?? ""}
                  onChange={(event) => patch({ notes: event.target.value || null })}
                />
              </CardContent>
            </Card>
          )}

          {props.showConfigPreview && (
            <Card>
              <CardHeader>
                <CardTitle>Config preview</CardTitle>
                <CardDescription>
                  The exact JSON sent to <code>POST /api/experiments/run</code> — reusable from
                  Python via <code>taf.experiments.run_experiment</code>.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="max-h-80 overflow-auto rounded-md bg-muted p-3 text-xs">
                  {JSON.stringify(config, null, 2)}
                </pre>
              </CardContent>
            </Card>
          )}

          {problems.length > 0 && (
            <Alert variant="destructive">
              <ShieldAlert className="h-4 w-4" />
              <AlertTitle>Fix before running</AlertTitle>
              <AlertDescription>
                <ul className="list-disc space-y-1 pl-4">
                  {problems.map((problem, index) => (
                    <li key={index}>{problem}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}
          {runError && (
            <Alert variant="destructive">
              <ShieldAlert className="h-4 w-4" />
              <AlertTitle>Request failed</AlertTitle>
              <AlertDescription>{runError}</AlertDescription>
            </Alert>
          )}

          <div className="flex flex-wrap items-center gap-2">
            <Button variant="outline" onClick={preview} disabled={planLoading || running}>
              {planLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Eye className="h-4 w-4" />}
              Preview experiment plan
            </Button>
            <Button size="lg" onClick={run} disabled={running}>
              {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <FlaskConical className="h-4 w-4" />}
              Run experiment
            </Button>
            {job && (
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
            )}
          </div>

          {plan && (
            <Card>
              <CardHeader>
                <CardTitle>Experiment plan</CardTitle>
              </CardHeader>
              <CardContent>
                <ExperimentPreview plan={plan} />
              </CardContent>
            </Card>
          )}

          {job && (
            <Card>
              <CardHeader>
                <CardTitle>
                  Progress — {rows.length}/{job.total_tasks || "?"} rows
                </CardTitle>
                {job.error && <CardDescription className="text-destructive">{job.error}</CardDescription>}
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
                <CardDescription>Methods × attacks, averaged over all rows.</CardDescription>
              </CardHeader>
              <CardContent>
                <RobustnessMatrix cells={summary.matrix as never} />
              </CardContent>
            </Card>
          )}

          {summary &&
            SUMMARY_SECTIONS.filter(
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

          {job && (
            <Card>
              <CardHeader className="flex-row items-start justify-between space-y-0">
                <div>
                  <CardTitle>Results</CardTitle>
                  <CardDescription>
                    One row per file × method × payload × repetition × attack variant.
                  </CardDescription>
                </div>
                <CsvExportButton experimentId={job.experiment_id} disabled={rows.length === 0} />
              </CardHeader>
              <CardContent>
                <ExperimentResultsTable rows={rows} />
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
