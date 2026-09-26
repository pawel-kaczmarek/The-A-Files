"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { use, useState, type ReactNode } from "react";
import { AlertTriangle, Archive, ArchiveRestore, Copy, Loader2, Pencil, Play, Trash2 } from "lucide-react";

import {
  Chip,
  EmptyState,
  ErrorNotice,
  KeyValues,
  LoadingLine,
  PageHeader,
  PropertyTag,
  RunStatusBadge,
  Section,
  Spec,
  formatDuration,
  runDuration,
} from "@/components/common";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";
import { useAsync, useCatalog, useInterval } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import type { ExperimentConfig } from "@/lib/types";

function SpecList({ items }: { items: string[] }) {
  const { t } = useI18n();
  if (!items.length) return <span className="text-muted-foreground">{t("common.none")}</span>;
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item) => (
        <Spec key={item}>{item}</Spec>
      ))}
    </div>
  );
}

export default function ExperimentPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { t, date } = useI18n();
  const router = useRouter();
  const catalog = useCatalog();
  const experiment = useAsync(() => api.experiment(id), [id]);
  const runs = useAsync(() => api.experimentRuns(id), [id]);
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const active = runs.data?.some((run) => run.status === "running" || run.status === "queued") ?? false;
  useInterval(runs.reload, 2000, active);

  async function act(name: string, action: () => Promise<void>) {
    setBusy(name);
    setActionError(null);
    try {
      await action();
    } catch (reason) {
      setActionError((reason as Error).message);
    } finally {
      setBusy(null);
    }
  }

  if (experiment.error) return <ErrorNotice error={experiment.error} onRetry={experiment.reload} />;
  if (!experiment.data) return <LoadingLine />;
  const current = experiment.data;
  const config = current.config as ExperimentConfig;
  const design = catalog.data?.designs.find((entry) => entry.type === current.experiment_type);

  return (
    <>
      <PageHeader
        eyebrow={
          <span className="flex items-center gap-3">
            <span>{t(`designs.${current.experiment_type}.title`)}</span>
            {design && <PropertyTag property={design.property} />}
            <span className="num normal-case tracking-normal">{t("experiment.version", { version: current.version })}</span>
          </span>
        }
        title={current.name}
        subtitle={t(`designs.${current.experiment_type}.question`)}
        actions={
          <>
            <Button variant="outline" size="sm" asChild>
              <Link href={`/experiments/${id}/edit`}>
                <Pencil className="h-3.5 w-3.5" /> {t("common.edit")}
              </Link>
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={busy !== null}
              onClick={() =>
                act("duplicate", async () => {
                  const copy = await api.duplicateExperiment(id);
                  router.push(`/experiments/${copy.id}/edit`);
                })
              }
            >
              <Copy className="h-3.5 w-3.5" /> {t("common.duplicate")}
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={busy !== null}
              onClick={() =>
                act("archive", async () => {
                  await api.archiveExperiment(id, !current.archived);
                  experiment.reload();
                })
              }
            >
              {current.archived ? <ArchiveRestore className="h-3.5 w-3.5" /> : <Archive className="h-3.5 w-3.5" />}
              {current.archived ? t("common.unarchive") : t("common.archive")}
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={busy !== null}
              onClick={() => {
                if (!window.confirm(t("common.confirmDelete"))) return;
                void act("delete", async () => {
                  await api.deleteExperiment(id);
                  router.push("/experiments");
                });
              }}
            >
              <Trash2 className="h-3.5 w-3.5" /> {t("common.delete")}
            </Button>
            <Button
              size="sm"
              disabled={busy !== null || current.problems.length > 0}
              onClick={() =>
                act("run", async () => {
                  const run = await api.startRun(id);
                  router.push(`/runs/${run.id}`);
                })
              }
            >
              {busy === "run" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              {current.run_count ? t("common.runAgain") : t("common.run")}
            </Button>
          </>
        }
      />

      {actionError && <ErrorNotice error={actionError} />}
      {current.problems.length > 0 && (
        <Alert variant="warning" className="mb-6">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <div className="font-medium">{t("experiment.notRunnable")}</div>
            <ul className="mt-1 list-disc pl-4">
              {current.problems.map((problem) => (
                <li key={problem}>{problem}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="space-y-6">
          <Section title={t("experiment.protocol")}>
            <div className="space-y-4 text-sm">
              <div>
                <div className="eyebrow mb-1">{t("experiment.question")}</div>
                <p className="leading-relaxed">{current.research_question || <span className="text-muted-foreground">–</span>}</p>
              </div>
              <div>
                <div className="eyebrow mb-1">{t("experiment.hypothesis")}</div>
                <p className="leading-relaxed">{current.hypothesis || <span className="text-muted-foreground">–</span>}</p>
              </div>
              {current.description && (
                <div>
                  <div className="eyebrow mb-1">{t("experiment.notes")}</div>
                  <p className="whitespace-pre-line leading-relaxed text-muted-foreground">{current.description}</p>
                </div>
              )}
              {current.tags.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {current.tags.map((tag) => (
                    <Chip key={tag}>{tag}</Chip>
                  ))}
                </div>
              )}
              <div className="text-xs text-muted-foreground">
                {t("common.created")} {date(current.created_at)} · {t("common.updated")} {date(current.updated_at)}
              </div>
            </div>
          </Section>

          <Section title={t("experiment.configuration")}>
            <KeyValues
              items={[
                [t("experiment.dataset"), <Spec key="d">{`${config.dataset_id}${config.file_limit ? ` · ${config.file_limit} ${t("common.files")}` : ""}`}</Spec>],
                [t("experiment.methods"), <SpecList key="m" items={config.methods ?? []} />],
                ...(config.method_sweep
                  ? [[t("experiment.sweep"), <Spec key="ms">{`${config.method_sweep.target} · ${config.method_sweep.parameter} = ${config.method_sweep.values.join(", ")}`}</Spec>] as [string, ReactNode]]
                  : []),
                ...(config.attack_sweep
                  ? [[t("experiment.sweep"), <Spec key="as">{`${config.attack_sweep.target} · ${config.attack_sweep.parameter} = ${config.attack_sweep.values.join(", ")}`}</Spec>] as [string, ReactNode]]
                  : []),
                [
                  t("experiment.attacks"),
                  config.attack_preset ? <Spec key="p">{`suite: ${config.attack_preset}`}</Spec> : <SpecList key="a" items={config.attacks ?? []} />,
                ],
                [t("experiment.metrics"), <SpecList key="x" items={config.metrics ?? []} />],
                [t("experiment.payloads"), <span key="pl" className="num">{config.payload_lengths.join(", ")} {t("units.bits")}</span>],
                [t("experiment.repetitions"), <span key="r" className="num">{config.repetitions}</span>],
                [t("experiment.seed"), <span key="s" className="num">{config.random_seed ?? "–"}</span>],
              ]}
            />
          </Section>
        </div>

        <Section title={t("experiment.runs")}>
          {runs.error && <ErrorNotice error={runs.error} onRetry={runs.reload} />}
          {!runs.data ? (
            <LoadingLine />
          ) : !runs.data.length ? (
            <EmptyState>{t("experiment.noRuns")}</EmptyState>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs text-muted-foreground">
                  <th className="py-2 pr-3 font-medium">#</th>
                  <th className="py-2 pr-3 font-medium">{t("common.status")}</th>
                  <th className="py-2 pr-3 font-medium">{t("common.version")}</th>
                  <th className="py-2 pr-3 font-medium">{t("experiment.trials")}</th>
                  <th className="py-2 pr-3 font-medium">{t("experiment.started")}</th>
                  <th className="py-2 text-right font-medium">{t("experiment.duration")}</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.map((run) => (
                  <tr key={run.id} className="border-b last:border-0 hover:bg-accent/40">
                    <td className="num py-2.5 pr-3">
                      <Link href={`/runs/${run.id}`} className="font-medium hover:underline">
                        #{run.number}
                      </Link>
                    </td>
                    <td className="py-2.5 pr-3">
                      <RunStatusBadge status={run.status} />
                    </td>
                    <td className="num py-2.5 pr-3 text-muted-foreground">v{run.experiment_version}</td>
                    <td className="py-2.5 pr-3">
                      <div className="num text-xs">
                        {run.total_rows || run.completed_rows ? `${run.completed_rows}/${run.total_rows || "?"}` : "–"}
                      </div>
                      {run.status === "running" && run.total_rows > 0 && (
                        <Progress value={(run.completed_rows / run.total_rows) * 100} className="mt-1 h-1" />
                      )}
                    </td>
                    <td className="py-2.5 pr-3 text-xs text-muted-foreground">{date(run.started_at ?? run.created_at)}</td>
                    <td className="num py-2.5 text-right text-xs text-muted-foreground">
                      {formatDuration(runDuration(run.started_at, run.finished_at))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>
      </div>
    </>
  );
}
