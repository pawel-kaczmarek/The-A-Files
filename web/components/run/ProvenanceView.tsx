"use client";

import { useState } from "react";
import { Download, FileCode2, FileJson, FileSpreadsheet, FileText } from "lucide-react";

import { DataTable } from "@/components/charts/base";
import { ErrorNotice, KeyValues, LoadingLine, Section, Spec } from "@/components/common";
import { Button } from "@/components/ui/button";
import { api, urls } from "@/lib/api";
import { useAsync } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import type { RunInfo } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Manifest {
  taf_version?: string;
  source?: { commit: string; dirty: boolean } | null;
  python?: string;
  platform?: string;
  packages?: Record<string, string | null>;
  ffmpeg?: string | null;
  random_seed?: number;
  resolved_attacks?: string[];
  timing_reliable?: boolean;
  created_at?: string;
  inputs?: { name: string; sha256: string | null; sample_rate: number; duration_seconds: number }[];
}

export function ProvenanceView({ run }: { run: RunInfo }) {
  const { t, date, number } = useI18n();
  const manifest = useAsync(() => api.manifest(run.id) as Promise<Manifest>, [run.id, run.status]);
  if (manifest.error) return <ErrorNotice error={manifest.error} onRetry={manifest.reload} />;
  if (!manifest.data) return <LoadingLine />;
  const data = manifest.data;
  const packages = Object.entries(data.packages ?? {}).filter(([, version]) => version);

  return (
    <div className="space-y-6">
      <Section title={t("run.manifest")}>
        <KeyValues
          items={[
            ["The A-Files", <span key="v" className="num">{data.taf_version}</span>],
            [
              t("run.source"),
              data.source ? (
                <span key="s">
                  <Spec>{data.source.commit.slice(0, 12)}</Spec>
                  {data.source.dirty && <span className="ml-2 text-xs" style={{ color: "var(--status-serious)" }}>{t("run.dirty")}</span>}
                </span>
              ) : (
                "–"
              ),
            ],
            [t("run.seed"), <span key="seed" className="num">{data.random_seed}</span>],
            ["Python", data.python],
            ["Platform", <span key="p" className="text-xs">{data.platform}</span>],
            ["FFmpeg", <span key="f" className="text-xs">{data.ffmpeg?.split(" Copyright")[0] ?? "–"}</span>],
            [t("run.timing"), data.timing_reliable ? t("common.yes") : t("common.no")],
            [t("common.created"), date(data.created_at)],
          ]}
        />
      </Section>
      <Section title={t("run.software")}>
        <div className="flex flex-wrap gap-2">
          {packages.map(([name, version]) => (
            <Spec key={name}>
              {name} {version}
            </Spec>
          ))}
        </div>
      </Section>
      {(data.resolved_attacks ?? []).length > 0 && (
        <Section title={t("experiment.attacks")}>
          <div className="flex flex-wrap gap-1.5">
            {data.resolved_attacks!.map((spec) => (
              <Spec key={spec}>{spec}</Spec>
            ))}
          </div>
        </Section>
      )}
      <Section title={`${t("run.inputs")} (${data.inputs?.length ?? 0})`}>
        <DataTable
          table={{
            columns: [t("common.file"), "Hz", t("datasets.duration"), "SHA-256"],
            rows: (data.inputs ?? []).map((input) => [
              input.name,
              input.sample_rate,
              `${number(input.duration_seconds, 2)} s`,
              <span key="h" className="font-mono text-[11px] text-muted-foreground">
                {input.sha256?.slice(0, 16)}…
              </span>,
            ]),
          }}
        />
      </Section>
      <Section title={t("experiment.configuration")}>
        <pre className="spec max-h-96 overflow-auto rounded-md bg-muted p-3 leading-relaxed">{JSON.stringify(run.config, null, 2)}</pre>
      </Section>
    </div>
  );
}

export function ReportView({ run }: { run: RunInfo }) {
  const { t } = useI18n();
  const [format, setFormat] = useState<"md" | "tex">("md");
  const completed = run.status === "completed";
  const report = useAsync(() => (completed ? api.report(run.id, format) : Promise.resolve("")), [run.id, format, completed]);

  const exports = [
    { href: urls.exportCsv(run.id), label: t("run.csv"), icon: FileSpreadsheet },
    { href: urls.exportSummaryCsv(run.id), label: t("run.summaryCsv"), icon: FileSpreadsheet },
    { href: urls.config(run.id), label: t("run.configJson"), icon: FileJson },
    { href: urls.manifest(run.id), label: t("run.manifestJson"), icon: FileJson },
    { href: urls.report(run.id, "tex"), label: t("run.latex"), icon: FileCode2, needsCompleted: true },
    { href: urls.report(run.id, "md"), label: t("run.markdown"), icon: FileText, needsCompleted: true },
  ];

  return (
    <div className="space-y-6">
      <Section title={t("run.exports")}>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {exports.map(({ href, label, icon: Icon, needsCompleted }) => {
            const disabled = needsCompleted && !completed;
            return (
              <a
                key={label}
                href={disabled ? undefined : href}
                download
                aria-disabled={disabled}
                className={cn(
                  "flex items-center gap-3 rounded-md border px-3 py-2.5 text-sm",
                  disabled ? "pointer-events-none opacity-50" : "hover:bg-accent"
                )}
              >
                <Icon className="h-4 w-4 text-muted-foreground" />
                <span className="flex-1">{label}</span>
                <Download className="h-3.5 w-3.5 text-muted-foreground" />
              </a>
            );
          })}
        </div>
      </Section>
      <Section
        title={t("run.reportPreview")}
        actions={
          <div className="inline-flex rounded-md border p-0.5">
            {(["md", "tex"] as const).map((entry) => (
              <Button key={entry} type="button" size="sm" variant={format === entry ? "secondary" : "ghost"} className="h-7" onClick={() => setFormat(entry)}>
                {entry === "md" ? "Markdown" : "LaTeX"}
              </Button>
            ))}
          </div>
        }
      >
        {!completed ? (
          <p className="text-sm text-muted-foreground">{t("run.reportOnlyCompleted")}</p>
        ) : report.error ? (
          <ErrorNotice error={report.error} />
        ) : report.loading ? (
          <LoadingLine />
        ) : (
          <pre className="spec max-h-[32rem] overflow-auto whitespace-pre-wrap rounded-md bg-muted p-4 leading-relaxed">{report.data}</pre>
        )}
      </Section>
    </div>
  );
}
