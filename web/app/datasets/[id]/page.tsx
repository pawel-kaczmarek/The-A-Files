"use client";

import Link from "next/link";
import { use } from "react";

import { DataTable } from "@/components/charts/base";
import { ErrorNotice, KeyValues, LoadingLine, PageHeader, Section, Spec } from "@/components/common";
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";
import { useAsync, useInterval } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

export default function DatasetPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { t, tOr, number, date } = useI18n();
  const dataset = useAsync(() => api.dataset(id), [id]);
  const busy = dataset.data ? ["pending", "downloading", "preparing"].includes(dataset.data.status) : false;
  useInterval(dataset.reload, 1500, busy);

  if (dataset.error) return <ErrorNotice error={dataset.error} onRetry={dataset.reload} />;
  if (!dataset.data) return <LoadingLine />;
  const current = dataset.data;
  const files = current.manifest.files ?? [];

  return (
    <>
      <PageHeader
        eyebrow={
          <Link href="/datasets" className="hover:underline">
            ← {t("nav.datasets")}
          </Link>
        }
        title={current.name}
        subtitle={current.description ?? undefined}
      >
        {busy && (
          <div className="mt-5 max-w-md space-y-1">
            <Progress value={current.progress * 100} className="h-1.5" />
            <div className="text-xs text-muted-foreground">
              {tOr(`datasetStatus.${current.status}`, current.status)} {current.stage ? `· ${current.stage}` : ""}
            </div>
          </div>
        )}
      </PageHeader>
      {current.error && <ErrorNotice error={current.error} />}
      <div className="grid gap-6 lg:grid-cols-2">
        <Section title={t("common.details")}>
          <KeyValues
            items={[
              [t("common.type"), tOr(`datasetKinds.${current.kind}`, current.kind)],
              [t("common.status"), tOr(`datasetStatus.${current.status}`, current.status)],
              [t("datasets.domain"), current.domain ? tOr(`domains.${current.domain}`, current.domain) : "–"],
              [t("datasets.language"), current.language ?? "–"],
              [t("common.license"), current.license ?? "–"],
              [t("common.files"), <span key="f" className="num">{current.file_count}</span>],
              [t("datasets.duration"), current.total_duration_seconds ? `${number(current.total_duration_seconds / 60, 1)} ${t("common.minutes")}` : "–"],
              [t("datasets.rate"), current.sample_rate ? `${current.sample_rate} Hz` : "–"],
              [t("datasets.speakers"), current.manifest.speakers ?? "–"],
              [t("datasets.path"), <Spec key="p" className="break-all">{current.path ?? "–"}</Spec>],
              [t("common.created"), date(current.created_at)],
              ["ID", <Spec key="id">{`library:${current.id}`}</Spec>],
            ]}
          />
        </Section>
        <Section title={t("datasets.manifest")}>
          <div className="space-y-4 text-sm">
            {current.citation && (
              <div>
                <div className="eyebrow mb-1">{t("common.citation")}</div>
                <p className="text-xs italic leading-relaxed">{current.citation}</p>
              </div>
            )}
            {Object.keys(current.rule ?? {}).length > 0 && (
              <div>
                <div className="eyebrow mb-1">{t("datasets.rule")}</div>
                <pre className="spec overflow-auto rounded-md bg-muted p-3">{JSON.stringify(current.rule, null, 2)}</pre>
              </div>
            )}
            {current.manifest.source_sha256 && (
              <div>
                <div className="eyebrow mb-1">{t("datasets.archiveDigest")}</div>
                <Spec className="break-all">{String(current.manifest.source_sha256)}</Spec>
              </div>
            )}
          </div>
        </Section>
      </div>
      {files.length > 0 && (
        <Section title={`${t("datasets.filesTitle")} (${files.length})`} className="mt-6">
          <DataTable
            table={{
              columns: [t("common.file"), t("datasets.speaker"), t("datasets.rate"), t("datasets.duration"), "Channels", "PCM bits / subtype", "Category", "Source", "SHA-256"],
              rows: files.map((file) => [
                <span key="f" className="text-xs" title={file.source ?? ""}>
                  {file.file}
                </span>,
                file.speaker ?? "–",
                file.sample_rate,
                `${number(file.duration_seconds, 2)} s`,
                file.channels ?? "—",
                `${file.bit_depth ?? "—"} / ${file.subtype ?? "—"}`,
                file.category ?? current.domain ?? "unknown",
                file.source ?? "—",
                <span key="h" className="font-mono text-[11px] text-muted-foreground">
                  {file.sha256 ? `${file.sha256.slice(0, 16)}…` : "–"}
                </span>,
              ]),
            }}
          />
        </Section>
      )}
    </>
  );
}
