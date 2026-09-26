"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { AlertTriangle, Plus } from "lucide-react";

import {
  Chip,
  EmptyState,
  ErrorNotice,
  LoadingLine,
  PageHeader,
  PROPERTY_ORDER,
  PropertyTag,
  RunStatusBadge,
} from "@/components/common";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { api } from "@/lib/api";
import { useAsync, useCatalog } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import type { Property } from "@/lib/types";

export default function ExperimentsPage() {
  const { t, date } = useI18n();
  const [archived, setArchived] = useState(false);
  const [property, setProperty] = useState<Property | "">("");
  const [search, setSearch] = useState("");
  const experiments = useAsync(() => api.experiments(archived), [archived]);
  const catalog = useCatalog();

  const propertyOf = (type: string): Property =>
    catalog.data?.designs.find((design) => design.type === type)?.property ?? "multi_criteria";

  const shown = useMemo(
    () =>
      (experiments.data ?? []).filter(
        (experiment) =>
          (!property || propertyOf(experiment.experiment_type) === property) &&
          (!search ||
            `${experiment.name} ${experiment.research_question ?? ""} ${experiment.tags.join(" ")}`
              .toLowerCase()
              .includes(search.toLowerCase()))
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [experiments.data, property, search, catalog.data]
  );

  return (
    <>
      <PageHeader
        title={t("experiments.title")}
        subtitle={t("experiments.subtitle")}
        actions={
          <Button asChild>
            <Link href="/experiments/new">
              <Plus className="h-4 w-4" /> {t("experiments.new")}
            </Link>
          </Button>
        }
      />

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <Input
          placeholder={t("common.search")}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          className="w-64"
        />
        <Select value={property} onChange={(event) => setProperty(event.target.value as Property | "")} className="w-56">
          <option value="">{t("experiments.filterAll")}</option>
          {PROPERTY_ORDER.map((entry) => (
            <option key={entry} value={entry}>
              {t(`properties.${entry}.name`)}
            </option>
          ))}
        </Select>
        <label className="flex items-center gap-2 text-sm text-muted-foreground">
          <input type="checkbox" checked={archived} onChange={(event) => setArchived(event.target.checked)} />
          {t("experiments.showArchived")}
        </label>
      </div>

      {experiments.error && <ErrorNotice error={experiments.error} onRetry={experiments.reload} />}
      {experiments.loading && !experiments.data ? (
        <LoadingLine />
      ) : !shown.length ? (
        <EmptyState
          action={
            <Button asChild size="sm">
              <Link href="/experiments/new">{t("experiments.new")}</Link>
            </Button>
          }
        >
          {t("experiments.empty")}
        </EmptyState>
      ) : (
        <div className="overflow-x-auto rounded-lg border bg-card">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-xs text-muted-foreground">
                <th className="px-4 py-2.5 font-medium">{t("common.name")}</th>
                <th className="px-4 py-2.5 font-medium">{t("experiments.design")}</th>
                <th className="px-4 py-2.5 text-right font-medium">{t("common.version")}</th>
                <th className="px-4 py-2.5 text-right font-medium">{t("experiments.runs")}</th>
                <th className="px-4 py-2.5 font-medium">{t("experiments.latest")}</th>
                <th className="px-4 py-2.5 text-right font-medium">{t("common.updated")}</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((experiment) => (
                <tr key={experiment.id} className="border-b align-top last:border-0 hover:bg-accent/40">
                  <td className="max-w-md px-4 py-3">
                    <Link href={`/experiments/${experiment.id}`} className="font-medium hover:underline">
                      {experiment.name}
                    </Link>
                    {experiment.research_question && (
                      <p className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">{experiment.research_question}</p>
                    )}
                    <div className="mt-1 flex flex-wrap gap-1">
                      {experiment.archived && <Chip>{t("common.archive")}</Chip>}
                      {experiment.problems.length > 0 && (
                        <Chip title={experiment.problems.join("\n")} className="gap-1">
                          <AlertTriangle className="h-3 w-3" style={{ color: "var(--status-serious)" }} />
                          {t("experiments.problems")}
                        </Chip>
                      )}
                      {experiment.tags.map((tag) => (
                        <Chip key={tag}>{tag}</Chip>
                      ))}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div>{t(`designs.${experiment.experiment_type}.title`)}</div>
                    <PropertyTag property={propertyOf(experiment.experiment_type)} />
                  </td>
                  <td className="num px-4 py-3 text-right text-muted-foreground">v{experiment.version}</td>
                  <td className="num px-4 py-3 text-right">{experiment.run_count}</td>
                  <td className="px-4 py-3">
                    {experiment.latest_run ? (
                      <Link href={`/runs/${experiment.latest_run.id}`} className="hover:underline">
                        <RunStatusBadge status={experiment.latest_run.status} />
                      </Link>
                    ) : (
                      <span className="text-xs text-muted-foreground">{t("experiments.never")}</span>
                    )}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-right text-xs text-muted-foreground">
                    {date(experiment.updated_at)}
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
