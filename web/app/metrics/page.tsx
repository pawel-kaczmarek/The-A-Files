"use client";

import Link from "next/link";
import { useState } from "react";
import { Input } from "@/components/ui/input";
import { CatalogueDescription } from "@/components/catalogue-description";
import { catalogueDescription } from "@/lib/catalogue-descriptions";
import { FlaskConical } from "lucide-react";

import { Chip, ErrorNotice, LoadingLine, PageHeader } from "@/components/common";
import { useCatalog } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

const CATEGORY_ORDER = ["speech_quality", "speech_intelligibility", "speech_reverberation", "ai_based", "unknown"];

export default function MetricsPage() {
  const { t, locale } = useI18n();
  const [search, setSearch] = useState("");
  const catalog = useCatalog();
  if (catalog.error) return <ErrorNotice error={catalog.error} onRetry={catalog.reload} />;
  if (!catalog.data) return <LoadingLine />;
  const metrics = catalog.data.metrics.filter((metric) =>
    `${metric.name} ${metric.abbreviation} ${metric.label} ${catalogueDescription("metrics", metric.name, locale)?.join(" ") ?? ""}`.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <>
      <PageHeader title={t("catalogue.metricsTitle")} subtitle={t("catalogue.metricsSubtitle")}>
        <Input className="mt-5 w-full sm:w-72" aria-label={t("common.search")} placeholder={t("common.search")} value={search} onChange={(event) => setSearch(event.target.value)} />
      </PageHeader>
      <div className="space-y-8">
        {metrics.length === 0 && <p className="text-sm text-muted-foreground">{t("common.notFound")}</p>}
        {CATEGORY_ORDER.filter((category) => metrics.some((metric) => metric.category === category)).map((category) => (
          <section key={category}>
            <h2 className="eyebrow mb-2">{t(`metricCategories.${category}`)}</h2>
            <div className="overflow-x-auto rounded-lg border bg-card focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring" tabIndex={0} role="region" aria-label={t(`metricCategories.${category}`)}>
              <table className="catalogue-table">
                <caption className="sr-only">{t(`metricCategories.${category}`)}</caption>
                <colgroup><col className="w-[42%]" /><col className="w-[13%]" /><col className="w-[12%]" /><col className="w-[17%]" /><col className="w-[16%]" /></colgroup>
                <thead>
                  <tr className="border-b text-left text-xs text-muted-foreground">
                    <th scope="col" className="px-4 py-2 font-medium">{t("common.name")}</th>
                    <th scope="col" className="px-4 py-2 font-medium">{t("editor.measures.direction")}</th>
                    <th scope="col" className="px-4 py-2 font-medium">{t("editor.measures.scale")}</th>
                    <th scope="col" className="px-4 py-2 font-medium">{t("common.reference")}</th>
                    <th scope="col">{locale === "pl" ? "Działanie" : "Action"}</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics
                    .filter((metric) => metric.category === category)
                    .map((metric) => (
                      <tr key={metric.name} className="border-b align-top last:border-0">
                        <td>
                          <div className="font-medium">{metric.abbreviation}</div>
                          <div className="text-xs text-muted-foreground">{metric.label}</div>
                          <CatalogueDescription kind="metrics" name={metric.name} fallback={metric.interpretation ?? metric.label} />
                          <div className="mt-1 flex flex-wrap gap-1">
                            <Chip>{metric.intrusive ? t("catalogue.intrusive") : t("catalogue.nonIntrusive")}</Chip>
                            {metric.domain && <Chip>{metric.domain}</Chip>}
                            {metric.components.length > 0 && (
                              <Chip>
                                {t("catalogue.components")}: {metric.components.join(" · ")}
                              </Chip>
                            )}
                            {metric.requires_tensorflow && <Chip>{t("catalogue.tensorflow")}</Chip>}
                          </div>
                        </td>
                        <td className="text-xs">
                          {metric.higher_is_better === null
                            ? t("common.notRanked")
                            : metric.higher_is_better
                              ? `↑ ${t("common.higherIsBetter")}`
                              : `↓ ${t("common.lowerIsBetter")}`}
                        </td>
                        <td className="text-xs">{metric.scale ?? "–"}</td>
                        <td className="text-xs">
                          {metric.reference} {metric.year && `(${metric.year})`}
                          {metric.url && <a className="ml-2 text-primary underline" href={metric.url}>{locale === "pl" ? "Źródło" : "Source"}</a>}
                        </td>
                        <td>
                          <Link href={`/experiments/new?metric=${metric.name}`} className="inline-flex items-center gap-1 text-xs text-primary hover:underline">
                            <FlaskConical className="h-3.5 w-3.5 shrink-0" /> {t("common.useInExperiment")}
                          </Link>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </section>
        ))}
      </div>
    </>
  );
}
