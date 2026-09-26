"use client";

import Link from "next/link";
import { FlaskConical } from "lucide-react";

import { Chip, ErrorNotice, LoadingLine, PageHeader } from "@/components/common";
import { useCatalog } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

const CATEGORY_ORDER = ["speech_quality", "speech_intelligibility", "speech_reverberation", "ai_based", "unknown"];

export default function MetricsPage() {
  const { t } = useI18n();
  const catalog = useCatalog();
  if (catalog.error) return <ErrorNotice error={catalog.error} onRetry={catalog.reload} />;
  if (!catalog.data) return <LoadingLine />;
  const metrics = catalog.data.metrics;

  return (
    <>
      <PageHeader title={t("catalogue.metricsTitle")} subtitle={t("catalogue.metricsSubtitle")} />
      <div className="space-y-8">
        {CATEGORY_ORDER.filter((category) => metrics.some((metric) => metric.category === category)).map((category) => (
          <section key={category}>
            <h2 className="eyebrow mb-2">{t(`metricCategories.${category}`)}</h2>
            <div className="overflow-x-auto rounded-lg border bg-card">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-xs text-muted-foreground">
                    <th className="px-4 py-2 font-medium">{t("common.name")}</th>
                    <th className="px-4 py-2 font-medium">{t("editor.measures.direction")}</th>
                    <th className="px-4 py-2 font-medium">{t("editor.measures.scale")}</th>
                    <th className="px-4 py-2 font-medium">{t("common.reference")}</th>
                    <th className="px-4 py-2" />
                  </tr>
                </thead>
                <tbody>
                  {metrics
                    .filter((metric) => metric.category === category)
                    .map((metric) => (
                      <tr key={metric.name} className="border-b align-top last:border-0">
                        <td className="px-4 py-3">
                          <div className="font-medium">{metric.abbreviation}</div>
                          <div className="text-xs text-muted-foreground">{metric.label}</div>
                          <div className="mt-1 flex flex-wrap gap-1">
                            <Chip>{metric.intrusive ? t("catalogue.intrusive") : t("catalogue.nonIntrusive")}</Chip>
                            {metric.components.length > 0 && (
                              <Chip>
                                {t("catalogue.components")}: {metric.components.join(" · ")}
                              </Chip>
                            )}
                            {metric.requires_tensorflow && <Chip>{t("catalogue.tensorflow")}</Chip>}
                          </div>
                        </td>
                        <td className="px-4 py-3 text-xs">
                          {metric.higher_is_better === null
                            ? t("common.notRanked")
                            : metric.higher_is_better
                              ? `↑ ${t("common.higherIsBetter")}`
                              : `↓ ${t("common.lowerIsBetter")}`}
                        </td>
                        <td className="whitespace-nowrap px-4 py-3 text-xs">{metric.scale ?? "–"}</td>
                        <td className="px-4 py-3 text-xs">
                          {metric.reference} {metric.year && `(${metric.year})`}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <Link href={`/experiments/new?metric=${metric.name}`} className="inline-flex items-center gap-1 whitespace-nowrap text-xs text-primary hover:underline">
                            <FlaskConical className="h-3.5 w-3.5" /> {t("common.useInExperiment")}
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
