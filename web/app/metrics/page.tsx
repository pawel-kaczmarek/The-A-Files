"use client";

import Link from "next/link";
import { useState } from "react";
import { FlaskConical } from "lucide-react";

import { Chip, ErrorNotice, LoadingLine, PageHeader } from "@/components/common";
import { CatalogueDescription, CatalogueReferences, RequirementChips } from "@/components/catalogue-description";
import { Input } from "@/components/ui/input";
import { groupsInOrder, localized, matchesSearch } from "@/lib/catalogue";
import { useCatalog } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

export default function MetricsPage() {
  const { t, locale } = useI18n();
  const [search, setSearch] = useState("");
  const catalog = useCatalog();
  if (catalog.error) return <ErrorNotice error={catalog.error} onRetry={catalog.reload} />;
  if (!catalog.data) return <LoadingLine />;
  const metrics = catalog.data.metrics.filter((metric) => matchesSearch(metric, search, locale) || metric.label.toLowerCase().includes(search.toLowerCase()));
  const groups = groupsInOrder(metrics, (metric) => metric.category, (metric) => metric.category_label);

  return (
    <>
      <PageHeader title={t("catalogue.metricsTitle")} subtitle={t("catalogue.metricsSubtitle")}>
        <Input className="mt-5 w-full sm:w-72" aria-label={t("common.search")} placeholder={t("common.search")} value={search} onChange={(event) => setSearch(event.target.value)} />
      </PageHeader>
      <div className="space-y-8">
        {metrics.length === 0 && <p className="text-sm text-muted-foreground">{t("common.notFound")}</p>}
        {groups.map((group) => {
          const label = localized(group.label, locale);
          return (
            <section key={group.key}>
              <h2 className="eyebrow mb-2">{label}</h2>
              <div className="overflow-x-auto rounded-lg border bg-card focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring" tabIndex={0} role="region" aria-label={label}>
                <table className="catalogue-table">
                  <caption className="sr-only">{label}</caption>
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
                    {group.items.map((metric) => (
                      <tr key={metric.name} className="border-b align-top last:border-0">
                        <td>
                          <div className="font-medium">{metric.abbreviation}</div>
                          <div className="text-xs text-muted-foreground">{localized(metric.title, locale) || metric.label}</div>
                          <CatalogueDescription entry={metric} name={metric.name} />
                          <div className="mt-1 flex flex-wrap gap-1">
                            <Chip>{metric.intrusive ? t("catalogue.intrusive") : t("catalogue.nonIntrusive")}</Chip>
                            {metric.domain && <Chip>{metric.domain}</Chip>}
                            {metric.components.length > 0 && (
                              <Chip>
                                {t("catalogue.components")}: {metric.components.join(" · ")}
                              </Chip>
                            )}
                            <RequirementChips entry={metric} />
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
                          <CatalogueReferences entry={metric} />
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
          );
        })}
      </div>
    </>
  );
}
