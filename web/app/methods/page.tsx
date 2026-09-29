"use client";

import Link from "next/link";
import { useState } from "react";

import { Chip, ErrorNotice, LoadingLine, PageHeader } from "@/components/common";
import { CatalogueDescription, CatalogueReferences, RequirementChips } from "@/components/catalogue-description";
import { Input } from "@/components/ui/input";
import { groupsInOrder, localized, matchesSearch } from "@/lib/catalogue";
import { useCatalog } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

export default function MethodsPage() {
  const { t, locale } = useI18n();
  const catalog = useCatalog();
  const [search, setSearch] = useState("");
  if (catalog.error) return <ErrorNotice error={catalog.error} onRetry={catalog.reload} />;
  if (!catalog.data) return <LoadingLine />;
  const methods = catalog.data.methods.filter((method) => matchesSearch(method, search, locale));
  const groups = groupsInOrder(methods, (method) => method.family ?? "plugin", (method) => method.family_label);

  return (
    <>
      <PageHeader title={t("catalogue.methodsTitle")} subtitle={t("catalogue.methodsSubtitle")}>
        <Input className="mt-5 w-full sm:w-72" aria-label={t("common.search")} placeholder={t("common.search")} value={search} onChange={(event) => setSearch(event.target.value)} />
      </PageHeader>
      <div className="space-y-8">
        {methods.length === 0 && <p className="text-sm text-muted-foreground">{t("common.notFound")}</p>}
        {groups.map((group) => {
          const label = localized(group.label, locale);
          return (
            <section key={group.key}>
              <h2 className="eyebrow mb-2">{label}</h2>
              <div className="overflow-x-auto rounded-lg border bg-card focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring" tabIndex={0} role="region" aria-label={label}>
                <table className="catalogue-table">
                  <caption className="sr-only">{label}</caption>
                  <colgroup><col className="w-[46%]" /><col className="w-[15%]" /><col className="w-[19%]" /><col className="w-[20%]" /></colgroup>
                  <thead><tr>
                    <th scope="col">{t("common.name")} / {locale === "pl" ? "Opis" : "Description"}</th>
                    <th scope="col">{t("catalogue.purpose")}</th>
                    <th scope="col">{t("common.reference")}</th>
                    <th scope="col">{t("common.parameters")}</th>
                  </tr></thead>
                  <tbody>
                    {group.items.map((method) => (
                      <tr key={method.name} className="border-b last:border-0 hover:bg-accent/40">
                        <td>
                          <Link href={`/methods/${method.name}`} className="font-medium hover:underline">
                            {localized(method.title, locale) || method.name}
                          </Link>
                          <div className="font-mono text-[11px] text-muted-foreground">{method.name}</div>
                          <CatalogueDescription entry={method} name={method.name} />
                        </td>
                        <td className="text-xs text-muted-foreground">{localized(method.purpose_label, locale) || "–"}</td>
                        <td className="text-xs">
                          <CatalogueReferences entry={method} />
                        </td>
                        <td>
                          <div className="flex flex-wrap gap-1">
                            {method.strength_parameter && <Chip>{method.strength_parameter}</Chip>}
                            <Chip>
                              {method.parameters.length} {t("common.parameters").toLowerCase()}
                            </Chip>
                            <RequirementChips entry={method} />
                            {!method.packaged && <Chip>{t("catalogue.plugin")}</Chip>}
                          </div>
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
