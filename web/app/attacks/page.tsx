"use client";

import Link from "next/link";
import { useState } from "react";
import { FlaskConical } from "lucide-react";

import { Chip, ErrorNotice, LoadingLine, PageHeader, Spec } from "@/components/common";
import { CatalogueDescription } from "@/components/catalogue-description";
import { Input } from "@/components/ui/input";
import { groupsInOrder, localized, matchesSearch } from "@/lib/catalogue";
import { useCatalog } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

export default function AttacksPage() {
  const { t, locale } = useI18n();
  const catalog = useCatalog();
  const [search, setSearch] = useState("");
  if (catalog.error) return <ErrorNotice error={catalog.error} onRetry={catalog.reload} />;
  if (!catalog.data) return <LoadingLine />;
  const attacks = catalog.data.attacks.filter((attack) => attack.name !== "codec" && matchesSearch(attack, search, locale));
  const groups = groupsInOrder(attacks, (attack) => attack.family, (attack) => attack.family_label);

  return (
    <>
      <PageHeader title={t("catalogue.attacksTitle")} subtitle={t("catalogue.attacksSubtitle")}>
        <Input className="mt-5 w-full sm:w-72" aria-label={t("common.search")} placeholder={t("common.search")} value={search} onChange={(event) => setSearch(event.target.value)} />
      </PageHeader>
      <div className="space-y-8">
        {attacks.length === 0 && <p className="text-sm text-muted-foreground">{t("common.notFound")}</p>}
        {groups.map((group) => {
          const label = localized(group.label, locale);
          return (
            <section key={group.key}>
              <h2 className="eyebrow mb-2">{label}</h2>
              <div className="overflow-x-auto rounded-lg border bg-card focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring" tabIndex={0} role="region" aria-label={label}>
                <table className="catalogue-table">
                  <caption className="sr-only">{label}</caption>
                  <colgroup><col className="w-[46%]" /><col className="w-[36%]" /><col className="w-[18%]" /></colgroup>
                  <thead><tr>
                    <th scope="col">{t("common.name")} / {locale === "pl" ? "Opis" : "Description"}</th>
                    <th scope="col">{t("common.parameters")}</th>
                    <th scope="col">{locale === "pl" ? "Działanie" : "Action"}</th>
                  </tr></thead>
                  <tbody>
                    {group.items.map((attack) => (
                      <tr key={attack.name} id={attack.name}>
                        <td>
                          <div className="font-medium">{localized(attack.title, locale) || attack.name}</div>
                          <div className="font-mono text-[11px] text-muted-foreground">{attack.name}</div>
                          <CatalogueDescription entry={attack} name={attack.name} />
                          <div className="mt-1.5 flex flex-wrap gap-1">
                            {attack.stochastic && <Chip>{t("catalogue.stochastic")}</Chip>}
                            {attack.changes_length_or_rate && <Chip>{t("catalogue.changesLength")}</Chip>}
                            {attack.has_severity && <Chip>mild · moderate · strong · extreme</Chip>}
                          </div>
                        </td>
                        <td className="space-y-1.5 text-xs">
                          <div className="flex flex-wrap gap-1">
                            {attack.parameters.map((parameter) => (
                              <Spec key={parameter.name}>
                                {parameter.name}={String(parameter.default)}
                              </Spec>
                            ))}
                          </div>
                          {attack.sweep && (
                            <div className="text-muted-foreground">
                              {t("catalogue.sweepLadder")}: <span className="font-mono">{attack.sweep.parameter}</span> ={" "}
                              <span className="num">{attack.sweep.values.join(", ")}</span> ({attack.sweep.unit})
                            </div>
                          )}
                        </td>
                        <td>
                          <Link
                            href={`/experiments/new?attack=${encodeURIComponent(attack.name)}`}
                            className="inline-flex items-center gap-1 self-start text-xs text-primary hover:underline"
                          >
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
