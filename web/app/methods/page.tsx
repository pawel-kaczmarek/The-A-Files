"use client";

import Link from "next/link";
import { useState } from "react";
import { ExternalLink } from "lucide-react";

import { Chip, ErrorNotice, LoadingLine, PageHeader } from "@/components/common";
import { CatalogueDescription } from "@/components/catalogue-description";
import { catalogueDescription } from "@/lib/catalogue-descriptions";
import { Input } from "@/components/ui/input";
import { useCatalog } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

const FAMILY_ORDER = ["lsb", "transform", "spread_spectrum", "echo", "phase", "quantization", "statistical", "adaptive", "reversible", "learned", "neural", "plugin"];

export default function MethodsPage() {
  const { t, locale } = useI18n();
  const catalog = useCatalog();
  const [search, setSearch] = useState("");
  if (catalog.error) return <ErrorNotice error={catalog.error} onRetry={catalog.reload} />;
  if (!catalog.data) return <LoadingLine />;
  const methods = catalog.data.methods.filter((method) =>
    `${method.name} ${method.description} ${method.reference ?? ""} ${catalogueDescription("methods", method.name, locale)?.join(" ") ?? ""}`.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <>
      <PageHeader title={t("catalogue.methodsTitle")} subtitle={t("catalogue.methodsSubtitle")}>
        <Input className="mt-5 w-full sm:w-72" aria-label={t("common.search")} placeholder={t("common.search")} value={search} onChange={(event) => setSearch(event.target.value)} />
      </PageHeader>
      <div className="space-y-8">
        {methods.length === 0 && <p className="text-sm text-muted-foreground">{t("common.notFound")}</p>}
        {FAMILY_ORDER.filter((family) => methods.some((method) => (method.family ?? "plugin") === family)).map((family) => (
          <section key={family}>
            <h2 className="eyebrow mb-2">{t(`families.${family}`)}</h2>
            <div className="overflow-x-auto rounded-lg border bg-card focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring" tabIndex={0} role="region" aria-label={t(`families.${family}`)}>
              <table className="catalogue-table">
                <caption className="sr-only">{t(`families.${family}`)}</caption>
                <colgroup><col className="w-[46%]" /><col className="w-[15%]" /><col className="w-[19%]" /><col className="w-[20%]" /></colgroup>
                <thead><tr>
                  <th scope="col">{t("common.name")} / {locale === "pl" ? "Opis" : "Description"}</th>
                  <th scope="col">{t("catalogue.purpose")}</th>
                  <th scope="col">{t("common.reference")}</th>
                  <th scope="col">{t("common.parameters")}</th>
                </tr></thead>
                <tbody>
                  {methods
                    .filter((method) => (method.family ?? "plugin") === family)
                    .map((method) => (
                      <tr key={method.name} className="border-b last:border-0 hover:bg-accent/40">
                        <td>
                          <Link href={`/methods/${method.name}`} className="font-medium hover:underline">
                            {method.description || method.name}
                          </Link>
                          <div className="font-mono text-[11px] text-muted-foreground">{method.name}</div>
                          <CatalogueDescription kind="methods" name={method.name} fallback={method.description} />
                        </td>
                        <td className="text-xs text-muted-foreground">{method.purpose ? t(`purposes.${method.purpose}`) : "–"}</td>
                        <td className="text-xs">
                          {method.reference ? `${method.reference} (${method.year})` : "–"}
                          {method.doi && (
                            <a href={`https://doi.org/${method.doi}`} target="_blank" rel="noreferrer" className="ml-1.5 inline-flex text-primary">
                              <ExternalLink className="h-3 w-3" />
                            </a>
                          )}
                        </td>
                        <td>
                          <div className="flex flex-wrap gap-1">
                            {method.strength_parameter && <Chip>{method.strength_parameter}</Chip>}
                            <Chip>
                              {method.parameters.length} {t("common.parameters").toLowerCase()}
                            </Chip>
                            {method.requires_tensorflow && <Chip>{t("catalogue.tensorflow")}</Chip>}
                            {!method.packaged && <Chip>{t("catalogue.plugin")}</Chip>}
                          </div>
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
