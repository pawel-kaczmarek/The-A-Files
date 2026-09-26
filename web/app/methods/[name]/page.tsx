"use client";

import Link from "next/link";
import { use } from "react";
import { ExternalLink, FlaskConical } from "lucide-react";

import { DataTable } from "@/components/charts/base";
import { Chip, ErrorNotice, KeyValues, LoadingLine, PageHeader, Section } from "@/components/common";
import { Button } from "@/components/ui/button";
import { useCatalog } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

export default function MethodPage({ params }: { params: Promise<{ name: string }> }) {
  const { name } = use(params);
  const { t } = useI18n();
  const catalog = useCatalog();
  if (catalog.error) return <ErrorNotice error={catalog.error} onRetry={catalog.reload} />;
  if (!catalog.data) return <LoadingLine />;
  const method = catalog.data.methods.find((entry) => entry.name === decodeURIComponent(name));
  if (!method) return <p className="text-sm text-muted-foreground">{t("common.notFound")}</p>;

  return (
    <>
      <PageHeader
        eyebrow={
          <Link href="/methods" className="hover:underline">
            ← {t("nav.methods")}
          </Link>
        }
        title={method.description || method.name}
        subtitle={<span className="font-mono">{method.name}</span>}
        actions={
          <Button asChild>
            <Link href={`/experiments/new?method=${encodeURIComponent(method.name)}`}>
              <FlaskConical className="h-4 w-4" /> {t("common.useInExperiment")}
            </Link>
          </Button>
        }
      />
      <div className="grid gap-6 lg:grid-cols-2">
        <Section title={t("common.details")}>
          <KeyValues
            items={[
              [t("catalogue.family"), method.family ? t(`families.${method.family}`) : t("families.plugin")],
              [t("catalogue.purpose"), method.purpose ? t(`purposes.${method.purpose}`) : "–"],
              [
                t("common.reference"),
                method.reference ? (
                  <span key="r">
                    {method.reference} ({method.year})
                    {method.doi && (
                      <a href={`https://doi.org/${method.doi}`} target="_blank" rel="noreferrer" className="ml-2 inline-flex items-center gap-1 text-primary">
                        {method.doi} <ExternalLink className="h-3 w-3" />
                      </a>
                    )}
                  </span>
                ) : (
                  "–"
                ),
              ],
              [t("catalogue.strength"), method.strength_parameter ? <span key="s" className="font-mono">{method.strength_parameter}</span> : "–"],
              [t("common.className"), <span key="c" className="font-mono text-xs">{method.class_name}</span>],
            ]}
          />
          <div className="mt-4 flex flex-wrap gap-1.5">
            {method.requires_tensorflow && <Chip>{t("catalogue.tensorflow")}</Chip>}
            {method.needs_long_input && <Chip>{t("catalogue.longInput")}</Chip>}
            {!method.packaged && <Chip>{t("catalogue.plugin")}</Chip>}
          </div>
        </Section>
        <Section title={t("common.parameters")}>
          {method.parameters.length ? (
            <DataTable
              table={{
                columns: [t("common.name"), t("common.default"), t("common.type")],
                rows: method.parameters.map((parameter) => [
                  <span key="n" className="flex items-center gap-1.5 font-mono text-xs">
                    {parameter.name}
                    {parameter.name === method.strength_parameter && <Chip>{t("catalogue.strength")}</Chip>}
                    {parameter.is_key && <Chip>{t("catalogue.key")}</Chip>}
                  </span>,
                  <span key="d" className="font-mono text-xs">{String(parameter.default)}</span>,
                  <span key="t" className="text-xs text-muted-foreground">{parameter.type ?? "–"}</span>,
                ]),
              }}
            />
          ) : (
            <p className="text-sm text-muted-foreground">{t("editor.methods.noParameters")}</p>
          )}
        </Section>
      </div>
    </>
  );
}
