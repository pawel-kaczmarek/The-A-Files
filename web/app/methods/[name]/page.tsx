"use client";

import Link from "next/link";
import { use } from "react";
import { FlaskConical } from "lucide-react";

import { CatalogueDescription, CatalogueReferences, RequirementChips } from "@/components/catalogue-description";
import { DataTable } from "@/components/charts/base";
import { Chip, ErrorNotice, KeyValues, LoadingLine, PageHeader, Section } from "@/components/common";
import { Button } from "@/components/ui/button";
import { localized } from "@/lib/catalogue";
import { useCatalog } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

export default function MethodPage({ params }: { params: Promise<{ name: string }> }) {
  const { name } = use(params);
  const { t, locale } = useI18n();
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
        title={localized(method.title, locale) || method.name}
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
          <CatalogueDescription entry={method} name={method.name} expanded />
          <div className="mt-5" />
          <KeyValues
            items={[
              [t("catalogue.family"), localized(method.family_label, locale)],
              [t("catalogue.purpose"), localized(method.purpose_label, locale) || "–"],
              [t("common.reference"), <CatalogueReferences key="r" entry={method} />],
              [t("catalogue.strength"), method.strength_parameter ? <span key="s" className="font-mono">{method.strength_parameter}</span> : "–"],
              [t("common.className"), <span key="c" className="font-mono text-xs">{method.class_name}</span>],
            ]}
          />
          <div className="mt-4 flex flex-wrap gap-1.5">
            <RequirementChips entry={method} />
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
