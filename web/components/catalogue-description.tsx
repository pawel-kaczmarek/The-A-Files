"use client";

import { catalogueDescription, type CatalogueKind } from "@/lib/catalogue-descriptions";
import { useI18n } from "@/lib/i18n";

export function CatalogueDescription({ kind, name, fallback, expanded = false }: {
  kind: CatalogueKind;
  name: string;
  fallback?: string;
  expanded?: boolean;
}) {
  const { locale } = useI18n();
  const description = catalogueDescription(kind, name, locale);
  if (!description) return fallback ? <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{fallback}</p> : null;
  return (
    <div className="mt-2 space-y-2 text-xs leading-relaxed text-muted-foreground">
      <p>{description[0]}</p>
      <details open={expanded || undefined} className="group">
        <summary className="w-fit cursor-pointer rounded-sm font-medium text-primary hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring">
          {locale === "pl" ? "Jak to działa i jak interpretować" : "How it works and how to interpret it"}
          <span className="sr-only">: {name}</span>
        </summary>
        <p className="mt-2">{description[1]}</p>
      </details>
    </div>
  );
}
