"use client";

import { Chip } from "@/components/common";
import { localized } from "@/lib/catalogue";
import { useI18n } from "@/lib/i18n";
import type { CatalogueCard } from "@/lib/types";

/** A component's summary and expandable mechanism, from its backend card. */
export function CatalogueDescription({ entry, name, expanded = false }: {
  entry: Pick<CatalogueCard, "summary" | "details">;
  name: string;
  expanded?: boolean;
}) {
  const { locale } = useI18n();
  const summary = localized(entry.summary, locale);
  const details = localized(entry.details, locale);
  if (!summary && !details) return null;
  return (
    <div className="mt-2 space-y-2 text-xs leading-relaxed text-muted-foreground">
      {summary && <p>{summary}</p>}
      {details && (
        <details open={expanded || undefined} className="group">
          <summary className="w-fit cursor-pointer rounded-sm font-medium text-primary hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring">
            {locale === "pl" ? "Jak to działa i jak interpretować" : "How it works and how to interpret it"}
            <span className="sr-only">: {name}</span>
          </summary>
          <p className="mt-2">{details}</p>
        </details>
      )}
    </div>
  );
}

/** Optional dependencies of a catalogue entry, and whether the server has them. */
export function RequirementChips({ entry }: { entry: Pick<CatalogueCard, "requires" | "extra" | "available"> }) {
  const { t } = useI18n();
  if (!entry.requires.length) return null;
  return (
    <>
      <Chip>
        {t("catalogue.requires", { modules: entry.requires.join(", ") })}
        {entry.extra && ` [${entry.extra}]`}
      </Chip>
      {!entry.available && <Chip>{t("catalogue.notInstalled")}</Chip>}
    </>
  );
}

/** Citation links of a catalogue entry. */
export function CatalogueReferences({ entry }: { entry: Pick<CatalogueCard, "references"> }) {
  if (!entry.references.length) return <>–</>;
  return (
    <span className="space-y-0.5">
      {entry.references.map((reference) => {
        const label = reference.year ? `${reference.citation} (${reference.year})` : reference.citation;
        return (
          <span key={`${reference.citation}-${reference.link ?? ""}`} className="block">
            {reference.link ? (
              <a href={reference.link} target="_blank" rel="noreferrer" className="text-primary hover:underline">
                {label}
              </a>
            ) : (
              label
            )}
          </span>
        );
      })}
    </span>
  );
}
