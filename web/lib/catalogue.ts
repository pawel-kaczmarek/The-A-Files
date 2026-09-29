import type { Locale } from "./i18n";
import type { CatalogueCard, LocalizedText } from "./types";

// Catalogue entries carry their titles, descriptions and group names from the
// backend (taf.models.card), in every interface language, and arrive ordered
// by group. These helpers only select the language and preserve that order.

/** A backend text in the interface language; the API fills every locale, English is the fallback. */
export function localized(text: LocalizedText | null | undefined, locale: Locale): string {
  if (!text) return "";
  return text[locale] || text.en;
}

export interface CatalogueGroup<T> {
  key: string;
  label: LocalizedText;
  items: T[];
}

/** Consecutive entries sharing a group, in the order the API lists them. */
export function groupsInOrder<T>(items: T[], key: (item: T) => string, label: (item: T) => LocalizedText): CatalogueGroup<T>[] {
  const groups = new Map<string, CatalogueGroup<T>>();
  for (const item of items) {
    const name = key(item);
    let group = groups.get(name);
    if (!group) {
      group = { key: name, label: label(item), items: [] };
      groups.set(name, group);
    }
    group.items.push(item);
  }
  return [...groups.values()];
}

/** Whether a catalogue entry matches a search, in its names and its card texts in the current language. */
export function matchesSearch(entry: CatalogueCard & { name: string }, search: string, locale: Locale): boolean {
  if (!search) return true;
  const haystack = [
    entry.name,
    entry.abbreviation,
    entry.reference ?? "",
    localized(entry.title, locale),
    localized(entry.summary, locale),
    localized(entry.details, locale),
  ].join(" ");
  return haystack.toLowerCase().includes(search.toLowerCase());
}
