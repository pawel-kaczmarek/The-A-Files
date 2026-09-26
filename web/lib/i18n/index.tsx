"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { en } from "./en";

type DeepString<T> = { [K in keyof T]: T[K] extends string ? string : DeepString<T[K]> };
export type Messages = DeepString<typeof en>;
export type Locale = "en" | "pl";

const STORAGE_KEY = "taf-locale";

// Loaded lazily so the default English bundle stays small.
const dictionaries: Record<Locale, () => Promise<Messages>> = {
  en: async () => en,
  pl: async () => (await import("./pl")).pl,
};

interface I18nContextValue {
  locale: Locale;
  messages: Messages;
  setLocale: (locale: Locale) => void;
}

const I18nContext = createContext<I18nContextValue>({ locale: "en", messages: en, setLocale: () => {} });

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("en");
  const [messages, setMessages] = useState<Messages>(en);

  const apply = useCallback(async (next: Locale) => {
    setMessages(await dictionaries[next]());
    setLocaleState(next);
    document.documentElement.lang = next;
  }, []);

  useEffect(() => {
    let stored: Locale = "en";
    try {
      const value = window.localStorage.getItem(STORAGE_KEY);
      if (value === "pl" || value === "en") stored = value;
    } catch {
      // storage unavailable: keep English
    }
    // A link may carry the language (?lang=pl), which then also becomes the stored choice.
    const requested = new URLSearchParams(window.location.search).get("lang");
    if (requested === "pl" || requested === "en") {
      stored = requested;
      try {
        window.localStorage.setItem(STORAGE_KEY, requested);
      } catch {
        // not persisted
      }
    }
    if (stored !== "en") void apply(stored);
  }, [apply]);

  const setLocale = useCallback(
    (next: Locale) => {
      try {
        window.localStorage.setItem(STORAGE_KEY, next);
      } catch {
        // not persisted
      }
      void apply(next);
    },
    [apply]
  );

  const value = useMemo(() => ({ locale, messages, setLocale }), [locale, messages, setLocale]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

function lookup(messages: Messages, key: string): string | undefined {
  let node: unknown = messages;
  for (const part of key.split(".")) {
    if (node && typeof node === "object" && part in (node as Record<string, unknown>)) {
      node = (node as Record<string, unknown>)[part];
    } else {
      return undefined;
    }
  }
  return typeof node === "string" ? node : undefined;
}

export function useI18n() {
  const { locale, messages, setLocale } = useContext(I18nContext);

  const t = useCallback(
    (key: string, vars?: Record<string, string | number>) => {
      const template = lookup(messages, key) ?? lookup(en as Messages, key) ?? key;
      if (!vars) return template;
      return template.replace(/\{(\w+)\}/g, (match, name: string) =>
        name in vars ? String(vars[name]) : match
      );
    },
    [messages]
  );

  /** Translation if the key exists, otherwise the fallback (for backend-provided names). */
  const tOr = useCallback(
    (key: string, fallback: string) => lookup(messages, key) ?? lookup(en as Messages, key) ?? fallback,
    [messages]
  );

  const number = useCallback(
    (value: number | null | undefined, digits = 3) =>
      value === null || value === undefined || !Number.isFinite(value)
        ? "–"
        : value.toLocaleString(locale === "pl" ? "pl-PL" : "en-US", {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits,
          }),
    [locale]
  );

  const percent = useCallback(
    (value: number | null | undefined, digits = 0) =>
      value === null || value === undefined || !Number.isFinite(value)
        ? "–"
        : (value * 100).toLocaleString(locale === "pl" ? "pl-PL" : "en-US", {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits,
          }) + "%",
    [locale]
  );

  const date = useCallback(
    (value: string | null | undefined) =>
      value
        ? new Date(value).toLocaleString(locale === "pl" ? "pl-PL" : "en-GB", {
            dateStyle: "medium",
            timeStyle: "short",
          })
        : "–",
    [locale]
  );

  return { t, tOr, locale, setLocale, number, percent, date, messages };
}
