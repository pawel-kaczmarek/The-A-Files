"use client";

import { useEffect, useState } from "react";

import { ErrorNotice, KeyValues, LoadingLine, PageHeader, Section, Spec } from "@/components/common";
import { Button } from "@/components/ui/button";
import { API_BASE, api, urls } from "@/lib/api";
import { useAsync } from "@/lib/hooks";
import { useI18n, type Locale } from "@/lib/i18n";

function applyTheme(theme: "light" | "dark") {
  document.documentElement.classList.toggle("dark", theme === "dark");
  try {
    window.localStorage.setItem("taf-theme", theme);
  } catch {
    // not persisted
  }
}

export default function SettingsPage() {
  const { t, locale, setLocale } = useI18n();
  const health = useAsync(() => api.health(), []);
  const [theme, setTheme] = useState<"light" | "dark">("light");
  useEffect(() => setTheme(document.documentElement.classList.contains("dark") ? "dark" : "light"), []);

  return (
    <>
      <PageHeader title={t("settings.title")} subtitle={t("settings.subtitle")} />
      <div className="grid gap-6 lg:grid-cols-2">
        <Section title={t("settings.appearance")}>
          <div className="space-y-5">
            <div className="space-y-2">
              <div className="text-sm font-medium">{t("common.language")}</div>
              <div className="flex gap-2">
                {(["en", "pl"] as Locale[]).map((entry) => (
                  <Button key={entry} variant={locale === entry ? "default" : "outline"} size="sm" onClick={() => setLocale(entry)}>
                    {entry === "en" ? "English" : "Polski"}
                  </Button>
                ))}
              </div>
            </div>
            <div className="space-y-2">
              <div className="text-sm font-medium">{t("common.theme")}</div>
              <div className="flex gap-2">
                {(["light", "dark"] as const).map((entry) => (
                  <Button
                    key={entry}
                    variant={theme === entry ? "default" : "outline"}
                    size="sm"
                    onClick={() => {
                      applyTheme(entry);
                      setTheme(entry);
                    }}
                  >
                    {t(`common.${entry}`)}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        </Section>
        <Section title={t("settings.services")}>
          {health.error && <ErrorNotice error={health.error} onRetry={health.reload} />}
          {!health.data && !health.error ? (
            <LoadingLine />
          ) : (
            <KeyValues
              items={[
                [t("settings.api"), <span key="a">{health.error ? t("settings.unreachable") : `${t("settings.reachable")} · v${health.data?.version}`} <Spec>{API_BASE}</Spec></span>],
                [
                  t("settings.database"),
                  health.data?.database.ok ? `${t("settings.reachable")} · ${health.data.database.version}` : health.data?.database.error ?? "–",
                ],
                [t("settings.dataDir"), <Spec key="d">{health.data?.data_dir ?? "–"}</Spec>],
                [t("nav.apiDocs"), <a key="docs" href={urls.docs()} target="_blank" rel="noreferrer" className="text-primary underline">{urls.docs()}</a>],
              ]}
            />
          )}
        </Section>
      </div>
    </>
  );
}
