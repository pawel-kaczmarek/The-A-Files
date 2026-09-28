"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import {
  BookOpen,
  Database,
  FlaskConical,
  Gauge,
  History,
  LayoutDashboard,
  Ruler,
  Settings,
  Swords,
  Waves,
} from "lucide-react";

import { ThemeToggle } from "@/components/theme-toggle";
import { LogoMark } from "@/components/logo";
import { api } from "@/lib/api";
import { useAsync, useInterval } from "@/lib/hooks";
import { useI18n, type Locale } from "@/lib/i18n";
import { cn } from "@/lib/utils";

const SECTIONS: { title: string; links: { href: string; label: string; icon: typeof Gauge }[] }[] = [
  {
    title: "nav.research",
    links: [
      { href: "/dashboard", label: "nav.dashboard", icon: LayoutDashboard },
      { href: "/experiments", label: "nav.experiments", icon: FlaskConical },
      { href: "/runs", label: "nav.runs", icon: History },
    ],
  },
  {
    title: "nav.catalogue",
    links: [
      { href: "/methods", label: "nav.methods", icon: Waves },
      { href: "/attacks", label: "nav.attacks", icon: Swords },
      { href: "/metrics", label: "nav.metrics", icon: Ruler },
    ],
  },
  {
    title: "nav.data",
    links: [{ href: "/datasets", label: "nav.datasets", icon: Database }],
  },
  {
    title: "nav.reference",
    links: [
      { href: "/methodology", label: "nav.methodology", icon: BookOpen },
      { href: "/settings", label: "nav.settings", icon: Settings },
    ],
  },
];

function Sidebar() {
  const pathname = usePathname();
  const { t } = useI18n();
  return (
    <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r bg-card md:flex">
      <Link href="/dashboard" className="flex items-center gap-2.5 border-b px-5 py-4">
        <LogoMark className="h-6 w-5 text-foreground" />
        <div className="leading-tight">
          <div className="text-sm font-semibold">{t("app.name")}</div>
          <div className="text-[11px] text-muted-foreground">{t("app.tagline")}</div>
        </div>
      </Link>
      <nav className="flex-1 space-y-6 overflow-y-auto px-3 py-5">
        {SECTIONS.map((section) => (
          <div key={section.title}>
            <div className="eyebrow mb-1.5 px-2">{t(section.title)}</div>
            <ul className="space-y-0.5">
              {section.links.map(({ href, label, icon: Icon }) => {
                const active = pathname === href || pathname.startsWith(`${href}/`);
                return (
                  <li key={href}>
                    <Link
                      href={href}
                      className={cn(
                        "flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm transition-colors",
                        active
                          ? "bg-accent font-medium text-foreground"
                          : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
                      )}
                    >
                      <Icon className="h-4 w-4" aria-hidden />
                      {t(label)}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>
    </aside>
  );
}

function LanguageSwitch() {
  const { locale, setLocale, t } = useI18n();
  const options: Locale[] = ["en", "pl"];
  return (
    <div className="inline-flex rounded-md border p-0.5" role="group" aria-label={t("common.language")}>
      {options.map((option) => (
        <button
          key={option}
          type="button"
          onClick={() => setLocale(option)}
          aria-pressed={locale === option}
          className={cn(
            "rounded px-2 py-0.5 text-xs font-medium uppercase transition-colors",
            locale === option ? "bg-accent text-foreground" : "text-muted-foreground hover:text-foreground"
          )}
        >
          {option}
        </button>
      ))}
    </div>
  );
}

function ServiceStatus() {
  const { t } = useI18n();
  const health = useAsync(() => api.health(), []);
  useInterval(health.reload, 30000);
  const ok = !health.error && health.data?.database.ok;
  const label = health.error
    ? `${t("settings.api")}: ${t("settings.unreachable")}`
    : `${t("settings.database")}: ${ok ? t("settings.reachable") : t("settings.unreachable")}`;
  return (
    <Link href="/settings" className="inline-flex items-center gap-1.5 text-xs text-muted-foreground" title={label}>
      <span
        className="h-2 w-2 rounded-full"
        style={{ background: health.loading && !health.data ? "var(--chart-muted)" : ok ? "var(--status-good)" : "var(--status-critical)" }}
        aria-hidden
      />
      <span className="hidden lg:inline">{label}</span>
    </Link>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="sticky top-0 z-20 flex h-12 items-center justify-end gap-3 border-b bg-background/90 px-6 backdrop-blur">
          <ServiceStatus />
          <LanguageSwitch />
          <ThemeToggle />
        </div>
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-6 py-8 lg:px-10">{children}</main>
      </div>
    </div>
  );
}
