"use client";

import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useI18n } from "@/lib/i18n";

type Theme = "light" | "dark";

function appliedTheme(): Theme {
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

export function ThemeToggle() {
  const { t } = useI18n();
  // Rendered only after mount so the icon always matches the applied theme.
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    setTheme(appliedTheme());
    const observer = new MutationObserver(() => setTheme(appliedTheme()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  function toggle() {
    const next: Theme = appliedTheme() === "dark" ? "light" : "dark";
    document.documentElement.classList.toggle("dark", next === "dark");
    try {
      window.localStorage.setItem("taf-theme", next);
    } catch {
      // not persisted
    }
    setTheme(next);
  }

  const label = `${t("common.theme")}: ${theme === "dark" ? t("common.dark") : t("common.light")}`;
  return (
    <Button variant="ghost" size="icon" onClick={toggle} title={label} aria-label={label} className="h-8 w-8">
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  );
}
