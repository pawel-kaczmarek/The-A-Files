"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  AudioWaveform,
  Database,
  FileDown,
  FlaskConical,
  Gauge,
  History,
  LayoutDashboard,
  Ruler,
  Scale,
  ShieldAlert,
  SlidersHorizontal,
  Swords,
  Wrench,
} from "lucide-react";

import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

const sections = [
  {
    title: null,
    links: [{ href: "/dashboard", label: "Dashboard", icon: LayoutDashboard }],
  },
  {
    title: "Experiments",
    links: [
      { href: "/experiments/dataset-benchmark", label: "Dataset Benchmark", icon: Gauge },
      { href: "/experiments/attack-robustness", label: "Attack Robustness", icon: ShieldAlert },
      { href: "/experiments/perceptual-quality", label: "Perceptual Quality", icon: Activity },
      { href: "/experiments/embedding-capacity", label: "Embedding Capacity", icon: Ruler },
      { href: "/experiments/method-comparison", label: "Method Comparison", icon: Scale },
      { href: "/experiments/research-experiment", label: "Research Experiment", icon: FlaskConical },
    ],
  },
  {
    title: "Results",
    links: [
      { href: "/results/history", label: "Experiment History", icon: History },
      { href: "/results/exports", label: "CSV Exports", icon: FileDown },
    ],
  },
  {
    title: "Settings",
    links: [
      { href: "/settings/methods", label: "Methods", icon: Wrench },
      { href: "/settings/metrics", label: "Metrics", icon: SlidersHorizontal },
      { href: "/settings/attacks", label: "Attacks", icon: Swords },
      { href: "/settings/datasets", label: "Datasets", icon: Database },
    ],
  },
];

export function AppSidebar() {
  const pathname = usePathname();
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r bg-card">
      <div className="flex items-center gap-2 border-b px-5 py-4">
        <AudioWaveform className="h-6 w-6 text-primary" />
        <div>
          <div className="text-sm font-semibold leading-tight">The A-Files</div>
          <div className="text-xs text-muted-foreground">
            audio steganography research
          </div>
        </div>
      </div>
      <nav className="flex flex-1 flex-col gap-4 overflow-y-auto p-3">
        {sections.map((section, index) => (
          <div key={index}>
            {section.title && (
              <div className="mb-1 px-3 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                {section.title}
              </div>
            )}
            <div className="flex flex-col gap-0.5">
              {section.links.map(({ href, label, icon: Icon }) => {
                const active = pathname === href || pathname.startsWith(`${href}/`);
                return (
                  <Link
                    key={href}
                    href={href}
                    className={cn(
                      "flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                      active
                        ? "bg-primary text-primary-foreground"
                        : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                    )}
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    {label}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>
      <div className="flex items-center justify-between border-t px-4 py-3">
        <span className="text-xs text-muted-foreground">Theme</span>
        <ThemeToggle />
      </div>
    </aside>
  );
}
