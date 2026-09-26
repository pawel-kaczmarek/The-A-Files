"use client";

import type { ReactNode } from "react";
import { AlertTriangle, CheckCircle2, CircleDashed, CircleSlash, Loader2, PauseCircle, XCircle } from "lucide-react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { API_BASE } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import type { Estimate, Property, RunStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

// ------------------------------------------------------------------ layout

export function PageHeader({
  eyebrow,
  title,
  subtitle,
  actions,
  children,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <header className="mb-8 border-b pb-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 max-w-3xl space-y-1.5">
          {eyebrow && <div className="eyebrow">{eyebrow}</div>}
          <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
          {subtitle && <p className="text-sm leading-relaxed text-muted-foreground">{subtitle}</p>}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {children}
    </header>
  );
}

export function Section({
  title,
  hint,
  actions,
  children,
  className,
  id,
}: {
  title?: ReactNode;
  hint?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <section id={id} className={cn("rounded-lg border bg-card", className)}>
      {(title || actions) && (
        <div className="flex flex-wrap items-start justify-between gap-3 border-b px-5 py-3.5">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold">{title}</h2>}
            {hint && <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">{hint}</p>}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function StatTile({ label, value, hint }: { label: ReactNode; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-lg border bg-card px-4 py-3.5">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}

export function KeyValues({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[minmax(8rem,auto)_1fr] gap-x-6 gap-y-2 text-sm">
      {items.map(([key, value], index) => (
        <div key={index} className="contents">
          <dt className="text-muted-foreground">{key}</dt>
          <dd className="min-w-0 break-words">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

// ------------------------------------------------------------ identities

/** Fixed categorical slot per property: identity never depends on rank. */
export const PROPERTY_SLOT: Record<Property, string> = {
  imperceptibility: "var(--series-1)",
  robustness: "var(--series-2)",
  capacity: "var(--series-3)",
  security: "var(--series-7)",
  multi_criteria: "var(--chart-muted)",
};

export const PROPERTY_ORDER: Property[] = ["imperceptibility", "robustness", "capacity", "security", "multi_criteria"];

export function PropertyTag({ property, className }: { property: Property; className?: string }) {
  const { t } = useI18n();
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-xs text-muted-foreground", className)}>
      <span className="h-2 w-2 rounded-full" style={{ background: PROPERTY_SLOT[property] }} aria-hidden />
      {t(`properties.${property}.name`)}
    </span>
  );
}

const STATUS_ICON: Record<RunStatus, typeof CheckCircle2> = {
  queued: CircleDashed,
  running: Loader2,
  completed: CheckCircle2,
  failed: XCircle,
  cancelled: CircleSlash,
  interrupted: PauseCircle,
};

const STATUS_COLOR: Record<RunStatus, string> = {
  queued: "var(--chart-muted)",
  running: "var(--series-1)",
  completed: "var(--status-good)",
  failed: "var(--status-critical)",
  cancelled: "var(--chart-muted)",
  interrupted: "var(--status-serious)",
};

export function RunStatusBadge({ status }: { status: RunStatus }) {
  const { t } = useI18n();
  const Icon = STATUS_ICON[status] ?? CircleDashed;
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-xs font-medium">
      <Icon
        className={cn("h-3.5 w-3.5", status === "running" && "animate-spin")}
        style={{ color: STATUS_COLOR[status] }}
        aria-hidden
      />
      {t(`runStatus.${status}`)}
    </span>
  );
}

export function Chip({ children, className, title }: { children: ReactNode; className?: string; title?: string }) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center rounded border bg-background px-1.5 py-0.5 text-[11px] leading-none text-muted-foreground",
        className
      )}
    >
      {children}
    </span>
  );
}

export function Spec({ children, className }: { children: ReactNode; className?: string }) {
  return <code className={cn("spec rounded bg-muted px-1.5 py-0.5 text-foreground", className)}>{children}</code>;
}

// --------------------------------------------------------------- numbers

export function EstimateText({ value, digits = 3, percent = false }: { value?: Estimate | null; digits?: number; percent?: boolean }) {
  const { number, percent: pct } = useI18n();
  if (!value || value.estimate === null || value.estimate === undefined) return <span className="text-muted-foreground">–</span>;
  const format = (v: number | null) => (percent ? pct(v, 1) : number(v, digits));
  return (
    <span className="num whitespace-nowrap">
      {format(value.estimate)}
      {value.ci95_low !== null && value.ci95_low !== undefined && (
        <span className="text-muted-foreground">
          {" "}
          [{format(value.ci95_low)}, {format(value.ci95_high)}]
        </span>
      )}
    </span>
  );
}

// ---------------------------------------------------------------- states

export function LoadingLine({ label }: { label?: string }) {
  const { t } = useI18n();
  return (
    <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" /> {label ?? t("common.loading")}
    </div>
  );
}

export function ErrorNotice({ error, onRetry }: { error: string; onRetry?: () => void }) {
  const { t } = useI18n();
  const unreachable = /Failed to fetch|NetworkError|Load failed/i.test(error);
  return (
    <Alert variant="destructive" className="my-4">
      <AlertTriangle className="h-4 w-4" />
      <AlertDescription className="flex flex-wrap items-center justify-between gap-3">
        <span>{unreachable ? t("common.apiUnavailable", { url: API_BASE }) : error}</span>
        {onRetry && (
          <button type="button" className="text-xs font-medium underline underline-offset-2" onClick={onRetry}>
            {t("common.retry")}
          </button>
        )}
      </AlertDescription>
    </Alert>
  );
}

export function EmptyState({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed px-6 py-12 text-center text-sm text-muted-foreground">
      <p>{children}</p>
      {action}
    </div>
  );
}

// -------------------------------------------------------------- helpers

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "–";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ${Math.round(seconds % 60)} s`;
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

export function runDuration(started: string | null, finished: string | null): number | null {
  if (!started) return null;
  const end = finished ? new Date(finished).getTime() : Date.now();
  return (end - new Date(started).getTime()) / 1000;
}
