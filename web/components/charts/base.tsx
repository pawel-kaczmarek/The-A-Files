"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { BarChart3, Table2 } from "lucide-react";

import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";

/** Categorical slot for the n-th entity. Past eight the series folds to muted. */
export function seriesColor(index: number): string {
  return index < 8 ? `var(--series-${index + 1})` : "var(--chart-muted)";
}

export const MAX_SERIES = 8;

export function useWidth<T extends HTMLElement>(fallback = 640) {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      const next = Math.floor(entries[0].contentRect.width);
      if (next > 0) setWidth(next);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return { ref, width };
}

export function linear(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (value: number) => r0 + ((value - d0) / span) * (r1 - r0);
}

/** Round tick values covering ``[min, max]``. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const raw = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? raw;
  const ticks: number[] = [];
  for (let value = Math.ceil(min / step) * step; value <= max + step * 1e-9; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }
  return ticks;
}

/** Decimal places that show every tick exactly (0.025 needs three, not two). */
export function tickDigits(ticks: number[]): number {
  if (ticks.length < 2) return 2;
  const step = Math.abs(ticks[1] - ticks[0]);
  let digits = 0;
  while (digits < 6 && Math.abs(step * 10 ** digits - Math.round(step * 10 ** digits)) > 1e-6) digits++;
  return digits;
}

/** An axis value in the page language's number format. */
export function formatTick(value: number, digits: number): string {
  const locale = typeof document !== "undefined" && document.documentElement.lang === "pl" ? "pl-PL" : "en-US";
  return value.toLocaleString(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** ``text`` cut to ``limit`` characters, keeping ``suffix`` whole. */
export function shorten(text: string, limit: number, suffix = ""): string {
  const room = limit - suffix.length;
  return (text.length > room ? `${text.slice(0, Math.max(8, room - 1))}…` : text) + suffix;
}

export function niceDomain(min: number, max: number): [number, number] {
  const ticks = niceTicks(min, max);
  if (ticks.length < 2) return [min, max];
  const step = ticks[1] - ticks[0];
  return [Math.min(min, Math.floor(min / step) * step), Math.max(max, Math.ceil(max / step) * step)];
}

export function Legend({ items, kind = "line" }: { items: { label: string; color: string; hollow?: boolean }[]; kind?: "line" | "dot" | "rect" }) {
  if (items.length < 2) return null;
  return (
    <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-1.5">
          {kind === "line" ? (
            <span className="h-0.5 w-4 rounded" style={{ background: item.color }} aria-hidden />
          ) : kind === "rect" ? (
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: item.color }} aria-hidden />
          ) : (
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={item.hollow ? { border: `2px solid ${item.color}` } : { background: item.color }}
              aria-hidden
            />
          )}
          <span className="max-w-[28rem] truncate" title={item.label}>
            {item.label}
          </span>
        </li>
      ))}
    </ul>
  );
}

export interface TableData {
  columns: string[];
  rows: (string | number | ReactNode)[][];
}

/** Title, hint, and a switch between the figure and its table (the accessible twin). */
export function ChartFrame({
  title,
  hint,
  table,
  children,
  actions,
}: {
  title: ReactNode;
  hint?: ReactNode;
  table?: TableData;
  children: ReactNode;
  actions?: ReactNode;
}) {
  const { t } = useI18n();
  const [showTable, setShowTable] = useState(false);
  return (
    <figure className="rounded-lg border bg-card">
      <figcaption className="flex flex-wrap items-start justify-between gap-3 border-b px-5 py-3.5">
        <div className="min-w-0">
          <div className="text-sm font-semibold">{title}</div>
          {hint && <div className="mt-0.5 text-xs leading-relaxed text-muted-foreground">{hint}</div>}
        </div>
        <div className="flex items-center gap-2">
          {actions}
          {table && (
            <button
              type="button"
              onClick={() => setShowTable((value) => !value)}
              className="inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
            >
              {showTable ? <BarChart3 className="h-3.5 w-3.5" /> : <Table2 className="h-3.5 w-3.5" />}
              {showTable ? t("common.showChart") : t("common.showTable")}
            </button>
          )}
        </div>
      </figcaption>
      <div className="p-5">{showTable && table ? <DataTable table={table} /> : children}</div>
    </figure>
  );
}

export function DataTable({ table, className }: { table: TableData; className?: string }) {
  return (
    <div className={cn("overflow-x-auto", className)}>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-xs text-muted-foreground">
            {table.columns.map((column, index) => (
              <th key={column} className={cn("whitespace-nowrap px-2 py-1.5 font-medium", index > 0 && "text-right")}>
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, rowIndex) => (
            <tr key={rowIndex} className="border-b last:border-0">
              {row.map((cell, index) => (
                <td key={index} className={cn("px-2 py-1.5", index > 0 ? "whitespace-nowrap text-right" : "max-w-[24rem]")}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A tooltip box positioned inside a relative container. */
export function Tooltip({ x, y, width, children }: { x: number; y: number; width: number; children: ReactNode }) {
  const left = Math.min(Math.max(x + 14, 0), Math.max(0, width - 260));
  return (
    <div
      className="pointer-events-none absolute z-10 w-max max-w-[260px] rounded-md border bg-card px-3 py-2 text-xs shadow-md"
      style={{ left, top: Math.max(0, y - 10) }}
    >
      {children}
    </div>
  );
}

export function TooltipRow({ color, label, value }: { color?: string; label: string; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="flex min-w-0 items-center gap-1.5 text-muted-foreground">
        {color && <span className="h-0.5 w-3 shrink-0 rounded" style={{ background: color }} />}
        <span className="truncate">{label}</span>
      </span>
      <span className="num font-semibold text-foreground">{value}</span>
    </div>
  );
}

export const AXIS_TEXT = { fontSize: 11, fill: "var(--chart-muted)" } as const;
