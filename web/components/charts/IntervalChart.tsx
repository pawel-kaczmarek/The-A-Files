"use client";

import { useState } from "react";

import { AXIS_TEXT, Tooltip, TooltipRow, formatTick, linear, niceDomain, niceTicks, shorten, tickDigits, useWidth } from "./base";

export interface IntervalRow {
  label: string;
  /** Unabbreviated label, shown on hover. */
  fullLabel?: string;
  estimate: number | null;
  lo?: number | null;
  hi?: number | null;
  note?: string;
  /** The true value is at least ``estimate``: drawn with an open arrow. */
  censored?: boolean;
}

/**
 * A forest plot: one row per entity, the estimate as a dot and its interval
 * as a whisker, against an optional reference line (e.g. chance level).
 */
export function IntervalChart({
  rows,
  axisTitle,
  domain,
  reference,
  referenceLabel,
  format = (value: number) => value.toFixed(3),
  color = "var(--series-1)",
}: {
  rows: IntervalRow[];
  axisTitle: string;
  domain?: [number, number];
  reference?: number;
  referenceLabel?: string;
  format?: (value: number) => string;
  color?: string;
}) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const labelWidth = Math.min(300, Math.max(140, width * 0.32));
  const margin = { top: 8, right: 24, bottom: 40, left: labelWidth };
  const rowHeight = 28;
  const plotWidth = Math.max(100, width - margin.left - margin.right);
  const height = margin.top + margin.bottom + rows.length * rowHeight;

  const values = rows.flatMap((row) => [row.estimate, row.lo, row.hi]).filter((v): v is number => v !== null && v !== undefined);
  const [lower, upper] =
    domain ?? niceDomain(Math.min(0, ...values, reference ?? 0), Math.max(...values, reference ?? 0) * 1.05);
  const ticks = niceTicks(lower, upper, 5).filter((tick) => tick >= lower - 1e-9 && tick <= upper + 1e-9);
  const x = linear([lower, upper], [0, plotWidth]);

  return (
    <div ref={ref} className="relative">
      <svg width={width} height={height} role="img" aria-label={axisTitle}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {ticks.map((tick) => (
            <g key={tick}>
              <line x1={x(tick)} x2={x(tick)} y1={0} y2={rows.length * rowHeight} stroke="var(--chart-grid)" />
              <text x={x(tick)} y={rows.length * rowHeight + 16} textAnchor="middle" {...AXIS_TEXT} className="num">
                {formatTick(tick, tickDigits(ticks))}
              </text>
            </g>
          ))}
          <text x={plotWidth / 2} y={rows.length * rowHeight + 34} textAnchor="middle" fontSize={11} fill="var(--chart-ink-2)">
            {axisTitle}
          </text>
          {reference !== undefined && (
            <g>
              <line x1={x(reference)} x2={x(reference)} y1={-4} y2={rows.length * rowHeight} stroke="var(--chart-ink-2)" strokeDasharray="4 4" />
              {referenceLabel && (
                <text x={x(reference) + 4} y={4} fontSize={10} fill="var(--chart-ink-2)">
                  {referenceLabel}
                </text>
              )}
            </g>
          )}
          {rows.map((row, index) => {
            const cy = index * rowHeight + rowHeight / 2;
            return (
              <g key={row.label + index} onPointerEnter={() => setHover(index)} onPointerLeave={() => setHover(null)}>
                <rect x={-labelWidth} y={index * rowHeight} width={labelWidth + plotWidth} height={rowHeight} fill={hover === index ? "var(--chart-grid)" : "transparent"} opacity={0.5} />
                <text x={-12} y={cy} dy="0.32em" textAnchor="end" fontSize={12} fill="var(--chart-ink)">
                  <title>{row.fullLabel ?? row.label}</title>
                  {shorten(row.label, Math.max(24, Math.floor(labelWidth / 6.4)))}
                </text>
                {row.lo !== null && row.lo !== undefined && row.hi !== null && row.hi !== undefined && (
                  <line x1={x(row.lo)} x2={x(row.hi)} y1={cy} y2={cy} stroke={color} strokeWidth={2} strokeLinecap="round" />
                )}
                {row.estimate !== null && (
                  <circle cx={x(row.estimate)} cy={cy} r={4.5} fill={color} stroke="var(--chart-surface)" strokeWidth={2} />
                )}
                {row.estimate !== null && row.censored && (
                  <path
                    d={`M${x(row.estimate) + 9},${cy - 5} L${x(row.estimate) + 15},${cy} L${x(row.estimate) + 9},${cy + 5}`}
                    fill="none"
                    stroke={color}
                    strokeWidth={2}
                    strokeLinecap="round"
                  />
                )}
              </g>
            );
          })}
        </g>
      </svg>
      {hover !== null && rows[hover] && (
        <Tooltip x={margin.left + plotWidth / 2} y={margin.top + (hover + 1) * rowHeight} width={width}>
          <div className="mb-1 font-medium">{rows[hover].fullLabel ?? rows[hover].label}</div>
          <TooltipRow label={axisTitle} value={rows[hover].estimate === null ? "–" : format(rows[hover].estimate as number)} />
          {rows[hover].lo !== null && rows[hover].lo !== undefined && (
            <TooltipRow label="95% CI" value={`[${format(rows[hover].lo as number)}, ${format(rows[hover].hi as number)}]`} />
          )}
          {rows[hover].note && <div className="mt-1 text-muted-foreground">{rows[hover].note}</div>}
        </Tooltip>
      )}
    </div>
  );
}
