"use client";

import { useState } from "react";

import type { BoxSummary } from "@/lib/types";

import { AXIS_TEXT, Tooltip, TooltipRow, formatTick, linear, niceDomain, niceTicks, shorten, tickDigits, useWidth } from "./base";

export interface BoxRow {
  label: string;
  fullLabel?: string;
  box: BoxSummary | null;
}

/**
 * Horizontal Tukey box plots, one per entity: box from Q1 to Q3, median bar,
 * whiskers to the most extreme values within 1.5 IQR, the mean as a small
 * diamond. Outliers are counted in the tooltip rather than drawn one by one.
 */
export function BoxPlot({
  rows,
  axisTitle,
  domain,
  format = (value: number) => value.toFixed(3),
  labels,
}: {
  rows: BoxRow[];
  axisTitle: string;
  domain?: [number, number];
  format?: (value: number) => string;
  labels: { median: string; quartiles: string; whiskers: string; mean: string; outliers: string; n: string };
}) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const labelWidth = Math.min(260, Math.max(120, width * 0.26));
  const margin = { top: 8, right: 24, bottom: 40, left: labelWidth };
  const rowHeight = 30;
  const plotWidth = Math.max(100, width - margin.left - margin.right);
  const values = rows.flatMap((row) => (row.box ? [row.box.min, row.box.max] : []));
  const [lower, upper] = domain ?? niceDomain(Math.min(0, ...values), Math.max(0.05, ...values));
  const ticks = niceTicks(lower, upper, 5).filter((tick) => tick >= lower - 1e-9 && tick <= upper + 1e-9);
  const x = linear([lower, upper], [0, plotWidth]);
  const height = margin.top + margin.bottom + rows.length * rowHeight;
  const color = "var(--series-1)";

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
          {rows.map((row, index) => {
            const cy = index * rowHeight + rowHeight / 2;
            const box = row.box;
            return (
              <g key={row.label + index} onPointerEnter={() => setHover(index)} onPointerLeave={() => setHover(null)}>
                <rect x={-labelWidth} y={index * rowHeight} width={labelWidth + plotWidth} height={rowHeight} fill={hover === index ? "var(--chart-grid)" : "transparent"} opacity={0.5} />
                <text x={-12} y={cy} dy="0.32em" textAnchor="end" fontSize={12} fill="var(--chart-ink)">
                  <title>{row.fullLabel ?? row.label}</title>
                  {shorten(row.label, Math.max(18, Math.floor(labelWidth / 6.4)))}
                </text>
                {box && (
                  <g>
                    <line x1={x(box.whisker_low)} x2={x(box.q1)} y1={cy} y2={cy} stroke="var(--chart-ink-2)" strokeWidth={1.5} />
                    <line x1={x(box.q3)} x2={x(box.whisker_high)} y1={cy} y2={cy} stroke="var(--chart-ink-2)" strokeWidth={1.5} />
                    <line x1={x(box.whisker_low)} x2={x(box.whisker_low)} y1={cy - 5} y2={cy + 5} stroke="var(--chart-ink-2)" strokeWidth={1.5} />
                    <line x1={x(box.whisker_high)} x2={x(box.whisker_high)} y1={cy - 5} y2={cy + 5} stroke="var(--chart-ink-2)" strokeWidth={1.5} />
                    <rect
                      x={x(box.q1)}
                      y={cy - 8}
                      width={Math.max(2, x(box.q3) - x(box.q1))}
                      height={16}
                      rx={3}
                      fill={color}
                      fillOpacity={0.18}
                      stroke={color}
                      strokeWidth={1.5}
                    />
                    <line x1={x(box.median)} x2={x(box.median)} y1={cy - 8} y2={cy + 8} stroke={color} strokeWidth={2.5} />
                    <rect
                      x={x(box.mean) - 3.5}
                      y={cy - 3.5}
                      width={7}
                      height={7}
                      transform={`rotate(45 ${x(box.mean)} ${cy})`}
                      fill="var(--chart-surface)"
                      stroke="var(--chart-ink)"
                      strokeWidth={1.5}
                    />
                  </g>
                )}
              </g>
            );
          })}
        </g>
      </svg>
      {hover !== null && rows[hover]?.box && (
        <Tooltip x={margin.left + plotWidth / 2} y={margin.top + (hover + 1) * rowHeight} width={width}>
          <div className="mb-1 font-medium">{rows[hover].fullLabel ?? rows[hover].label}</div>
          <TooltipRow label={labels.median} value={format(rows[hover].box!.median)} />
          <TooltipRow label={labels.quartiles} value={`${format(rows[hover].box!.q1)} – ${format(rows[hover].box!.q3)}`} />
          <TooltipRow label={labels.whiskers} value={`${format(rows[hover].box!.whisker_low)} – ${format(rows[hover].box!.whisker_high)}`} />
          <TooltipRow label={labels.mean} value={format(rows[hover].box!.mean)} />
          <TooltipRow label={labels.outliers} value={String(rows[hover].box!.outliers)} />
          <TooltipRow label={labels.n} value={String(rows[hover].box!.n)} />
        </Tooltip>
      )}
    </div>
  );
}
