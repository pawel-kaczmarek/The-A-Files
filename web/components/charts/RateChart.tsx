"use client";

import { useState } from "react";

import type { Estimate } from "@/lib/types";

import { AXIS_TEXT, Legend, Tooltip, TooltipRow, shorten, useWidth } from "./base";

/**
 * Nested rates per row (e.g. BER = 0 within BER <= 1% within BER <= 5%), each
 * as a dot with its 95% interval. The rates are ordered, so they take steps of
 * one sequential hue, strictest darkest, and are also offset vertically so no
 * dot hides another.
 */
export function RateChart({
  rows,
  series,
  axisTitle,
  percent,
}: {
  rows: { label: string; fullLabel?: string; values: (Estimate | null)[]; note?: string }[];
  series: string[];
  axisTitle: string;
  percent: (value: number | null) => string;
}) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const labelWidth = Math.min(260, Math.max(120, width * 0.26));
  const margin = { top: 8, right: 24, bottom: 40, left: labelWidth };
  const rowHeight = 40;
  const plotWidth = Math.max(100, width - margin.left - margin.right);
  const x = (value: number) => value * plotWidth;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const height = margin.top + margin.bottom + rows.length * rowHeight;
  const colors = ["var(--seq-700)", "var(--seq-500)", "var(--seq-300)"];
  const offsets = series.map((_, index) => (index - (series.length - 1) / 2) * 9);

  return (
    <div ref={ref} className="relative">
      <svg width={width} height={height} role="img" aria-label={axisTitle}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {ticks.map((tick) => (
            <g key={tick}>
              <line x1={x(tick)} x2={x(tick)} y1={0} y2={rows.length * rowHeight} stroke="var(--chart-grid)" />
              <text x={x(tick)} y={rows.length * rowHeight + 16} textAnchor="middle" {...AXIS_TEXT} className="num">
                {percent(tick)}
              </text>
            </g>
          ))}
          <text x={plotWidth / 2} y={rows.length * rowHeight + 34} textAnchor="middle" fontSize={11} fill="var(--chart-ink-2)">
            {axisTitle}
          </text>
          {rows.map((row, index) => {
            const cy = index * rowHeight + rowHeight / 2;
            return (
              <g key={row.label + index} onPointerEnter={() => setHover(index)} onPointerLeave={() => setHover(null)}>
                <rect x={-labelWidth} y={index * rowHeight} width={labelWidth + plotWidth} height={rowHeight} fill={hover === index ? "var(--chart-grid)" : "transparent"} opacity={0.5} />
                <text x={-12} y={cy} dy="0.32em" textAnchor="end" fontSize={12} fill="var(--chart-ink)">
                  <title>{row.fullLabel ?? row.label}</title>
                  {shorten(row.label, Math.max(18, Math.floor(labelWidth / 6.4)))}
                </text>
                {row.values.map((value, position) =>
                  value && value.estimate !== null ? (
                    <g key={position}>
                      {value.ci95_low !== null && value.ci95_high !== null && (
                        <line
                          x1={x(value.ci95_low)}
                          x2={x(value.ci95_high)}
                          y1={cy + offsets[position]}
                          y2={cy + offsets[position]}
                          stroke={colors[position]}
                          strokeWidth={2}
                          strokeLinecap="round"
                        />
                      )}
                      <circle cx={x(value.estimate)} cy={cy + offsets[position]} r={4} fill={colors[position]} stroke="var(--chart-surface)" strokeWidth={2} />
                    </g>
                  ) : null
                )}
              </g>
            );
          })}
        </g>
      </svg>
      {hover !== null && rows[hover] && (
        <Tooltip x={margin.left + plotWidth / 2} y={margin.top + (hover + 1) * rowHeight} width={width}>
          <div className="mb-1 font-medium">{rows[hover].fullLabel ?? rows[hover].label}</div>
          {rows[hover].values.map((value, position) => (
            <TooltipRow
              key={position}
              color={colors[position]}
              label={series[position]}
              value={
                value && value.estimate !== null
                  ? `${percent(value.estimate)}${value.ci95_low !== null ? ` [${percent(value.ci95_low)}, ${percent(value.ci95_high)}]` : ""}`
                  : "–"
              }
            />
          ))}
          {rows[hover].note && <div className="mt-1 text-muted-foreground">{rows[hover].note}</div>}
        </Tooltip>
      )}
      <Legend kind="dot" items={series.map((label, index) => ({ label, color: colors[index] }))} />
    </div>
  );
}
