"use client";

import { useState } from "react";

import { AXIS_TEXT, Legend, Tooltip, TooltipRow, formatTick, linear, niceDomain, niceTicks, seriesColor, tickDigits, useWidth } from "./base";

export interface TradeoffPoint {
  label: string;
  x: number | null;
  xLo?: number | null;
  xHi?: number | null;
  y: number | null;
  yLo?: number | null;
  yHi?: number | null;
  optimal?: boolean;
}

/**
 * The sweep settings of one method as a path through (quality, BER) space,
 * with 95% whiskers on both axes, the setting labelled at each point, and
 * reference methods as separate markers. Pareto-optimal settings are filled,
 * dominated ones hollow.
 */
export function TradeoffChart({
  points,
  references,
  xTitle,
  yTitle,
  xHigherIsBetter,
  seriesLabel,
  optimalLabel,
  dominatedLabel,
  betterWord = "better",
  outsideLabel = "Outside the plotted range",
  formatX = (value: number) => value.toFixed(2),
  formatY = (value: number) => value.toFixed(3),
  height = 360,
}: {
  points: TradeoffPoint[];
  references: TradeoffPoint[];
  xTitle: string;
  yTitle: string;
  xHigherIsBetter: boolean | null;
  seriesLabel: string;
  optimalLabel: string;
  dominatedLabel: string;
  betterWord?: string;
  outsideLabel?: string;
  formatX?: (value: number) => string;
  formatY?: (value: number) => string;
  height?: number;
}) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ point: TradeoffPoint; reference: boolean } | null>(null);
  const margin = { top: 16, right: 28, bottom: 46, left: 56 };
  const plotWidth = Math.max(120, width - margin.left - margin.right);
  const plotHeight = height - margin.top - margin.bottom;

  // The axes frame the swept settings. A reference far outside them (LSB
  // has an SNR near 200 dB) would squeeze the curve into a corner, so it is
  // listed under the chart instead of stretching the axis.
  const finite = (values: (number | null | undefined)[]) =>
    values.filter((v): v is number => v !== null && v !== undefined && Number.isFinite(v));
  const xs = finite(points.flatMap((p) => [p.x, p.xLo, p.xHi]));
  const ys = finite(points.flatMap((p) => [p.y, p.yLo, p.yHi]));
  const xSpan = Math.max(1e-9, Math.max(...xs) - Math.min(...xs));
  const [xLow, xHigh] = niceDomain(Math.min(...xs) - xSpan * 0.08, Math.max(...xs) + xSpan * 0.08);
  const [, yHigh] = niceDomain(0, Math.max(0.02, ...ys) * 1.1);
  const xTicks = niceTicks(xLow, xHigh, 6).filter((tick) => tick >= xLow && tick <= xHigh);
  const yTicks = niceTicks(0, yHigh, 5).filter((tick) => tick <= yHigh);
  const x = linear([xLow, xHigh], [0, plotWidth]);
  const y = linear([0, yHigh], [plotHeight, 0]);
  const inside = (point: TradeoffPoint) =>
    point.x !== null && point.y !== null && point.x >= xLow && point.x <= xHigh && point.y <= yHigh;
  const shownReferences = references.filter(inside);
  const outsideReferences = references.filter((point) => !inside(point));
  const color = seriesColor(0);
  const referenceColor = seriesColor(1);

  const valid = points.filter((p) => p.x !== null && p.y !== null);
  const path = valid.map((p, index) => `${index ? "L" : "M"}${x(p.x as number)},${y(p.y as number)}`).join("");

  function whiskers(point: TradeoffPoint, stroke: string) {
    if (point.x === null || point.y === null) return null;
    return (
      <g stroke={stroke} strokeWidth={1.5} opacity={0.6}>
        {point.xLo !== null && point.xLo !== undefined && point.xHi !== null && point.xHi !== undefined && (
          <line x1={x(point.xLo)} x2={x(point.xHi)} y1={y(point.y)} y2={y(point.y)} />
        )}
        {point.yLo !== null && point.yLo !== undefined && point.yHi !== null && point.yHi !== undefined && (
          <line x1={x(point.x)} x2={x(point.x)} y1={y(point.yLo)} y2={y(point.yHi)} />
        )}
      </g>
    );
  }

  return (
    <div ref={ref} className="relative">
      <svg width={width} height={height} role="img" aria-label={`${yTitle} / ${xTitle}`}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {yTicks.map((tick) => (
            <g key={`y${tick}`}>
              <line x1={0} x2={plotWidth} y1={y(tick)} y2={y(tick)} stroke="var(--chart-grid)" />
              <text x={-8} y={y(tick)} dy="0.32em" textAnchor="end" {...AXIS_TEXT} className="num">
                {formatTick(tick, tickDigits(yTicks))}
              </text>
            </g>
          ))}
          {xTicks.map((tick) => (
            <text key={`x${tick}`} x={x(tick)} y={plotHeight + 16} textAnchor="middle" {...AXIS_TEXT} className="num">
              {formatTick(tick, tickDigits(xTicks))}
            </text>
          ))}
          <line x1={0} x2={plotWidth} y1={plotHeight} y2={plotHeight} stroke="var(--chart-axis)" />
          <text x={plotWidth / 2} y={plotHeight + 38} textAnchor="middle" fontSize={11} fill="var(--chart-ink-2)">
            {xTitle}
            {xHigherIsBetter === null ? "" : xHigherIsBetter ? `  →  ${betterWord}` : `  ←  ${betterWord}`}
          </text>
          <text transform={`translate(${-44},${plotHeight / 2}) rotate(-90)`} textAnchor="middle" fontSize={11} fill="var(--chart-ink-2)">
            {yTitle}  ↓ {betterWord}
          </text>

          <path d={path} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
          {points.map((point) => (
            <g key={`w${point.label}`}>{whiskers(point, color)}</g>
          ))}
          {shownReferences.map((point) => (
            <g key={`rw${point.label}`}>{whiskers(point, referenceColor)}</g>
          ))}
          {points.map((point) =>
            point.x === null || point.y === null ? null : (
              <g
                key={point.label}
                onPointerEnter={() => setHover({ point, reference: false })}
                onPointerLeave={() => setHover(null)}
              >
                <circle cx={x(point.x)} cy={y(point.y)} r={12} fill="transparent" />
                <circle
                  cx={x(point.x)}
                  cy={y(point.y)}
                  r={5}
                  fill={point.optimal ? color : "var(--chart-surface)"}
                  stroke={point.optimal ? "var(--chart-surface)" : color}
                  strokeWidth={2}
                />
                <text
                  x={x(point.x) > plotWidth - 90 ? x(point.x) - 8 : x(point.x) + 8}
                  y={y(point.y) - 8}
                  textAnchor={x(point.x) > plotWidth - 90 ? "end" : "start"}
                  fontSize={11}
                  fill="var(--chart-ink)"
                  className="num"
                >
                  {point.label}
                </text>
              </g>
            )
          )}
          {shownReferences.map((point) =>
            point.x === null || point.y === null ? null : (
              <g
                key={`ref-${point.label}`}
                onPointerEnter={() => setHover({ point, reference: true })}
                onPointerLeave={() => setHover(null)}
              >
                <rect x={x(point.x) - 12} y={y(point.y) - 12} width={24} height={24} fill="transparent" />
                <rect
                  x={x(point.x) - 5}
                  y={y(point.y) - 5}
                  width={10}
                  height={10}
                  transform={`rotate(45 ${x(point.x)} ${y(point.y)})`}
                  fill={referenceColor}
                  stroke="var(--chart-surface)"
                  strokeWidth={2}
                />
                <text x={x(point.x) + 9} y={y(point.y) + 4} fontSize={11} fill="var(--chart-ink-2)">
                  {point.label.length > 28 ? `${point.label.slice(0, 26)}…` : point.label}
                </text>
              </g>
            )
          )}
        </g>
      </svg>
      {hover && hover.point.x !== null && hover.point.y !== null && (
        <Tooltip x={margin.left + x(hover.point.x)} y={margin.top + y(hover.point.y)} width={width}>
          <div className="mb-1 font-medium">{hover.point.label}</div>
          <TooltipRow label={xTitle} value={formatX(hover.point.x)} />
          <TooltipRow label={yTitle} value={formatY(hover.point.y)} />
          {!hover.reference && <div className="mt-1 text-muted-foreground">{hover.point.optimal ? optimalLabel : dominatedLabel}</div>}
        </Tooltip>
      )}
      {outsideReferences.length > 0 && (
        <p className="mt-2 text-xs text-muted-foreground">
          {outsideLabel}:{" "}
          {outsideReferences
            .map((point) => `${point.label} (${point.x === null ? "–" : formatX(point.x)}, ${point.y === null ? "–" : formatY(point.y)})`)
            .join("; ")}
        </p>
      )}
      <Legend
        kind="dot"
        items={[
          { label: `${seriesLabel} — ${optimalLabel}`, color },
          { label: `${seriesLabel} — ${dominatedLabel}`, color, hollow: true },
          ...(references.length ? [{ label: references.map((r) => r.label).join(", "), color: referenceColor }] : []),
        ]}
      />
    </div>
  );
}
