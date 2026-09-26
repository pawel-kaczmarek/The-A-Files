"use client";

import { useState } from "react";

import { AXIS_TEXT, Legend, Tooltip, TooltipRow, formatTick, linear, niceDomain, niceTicks, seriesColor, tickDigits, useWidth } from "./base";

export interface ScatterPoint {
  /** Position in the fixed entity order: the colour follows the entity. */
  slot: number;
  label: string;
  fullLabel?: string;
  x: number | null;
  xLo?: number | null;
  xHi?: number | null;
  y: number | null;
  yLo?: number | null;
  yHi?: number | null;
}

/**
 * One point per entity in the plane of two measures, with 95% whiskers on
 * both axes and a direct label per point; the better direction of each axis
 * is written on the axis title.
 */
export function ScatterChart({
  points,
  xTitle,
  yTitle,
  xBetter,
  yBetter,
  betterWord,
  formatX = (value: number) => value.toFixed(2),
  formatY = (value: number) => value.toFixed(3),
  height = 340,
}: {
  points: ScatterPoint[];
  xTitle: string;
  yTitle: string;
  /** "up" when higher is better, "down" when lower is, null when undeclared. */
  xBetter: "up" | "down" | null;
  yBetter: "up" | "down" | null;
  betterWord: string;
  formatX?: (value: number) => string;
  formatY?: (value: number) => string;
  height?: number;
}) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<ScatterPoint | null>(null);
  const margin = { top: 16, right: 28, bottom: 46, left: 56 };
  const plotWidth = Math.max(120, width - margin.left - margin.right);
  const plotHeight = height - margin.top - margin.bottom;
  const finite = (values: (number | null | undefined)[]) => values.filter((v): v is number => v !== null && v !== undefined && Number.isFinite(v));
  const shown = points.filter((p) => p.x !== null && p.y !== null);
  const xs = finite(shown.flatMap((p) => [p.x, p.xLo, p.xHi]));
  const ys = finite(shown.flatMap((p) => [p.y, p.yLo, p.yHi]));
  if (!shown.length) return null;
  const xSpan = Math.max(1e-9, Math.max(...xs) - Math.min(...xs));
  const [xLow, xHigh] = niceDomain(Math.min(...xs) - xSpan * 0.08, Math.max(...xs) + xSpan * 0.08);
  const [, yHigh] = niceDomain(0, Math.max(0.02, ...ys) * 1.1);
  const xTicks = niceTicks(xLow, xHigh, 6).filter((tick) => tick >= xLow && tick <= xHigh);
  const yTicks = niceTicks(0, yHigh, 5).filter((tick) => tick <= yHigh);
  const x = linear([xLow, xHigh], [0, plotWidth]);
  const y = linear([0, yHigh], [plotHeight, 0]);
  // Greedy label placement: above the point, else below, else further up.
  const placed: { x0: number; x1: number; y: number }[] = [];
  const placements = shown.map((point) => {
    const cx = x(point.x!);
    const cy = y(point.y!);
    const widthGuess = point.label.length * 6.5;
    const right = cx > plotWidth - 110;
    const x0 = right ? cx - 8 - widthGuess : cx + 8;
    const x1 = x0 + widthGuess;
    const candidates = [cy - 8, cy + 16, cy - 22, cy + 30];
    const free = candidates.find((candidate) => !placed.some((box) => x0 < box.x1 && x1 > box.x0 && Math.abs(box.y - candidate) < 12));
    const chosen = free ?? candidates[0];
    placed.push({ x0, x1, y: chosen });
    return chosen;
  });
  const arrow = (better: "up" | "down" | null, horizontal: boolean) =>
    better === null ? "" : `  ${better === "up" ? (horizontal ? "→" : "↑") : horizontal ? "←" : "↓"} ${betterWord}`;

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
            {arrow(xBetter, true)}
          </text>
          <text transform={`translate(${-44},${plotHeight / 2}) rotate(-90)`} textAnchor="middle" fontSize={11} fill="var(--chart-ink-2)">
            {yTitle}
            {arrow(yBetter, false)}
          </text>
          {shown.map((point) => {
            const color = seriesColor(point.slot);
            return (
              <g key={`w${point.label}`} stroke={color} strokeWidth={1.5} opacity={0.55}>
                {point.xLo != null && point.xHi != null && <line x1={x(point.xLo)} x2={x(point.xHi)} y1={y(point.y!)} y2={y(point.y!)} />}
                {point.yLo != null && point.yHi != null && <line x1={x(point.x!)} x2={x(point.x!)} y1={y(point.yLo)} y2={y(point.yHi)} />}
              </g>
            );
          })}
          {shown.map((point, index) => {
            const cx = x(point.x!);
            const cy = y(point.y!);
            const right = cx > plotWidth - 110;
            const labelY = placements[index];
            return (
              <g key={point.label} onPointerEnter={() => setHover(point)} onPointerLeave={() => setHover(null)}>
                <circle cx={cx} cy={cy} r={12} fill="transparent" />
                <circle cx={cx} cy={cy} r={5} fill={seriesColor(point.slot)} stroke="var(--chart-surface)" strokeWidth={2} />
                <text x={right ? cx - 8 : cx + 8} y={labelY} textAnchor={right ? "end" : "start"} fontSize={11} fill="var(--chart-ink)">
                  {point.label}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
      {hover && (
        <Tooltip x={margin.left + x(hover.x!)} y={margin.top + y(hover.y!)} width={width}>
          <div className="mb-1 font-medium">{hover.fullLabel ?? hover.label}</div>
          <TooltipRow
            label={xTitle}
            value={`${formatX(hover.x!)}${hover.xLo != null && hover.xHi != null ? ` [${formatX(hover.xLo)}, ${formatX(hover.xHi)}]` : ""}`}
          />
          <TooltipRow
            label={yTitle}
            value={`${formatY(hover.y!)}${hover.yLo != null && hover.yHi != null ? ` [${formatY(hover.yLo)}, ${formatY(hover.yHi)}]` : ""}`}
          />
        </Tooltip>
      )}
      <Legend kind="dot" items={points.map((point) => ({ label: point.fullLabel ?? point.label, color: seriesColor(point.slot) }))} />
    </div>
  );
}
