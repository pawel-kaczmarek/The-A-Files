"use client";

import { useState } from "react";

import { useI18n } from "@/lib/i18n";

import { AXIS_TEXT, Legend, Tooltip, TooltipRow, formatTick, linear, niceTicks, seriesColor, tickDigits, useWidth } from "./base";

export interface CurvePoint {
  y: number | null;
  lo?: number | null;
  hi?: number | null;
}

export interface CurveSeries {
  label: string;
  points: CurvePoint[];
}

/**
 * Lines over an ordinal x axis (a sweep read in order), each with a 95% band.
 * The x positions are equally spaced: the order of the settings is the
 * information, their numeric spacing is not.
 */
export function CurveChart({
  xLabels,
  xTitle,
  yTitle,
  series,
  yMax,
  threshold,
  format = (value: number) => value.toFixed(3),
  height = 320,
}: {
  xLabels: string[];
  xTitle: string;
  yTitle: string;
  series: CurveSeries[];
  yMax?: number;
  threshold?: number;
  format?: (value: number) => string;
  height?: number;
}) {
  const { t } = useI18n();
  const { ref, width } = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const margin = { top: 12, right: 20, bottom: 44, left: 52 };
  const plotWidth = Math.max(120, width - margin.left - margin.right);
  const plotHeight = height - margin.top - margin.bottom;
  const values = series.flatMap((s) => s.points.flatMap((p) => [p.y, p.hi]).filter((v): v is number => v !== null && v !== undefined));
  const top = yMax ?? Math.max(0.05, ...values, threshold ?? 0) * 1.05;
  const ticks = niceTicks(0, top, 5);
  const domainTop = Math.max(top, ticks[ticks.length - 1] ?? top);
  const y = linear([0, domainTop], [plotHeight, 0]);
  const step = xLabels.length > 1 ? plotWidth / (xLabels.length - 1) : 0;
  const x = (index: number) => (xLabels.length > 1 ? index * step : plotWidth / 2);

  function path(points: CurvePoint[]) {
    let d = "";
    let open = false;
    points.forEach((point, index) => {
      if (point.y === null || point.y === undefined) {
        open = false;
        return;
      }
      d += `${open ? "L" : "M"}${x(index)},${y(point.y)}`;
      open = true;
    });
    return d;
  }

  function band(points: CurvePoint[]) {
    const valid = points.map((p, i) => ({ ...p, i })).filter((p) => p.lo !== null && p.lo !== undefined && p.hi !== null && p.hi !== undefined);
    if (valid.length < 2) return "";
    const upper = valid.map((p) => `${x(p.i)},${y(p.hi as number)}`).join("L");
    const lower = [...valid].reverse().map((p) => `${x(p.i)},${y(p.lo as number)}`).join("L");
    return `M${upper}L${lower}Z`;
  }

  function onMove(event: React.PointerEvent<SVGRectElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const position = event.clientX - box.left;
    const index = step ? Math.round(position / step) : 0;
    setHover(Math.max(0, Math.min(xLabels.length - 1, index)));
  }

  return (
    <div ref={ref} className="relative">
      <svg width={width} height={height} role="img" aria-label={`${yTitle} / ${xTitle}`}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {ticks.map((tick) => (
            <g key={tick}>
              <line x1={0} x2={plotWidth} y1={y(tick)} y2={y(tick)} stroke="var(--chart-grid)" />
              <text x={-8} y={y(tick)} dy="0.32em" textAnchor="end" {...AXIS_TEXT} className="num">
                {formatTick(tick, tickDigits(ticks))}
              </text>
            </g>
          ))}
          <line x1={0} x2={plotWidth} y1={plotHeight} y2={plotHeight} stroke="var(--chart-axis)" />
          {xLabels.map((label, index) => (
            <text key={label + index} x={x(index)} y={plotHeight + 16} textAnchor="middle" {...AXIS_TEXT} className="num">
              {label}
            </text>
          ))}
          <text x={plotWidth / 2} y={plotHeight + 36} textAnchor="middle" fontSize={11} fill="var(--chart-ink-2)">
            {xTitle}
          </text>
          <text transform={`translate(${-40},${plotHeight / 2}) rotate(-90)`} textAnchor="middle" fontSize={11} fill="var(--chart-ink-2)">
            {yTitle}
          </text>

          {threshold !== undefined && threshold <= domainTop && (
            <line x1={0} x2={plotWidth} y1={y(threshold)} y2={y(threshold)} stroke="var(--chart-ink-2)" strokeDasharray="4 4" strokeWidth={1} />
          )}

          {series.map((s, index) => (
            <path key={`band-${s.label}`} d={band(s.points)} fill={seriesColor(index)} opacity={0.1} />
          ))}
          {series.map((s, index) => (
            <path
              key={`line-${s.label}`}
              d={path(s.points)}
              fill="none"
              stroke={seriesColor(index)}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
          {series.map((s, index) =>
            s.points.map((point, pointIndex) =>
              point.y === null || point.y === undefined ? null : (
                <circle
                  key={`${s.label}-${pointIndex}`}
                  cx={x(pointIndex)}
                  cy={y(point.y)}
                  r={hover === pointIndex ? 5 : 4}
                  fill={seriesColor(index)}
                  stroke="var(--chart-surface)"
                  strokeWidth={2}
                />
              )
            )
          )}
          {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={plotHeight} stroke="var(--chart-axis)" />}
          <rect
            width={plotWidth + 20}
            x={-10}
            height={plotHeight}
            fill="transparent"
            onPointerMove={onMove}
            onPointerLeave={() => setHover(null)}
          />
        </g>
      </svg>
      {hover !== null && (
        <Tooltip x={margin.left + x(hover)} y={margin.top} width={width}>
          <div className="mb-1 font-medium">
            {xTitle}: {xLabels[hover]}
          </div>
          <div className="space-y-0.5">
            {series.map((s, index) => {
              const point = s.points[hover];
              return (
                <TooltipRow
                  key={s.label}
                  color={seriesColor(index)}
                  label={s.label}
                  value={
                    point?.y === null || point?.y === undefined
                      ? "–"
                      : `${format(point.y)}${point.lo !== null && point.lo !== undefined ? ` [${format(point.lo)}, ${format(point.hi as number)}]` : ""}`
                  }
                />
              );
            })}
          </div>
          <div className="mt-1 text-[10px] text-muted-foreground">{t("common.ci")}</div>
        </Tooltip>
      )}
      <Legend items={series.map((s, index) => ({ label: s.label, color: seriesColor(index) }))} />
    </div>
  );
}
