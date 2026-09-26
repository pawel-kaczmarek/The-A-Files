"use client";

import { useState } from "react";

import { Tooltip, TooltipRow, useWidth } from "./base";

const STEPS = ["--seq-100", "--seq-200", "--seq-300", "--seq-400", "--seq-500", "--seq-600", "--seq-700"];

/** Sequential single-hue class for a value in [0, max]; darker means more. */
function stepFor(value: number, max: number): number {
  const fraction = Math.max(0, Math.min(1, value / (max || 1)));
  return Math.min(STEPS.length - 1, Math.floor(fraction * STEPS.length));
}

/**
 * Rows x columns of one magnitude on a sequential scale, with the value
 * printed in each cell (ink chosen by the step) and a scale legend.
 */
export function Heatmap({
  rows,
  columns,
  value,
  max,
  format,
  detail,
  scaleLabel,
}: {
  rows: string[];
  columns: string[];
  value: (row: string, column: string) => number | null;
  max: number;
  format: (value: number) => string;
  detail?: (row: string, column: string) => { label: string; value: string }[];
  scaleLabel: string;
}) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ row: string; column: string; x: number; y: number } | null>(null);
  const labelWidth = Math.min(280, Math.max(140, width * 0.28));
  const cellWidth = Math.max(44, Math.min(96, (width - labelWidth) / Math.max(1, columns.length)));
  const cellHeight = 30;

  return (
    <div ref={ref} className="relative overflow-x-auto">
      <div style={{ minWidth: labelWidth + cellWidth * columns.length }}>
        <div className="flex" style={{ paddingLeft: labelWidth }}>
          {columns.map((column) => (
            <div
              key={column}
              className="flex items-end justify-center px-0.5 pb-1.5 text-center text-[11px] leading-tight text-muted-foreground"
              style={{ width: cellWidth, height: 56 }}
              title={column}
            >
              <span className="line-clamp-3">{column.replace(/([@:,=_])/g, "$1\u200b")}</span>
            </div>
          ))}
        </div>
        {rows.map((row) => (
          <div key={row} className="flex items-center">
            <div className="truncate pr-3 text-xs" style={{ width: labelWidth }} title={row}>
              {row}
            </div>
            {columns.map((column) => {
              const cell = value(row, column);
              const step = cell === null ? null : stepFor(cell, max);
              return (
                <div
                  key={column}
                  className="flex items-center justify-center text-[11px] num"
                  style={{
                    width: cellWidth,
                    height: cellHeight,
                    padding: 1,
                  }}
                  onPointerEnter={(event) => {
                    if (!ref.current) return;
                    const box = ref.current.getBoundingClientRect();
                    const cellBox = event.currentTarget.getBoundingClientRect();
                    setHover({ row, column, x: cellBox.left - box.left + cellWidth / 2, y: cellBox.top - box.top + cellHeight });
                  }}
                  onPointerLeave={() => setHover(null)}
                >
                  <div
                    className="flex h-full w-full items-center justify-center rounded-[3px]"
                    style={{
                      background: step === null ? "transparent" : `var(${STEPS[step]})`,
                      color: step === null ? "var(--chart-muted)" : `var(${STEPS[step].replace("seq-", "seq-ink-")})`,
                      outline: hover?.row === row && hover?.column === column ? "2px solid var(--chart-ink)" : undefined,
                    }}
                  >
                    {cell === null ? "–" : format(cell)}
                  </div>
                </div>
              );
            })}
          </div>
        ))}
        <div className="mt-3 flex items-center gap-2 text-[11px] text-muted-foreground" style={{ paddingLeft: labelWidth }}>
          <span className="num">{format(0)}</span>
          <div className="flex">
            {STEPS.map((step) => (
              <span key={step} className="h-2.5 w-6" style={{ background: `var(${step})` }} />
            ))}
          </div>
          <span className="num">{format(max)}</span>
          <span className="ml-2">{scaleLabel}</span>
        </div>
      </div>
      {hover && (
        <Tooltip x={hover.x} y={hover.y} width={width}>
          <div className="mb-1 font-medium">{hover.row}</div>
          <div className="mb-1 text-muted-foreground">{hover.column}</div>
          {(detail?.(hover.row, hover.column) ?? []).map((entry) => (
            <TooltipRow key={entry.label} label={entry.label} value={entry.value} />
          ))}
        </Tooltip>
      )}
    </div>
  );
}
