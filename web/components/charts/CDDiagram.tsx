"use client";

import { AXIS_TEXT, linear, useWidth } from "./base";

/**
 * Critical-difference diagram (Demšar, 2006): methods placed by mean rank
 * (1 = best, on the left), with a bar joining every maximal group whose
 * ranks differ by less than the Nemenyi critical difference.
 */
export function CDDiagram({ ranks, cd, cdLabel }: { ranks: Record<string, number>; cd: number; cdLabel: string }) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const entries = Object.entries(ranks).sort((a, b) => a[1] - b[1]);
  const count = entries.length;
  if (count < 2) return null;

  const half = Math.ceil(count / 2);
  const labelSpace = Math.min(320, Math.max(130, width * 0.3));
  const margin = { top: 58, left: labelSpace, right: labelSpace };
  const plotWidth = Math.max(160, width - margin.left - margin.right);
  const x = linear([1, count], [0, plotWidth]);
  const rowGap = 22;

  // Maximal cliques of consecutive methods whose rank span is below the CD.
  const cliques: [number, number][] = [];
  for (let start = 0; start < count; start++) {
    let end = start;
    while (end + 1 < count && entries[end + 1][1] - entries[start][1] < cd) end++;
    if (end > start && !cliques.some(([s, e]) => s <= start && e >= end)) cliques.push([start, end]);
  }
  const cliqueTop = 18;
  const labelsTop = cliqueTop + cliques.length * 8 + 18;
  const height = margin.top + labelsTop + Math.max(half, count - half) * rowGap + 10;

  return (
    <div ref={ref}>
      <svg width={width} height={height} role="img" aria-label="Critical difference diagram">
        <g transform={`translate(${margin.left},${margin.top})`}>
          {/* CD scale */}
          <g transform={`translate(0,${-30})`}>
            <line x1={0} x2={x(1 + cd) - x(1)} y1={0} y2={0} stroke="var(--chart-ink)" strokeWidth={1.5} />
            <line x1={0} x2={0} y1={-4} y2={4} stroke="var(--chart-ink)" />
            <line x1={x(1 + cd) - x(1)} x2={x(1 + cd) - x(1)} y1={-4} y2={4} stroke="var(--chart-ink)" />
            <text x={(x(1 + cd) - x(1)) / 2} y={-8} textAnchor="middle" fontSize={11} fill="var(--chart-ink-2)" className="num">
              {cdLabel}
            </text>
          </g>
          {/* rank axis */}
          <line x1={0} x2={plotWidth} y1={0} y2={0} stroke="var(--chart-axis)" />
          {Array.from({ length: count }, (_, index) => index + 1).map((rank) => (
            <g key={rank}>
              <line x1={x(rank)} x2={x(rank)} y1={-4} y2={0} stroke="var(--chart-axis)" />
              <text x={x(rank)} y={-8} textAnchor="middle" {...AXIS_TEXT} className="num">
                {rank}
              </text>
            </g>
          ))}
          {/* non-significant groups */}
          {cliques.map(([start, end], index) => (
            <line
              key={`${start}-${end}`}
              x1={x(entries[start][1]) - 3}
              x2={x(entries[end][1]) + 3}
              y1={cliqueTop + index * 8}
              y2={cliqueTop + index * 8}
              stroke="var(--chart-ink)"
              strokeWidth={3}
              strokeLinecap="round"
            />
          ))}
          {/* methods */}
          {entries.map(([name, rank], index) => {
            const left = index < half;
            const row = left ? index : count - 1 - index;
            const yLabel = labelsTop + row * rowGap;
            const xEnd = left ? -12 : plotWidth + 12;
            return (
              <g key={name}>
                <polyline
                  points={`${x(rank)},0 ${x(rank)},${yLabel} ${xEnd},${yLabel}`}
                  fill="none"
                  stroke="var(--chart-ink-2)"
                  strokeWidth={1}
                />
                <circle cx={x(rank)} cy={0} r={3} fill="var(--chart-ink)" />
                <text
                  x={left ? xEnd - 6 : xEnd + 6}
                  y={yLabel}
                  dy="0.32em"
                  textAnchor={left ? "end" : "start"}
                  fontSize={12}
                  fill="var(--chart-ink)"
                >
                  <title>{name}</title>
                  {name.length > 44 ? `${name.slice(0, 42)}…` : name}
                  <tspan fill="var(--chart-muted)" className="num">{`  ${rank.toFixed(2)}`}</tspan>
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}
