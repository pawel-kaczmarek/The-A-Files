"use client";

/**
 * Lightweight bar charts for experiment summaries.
 *
 * Colors come from the validated categorical palette exposed as
 * `--chart-1..6` in globals.css (separately stepped for light and dark).
 * Series are assigned slots in fixed order and never cycled; the detailed
 * results table on the same page is the accessible fallback view.
 */

const SERIES_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
  "var(--chart-6)",
];

export const MAX_CHART_SERIES = SERIES_COLORS.length;

export interface BarDatum {
  series: string;
  value: number;
  /** Full text for the hover tooltip. */
  detail?: string;
}

export interface BarGroup {
  label: string;
  bars: BarDatum[];
}

interface BarChartProps {
  groups: BarGroup[];
  series: string[];
  /** Upper bound of the value axis; defaults to the data maximum. */
  domainMax?: number;
  formatValue: (value: number) => string;
}

export function GroupedBarChart({ groups, series, domainMax, formatValue }: BarChartProps) {
  const values = groups.flatMap((group) => group.bars.map((bar) => bar.value));
  if (values.length === 0) {
    return <p className="py-4 text-sm text-muted-foreground">No data yet.</p>;
  }
  const max = domainMax ?? Math.max(...values, 0);
  const min = Math.min(...values, 0);
  const span = max - min || 1;
  const zero = ((0 - min) / span) * 100; // % offset of the zero baseline

  const colorOf = (name: string) => SERIES_COLORS[series.indexOf(name) % SERIES_COLORS.length];

  return (
    <div>
      {series.length > 1 && (
        <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1">
          {series.map((name) => (
            <span key={name} className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ background: colorOf(name) }}
              />
              {name}
            </span>
          ))}
        </div>
      )}
      <div className="space-y-3">
        {groups.map((group) => (
          <div key={group.label} className="grid grid-cols-[8rem_1fr] items-center gap-3">
            <div className="truncate text-xs text-muted-foreground" title={group.label}>
              {group.label}
            </div>
            <div
              className="relative space-y-0.5 border-l"
              style={{ borderColor: "var(--chart-axis)" }}
            >
              {group.bars.map((bar) => {
                const clamped = Math.max(min, Math.min(max, bar.value));
                const negative = clamped < 0;
                const left = negative ? ((clamped - min) / span) * 100 : zero;
                const width = (Math.abs(clamped) / span) * 100;
                return (
                  <div
                    key={bar.series}
                    className="flex h-3.5 items-center"
                    title={bar.detail ?? `${group.label} · ${bar.series}: ${formatValue(bar.value)}`}
                  >
                    {/* mr-12 reserves room for the value label at a full-width bar's tip */}
                    <div className="relative mr-12 h-3 flex-1">
                      <div
                        className="absolute inset-y-0"
                        style={{
                          left: `${left}%`,
                          width: `calc(${width}% )`,
                          minWidth: bar.value !== 0 ? "2px" : "0",
                          background: colorOf(bar.series),
                          borderRadius: negative ? "4px 0 0 4px" : "0 4px 4px 0",
                        }}
                      />
                      <span
                        className="absolute top-1/2 -translate-y-1/2 whitespace-nowrap pl-1.5 text-[11px] leading-none text-muted-foreground"
                        style={{ left: `${negative ? zero : left + width}%` }}
                      >
                        {formatValue(bar.value)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function StatTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}
