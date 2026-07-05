"use client";

import { StatTile } from "@/components/charts";
import type { GroupStats, ScenarioSummary } from "@/lib/types";

function pct(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${Math.round(value * 100)}%`;
}

function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

// Generic tiles from summary.overall plus scenario-specific highlights.
// The backend computes all values; this component only picks what to show.
export function ExperimentSummaryCards({ summary }: { summary: ScenarioSummary }) {
  const overall = summary.overall as GroupStats | undefined;
  const tiles: { label: string; value: string; hint?: string }[] = [];

  if (overall) {
    tiles.push(
      {
        label: "Decode success rate",
        value: pct(overall.decode_success_rate),
        hint: `${overall.rows} rows, ${overall.error_rows} errors`,
      },
      { label: "Avg bit accuracy", value: pct(overall.avg_bit_accuracy) },
      { label: "Avg BER", value: num(overall.avg_ber) }
    );
  }
  if (typeof summary.most_robust_method === "string") {
    tiles.push({ label: "Most robust method", value: summary.most_robust_method });
  }
  if (typeof summary.worst_attack === "string") {
    tiles.push({ label: "Worst attack", value: summary.worst_attack.replaceAll("_", " ") });
  }
  if (typeof summary.best_method === "string") {
    tiles.push({ label: "Best method overall", value: summary.best_method });
  }
  if (typeof summary.best_capacity_method === "string") {
    tiles.push({ label: "Best capacity method", value: summary.best_capacity_method });
  }
  if (typeof summary.highest_stable_payload === "number") {
    tiles.push({
      label: "Highest stable payload",
      value: `${summary.highest_stable_payload} bits`,
    });
  }
  if (typeof summary.average_capacity === "number") {
    tiles.push({ label: "Average capacity", value: `${summary.average_capacity.toFixed(1)} bits` });
  }
  const ranking = summary.method_ranking as string[] | undefined;
  if (Array.isArray(ranking) && ranking.length > 0 && typeof ranking[0] === "string") {
    tiles.push({ label: "Top method (bit accuracy)", value: ranking[0] });
  }
  const quality = summary.quality_ranking as { method: string }[] | undefined;
  if (Array.isArray(quality) && quality.length > 0 && quality[0]?.method) {
    tiles.push({ label: "Best quality method", value: quality[0].method });
  }

  if (tiles.length === 0) return null;
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {tiles.slice(0, 8).map((tile) => (
        <StatTile key={tile.label} label={tile.label} value={tile.value} hint={tile.hint} />
      ))}
    </div>
  );
}
