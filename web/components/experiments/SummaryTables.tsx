"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ScenarioSummary } from "@/lib/types";

function pct(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value * 100)}%` : "—";
}

function num(value: unknown, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

// Conditional background for the robustness matrix: green (accurate) → red.
function accuracyBackground(value: unknown): string | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  const hue = 120 * Math.max(0, Math.min(1, value)); // 0 = red, 120 = green
  return `hsla(${hue}, 70%, 45%, 0.18)`;
}

interface MatrixCell {
  method: string;
  attack: string | null;
  avg_bit_accuracy: number | null;
  avg_ber: number | null;
  decode_success_rate: number | null;
}

export function RobustnessMatrix({ cells }: { cells: MatrixCell[] }) {
  const methods = [...new Set(cells.map((cell) => cell.method))].sort();
  const attacks = [...new Set(cells.map((cell) => cell.attack ?? "baseline"))].sort((a, b) =>
    a === "baseline" ? -1 : b === "baseline" ? 1 : a.localeCompare(b)
  );
  const lookup = new Map(cells.map((cell) => [`${cell.method}|${cell.attack ?? "baseline"}`, cell]));

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Method</TableHead>
            {attacks.map((attack) => (
              <TableHead key={attack} className="whitespace-nowrap">
                {attack.replaceAll("_", " ")}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {methods.map((method) => (
            <TableRow key={method}>
              <TableCell className="whitespace-nowrap font-medium">{method}</TableCell>
              {attacks.map((attack) => {
                const cell = lookup.get(`${method}|${attack}`);
                return (
                  <TableCell
                    key={attack}
                    className="whitespace-nowrap"
                    style={{ background: accuracyBackground(cell?.avg_bit_accuracy) }}
                    title={
                      cell
                        ? `bit accuracy ${pct(cell.avg_bit_accuracy)} · BER ${num(cell.avg_ber)} · decode ${pct(cell.decode_success_rate)}`
                        : undefined
                    }
                  >
                    {cell ? pct(cell.avg_bit_accuracy) : "—"}
                  </TableCell>
                );
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <p className="mt-1 text-xs text-muted-foreground">
        Cell value = average bit accuracy (green = robust, red = broken); hover for BER and decode
        success rate.
      </p>
    </div>
  );
}

// Renders any list-of-objects summary section as a table (backend-shaped).
export function SummarySectionTable({ rows }: { rows: Record<string, unknown>[] }) {
  if (rows.length === 0) return null;
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))].filter(
    (column) => column !== "avg_metrics" && column !== "payloads_tested"
  );
  const metricColumns = [
    ...new Set(
      rows.flatMap((row) =>
        typeof row.avg_metrics === "object" && row.avg_metrics
          ? Object.keys(row.avg_metrics as Record<string, number>)
          : []
      )
    ),
  ];

  function renderValue(column: string, value: unknown): string {
    if (value === null || value === undefined) return "—";
    if (typeof value === "boolean") return value ? "✓" : "✗";
    if (typeof value === "number") {
      if (column.includes("rate") || column.includes("accuracy") || column.endsWith("_score"))
        return pct(value);
      if (Number.isInteger(value)) return String(value);
      return num(value);
    }
    return String(value).replaceAll("_", " ");
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            {columns.map((column) => (
              <TableHead key={column} className="whitespace-nowrap">
                {column.replaceAll("_", " ")}
              </TableHead>
            ))}
            {metricColumns.map((column) => (
              <TableHead key={column} className="whitespace-nowrap">
                {column}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, index) => (
            <TableRow key={index}>
              {columns.map((column) => (
                <TableCell key={column} className="whitespace-nowrap">
                  {renderValue(column, row[column])}
                </TableCell>
              ))}
              {metricColumns.map((column) => (
                <TableCell key={column} className="whitespace-nowrap">
                  {num((row.avg_metrics as Record<string, number> | undefined)?.[column])}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
