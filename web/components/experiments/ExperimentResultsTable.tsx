"use client";

import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ResultRow } from "@/lib/types";

const PAGE_SIZE = 25;

type SortKey =
  | "file_name"
  | "method"
  | "attack"
  | "payload_length"
  | "bit_accuracy"
  | "ber"
  | "encode_time_seconds"
  | "decode_time_seconds";

function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

export function ExperimentResultsTable({ rows }: { rows: ResultRow[] }) {
  const [filter, setFilter] = useState("");
  const [methodFilter, setMethodFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("file_name");
  const [sortAsc, setSortAsc] = useState(true);
  const [page, setPage] = useState(0);

  const methods = useMemo(() => [...new Set(rows.map((row) => row.method))].sort(), [rows]);
  const metricNames = useMemo(
    () => [...new Set(rows.flatMap((row) => Object.keys(row.metrics)))].sort(),
    [rows]
  );
  const hasAttacks = rows.some((row) => row.attack !== null);

  const filtered = useMemo(() => {
    const query = filter.trim().toLowerCase();
    let result = rows;
    if (methodFilter !== "all") result = result.filter((row) => row.method === methodFilter);
    if (statusFilter !== "all") {
      result = result.filter((row) =>
        statusFilter === "error" ? row.status === "error" : row.status === "ok"
      );
    }
    if (query) {
      result = result.filter((row) =>
        [row.file_name, row.method, row.attack ?? "", row.error ?? ""]
          .join(" ")
          .toLowerCase()
          .includes(query)
      );
    }
    const direction = sortAsc ? 1 : -1;
    return [...result].sort((a, b) => {
      const left = a[sortKey];
      const right = b[sortKey];
      if (left === null || left === undefined) return 1;
      if (right === null || right === undefined) return -1;
      if (typeof left === "number" && typeof right === "number") return (left - right) * direction;
      return String(left).localeCompare(String(right)) * direction;
    });
  }, [rows, filter, methodFilter, statusFilter, sortKey, sortAsc]);

  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const current = Math.min(page, pages - 1);
  const visible = filtered.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE);

  function sortBy(key: SortKey) {
    if (key === sortKey) setSortAsc((asc) => !asc);
    else {
      setSortKey(key);
      setSortAsc(true);
    }
    setPage(0);
  }

  function SortHeader({ label, column }: { label: string; column: SortKey }) {
    const active = sortKey === column;
    const Icon = active ? (sortAsc ? ArrowUp : ArrowDown) : ArrowUpDown;
    return (
      <button
        type="button"
        className="inline-flex items-center gap-1 hover:text-foreground"
        onClick={() => sortBy(column)}
      >
        {label}
        <Icon className="h-3 w-3" />
      </button>
    );
  }

  if (rows.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        No result rows yet — run the experiment to populate this table.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Input
          placeholder="Filter by file, method, attack, error…"
          className="w-72"
          value={filter}
          onChange={(event) => {
            setFilter(event.target.value);
            setPage(0);
          }}
        />
        <Select
          className="w-48"
          value={methodFilter}
          onChange={(event) => {
            setMethodFilter(event.target.value);
            setPage(0);
          }}
        >
          <option value="all">All methods</option>
          {methods.map((method) => (
            <option key={method} value={method}>
              {method}
            </option>
          ))}
        </Select>
        <Select
          className="w-36"
          value={statusFilter}
          onChange={(event) => {
            setStatusFilter(event.target.value);
            setPage(0);
          }}
        >
          <option value="all">All statuses</option>
          <option value="ok">ok</option>
          <option value="error">error</option>
        </Select>
        <span className="ml-auto text-xs text-muted-foreground">
          {filtered.length} of {rows.length} rows
        </span>
      </div>

      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead><SortHeader label="File" column="file_name" /></TableHead>
              <TableHead><SortHeader label="Method" column="method" /></TableHead>
              {hasAttacks && <TableHead><SortHeader label="Attack" column="attack" /></TableHead>}
              <TableHead><SortHeader label="Bits" column="payload_length" /></TableHead>
              <TableHead>Rep</TableHead>
              <TableHead>Status</TableHead>
              <TableHead><SortHeader label="Bit acc." column="bit_accuracy" /></TableHead>
              <TableHead><SortHeader label="BER" column="ber" /></TableHead>
              <TableHead><SortHeader label="Enc (s)" column="encode_time_seconds" /></TableHead>
              <TableHead><SortHeader label="Dec (s)" column="decode_time_seconds" /></TableHead>
              {metricNames.map((name) => (
                <TableHead key={name} className="whitespace-nowrap">
                  {name}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {visible.map((row, index) => (
              <TableRow key={`${current}-${index}`}>
                <TableCell className="whitespace-nowrap">{row.file_name}</TableCell>
                <TableCell className="whitespace-nowrap">{row.method}</TableCell>
                {hasAttacks && (
                  <TableCell className="whitespace-nowrap">
                    {row.attack ?? <span className="text-muted-foreground">baseline</span>}
                  </TableCell>
                )}
                <TableCell>{row.payload_length}</TableCell>
                <TableCell>{row.repetition}</TableCell>
                <TableCell>
                  {row.status === "error" ? (
                    <Badge variant="destructive" title={row.error ?? ""}>
                      error
                    </Badge>
                  ) : row.decode_success ? (
                    <Badge variant="success">ok</Badge>
                  ) : (
                    <Badge variant="secondary">mismatch</Badge>
                  )}
                </TableCell>
                <TableCell>{formatNumber(row.bit_accuracy)}</TableCell>
                <TableCell>{formatNumber(row.ber)}</TableCell>
                <TableCell>{formatNumber(row.encode_time_seconds)}</TableCell>
                <TableCell>{formatNumber(row.decode_time_seconds)}</TableCell>
                {metricNames.map((name) => (
                  <TableCell key={name} className="whitespace-nowrap">
                    {name in row.metric_errors ? (
                      <span title={row.metric_errors[name]}>⚠</span>
                    ) : (
                      formatNumber(row.metrics[name])
                    )}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {pages > 1 && (
        <div className="flex items-center justify-end gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={current === 0}
            onClick={() => setPage(current - 1)}
          >
            Previous
          </Button>
          <span className="text-xs text-muted-foreground">
            Page {current + 1} of {pages}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={current >= pages - 1}
            onClick={() => setPage(current + 1)}
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
}
