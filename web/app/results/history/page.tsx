"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { PageHeader } from "@/components/layout/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { api } from "@/lib/api";
import { EXPERIMENTS, experimentLabel } from "@/lib/experimentLabels";
import type { JobSummary } from "@/lib/types";

function statusVariant(status: JobSummary["status"]) {
  switch (status) {
    case "completed":
      return "success" as const;
    case "failed":
      return "destructive" as const;
    case "running":
      return "default" as const;
    default:
      return "secondary" as const;
  }
}

export default function HistoryPage() {
  const [experiments, setExperiments] = useState<JobSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    api
      .history()
      .then((rows) => {
        if (!cancelled) {
          setExperiments(rows);
          setError(null);
        }
      })
      .catch((err: Error) => !cancelled && setError(err.message));

    // Live updates via the global SSE stream; reconnects re-sync via snapshot.
    const source = new EventSource(api.allEventsUrl());
    source.addEventListener("experiments", (event) => {
      setExperiments(JSON.parse((event as MessageEvent).data) as JobSummary[]);
      setError(null);
    });
    source.addEventListener("experiment", (event) => {
      const summary = JSON.parse((event as MessageEvent).data) as JobSummary;
      setExperiments((previous) => {
        if (previous === null) return [summary];
        const index = previous.findIndex((row) => row.experiment_id === summary.experiment_id);
        if (index === -1) return [summary, ...previous];
        const next = [...previous];
        next[index] = summary;
        return next;
      });
    });

    return () => {
      cancelled = true;
      source.close();
    };
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Experiment History"
        description="All runs from this API session, updating live. Experiments are held in memory — export CSVs to persist results."
      />
      {error && <p className="text-sm text-destructive">{error}</p>}
      <Card>
        <CardContent className="pt-6">
          {experiments === null ? (
            <Skeleton className="h-40 w-full" />
          ) : experiments.length === 0 ? (
            <div className="py-10 text-center text-sm text-muted-foreground">
              <p>No experiments yet.</p>
              <p className="mt-1">
                Start one from the sidebar, e.g.{" "}
                <Link href={EXPERIMENTS[0].route} className="text-primary hover:underline">
                  {EXPERIMENTS[0].label}
                </Link>
                .
              </p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Dataset</TableHead>
                  <TableHead>Methods</TableHead>
                  <TableHead>Attacks</TableHead>
                  <TableHead>Progress</TableHead>
                  <TableHead>Success</TableHead>
                  <TableHead>Started</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {experiments.map((experiment) => (
                  <TableRow key={experiment.experiment_id}>
                    <TableCell>
                      <Link
                        href={`/experiments/${experiment.experiment_id}`}
                        className="font-medium text-primary hover:underline"
                      >
                        {experiment.name}
                      </Link>
                    </TableCell>
                    <TableCell className="whitespace-nowrap">
                      {experimentLabel(experiment.experiment_type)}
                    </TableCell>
                    <TableCell>
                      <Badge variant={statusVariant(experiment.status)}>{experiment.status}</Badge>
                    </TableCell>
                    <TableCell className="whitespace-nowrap">{experiment.dataset}</TableCell>
                    <TableCell>{experiment.methods.length}</TableCell>
                    <TableCell>{experiment.attacks.length}</TableCell>
                    <TableCell>
                      {experiment.completed_rows}/{experiment.total_tasks || "?"}
                    </TableCell>
                    <TableCell>
                      {experiment.completed_rows > 0
                        ? `${Math.round((experiment.success_rows / experiment.completed_rows) * 100)}%`
                        : "—"}
                    </TableCell>
                    <TableCell className="whitespace-nowrap text-muted-foreground">
                      {new Date(experiment.created_at).toLocaleTimeString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
