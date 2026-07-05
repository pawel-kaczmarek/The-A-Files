"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, ShieldAlert } from "lucide-react";

import { PageHeader } from "@/components/layout/PageHeader";
import { useCatalog } from "@/components/experiments/useCatalog";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { api, API_BASE } from "@/lib/api";
import { EXPERIMENTS, experimentLabel } from "@/lib/experimentLabels";
import type { JobSummary } from "@/lib/types";

function Stat({ label, value, href }: { label: string; value: number | string; href: string }) {
  return (
    <Link href={href}>
      <Card className="transition-colors hover:bg-accent/50">
        <CardHeader>
          <CardTitle className="text-4xl">{value}</CardTitle>
          <CardDescription>{label}</CardDescription>
        </CardHeader>
      </Card>
    </Link>
  );
}

export default function DashboardPage() {
  const { catalog, error } = useCatalog();
  const [recent, setRecent] = useState<JobSummary[] | null>(null);

  useEffect(() => {
    api
      .history()
      .then((rows) => setRecent(rows.slice(0, 5)))
      .catch(() => setRecent([]));
  }, []);

  if (error) {
    return (
      <Card className="max-w-xl border-destructive">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ShieldAlert className="h-5 w-5 text-destructive" /> API unreachable
          </CardTitle>
          <CardDescription>
            Could not reach the taf API at <code>{API_BASE}</code>. Start it with{" "}
            <code>taf-api</code> or <code>uvicorn taf.api.main:app --reload</code> (requires{" "}
            <code>pip install -e ".[experiments]"</code>).
          </CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">{error}</CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-8">
      <PageHeader
        title="Dashboard"
        description="Benchmark audio steganography methods against quality metrics and robustness attacks."
      />

      {!catalog ? (
        <Skeleton className="h-28 w-full" />
      ) : (
        <div className="grid gap-4 md:grid-cols-4">
          <Stat label="steganography methods" value={catalog.methods.length} href="/settings/methods" />
          <Stat label="quality metrics" value={catalog.metrics.length} href="/settings/metrics" />
          <Stat label="robustness attacks" value={catalog.attacks.length} href="/settings/attacks" />
          <Stat label="datasets" value={catalog.datasets.length} href="/settings/datasets" />
        </div>
      )}

      <div>
        <h2 className="mb-3 text-lg font-semibold">Experiments</h2>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {EXPERIMENTS.map((experiment) => (
            <Link key={experiment.type} href={experiment.route}>
              <Card className="h-full transition-colors hover:bg-accent/50">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between text-base">
                    {experiment.label}
                    <ArrowRight className="h-4 w-4 text-muted-foreground" />
                  </CardTitle>
                  <CardDescription>{experiment.question}</CardDescription>
                </CardHeader>
              </Card>
            </Link>
          ))}
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent experiments</CardTitle>
          <CardDescription>
            The five most recent runs — see{" "}
            <Link href="/results/history" className="text-primary hover:underline">
              Experiment History
            </Link>{" "}
            for all.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {recent === null ? (
            <Skeleton className="h-24 w-full" />
          ) : recent.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted-foreground">No runs yet.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Rows</TableHead>
                  <TableHead>CSV</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recent.map((experiment) => (
                  <TableRow key={experiment.experiment_id}>
                    <TableCell>
                      <Link
                        href={`/experiments/${experiment.experiment_id}`}
                        className="font-medium text-primary hover:underline"
                      >
                        {experiment.name}
                      </Link>
                    </TableCell>
                    <TableCell>{experimentLabel(experiment.experiment_type)}</TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          experiment.status === "completed"
                            ? "success"
                            : experiment.status === "failed"
                              ? "destructive"
                              : "default"
                        }
                      >
                        {experiment.status}
                      </Badge>
                    </TableCell>
                    <TableCell>{experiment.completed_rows}</TableCell>
                    <TableCell>
                      <a
                        className="text-primary hover:underline"
                        href={api.exportCsvUrl(experiment.experiment_id)}
                        download
                      >
                        download
                      </a>
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
