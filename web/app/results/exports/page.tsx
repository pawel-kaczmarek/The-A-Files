"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Download, FileJson } from "lucide-react";

import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
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
import { experimentLabel } from "@/lib/experimentLabels";
import type { JobSummary } from "@/lib/types";

export default function ExportsPage() {
  const [experiments, setExperiments] = useState<JobSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .history()
      .then((rows) => setExperiments(rows.filter((row) => row.completed_rows > 0)))
      .catch((err: Error) => setError(err.message));
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader
        title="CSV Exports"
        description="Download normalized detailed results, summary tables, and experiment configs for every finished run. Filenames include experiment type, id and timestamp."
      />
      {error && <p className="text-sm text-destructive">{error}</p>}
      <Card>
        <CardContent className="pt-6">
          {experiments === null ? (
            <Skeleton className="h-40 w-full" />
          ) : experiments.length === 0 ? (
            <p className="py-10 text-center text-sm text-muted-foreground">
              Nothing to export yet — run an experiment first.
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Experiment</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Rows</TableHead>
                  <TableHead>Downloads</TableHead>
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
                    <TableCell>{experiment.completed_rows}</TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-2">
                        <Button asChild variant="outline" size="sm">
                          <a href={api.exportCsvUrl(experiment.experiment_id)} download>
                            <Download className="h-4 w-4" /> detailed_results.csv
                          </a>
                        </Button>
                        <Button asChild variant="outline" size="sm">
                          <a href={api.exportSummaryCsvUrl(experiment.experiment_id)} download>
                            <Download className="h-4 w-4" /> summary_results.csv
                          </a>
                        </Button>
                        <Button asChild variant="ghost" size="sm">
                          <a href={api.exportConfigUrl(experiment.experiment_id)} download>
                            <FileJson className="h-4 w-4" /> config.json
                          </a>
                        </Button>
                      </div>
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
