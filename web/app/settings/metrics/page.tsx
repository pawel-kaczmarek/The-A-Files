"use client";

import { PageHeader } from "@/components/layout/PageHeader";
import { useCatalog } from "@/components/experiments/useCatalog";
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

export default function MetricsSettingsPage() {
  const { catalog, error } = useCatalog();
  return (
    <div className="space-y-6">
      <PageHeader
        title="Metrics"
        description="Quality, intelligibility and reverberation measures discovered from the MetricFactory registry. All metrics compare the original signal against the processed one."
      />
      {error && <p className="text-sm text-destructive">{error}</p>}
      <Card>
        <CardContent className="pt-6">
          {!catalog ? (
            <Skeleton className="h-64 w-full" />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Category</TableHead>
                  <TableHead>Compares original</TableHead>
                  <TableHead>Supports attacked audio</TableHead>
                  <TableHead>Notes</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {catalog.metrics.map((metric) => (
                  <TableRow key={metric.name}>
                    <TableCell className="whitespace-nowrap font-medium">{metric.name}</TableCell>
                    <TableCell className="whitespace-nowrap">
                      {metric.category.replaceAll("_", " ")}
                    </TableCell>
                    <TableCell>{metric.compares_original ? "✓" : "✗"}</TableCell>
                    <TableCell>
                      {metric.supports_attacked_audio ? "✓ (same length/rate)" : "✗"}
                    </TableCell>
                    <TableCell>
                      {metric.requires_tensorflow ? (
                        <Badge variant="secondary" title="Install with: pip install 'the-a-files[ai]'">
                          requires TensorFlow
                        </Badge>
                      ) : (
                        <span className="text-xs text-muted-foreground">—</span>
                      )}
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
