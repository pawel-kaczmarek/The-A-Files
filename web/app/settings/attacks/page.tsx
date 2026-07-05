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

export default function AttacksSettingsPage() {
  const { catalog, error } = useCatalog();
  return (
    <div className="space-y-6">
      <PageHeader
        title="Attacks"
        description="Signal attacks discovered from CorruptedWavFile. Experiments currently run each attack with its default parameters, applied to the watermarked signal before decoding."
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
                  <TableHead>Description</TableHead>
                  <TableHead>Parameters (defaults)</TableHead>
                  <TableHead>Notes</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {catalog.attacks.map((attack) => (
                  <TableRow key={attack.name}>
                    <TableCell className="whitespace-nowrap font-medium">
                      {attack.name.replaceAll("_", " ")}
                    </TableCell>
                    <TableCell>{attack.description}</TableCell>
                    <TableCell className="font-mono text-xs">
                      {attack.parameters.length === 0
                        ? "—"
                        : attack.parameters
                            .map((param) => `${param.name}=${String(param.default)}`)
                            .join(", ")}
                    </TableCell>
                    <TableCell>
                      {attack.changes_length_or_rate ? (
                        <Badge
                          variant="secondary"
                          title="Changes signal length or sample rate; decodes usually fail and metrics are recorded as row-level errors"
                        >
                          changes length/rate
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
