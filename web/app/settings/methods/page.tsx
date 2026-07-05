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

export default function MethodsSettingsPage() {
  const { catalog, error } = useCatalog();
  return (
    <div className="space-y-6">
      <PageHeader
        title="Methods"
        description="Steganography algorithms discovered from the SteganographyMethodFactory registry. All methods implement encode(samples, bits) / decode(samples, length) / type()."
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
                  <TableHead>Class</TableHead>
                  <TableHead>Description</TableHead>
                  <TableHead>Notes / limitations</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {catalog.methods.map((method) => (
                  <TableRow key={method.name}>
                    <TableCell className="whitespace-nowrap font-medium">{method.name}</TableCell>
                    <TableCell className="whitespace-nowrap font-mono text-xs">
                      {method.class_name}
                    </TableCell>
                    <TableCell>{method.description}</TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {method.requires_tensorflow && (
                          <Badge variant="secondary" title="Install with: pip install 'the-a-files[ai]'">
                            requires TensorFlow
                          </Badge>
                        )}
                        {method.needs_long_input && (
                          <Badge
                            variant="secondary"
                            title="Reserves floor(len/8192) frames and needs at least 8; short files produce failed rows"
                          >
                            needs long input
                          </Badge>
                        )}
                        {!method.requires_tensorflow && !method.needs_long_input && (
                          <span className="text-xs text-muted-foreground">—</span>
                        )}
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
