"use client";

import { Download } from "lucide-react";

import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

export function CsvExportButton({
  experimentId,
  disabled = false,
  withSummary = true,
}: {
  experimentId: string | null;
  disabled?: boolean;
  withSummary?: boolean;
}) {
  if (!experimentId) return null;
  return (
    <div className="flex flex-wrap gap-2">
      <Button asChild variant="outline" size="sm" disabled={disabled}>
        <a href={api.exportCsvUrl(experimentId)} download>
          <Download className="h-4 w-4" /> Detailed CSV
        </a>
      </Button>
      {withSummary && (
        <Button asChild variant="outline" size="sm" disabled={disabled}>
          <a href={api.exportSummaryCsvUrl(experimentId)} download>
            <Download className="h-4 w-4" /> Summary CSV
          </a>
        </Button>
      )}
    </div>
  );
}
