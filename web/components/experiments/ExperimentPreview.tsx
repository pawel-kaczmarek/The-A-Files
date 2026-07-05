"use client";

import { AlertTriangle } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { ExperimentPlan } from "@/lib/types";

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-md border bg-card px-3 py-2">
      <div className="text-lg font-semibold">{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </div>
  );
}

export function ExperimentPreview({ plan }: { plan: ExperimentPlan }) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
        <Stat label="files" value={plan.file_count} />
        <Stat label="methods" value={plan.method_count} />
        <Stat label="payload lengths" value={plan.payload_length_count} />
        <Stat label="repetitions" value={plan.repetitions} />
        <Stat label="attack variants" value={plan.attack_variant_count} />
        <Stat label="encode operations" value={plan.encode_operations} />
        <Stat label="estimated rows" value={plan.estimated_result_rows} />
      </div>
      {plan.metric_count > 0 && (
        <p className="text-xs text-muted-foreground">
          ≈ {plan.estimated_metric_calculations} metric calculations ({plan.metric_count} metric
          {plan.metric_count === 1 ? "" : "s"} per row).
        </p>
      )}
      {plan.warnings.length > 0 && (
        <Alert variant="warning">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Plan warnings</AlertTitle>
          <AlertDescription>
            <ul className="list-disc space-y-1 pl-4">
              {plan.warnings.map((warning, index) => (
                <li key={index}>{warning.message}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
