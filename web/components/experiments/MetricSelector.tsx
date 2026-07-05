"use client";

import { useMemo } from "react";

import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import type { MetricInfo } from "@/lib/types";

export function MetricSelector({
  metrics,
  selected,
  onChange,
}: {
  metrics: MetricInfo[];
  selected: string[];
  onChange: (metrics: string[]) => void;
}) {
  const set = new Set(selected);
  const byCategory = useMemo(() => {
    const groups: Record<string, MetricInfo[]> = {};
    for (const metric of metrics) (groups[metric.category] ??= []).push(metric);
    return groups;
  }, [metrics]);

  function toggle(name: string) {
    const next = new Set(set);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    onChange([...next]);
  }

  return (
    <div className="space-y-4">
      {Object.entries(byCategory).map(([category, rows]) => (
        <div key={category}>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {category.replaceAll("_", " ")}
          </div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {rows.map((metric) => (
              <label key={metric.name} className="flex items-center gap-2 text-sm">
                <Checkbox checked={set.has(metric.name)} onCheckedChange={() => toggle(metric.name)} />
                {metric.name}
                {metric.requires_tensorflow && (
                  <Badge variant="secondary" title="Requires the 'ai' extra (TensorFlow)">
                    TF
                  </Badge>
                )}
              </label>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
