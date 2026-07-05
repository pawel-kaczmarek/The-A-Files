"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import type { MethodInfo } from "@/lib/types";

export function MethodSelector({
  methods,
  selected,
  onChange,
}: {
  methods: MethodInfo[];
  selected: string[];
  onChange: (methods: string[]) => void;
}) {
  const set = new Set(selected);

  function toggle(name: string) {
    const next = new Set(set);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    onChange([...next]);
  }

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => onChange(methods.filter((m) => !m.requires_tensorflow).map((m) => m.name))}
        >
          Select all
        </Button>
        <Button type="button" variant="ghost" size="sm" onClick={() => onChange([])}>
          Clear
        </Button>
      </div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {methods.map((method) => (
          <label key={method.name} className="flex items-start gap-2 text-sm">
            <Checkbox checked={set.has(method.name)} onCheckedChange={() => toggle(method.name)} />
            <span>
              <span className="font-medium">{method.name}</span>
              {method.requires_tensorflow && (
                <Badge variant="secondary" className="ml-1.5" title="Requires the 'ai' extra (TensorFlow)">
                  TF
                </Badge>
              )}
              {method.needs_long_input && (
                <Badge
                  variant="secondary"
                  className="ml-1.5"
                  title="Needs long inputs (~65k+ samples); short files produce failed rows"
                >
                  long input
                </Badge>
              )}
              {method.description && (
                <span className="block text-xs text-muted-foreground">{method.description}</span>
              )}
            </span>
          </label>
        ))}
      </div>
    </div>
  );
}
