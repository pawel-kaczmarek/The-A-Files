"use client";

import { useState } from "react";
import { Plus, X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { PAYLOAD_PRESETS } from "@/lib/experimentLabels";

export function PayloadLengthSelector({
  value,
  onChange,
}: {
  value: number[];
  onChange: (lengths: number[]) => void;
}) {
  const [custom, setCustom] = useState("");
  const set = new Set(value);

  function toggle(length: number) {
    const next = new Set(set);
    if (next.has(length)) next.delete(length);
    else next.add(length);
    onChange([...next].sort((a, b) => a - b));
  }

  function addCustom() {
    const length = Number(custom);
    if (Number.isInteger(length) && length >= 4 && length <= 120) {
      toggle(length);
      setCustom("");
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        {PAYLOAD_PRESETS.map((preset) => (
          <Button
            key={preset}
            type="button"
            size="sm"
            variant={set.has(preset) ? "default" : "outline"}
            onClick={() => toggle(preset)}
          >
            {preset} bits
          </Button>
        ))}
      </div>
      <div className="flex items-center gap-2">
        <Input
          type="number"
          min={4}
          max={120}
          placeholder="custom (4–120)"
          className="w-40"
          value={custom}
          onChange={(event) => setCustom(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              addCustom();
            }
          }}
        />
        <Button type="button" variant="outline" size="sm" onClick={addCustom}>
          <Plus className="h-4 w-4" /> Add
        </Button>
      </div>
      {value.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-muted-foreground">Selected:</span>
          {value.map((length) => (
            <Badge key={length} variant="secondary" className="gap-1">
              {length}
              <button type="button" onClick={() => toggle(length)} aria-label={`Remove ${length}`}>
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}
