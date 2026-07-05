"use client";

import { Accordion, AccordionItem } from "@/components/ui/accordion";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import type { ExperimentConfig } from "@/lib/types";

export function AdvancedSettings({
  config,
  onChange,
  showOutputOptions = true,
}: {
  config: ExperimentConfig;
  onChange: (patch: Partial<ExperimentConfig>) => void;
  showOutputOptions?: boolean;
}) {
  return (
    <Accordion>
      <AccordionItem title="Advanced settings">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div className="space-y-1">
            <Label htmlFor="repetitions">Repetitions per payload</Label>
            <Input
              id="repetitions"
              type="number"
              min={1}
              max={50}
              value={config.repetitions ?? 1}
              onChange={(event) => onChange({ repetitions: Number(event.target.value) })}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="seed">Random seed</Label>
            <Input
              id="seed"
              type="number"
              placeholder="random"
              value={config.random_seed ?? ""}
              onChange={(event) =>
                onChange({
                  random_seed: event.target.value === "" ? null : Number(event.target.value),
                })
              }
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="workers">Parallel workers</Label>
            <Input
              id="workers"
              type="number"
              min={1}
              max={16}
              value={config.max_workers ?? 2}
              onChange={(event) => onChange({ max_workers: Number(event.target.value) })}
            />
          </div>
          {showOutputOptions && (
            <>
              <div className="flex items-center justify-between gap-2 rounded-md border px-3 py-2">
                <Label className="text-sm font-normal">Save encoded audio</Label>
                <Switch
                  checked={config.save_encoded_audio ?? false}
                  onCheckedChange={(checked) => onChange({ save_encoded_audio: checked })}
                />
              </div>
              <div className="space-y-1 sm:col-span-2">
                <Label htmlFor="outputDir">Output directory (backend path)</Label>
                <Input
                  id="outputDir"
                  placeholder="default: system temp"
                  value={config.output_directory ?? ""}
                  onChange={(event) =>
                    onChange({ output_directory: event.target.value || null })
                  }
                />
              </div>
            </>
          )}
        </div>
      </AccordionItem>
    </Accordion>
  );
}
