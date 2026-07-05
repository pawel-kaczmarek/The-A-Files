"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";

// Both components only edit config.advanced_options; the backend interprets
// the values (thresholds in the capacity scenario, weights in comparison).

export function ThresholdControls({
  advanced,
  setAdvanced,
}: {
  advanced: Record<string, unknown>;
  setAdvanced: (patch: Record<string, unknown>) => void;
}) {
  const minAccuracy = Number(advanced.min_bit_accuracy ?? 0.95);
  const maxBer = Number(advanced.max_ber ?? 0.05);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Pass/fail thresholds</CardTitle>
        <CardDescription>
          A payload length passes for a method when its average bit accuracy and BER meet both
          thresholds.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6 sm:grid-cols-2">
        <div className="space-y-2">
          <Label>Minimum bit accuracy: {(minAccuracy * 100).toFixed(0)}%</Label>
          <Slider
            min={50}
            max={100}
            step={1}
            value={Math.round(minAccuracy * 100)}
            onChange={(event) => setAdvanced({ min_bit_accuracy: Number(event.target.value) / 100 })}
          />
        </div>
        <div className="space-y-2">
          <Label>Maximum BER: {(maxBer * 100).toFixed(0)}%</Label>
          <Slider
            min={0}
            max={50}
            step={1}
            value={Math.round(maxBer * 100)}
            onChange={(event) => setAdvanced({ max_ber: Number(event.target.value) / 100 })}
          />
        </div>
      </CardContent>
    </Card>
  );
}

const WEIGHT_KEYS = ["quality", "robustness", "accuracy", "speed"] as const;
const WEIGHT_DEFAULTS: Record<(typeof WEIGHT_KEYS)[number], number> = {
  quality: 0.3,
  robustness: 0.3,
  accuracy: 0.3,
  speed: 0.1,
};

export function WeightControls({
  advanced,
  setAdvanced,
}: {
  advanced: Record<string, unknown>;
  setAdvanced: (patch: Record<string, unknown>) => void;
}) {
  const weights = { ...WEIGHT_DEFAULTS, ...((advanced.weights as object) ?? {}) } as Record<
    string,
    number
  >;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Score weights</CardTitle>
        <CardDescription>
          The overall ranking is a weighted mean of normalized component scores; the backend
          renormalizes the weights to sum to 1.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {WEIGHT_KEYS.map((key) => (
          <div key={key} className="space-y-1">
            <Label htmlFor={`weight-${key}`} className="capitalize">
              {key}
            </Label>
            <Input
              id={`weight-${key}`}
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={weights[key]}
              onChange={(event) =>
                setAdvanced({ weights: { ...weights, [key]: Number(event.target.value) } })
              }
            />
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
