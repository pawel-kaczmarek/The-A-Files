"use client";

import { ExperimentLayout } from "@/components/experiments/ExperimentLayout";
import { WeightControls } from "@/components/experiments/ScenarioOptions";

export default function MethodComparisonPage() {
  return (
    <ExperimentLayout
      type="method_comparison"
      extraCards={(context) => <WeightControls {...context} />}
    />
  );
}
