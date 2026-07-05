"use client";

import { ExperimentLayout } from "@/components/experiments/ExperimentLayout";
import { ThresholdControls } from "@/components/experiments/ScenarioOptions";

export default function EmbeddingCapacityPage() {
  return (
    <ExperimentLayout
      type="embedding_capacity"
      showAttacks={false}
      defaultPayloads={[4, 8, 16, 32, 64, 120]}
      extraCards={(context) => <ThresholdControls {...context} />}
    />
  );
}
