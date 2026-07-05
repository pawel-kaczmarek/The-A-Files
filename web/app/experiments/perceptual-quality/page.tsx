"use client";

import { ExperimentLayout } from "@/components/experiments/ExperimentLayout";

export default function PerceptualQualityPage() {
  return <ExperimentLayout type="perceptual_quality" requireMetrics showAttacks={false} />;
}
