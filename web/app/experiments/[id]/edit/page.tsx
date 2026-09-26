"use client";

import { use } from "react";

import { ErrorNotice, LoadingLine, PageHeader } from "@/components/common";
import { ExperimentEditor } from "@/components/editor/ExperimentEditor";
import { api } from "@/lib/api";
import { useAsync, useCatalog } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import type { ExperimentConfig } from "@/lib/types";

export default function EditExperimentPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { t } = useI18n();
  const catalog = useCatalog();
  const experiment = useAsync(() => api.experiment(id), [id]);

  const error = catalog.error ?? experiment.error;
  if (error) return <ErrorNotice error={error} onRetry={experiment.reload} />;
  if (!catalog.data || !experiment.data) return <LoadingLine />;

  const current = experiment.data;
  const config = current.config as ExperimentConfig;
  return (
    <>
      <PageHeader eyebrow={t("editor.editTitle")} title={current.name} subtitle={t(`designs.${current.experiment_type}.question`)} />
      <ExperimentEditor
        catalog={catalog.data}
        experimentId={current.id}
        version={current.version}
        hasRuns={current.run_count > 0}
        initial={{
          name: current.name,
          experiment_type: current.experiment_type,
          research_question: current.research_question ?? "",
          hypothesis: current.hypothesis ?? "",
          description: current.description ?? "",
          tags: current.tags,
          config: {
            ...config,
            methods: config.methods ?? [],
            metrics: config.metrics ?? [],
            attacks: config.attacks ?? [],
            advanced_options: config.advanced_options ?? {},
          },
        }}
      />
    </>
  );
}
