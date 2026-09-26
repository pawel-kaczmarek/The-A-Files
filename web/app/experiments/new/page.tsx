"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useMemo } from "react";

import { ErrorNotice, LoadingLine, PageHeader } from "@/components/common";
import { DesignGallery } from "@/components/editor/DesignGallery";
import { defaultDraft } from "@/components/editor/draft";
import { ExperimentEditor } from "@/components/editor/ExperimentEditor";
import { useCatalog } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import type { ExperimentType } from "@/lib/types";

function NewExperiment() {
  const { t } = useI18n();
  const router = useRouter();
  const params = useSearchParams();
  const catalog = useCatalog();
  const type = params.get("design") as ExperimentType | null;

  const initial = useMemo(() => {
    if (!type || !catalog.data) return null;
    const design = catalog.data.designs.find((entry) => entry.type === type);
    const draft = defaultDraft(type, design);
    // Arriving from a catalogue page: start from that component.
    const method = params.get("method");
    const attack = params.get("attack");
    const metric = params.get("metric");
    if (method) {
      if (design?.requires_sweep === "method" && draft.config.method_sweep) draft.config.method_sweep.target = method;
      else draft.config.methods = [method, ...draft.config.methods.filter((entry) => entry !== method)];
    }
    if (attack) {
      if (design?.requires_sweep === "attack" && draft.config.attack_sweep) {
        const attackInfo = catalog.data.attacks.find((entry) => entry.name === attack);
        if (attackInfo?.sweep) draft.config.attack_sweep = { target: attack, parameter: attackInfo.sweep.parameter, values: attackInfo.sweep.values };
      } else if (!draft.config.attacks.some((spec) => spec.startsWith(attack))) {
        draft.config.attacks = [...draft.config.attacks, `${attack}@moderate`];
      }
    }
    if (metric && !draft.config.metrics.includes(metric)) draft.config.metrics = [...draft.config.metrics, metric];
    return draft;
  }, [type, catalog.data, params]);

  if (catalog.error) return <ErrorNotice error={catalog.error} onRetry={catalog.reload} />;
  if (!catalog.data) return <LoadingLine />;

  if (!type || !initial) {
    const carried = ["method", "attack", "metric"]
      .map((key) => (params.get(key) ? `&${key}=${encodeURIComponent(params.get(key) as string)}` : ""))
      .join("");
    return (
      <>
        <PageHeader eyebrow={t("editor.newTitle")} title={t("gallery.title")} subtitle={t("gallery.subtitle")} />
        <DesignGallery designs={catalog.data.designs} onChoose={(choice) => router.push(`/experiments/new?design=${choice}${carried}`)} />
      </>
    );
  }

  return (
    <>
      <PageHeader eyebrow={t("editor.newTitle")} title={t(`designs.${type}.title`)} subtitle={t(`designs.${type}.description`)} />
      <ExperimentEditor key={type} catalog={catalog.data} initial={initial} />
    </>
  );
}

export default function NewExperimentPage() {
  return (
    <Suspense fallback={<LoadingLine />}>
      <NewExperiment />
    </Suspense>
  );
}
