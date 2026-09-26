"use client";

import Link from "next/link";
import { useState } from "react";
import { FlaskConical } from "lucide-react";

import { Chip, ErrorNotice, LoadingLine, PageHeader, Spec } from "@/components/common";
import { Input } from "@/components/ui/input";
import { useCatalog } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

const FAMILY_ORDER = ["noise", "codec", "filtering", "resampling", "quantization", "amplitude", "temporal", "acoustic"];

export default function AttacksPage() {
  const { t } = useI18n();
  const catalog = useCatalog();
  const [search, setSearch] = useState("");
  if (catalog.error) return <ErrorNotice error={catalog.error} onRetry={catalog.reload} />;
  if (!catalog.data) return <LoadingLine />;
  const attacks = catalog.data.attacks.filter(
    (attack) => attack.name !== "codec" && `${attack.name} ${attack.description}`.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <>
      <PageHeader title={t("catalogue.attacksTitle")} subtitle={t("catalogue.attacksSubtitle")}>
        <Input className="mt-5 w-72" placeholder={t("common.search")} value={search} onChange={(event) => setSearch(event.target.value)} />
      </PageHeader>
      <div className="space-y-8">
        {FAMILY_ORDER.filter((family) => attacks.some((attack) => attack.family === family)).map((family) => (
          <section key={family}>
            <h2 className="eyebrow mb-2">{t(`attackFamilies.${family}`)}</h2>
            <div className="divide-y rounded-lg border bg-card">
              {attacks
                .filter((attack) => attack.family === family)
                .map((attack) => (
                  <div key={attack.name} id={attack.name} className="grid gap-3 px-4 py-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)_auto]">
                    <div>
                      <div className="font-mono text-sm font-medium">{attack.name}</div>
                      <div className="text-xs text-muted-foreground">{attack.description}</div>
                      <div className="mt-1.5 flex flex-wrap gap-1">
                        {attack.stochastic && <Chip>{t("catalogue.stochastic")}</Chip>}
                        {attack.changes_length_or_rate && <Chip>{t("catalogue.changesLength")}</Chip>}
                        {attack.has_severity && <Chip>mild · moderate · strong · extreme</Chip>}
                      </div>
                    </div>
                    <div className="space-y-1.5 text-xs">
                      <div className="flex flex-wrap gap-1">
                        {attack.parameters.map((parameter) => (
                          <Spec key={parameter.name}>
                            {parameter.name}={String(parameter.default)}
                          </Spec>
                        ))}
                      </div>
                      {attack.sweep && (
                        <div className="text-muted-foreground">
                          {t("catalogue.sweepLadder")}: <span className="font-mono">{attack.sweep.parameter}</span> ={" "}
                          <span className="num">{attack.sweep.values.join(", ")}</span> ({attack.sweep.unit})
                        </div>
                      )}
                    </div>
                    <Link
                      href={`/experiments/new?attack=${encodeURIComponent(attack.name)}`}
                      className="inline-flex items-center gap-1 self-start text-xs text-primary hover:underline"
                    >
                      <FlaskConical className="h-3.5 w-3.5" /> {t("common.useInExperiment")}
                    </Link>
                  </div>
                ))}
            </div>
          </section>
        ))}
      </div>
    </>
  );
}
