"use client";

import Link from "next/link";
import { use } from "react";
import { AlertTriangle, CheckCircle2 } from "lucide-react";

import { Spectrogram, SpectrogramScale, Waveform } from "@/components/charts/Spectrogram";
import { ErrorNotice, KeyValues, LoadingLine, PageHeader, Section, Spec } from "@/components/common";
import { api, urls } from "@/lib/api";
import { useAsync } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

const ORDER = ["cover", "stego", "attacked", "residual"] as const;

function BitString({ reference, bits }: { reference: string; bits: string }) {
  // Wrong bits are marked with an underline and weight, not colour alone.
  return (
    <span className="spec break-all leading-relaxed">
      {bits.split("").map((bit, index) => {
        const wrong = bit !== reference[index];
        return (
          <span
            key={index}
            className={wrong ? "font-bold underline decoration-2 underline-offset-2" : ""}
            style={wrong ? { color: "var(--status-critical)" } : undefined}
          >
            {bit}
          </span>
        );
      })}
    </span>
  );
}

export default function TrialPage({ params }: { params: Promise<{ id: string; rowId: string }> }) {
  const { id, rowId } = use(params);
  const { t, number } = useI18n();
  const inspection = useAsync(() => api.inspect(id, Number(rowId)), [id, rowId]);

  return (
    <>
      <PageHeader
        eyebrow={
          <Link href={`/runs/${id}`} className="hover:underline">
            ← {t("nav.runs")}
          </Link>
        }
        title={t("inspector.title")}
        subtitle={t("inspector.subtitle")}
      />
      {inspection.error && <ErrorNotice error={inspection.error} onRetry={inspection.reload} />}
      {!inspection.data ? (
        !inspection.error && <LoadingLine label={t("inspector.loading")} />
      ) : (
        (() => {
          const data = inspection.data;
          const row = data.row;
          return (
            <div className="space-y-6">
              <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                <Section title={t("common.details")}>
                  <KeyValues
                    items={[
                      [t("common.file"), row.file_name],
                      [t("run.filterMethod"), <Spec key="m">{row.method}</Spec>],
                      [t("run.filterAttack"), row.attack ? <Spec key="a">{row.attack}</Spec> : t("run.baseline")],
                      [t("run.bits"), <span key="b" className="num">{row.payload_length} · rep {row.repetition}</span>],
                      ["BER", <span key="ber" className="num">{number(row.ber, 4)}</span>],
                      [t("datasets.rate"), <span key="r" className="num">{data.sample_rate} Hz · {number(data.duration_seconds, 2)} s</span>],
                    ]}
                  />
                </Section>
                <Section title={t("inspector.message")}>
                  <div className="space-y-3">
                    <div
                      className="flex items-center gap-2 text-sm font-medium"
                      style={{ color: data.reproduced ? "hsl(var(--success))" : "var(--status-serious)" }}
                    >
                      {data.reproduced ? <CheckCircle2 className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}
                      {data.reproduced ? t("inspector.reproduced") : t("inspector.notReproduced")}
                    </div>
                    <div>
                      <div className="eyebrow mb-1">{t("inspector.message")}</div>
                      <span className="spec break-all">{row.message_bits}</span>
                    </div>
                    <div>
                      <div className="eyebrow mb-1">{t("inspector.decoded")}</div>
                      <BitString reference={row.message_bits ?? ""} bits={data.decoded_bits} />
                    </div>
                  </div>
                </Section>
              </div>

              <Section
                title={t("inspector.spectrogram")}
                hint={t("inspector.spectrogramHint")}
                actions={<SpectrogramScale />}
              >
                <div className="grid gap-6 xl:grid-cols-2">
                  {ORDER.filter((name) => data.signals[name]).map((name) => (
                    <div key={name} className="space-y-2">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="text-sm font-medium">{t(`inspector.signals.${name}`)}</div>
                        <audio controls preload="none" src={urls.audio(id, Number(rowId), name)} className="h-8" />
                      </div>
                      {name === "residual" && <p className="text-xs text-muted-foreground">{t("inspector.residualNote")}</p>}
                      <Waveform envelope={data.signals[name].envelope} />
                      <Spectrogram data={data.signals[name].spectrogram} timeLabel="s" frequencyLabel="kHz" />
                    </div>
                  ))}
                </div>
              </Section>

              {Object.keys(row.metrics).length > 0 && (
                <Section title={t("experiment.metrics")}>
                  <KeyValues items={Object.entries(row.metrics).map(([name, value]) => [name, <span key={name} className="num">{number(value, 3)}</span>])} />
                </Section>
              )}
            </div>
          );
        })()
      )}
    </>
  );
}
