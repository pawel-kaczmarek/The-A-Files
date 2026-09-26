"use client";

import { PageHeader, PROPERTY_ORDER, PROPERTY_SLOT } from "@/components/common";
import { EvaluationModel } from "@/components/evaluation-model";
import { useI18n } from "@/lib/i18n";

const SECTIONS = ["model", "properties", "randomness", "failures", "statistics", "multi", "provenance", "data"] as const;

export default function MethodologyPage() {
  const { t } = useI18n();
  return (
    <>
      <PageHeader title={t("methodology.title")} subtitle={t("methodology.subtitle")} />
      <div className="grid gap-10 lg:grid-cols-[200px_minmax(0,1fr)]">
        <nav className="hidden lg:block">
          <ol className="sticky top-20 space-y-1.5 text-sm">
            {SECTIONS.map((section, index) => (
              <li key={section}>
                <a href={`#${section}`} className="text-muted-foreground hover:text-foreground">
                  <span className="num mr-2 text-xs">{index + 1}.</span>
                  {t(`methodology.sections.${section}.title`)}
                </a>
              </li>
            ))}
          </ol>
        </nav>
        <article className="max-w-3xl space-y-10">
          {SECTIONS.map((section, index) => (
            <section key={section} id={section} className="scroll-mt-20">
              <h2 className="mb-3 text-lg font-semibold">
                <span className="num mr-2 text-muted-foreground">{index + 1}.</span>
                {t(`methodology.sections.${section}.title`)}
              </h2>
              <p className="leading-relaxed">{t(`methodology.sections.${section}.body`)}</p>
              {section === "properties" && (
                <div className="mt-5 grid gap-3 sm:grid-cols-2">
                  {PROPERTY_ORDER.filter((property) => property !== "multi_criteria").map((property) => (
                    <div key={property} className="rounded-lg border bg-card p-4">
                      <div className="flex items-center gap-2 text-sm font-semibold">
                        <span className="h-2 w-2 rounded-full" style={{ background: PROPERTY_SLOT[property] }} />
                        {t(`properties.${property}.name`)}
                      </div>
                      <p className="mt-1 text-sm text-muted-foreground">{t(`properties.${property}.short`)}</p>
                    </div>
                  ))}
                </div>
              )}
              {section === "model" && (
                <div className="mt-5 rounded-lg border bg-card p-5">
                  <EvaluationModel />
                </div>
              )}
              {section === "model" && (
                <pre className="spec mt-4 rounded-md border bg-card p-4 leading-loose">
                  {"y[n] = E(x[n], b)\nz[n] = A_θ(y[n])\nb̂ = D(z[n], L)\nBER = (1/L) Σ 1[b_i ≠ b̂_i]"}
                </pre>
              )}
            </section>
          ))}
        </article>
      </div>
    </>
  );
}
