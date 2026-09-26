"use client";

import { ArrowRight } from "lucide-react";

import { Chip, PROPERTY_ORDER, PROPERTY_SLOT } from "@/components/common";
import { BlurFade } from "@/components/magicui/blur-fade";
import { MagicCard } from "@/components/magicui/magic-card";
import { useI18n } from "@/lib/i18n";
import type { DesignInfo, ExperimentType } from "@/lib/types";

export function DesignGallery({ designs, onChoose }: { designs: DesignInfo[]; onChoose: (type: ExperimentType) => void }) {
  const { t, tOr } = useI18n();
  return (
    <div className="space-y-10">
      {PROPERTY_ORDER.map((property, position) => {
        const group = designs.filter((design) => design.property === property);
        if (!group.length) return null;
        return (
          <BlurFade key={property} delay={position * 0.06}>
          <section>
            <div className="mb-3 flex items-baseline gap-3 border-b pb-2">
              <span className="h-2.5 w-2.5 translate-y-[-1px] rounded-full" style={{ background: PROPERTY_SLOT[property] }} />
              <h2 className="text-base font-semibold">{t(`properties.${property}.name`)}</h2>
              <span className="text-sm text-muted-foreground">{t(`properties.${property}.short`)}</span>
            </div>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {group.map((design) => (
                <MagicCard
                  key={design.type}
                  gradientFrom={PROPERTY_SLOT[property]}
                  gradientTo="hsl(var(--primary))"
                  spotlight={`color-mix(in srgb, ${PROPERTY_SLOT[property]} 13%, transparent)`}
                >
                <button
                  type="button"
                  onClick={() => onChoose(design.type)}
                  className="group flex h-full w-full flex-col rounded-lg p-5 text-left"
                >
                  <div className="text-sm font-semibold">{tOr(`designs.${design.type}.title`, design.title)}</div>
                  <div className="mt-1 text-sm italic text-muted-foreground">{t(`designs.${design.type}.question`)}</div>
                  <p className="mt-3 flex-1 text-xs leading-relaxed text-muted-foreground">
                    {tOr(`designs.${design.type}.description`, design.description)}
                  </p>
                  <dl className="mt-4 space-y-1.5 text-[11px]">
                    {(
                      [
                        ["gallery.factors", design.factors, "factors"],
                        ["gallery.measures", design.measures, "measures"],
                        ["gallery.analyses", design.analyses, "analyses"],
                      ] as const
                    ).map(([label, items, prefix]) => (
                      <div key={label} className="flex flex-wrap items-center gap-1">
                        <dt className="w-16 shrink-0 text-muted-foreground">{t(label)}</dt>
                        {items.map((item) => (
                          <Chip key={item}>{tOr(`${prefix}.${item}`, item)}</Chip>
                        ))}
                      </div>
                    ))}
                  </dl>
                  <span className="mt-4 inline-flex items-center gap-1 text-xs font-medium text-primary">
                    {t("gallery.choose")} <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
                  </span>
                </button>
                </MagicCard>
              ))}
            </div>
          </section>
          </BlurFade>
        );
      })}
    </div>
  );
}
