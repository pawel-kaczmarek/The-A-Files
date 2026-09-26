"use client";

import Link from "next/link";
import { useRef, type ReactNode, type RefObject } from "react";
import { AudioLines, AudioWaveform, Binary, Combine, RadioTower, ScanSearch } from "lucide-react";

import { PROPERTY_SLOT } from "@/components/common";
import { AnimatedBeam } from "@/components/magicui/animated-beam";
import { useI18n } from "@/lib/i18n";
import type { Property } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * One trial of the evaluation model, from cover to decoded bits, and where each
 * of the four properties is measured. Solid connectors are processing steps,
 * dashed ones measurements; each measurement opens the design that studies it.
 */
export function EvaluationModel({ className }: { className?: string }) {
  const { t } = useI18n();
  const container = useRef<HTMLDivElement>(null);
  const cover = useRef<HTMLDivElement>(null);
  const payload = useRef<HTMLDivElement>(null);
  const embed = useRef<HTMLDivElement>(null);
  const stego = useRef<HTMLDivElement>(null);
  const channel = useRef<HTMLDivElement>(null);
  const decode = useRef<HTMLDivElement>(null);
  const imperceptibility = useRef<HTMLAnchorElement>(null);
  const security = useRef<HTMLAnchorElement>(null);
  const robustness = useRef<HTMLAnchorElement>(null);
  const capacity = useRef<HTMLAnchorElement>(null);

  const chain: [RefObject<HTMLElement | null>, RefObject<HTMLElement | null>, number][] = [
    [cover, embed, 0],
    [payload, embed, 0.4],
    [embed, stego, 0.8],
    [stego, channel, 1.2],
    [channel, decode, 1.6],
  ];
  const measures: [RefObject<HTMLElement | null>, RefObject<HTMLElement | null>, Property, number][] = [
    [cover, imperceptibility, "imperceptibility", 0.3],
    [stego, imperceptibility, "imperceptibility", 0.3],
    [stego, security, "security", 1.0],
    [decode, robustness, "robustness", 2.0],
    [decode, capacity, "capacity", 2.2],
  ];

  return (
    <div className={cn("overflow-x-auto", className)}>
      <div ref={container} className="relative mx-auto grid min-w-[700px] grid-cols-5 grid-rows-[auto_auto_auto] items-center gap-x-6 gap-y-10 py-2">
        <div className="col-start-2 row-start-1 flex justify-center">
          <Node ref={payload} icon={<Binary />} title={t("model.nodes.payload.title")} detail={t("model.nodes.payload.detail")} />
        </div>
        <div className="col-start-5 row-start-1 flex justify-center">
          <Measure ref={capacity} property="capacity" design="embedding_capacity" />
        </div>

        <div className="col-start-1 row-start-2 flex justify-center">
          <Node ref={cover} icon={<AudioLines />} title={t("model.nodes.cover.title")} detail={t("model.nodes.cover.detail")} />
        </div>
        <div className="col-start-2 row-start-2 flex justify-center">
          <Node ref={embed} icon={<Combine />} title={t("model.nodes.embed.title")} detail={t("model.nodes.embed.detail")} emphasis />
        </div>
        <div className="col-start-3 row-start-2 flex justify-center">
          <Node ref={stego} icon={<AudioWaveform />} title={t("model.nodes.stego.title")} detail={t("model.nodes.stego.detail")} />
        </div>
        <div className="col-start-4 row-start-2 flex justify-center">
          <Node ref={channel} icon={<RadioTower />} title={t("model.nodes.channel.title")} detail={t("model.nodes.channel.detail")} />
        </div>
        <div className="col-start-5 row-start-2 flex justify-center">
          <Node ref={decode} icon={<ScanSearch />} title={t("model.nodes.decode.title")} detail={t("model.nodes.decode.detail")} emphasis />
        </div>

        <div className="col-start-2 row-start-3 flex justify-center">
          <Measure ref={imperceptibility} property="imperceptibility" design="perceptual_quality" />
        </div>
        <div className="col-start-4 row-start-3 flex justify-center">
          <Measure ref={security} property="security" design="detectability" />
        </div>
        <div className="col-start-5 row-start-3 flex justify-center">
          <Measure ref={robustness} property="robustness" design="robustness_curve" />
        </div>

        {chain.map(([from, to, delay], index) => (
          <AnimatedBeam key={`chain-${index}`} containerRef={container} fromRef={from} toRef={to} delay={delay} duration={3.2} />
        ))}
        {measures.map(([from, to, property, delay], index) => (
          <AnimatedBeam
            key={`measure-${index}`}
            containerRef={container}
            fromRef={from}
            toRef={to}
            delay={delay}
            duration={3.6}
            dashed
            curvature={from === stego && to === security ? -30 : 0}
            gradientStartColor={PROPERTY_SLOT[property]}
            gradientStopColor={PROPERTY_SLOT[property]}
          />
        ))}
      </div>
      <div className="mt-4 flex min-w-[700px] items-center justify-center gap-6 text-[11px] text-muted-foreground">
        <span className="flex items-center gap-2">
          <svg width="28" height="6" aria-hidden>
            <line x1="1" x2="27" y1="3" y2="3" stroke="var(--chart-axis)" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          {t("model.legendProcess")}
        </span>
        <span className="flex items-center gap-2">
          <svg width="28" height="6" aria-hidden>
            <line x1="1" x2="27" y1="3" y2="3" stroke="var(--chart-axis)" strokeWidth="1.5" strokeDasharray="3 4" strokeLinecap="round" />
          </svg>
          {t("model.legendMeasure")}
        </span>
      </div>
    </div>
  );
}

function Node({
  ref,
  icon,
  title,
  detail,
  emphasis = false,
}: {
  ref: RefObject<HTMLDivElement | null>;
  icon: ReactNode;
  title: string;
  detail: string;
  emphasis?: boolean;
}) {
  return (
    <div
      ref={ref}
      className={cn(
        "relative z-10 w-full max-w-[170px] rounded-lg border bg-card px-3 py-2.5 shadow-sm",
        emphasis && "border-primary/40 ring-1 ring-primary/10"
      )}
    >
      <div className="flex items-center gap-2 text-xs font-semibold">
        <span className="text-primary [&>svg]:h-4 [&>svg]:w-4" aria-hidden>
          {icon}
        </span>
        {title}
      </div>
      <div className="spec mt-1 text-[11px] leading-snug text-muted-foreground">{detail}</div>
    </div>
  );
}

function Measure({ ref, property, design }: { ref: RefObject<HTMLAnchorElement | null>; property: Property; design: string }) {
  const { t } = useI18n();
  return (
    <Link
      ref={ref}
      href={`/experiments/new?design=${design}`}
      title={t(`designs.${design}.title`)}
      className="relative z-10 w-full max-w-[170px] rounded-lg border border-dashed bg-card px-3 py-2.5 transition-colors hover:border-solid"
      style={{ borderColor: PROPERTY_SLOT[property] }}
    >
      <div className="flex items-center gap-2 text-xs font-semibold">
        <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: PROPERTY_SLOT[property] }} aria-hidden />
        {t(`properties.${property}.name`)}
      </div>
      <div className="spec mt-1 text-[11px] text-foreground">{t(`model.measures.${property}.formula`)}</div>
      <div className="mt-0.5 text-[11px] leading-snug text-muted-foreground">{t(`model.measures.${property}.detail`)}</div>
    </Link>
  );
}
