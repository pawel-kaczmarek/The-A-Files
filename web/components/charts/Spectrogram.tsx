"use client";

import { useEffect, useRef, useState } from "react";

import type { SpectrogramData } from "@/lib/types";

import { AXIS_TEXT, useWidth } from "./base";

const STEPS = ["--seq-100", "--seq-200", "--seq-300", "--seq-400", "--seq-500", "--seq-600", "--seq-700"];

function parseHex(hex: string): [number, number, number] {
  const value = hex.trim().replace("#", "");
  return [parseInt(value.slice(0, 2), 16), parseInt(value.slice(2, 4), 16), parseInt(value.slice(4, 6), 16)];
}

/** Interpolated sequential ramp read from the current theme's tokens. */
function useRamp(): [number, number, number][] | null {
  const [ramp, setRamp] = useState<[number, number, number][] | null>(null);
  useEffect(() => {
    function read() {
      const style = getComputedStyle(document.documentElement);
      const surface = parseHex(style.getPropertyValue("--chart-surface") || "#fcfcfb");
      setRamp([surface, ...STEPS.map((step) => parseHex(style.getPropertyValue(step)))]);
    }
    read();
    const observer = new MutationObserver(read);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);
  return ramp;
}

function colorAt(ramp: [number, number, number][], t: number): [number, number, number] {
  const position = Math.max(0, Math.min(1, t)) * (ramp.length - 1);
  const index = Math.min(ramp.length - 2, Math.floor(position));
  const f = position - index;
  const a = ramp[index];
  const b = ramp[index + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/**
 * Magnitude spectrogram on a single-hue sequential ramp from the surface
 * (quiet) to the darkest step (loud), over a fixed dB range so that several
 * spectrograms on one page share one scale.
 */
export function Spectrogram({
  data,
  dbRange = [-90, 0],
  height = 180,
  timeLabel,
  frequencyLabel,
}: {
  data: SpectrogramData;
  dbRange?: [number, number];
  height?: number;
  timeLabel: string;
  frequencyLabel: string;
}) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const ramp = useRamp();
  const margin = { left: 44, bottom: 30, top: 4, right: 8 };
  const plotWidth = Math.max(100, width - margin.left - margin.right);
  const plotHeight = height - margin.top - margin.bottom;

  useEffect(() => {
    const element = canvas.current;
    if (!element || !ramp) return;
    const bins = data.db.length;
    const frames = data.db[0]?.length ?? 0;
    element.width = frames;
    element.height = bins;
    const context = element.getContext("2d");
    if (!context || !frames) return;
    const image = context.createImageData(frames, bins);
    const [low, high] = dbRange;
    for (let bin = 0; bin < bins; bin++) {
      for (let frame = 0; frame < frames; frame++) {
        const value = data.db[bin][frame];
        const [r, g, b] = colorAt(ramp, (value - low) / (high - low));
        const offset = ((bins - 1 - bin) * frames + frame) * 4;
        image.data[offset] = r;
        image.data[offset + 1] = g;
        image.data[offset + 2] = b;
        image.data[offset + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [data, ramp, dbRange]);

  const duration = data.times[data.times.length - 1] ?? 0;
  const nyquist = data.frequencies[data.frequencies.length - 1] ?? 0;
  const timeTicks = [0, duration / 2, duration];
  const frequencyTicks = [0, nyquist / 2, nyquist];

  return (
    <div ref={ref} className="relative" style={{ height }}>
      <canvas
        ref={canvas}
        className="absolute rounded-sm"
        style={{ left: margin.left, top: margin.top, width: plotWidth, height: plotHeight, imageRendering: "pixelated" }}
        role="img"
        aria-label={`${frequencyLabel} / ${timeLabel}`}
      />
      <svg width={width} height={height} className="pointer-events-none absolute left-0 top-0">
        {frequencyTicks.map((tick, index) => (
          <text key={index} x={margin.left - 6} y={margin.top + plotHeight * (1 - index / 2)} dy="0.32em" textAnchor="end" {...AXIS_TEXT} className="num">
            {(tick / 1000).toFixed(1)}
          </text>
        ))}
        {timeTicks.map((tick, index) => (
          <text key={index} x={margin.left + plotWidth * (index / 2)} y={margin.top + plotHeight + 14} textAnchor={index === 0 ? "start" : index === 2 ? "end" : "middle"} {...AXIS_TEXT} className="num">
            {tick.toFixed(1)}
          </text>
        ))}
        <text x={margin.left + plotWidth / 2} y={height - 2} textAnchor="middle" fontSize={10} fill="var(--chart-ink-2)">
          {timeLabel}
        </text>
        <text transform={`translate(10,${margin.top + plotHeight / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--chart-ink-2)">
          {frequencyLabel}
        </text>
      </svg>
    </div>
  );
}

export function SpectrogramScale({ dbRange = [-90, 0] }: { dbRange?: [number, number] }) {
  return (
    <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
      <span className="num">{dbRange[0]} dB</span>
      <div className="flex">
        <span className="h-2.5 w-6 border" style={{ background: "var(--chart-surface)" }} />
        {STEPS.map((step) => (
          <span key={step} className="h-2.5 w-6" style={{ background: `var(${step})` }} />
        ))}
      </div>
      <span className="num">{dbRange[1]} dB</span>
    </div>
  );
}

export function Waveform({ envelope, height = 56 }: { envelope: { min: number[]; max: number[] }; height?: number }) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const count = envelope.max.length;
  const peak = Math.max(1e-9, ...envelope.max.map(Math.abs), ...envelope.min.map(Math.abs));
  const mid = height / 2;
  const scale = (value: number) => mid - (value / peak) * (mid - 2);
  const step = width / Math.max(1, count);
  let d = "";
  for (let index = 0; index < count; index++) {
    const xPosition = index * step + step / 2;
    d += `M${xPosition},${scale(envelope.max[index])}L${xPosition},${scale(envelope.min[index])}`;
  }
  return (
    <div ref={ref}>
      <svg width={width} height={height} aria-hidden>
        <line x1={0} x2={width} y1={mid} y2={mid} stroke="var(--chart-grid)" />
        <path d={d} stroke="var(--series-1)" strokeWidth={Math.max(1, step * 0.8)} />
      </svg>
    </div>
  );
}
