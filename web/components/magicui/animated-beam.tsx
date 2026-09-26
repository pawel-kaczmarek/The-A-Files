"use client";

// Adapted from Magic UI (https://magicui.design, MIT): a curved connector between
// two elements of a container, with a pulse travelling along it. With reduced
// motion only the static connector is drawn.

import { useEffect, useId, useState, type RefObject } from "react";
import { motion, useReducedMotion } from "motion/react";

import { cn } from "@/lib/utils";

export function AnimatedBeam({
  className,
  containerRef,
  fromRef,
  toRef,
  curvature = 0,
  reverse = false,
  duration = 4,
  delay = 0,
  pathColor = "var(--chart-axis)",
  pathWidth = 1.5,
  pathOpacity = 1,
  gradientStartColor = "hsl(var(--primary))",
  gradientStopColor = "var(--series-7)",
  startXOffset = 0,
  startYOffset = 0,
  endXOffset = 0,
  endYOffset = 0,
  dashed = false,
}: {
  className?: string;
  containerRef: RefObject<HTMLElement | null>;
  fromRef: RefObject<HTMLElement | null>;
  toRef: RefObject<HTMLElement | null>;
  curvature?: number;
  reverse?: boolean;
  duration?: number;
  delay?: number;
  pathColor?: string;
  pathWidth?: number;
  pathOpacity?: number;
  gradientStartColor?: string;
  gradientStopColor?: string;
  startXOffset?: number;
  startYOffset?: number;
  endXOffset?: number;
  endYOffset?: number;
  /** Dashed connector: a measurement taken from a signal, not a processing step. */
  dashed?: boolean;
}) {
  const id = useId();
  const reduced = useReducedMotion();
  const [path, setPath] = useState("");
  const [size, setSize] = useState({ width: 0, height: 0 });

  const coordinates = reverse
    ? { x1: ["90%", "-10%"], x2: ["100%", "0%"], y1: ["0%", "0%"], y2: ["0%", "0%"] }
    : { x1: ["10%", "110%"], x2: ["0%", "100%"], y1: ["0%", "0%"], y2: ["0%", "0%"] };

  useEffect(() => {
    const update = () => {
      if (!containerRef.current || !fromRef.current || !toRef.current) return;
      const container = containerRef.current.getBoundingClientRect();
      const a = fromRef.current.getBoundingClientRect();
      const b = toRef.current.getBoundingClientRect();
      setSize({ width: container.width, height: container.height });
      const startX = a.left - container.left + a.width / 2 + startXOffset;
      const startY = a.top - container.top + a.height / 2 + startYOffset;
      const endX = b.left - container.left + b.width / 2 + endXOffset;
      const endY = b.top - container.top + b.height / 2 + endYOffset;
      setPath(`M ${startX},${startY} Q ${(startX + endX) / 2},${startY - curvature} ${endX},${endY}`);
    };
    const observer = new ResizeObserver(update);
    if (containerRef.current) observer.observe(containerRef.current);
    update();
    return () => observer.disconnect();
  }, [containerRef, fromRef, toRef, curvature, startXOffset, startYOffset, endXOffset, endYOffset]);

  return (
    <svg
      fill="none"
      width={size.width}
      height={size.height}
      aria-hidden
      className={cn("pointer-events-none absolute left-0 top-0 transform-gpu", className)}
      viewBox={`0 0 ${size.width} ${size.height}`}
    >
      <path
        d={path}
        stroke={pathColor}
        strokeWidth={pathWidth}
        strokeOpacity={pathOpacity}
        strokeLinecap="round"
        strokeDasharray={dashed ? "3 4" : undefined}
      />
      {!reduced && (
        <>
          <path d={path} strokeWidth={pathWidth + 0.5} stroke={`url(#${id})`} strokeLinecap="round" />
          <defs>
            <motion.linearGradient
              id={id}
              gradientUnits="userSpaceOnUse"
              initial={{ x1: "0%", x2: "0%", y1: "0%", y2: "0%" }}
              animate={coordinates}
              transition={{ delay, duration, ease: [0.16, 1, 0.3, 1], repeat: Infinity, repeatDelay: 0.6 }}
            >
              <stop stopColor={gradientStartColor} stopOpacity="0" />
              <stop stopColor={gradientStartColor} />
              <stop offset="32.5%" stopColor={gradientStopColor} />
              <stop offset="100%" stopColor={gradientStopColor} stopOpacity="0" />
            </motion.linearGradient>
          </defs>
        </>
      )}
    </svg>
  );
}
