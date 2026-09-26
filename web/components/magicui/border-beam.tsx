"use client";

// Adapted from Magic UI (https://magicui.design, MIT), rewritten for Tailwind 3.
// A light travelling along the border of its (relative, rounded) parent; used to
// mark work in progress. Not rendered with reduced motion.

import { motion, useReducedMotion } from "motion/react";
import type { CSSProperties } from "react";

export function BorderBeam({
  size = 80,
  duration = 6,
  delay = 0,
  colorFrom = "hsl(var(--primary))",
  colorTo = "var(--series-7)",
  borderWidth = 1.5,
}: {
  size?: number;
  duration?: number;
  delay?: number;
  colorFrom?: string;
  colorTo?: string;
  borderWidth?: number;
}) {
  const reduced = useReducedMotion();
  if (reduced) return null;
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 rounded-[inherit] border-solid border-transparent"
      style={{
        borderWidth,
        WebkitMask: "linear-gradient(transparent, transparent) padding-box, linear-gradient(#000, #000) border-box",
        mask: "linear-gradient(transparent, transparent) padding-box, linear-gradient(#000, #000) border-box",
        WebkitMaskComposite: "source-in",
        maskComposite: "intersect",
      }}
    >
      <motion.div
        className="absolute aspect-square"
        style={
          {
            width: size,
            offsetPath: `rect(0 auto auto 0 round ${size}px)`,
            background: `linear-gradient(to left, ${colorFrom}, ${colorTo}, transparent)`,
          } as CSSProperties
        }
        initial={{ offsetDistance: "0%" }}
        animate={{ offsetDistance: ["0%", "100%"] }}
        transition={{ repeat: Infinity, ease: "linear", duration, delay: -delay }}
      />
    </div>
  );
}
