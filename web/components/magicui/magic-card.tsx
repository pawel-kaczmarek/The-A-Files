"use client";

// Adapted from Magic UI (https://magicui.design, MIT): a card whose border and
// surface light up around the pointer. Simplified to the gradient mode and bound
// to the platform's tokens, so it follows the light and the dark theme.

import { useCallback, type ReactNode } from "react";
import { motion, useMotionTemplate, useMotionValue } from "motion/react";

import { cn } from "@/lib/utils";

export function MagicCard({
  children,
  className,
  gradientSize = 220,
  gradientFrom = "hsl(var(--primary))",
  gradientTo = "var(--series-7)",
  spotlight = "hsl(var(--primary) / 0.07)",
}: {
  children: ReactNode;
  className?: string;
  gradientSize?: number;
  gradientFrom?: string;
  gradientTo?: string;
  spotlight?: string;
}) {
  const x = useMotionValue(-gradientSize);
  const y = useMotionValue(-gradientSize);

  const move = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const box = event.currentTarget.getBoundingClientRect();
      x.set(event.clientX - box.left);
      y.set(event.clientY - box.top);
    },
    [x, y]
  );
  const leave = useCallback(() => {
    x.set(-gradientSize);
    y.set(-gradientSize);
  }, [x, y, gradientSize]);

  const border = useMotionTemplate`
    linear-gradient(hsl(var(--card)) 0 0) padding-box,
    radial-gradient(${gradientSize}px circle at ${x}px ${y}px, ${gradientFrom}, ${gradientTo}, hsl(var(--border)) 100%) border-box`;
  const glow = useMotionTemplate`radial-gradient(${gradientSize}px circle at ${x}px ${y}px, ${spotlight}, transparent 100%)`;

  return (
    <motion.div
      className={cn("group relative isolate rounded-lg border border-transparent", className)}
      onPointerMove={move}
      onPointerLeave={leave}
      style={{ background: border }}
    >
      <motion.div
        aria-hidden
        className="pointer-events-none absolute inset-px z-10 rounded-[inherit] opacity-0 transition-opacity duration-300 group-hover:opacity-100"
        style={{ background: glow }}
      />
      <div className="relative z-20 h-full">{children}</div>
    </motion.div>
  );
}
