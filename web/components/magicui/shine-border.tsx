// Adapted from Magic UI (https://magicui.design, MIT): a slow shine running
// around the border of its (relative, rounded) parent. Static with reduced motion.

import type { CSSProperties } from "react";

import { cn } from "@/lib/utils";

export function ShineBorder({
  borderWidth = 1,
  duration = 14,
  colors = ["hsl(var(--primary))", "var(--series-7)", "var(--series-3)"],
  className,
}: {
  borderWidth?: number;
  duration?: number;
  colors?: string[];
  className?: string;
}) {
  return (
    <div
      aria-hidden
      style={
        {
          "--duration": `${duration}s`,
          backgroundImage: `radial-gradient(transparent, transparent, ${colors.join(",")}, transparent, transparent)`,
          backgroundSize: "300% 300%",
          mask: "linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)",
          WebkitMask: "linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)",
          WebkitMaskComposite: "xor",
          maskComposite: "exclude",
          padding: borderWidth,
        } as CSSProperties
      }
      className={cn(
        "pointer-events-none absolute inset-0 h-full w-full rounded-[inherit] will-change-[background-position] motion-safe:animate-shine",
        className
      )}
    />
  );
}
