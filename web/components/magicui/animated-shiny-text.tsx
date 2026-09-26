// Adapted from Magic UI (https://magicui.design, MIT), rewritten for Tailwind 3:
// muted text with a highlight sweeping across it now and then.

import type { CSSProperties, ReactNode } from "react";

import { cn } from "@/lib/utils";

export function AnimatedShinyText({ children, className, shimmerWidth = 100 }: { children: ReactNode; className?: string; shimmerWidth?: number }) {
  return (
    <span
      style={{ "--shiny-width": `${shimmerWidth}px` } as CSSProperties}
      className={cn(
        "text-muted-foreground",
        "bg-clip-text bg-no-repeat [background-position:0_0] [background-size:var(--shiny-width)_100%] motion-safe:animate-shiny-text",
        "bg-gradient-to-r from-transparent via-foreground/80 via-50% to-transparent",
        className
      )}
    >
      {children}
    </span>
  );
}
