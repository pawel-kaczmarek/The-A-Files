// Adapted from Magic UI (https://magicui.design, MIT), rewritten for Tailwind 3:
// the primary call to action, with a spark circling its edge. Keyframes live in
// tailwind.config.ts; the spark is hidden with reduced motion.

import { forwardRef, type ComponentPropsWithoutRef, type CSSProperties } from "react";

import { cn } from "@/lib/utils";

export const ShimmerButton = forwardRef<
  HTMLButtonElement,
  ComponentPropsWithoutRef<"button"> & { shimmerColor?: string; background?: string }
>(({ shimmerColor = "rgba(255,255,255,0.9)", background = "hsl(var(--primary))", className, children, style, ...props }, ref) => (
  <button
    ref={ref}
    style={
      {
        "--spread": "90deg",
        "--shimmer-color": shimmerColor,
        "--speed": "3s",
        "--cut": "0.08em",
        "--bg": background,
        ...style,
      } as CSSProperties
    }
    className={cn(
      "group relative z-0 inline-flex h-9 cursor-pointer items-center justify-center gap-2 overflow-hidden whitespace-nowrap rounded-md border border-white/10 px-4 text-sm font-medium text-primary-foreground [background:var(--bg)]",
      "transform-gpu transition-transform duration-300 ease-in-out active:translate-y-px disabled:cursor-not-allowed disabled:opacity-50",
      className
    )}
    {...props}
  >
    <span className="absolute inset-0 -z-30 overflow-visible blur-[2px] [container-type:size] motion-reduce:hidden">
      <span className="absolute inset-0 aspect-square h-[100cqh] animate-shimmer-slide rounded-none [mask:none]">
        <span className="absolute -inset-full w-auto rotate-0 animate-spin-around [background:conic-gradient(from_calc(270deg-(var(--spread)*0.5)),transparent_0,var(--shimmer-color)_var(--spread),transparent_var(--spread))] [translate:0_0]" />
      </span>
    </span>
    {children}
    <span className="absolute inset-0 rounded-md shadow-[inset_0_-8px_10px_#ffffff1f] transition-all duration-300 group-hover:shadow-[inset_0_-6px_10px_#ffffff3f] group-active:shadow-[inset_0_-10px_10px_#ffffff3f]" />
    <span className="absolute inset-[var(--cut)] -z-20 rounded-md [background:var(--bg)]" />
  </button>
));
ShimmerButton.displayName = "ShimmerButton";
