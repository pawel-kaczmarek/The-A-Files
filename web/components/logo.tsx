import { useId } from "react";

import { cn } from "@/lib/utils";

/** The A-Files mark: a spectrogram "A" whose crossbar is a notch carrying the embedded payload. */
export function LogoMark({ className, title }: { className?: string; title?: string }) {
  const id = useId();
  return (
    <svg
      viewBox="0 0 84 100"
      className={cn("shrink-0", className)}
      role={title ? "img" : undefined}
      aria-hidden={title ? undefined : true}
    >
      {title ? <title>{title}</title> : null}
      <defs>
        <mask id={id}>
          <rect width="84" height="100" fill="#fff" />
          <rect y="58" width="84" height="10" fill="#000" />
        </mask>
      </defs>
      <g fill="currentColor" mask={`url(#${id})`}>
        {[
          [0, 75], [11, 50], [22, 25], [33, 0], [44, 0], [55, 25], [66, 50], [77, 75],
        ].map(([x, y]) => (
          <rect key={x} x={x} y={y} width="7" height={100 - y} />
        ))}
      </g>
      <rect x="33" y="60" width="18" height="6" fill="var(--series-3)" />
    </svg>
  );
}
