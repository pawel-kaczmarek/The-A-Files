// Adapted from Magic UI's animated circular progress bar
// (https://magicui.design, MIT): a gauge with a gap between the done and the
// remaining arc; the arcs ease between values.

import { cn } from "@/lib/utils";

export function CircularProgress({
  value,
  label,
  className,
  primary = "hsl(var(--primary))",
  secondary = "hsl(var(--muted))",
}: {
  /** Percentage, 0-100. */
  value: number;
  label?: string;
  className?: string;
  primary?: string;
  secondary?: string;
}) {
  const percent = Math.max(0, Math.min(100, Math.round(value)));
  const circumference = 2 * Math.PI * 45;
  const gap = percent > 0 && percent < 100 ? 4 : 0;
  const done = (Math.max(0, percent - gap / 2) / 100) * circumference;
  const rest = (Math.max(0, 100 - percent - gap / 2) / 100) * circumference;
  const arc = { transition: "stroke-dasharray 0.8s ease, stroke-dashoffset 0.8s ease" };
  return (
    <div
      className={cn("relative h-14 w-14", className)}
      role="progressbar"
      aria-valuenow={percent}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
    >
      <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90" fill="none">
        <circle
          cx="50"
          cy="50"
          r="45"
          strokeWidth="9"
          strokeLinecap="round"
          stroke={secondary}
          strokeDasharray={`${rest} ${circumference}`}
          strokeDashoffset={-(done + (gap / 100) * circumference)}
          style={arc}
        />
        <circle cx="50" cy="50" r="45" strokeWidth="9" strokeLinecap="round" stroke={primary} strokeDasharray={`${done} ${circumference}`} style={arc} />
      </svg>
      <span className="num absolute inset-0 flex items-center justify-center text-xs font-semibold">{percent}%</span>
    </div>
  );
}
