"use client";

// Adapted from Magic UI (https://magicui.design, MIT): counts up to a value
// once it scrolls into view. Formatting follows the page language; with
// reduced motion the value is shown at once.

import { useEffect, useRef, type ComponentPropsWithoutRef } from "react";
import { useInView, useMotionValue, useReducedMotion, useSpring } from "motion/react";

import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";

interface NumberTickerProps extends ComponentPropsWithoutRef<"span"> {
  value: number;
  delay?: number;
  decimalPlaces?: number;
}

export function NumberTicker({ value, delay = 0, decimalPlaces = 0, className, ...props }: NumberTickerProps) {
  const { locale } = useI18n();
  const ref = useRef<HTMLSpanElement>(null);
  const reduced = useReducedMotion();
  const motionValue = useMotionValue(reduced ? value : 0);
  const spring = useSpring(motionValue, { damping: 60, stiffness: 120 });
  const inView = useInView(ref, { once: true });

  const format = (number: number) =>
    new Intl.NumberFormat(locale === "pl" ? "pl-PL" : "en-US", {
      minimumFractionDigits: decimalPlaces,
      maximumFractionDigits: decimalPlaces,
    }).format(Number(number.toFixed(decimalPlaces)));

  useEffect(() => {
    if (!inView) return;
    if (reduced) {
      motionValue.jump(value);
      spring.jump(value);
      if (ref.current) ref.current.textContent = format(value);
      return;
    }
    const timer = setTimeout(() => motionValue.set(value), delay * 1000);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, value, delay, reduced]);

  useEffect(
    () =>
      spring.on("change", (latest) => {
        if (ref.current) ref.current.textContent = format(latest);
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [spring, decimalPlaces, locale]
  );

  return (
    <span ref={ref} className={cn("num inline-block tabular-nums", className)} aria-label={format(value)} {...props}>
      {format(reduced ? value : 0)}
    </span>
  );
}
