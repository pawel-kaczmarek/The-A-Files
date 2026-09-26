"use client";

// Adapted from Magic UI (https://magicui.design, MIT). The filter and the
// transform are removed once the entrance ends: a lingering `filter` would make
// the element the containing block of fixed descendants such as modals.

import { useRef, type ReactNode } from "react";
import { motion, useInView, useReducedMotion, type Variants } from "motion/react";

export function BlurFade({
  children,
  className,
  delay = 0,
  duration = 0.35,
  offset = 6,
  inView = false,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
  duration?: number;
  offset?: number;
  /** Start when scrolled into view instead of on mount. */
  inView?: boolean;
}) {
  const ref = useRef(null);
  const seen = useInView(ref, { once: true, margin: "-40px" });
  const reduced = useReducedMotion();
  const visible = !inView || seen;

  const variants: Variants = {
    hidden: { y: reduced ? 0 : offset, opacity: 0, filter: reduced ? "none" : "blur(4px)" },
    visible: { y: 0, opacity: 1, filter: "blur(0px)", transitionEnd: { filter: "none", transform: "none" } },
  };

  return (
    <motion.div
      ref={ref}
      initial="hidden"
      animate={visible ? "visible" : "hidden"}
      variants={variants}
      transition={{ delay: 0.04 + delay, duration, ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
