"use client";

import type { ReactNode } from "react";
import { MotionConfig } from "motion/react";

/** Every motion animation honours the system's "reduce motion" setting. */
export function MotionProvider({ children }: { children: ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}
