"use client";

// Adapted from Magic UI (https://magicui.design, MIT): cycles through words in
// place. With reduced motion the words change without movement.

import { useEffect, useState, type ReactNode } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";

import { cn } from "@/lib/utils";

export function WordRotate({
  words,
  duration = 2600,
  className,
}: {
  words: { key: string; content: ReactNode }[];
  duration?: number;
  className?: string;
}) {
  const [index, setIndex] = useState(0);
  const reduced = useReducedMotion();
  useEffect(() => {
    const interval = setInterval(() => setIndex((value) => (value + 1) % words.length), duration);
    return () => clearInterval(interval);
  }, [words.length, duration]);
  const word = words[index];
  return (
    <span className={cn("relative -mb-3 inline-flex overflow-hidden pb-3 align-bottom", className)}>
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={word.key}
          initial={reduced ? { opacity: 0 } : { opacity: 0, y: "-60%" }}
          animate={{ opacity: 1, y: 0 }}
          exit={reduced ? { opacity: 0 } : { opacity: 0, y: "60%" }}
          transition={{ duration: 0.25, ease: "easeOut" }}
          className="inline-block"
        >
          {word.content}
        </motion.span>
      </AnimatePresence>
    </span>
  );
}
