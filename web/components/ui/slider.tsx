"use client";

import * as React from "react";

import { cn } from "@/lib/utils";

// Native range input styled to match the theme.
function Slider({
  className,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      type="range"
      className={cn("h-2 w-full cursor-pointer appearance-none rounded-full bg-muted accent-[hsl(var(--primary))]", className)}
      {...props}
    />
  );
}

export { Slider };
