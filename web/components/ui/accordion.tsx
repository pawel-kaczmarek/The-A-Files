"use client";

import * as React from "react";
import { ChevronDown } from "lucide-react";

import { cn } from "@/lib/utils";

// Minimal accordion built on <details>, styled to match shadcn.
function Accordion({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("divide-y rounded-lg border", className)} {...props} />;
}

function AccordionItem({
  title,
  defaultOpen = false,
  children,
  className,
}: {
  title: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <details open={defaultOpen} className={cn("group px-4", className)}>
      <summary className="flex cursor-pointer select-none items-center justify-between py-3 text-sm font-medium [&::-webkit-details-marker]:hidden">
        {title}
        <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-180" />
      </summary>
      <div className="pb-4">{children}</div>
    </details>
  );
}

export { Accordion, AccordionItem };
