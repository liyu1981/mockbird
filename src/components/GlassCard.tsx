"use client";

import { cn } from "cn";
/**
 * Frosted-glass surface — copied from the Shoufa project (~/shoufa).
 * Used for the main centred "dialog" and its cards.
 */
import type { ReactNode } from "react";

export function GlassCard({
  children,
  className,
  hover = false,
}: {
  children: ReactNode;
  className?: string;
  hover?: boolean;
}) {
  return (
    <div
      className={cn(
        "glass-control rounded-xl",
        hover &&
          "transition-all duration-200 hover:brightness-[1.04] hover:-translate-y-px active:scale-[0.98]",
        className,
      )}
    >
      {children}
    </div>
  );
}
