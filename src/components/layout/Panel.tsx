"use client";

import { cn } from "cn";
/**
 * The centred "dialog" surface every tab renders into: a glass card with a
 * title, optional description and a footer. Keeps the three tabs visually
 * identical and gives casual users one obvious place to look.
 */
import type { ReactNode } from "react";
import { GlassCard } from "@/components/GlassCard";

export function Panel({
  title,
  description,
  actions,
  children,
  footer,
  className,
  grow = false,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  className?: string;
  /**
   * Fill the leftover space of a *height-bounded* flex column (see
   * `.bounded-main`). Only safe there: with an unbounded parent, flex-1 would
   * stretch the card instead of merely filling what is left.
   */
  grow?: boolean;
}) {
  return (
    <GlassCard className={cn("p-0", grow && "flex min-h-0 flex-1 flex-col", className)}>
      <div className="flex flex-wrap items-start gap-3 px-6 pt-6">
        <div className="min-w-0 flex-1">
          <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
          {description ? (
            <p className="mt-1 text-sm text-muted-foreground">{description}</p>
          ) : null}
        </div>
        {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
      </div>
      <div className={cn("px-6 py-5", grow && "flex min-h-0 flex-1 flex-col")}>{children}</div>
      {footer ? <div className="border-t px-6 py-3">{footer}</div> : null}
    </GlassCard>
  );
}

/** Disclosure used for everything technical, so it is opt-in. */
export function Details({
  summary,
  description,
  children,
  defaultOpen = false,
}: {
  summary: string;
  description?: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details
      open={defaultOpen}
      className="group rounded-lg border border-border/60 bg-background/40"
    >
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-3 text-sm font-medium transition-colors hover:bg-foreground/5 [&::-webkit-details-marker]:hidden">
        <span className="text-xs text-muted-foreground transition-transform group-open:rotate-90">
          ›
        </span>
        {summary}
      </summary>
      {description ? (
        <p className="px-4 pb-2 text-xs text-muted-foreground">{description}</p>
      ) : null}
      <div className="space-y-3 border-t border-border/60 px-4 py-4">{children}</div>
    </details>
  );
}
