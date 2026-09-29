"use client";

import { cn } from "cn";
import { Check, Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
/**
 * Light / dark / system switch. Mirrors Shoufa's toggle, minus the language
 * dropdown (Mockbird is English-only for now).
 */
import { useEffect, useRef, useState } from "react";

const THEMES = [
  { value: "light", icon: Sun, label: "Light" },
  { value: "dark", icon: Moon, label: "Dark" },
  { value: "system", icon: Monitor, label: "System" },
] as const;

export function ThemeToggle({ className }: { className?: string }) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  if (!mounted) return <div className={cn("h-9 w-9", className)} />;

  const Current = THEMES.find((item) => item.value === theme)?.icon ?? Monitor;

  return (
    <div ref={ref} className={cn("relative", className)}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-label="Change theme"
        aria-expanded={open}
        className="glass-control flex size-9 items-center justify-center rounded-lg transition-all hover:brightness-[1.06]"
      >
        <Current className="size-4" />
      </button>

      {/* menu-surface (opaque) rather than glass-control: this menu opens over the
          header and the page, so a translucent background let the labels behind it
          bleed through. */}
      {open && (
        <div className="menu-surface absolute right-0 top-full z-50 mt-1 min-w-36 rounded-xl py-1">
          {THEMES.map(({ value, icon: Icon, label }) => (
            <button
              key={value}
              type="button"
              onClick={() => {
                setTheme(value);
                setOpen(false);
              }}
              className={cn(
                "flex w-full items-center gap-2 px-3 py-2 text-left text-sm transition-all hover:bg-foreground/5",
                theme === value ? "text-foreground" : "text-muted-foreground",
              )}
            >
              <Icon className="size-4" />
              <span className="flex-1">{label}</span>
              {theme === value && <Check className="size-3.5" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
