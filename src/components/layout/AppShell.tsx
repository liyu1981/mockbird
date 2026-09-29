"use client";

/**
 * The whole app chrome: an ambient background, a slim top bar, and one large
 * centred glass panel that behaves like a tabbed dialog.
 *
 * Design language copied from the Shoufa project (~/shoufa). There is no
 * sidebar — the three jobs (generate / clone / settings) are tabs, so a casual
 * user sees one card and three obvious choices.
 */

import { cn } from "cn";
import { useAtomValue } from "jotai";
import { AudioLines, Mic2, Settings2, Sparkles } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { lazy, Suspense } from "react";
import { ThemeToggle } from "@/components/ThemeToggle";
import { engineAtom, enginePhaseAtom } from "@/state/modelAtoms";
import { clonedVoicesAtom } from "@/state/voiceAtoms";

const TABS = [
  { href: "/", label: "Generate", icon: Sparkles, hint: "Write something and hear it" },
  { href: "/clone", label: "Clone", icon: Mic2, hint: "Record a voice of your own" },
  {
    href: "/settings",
    label: "Settings",
    icon: Settings2,
    hint: "Model, storage, troubleshooting",
  },
] as const;

const AmbientBackground = lazy(() => import("@/components/AmbientBackground"));

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const phase = useAtomValue(enginePhaseAtom);
  const engine = useAtomValue(engineAtom);
  const cloned = useAtomValue(clonedVoicesAtom);

  return (
    <div className="ambient-bg min-h-screen w-full overflow-x-clip">
      {/* Decorative blob field, behind the glass surfaces. */}
      <Suspense fallback={null}>
        <AmbientBackground />
      </Suspense>

      <div className="relative z-10 mx-auto flex min-h-dvh w-full max-w-4xl flex-col px-4 pb-10">
        {/* z-30: the header hosts popovers (theme menu). `glass-control` sets a
            backdrop-filter, which creates a stacking context, so the header needs
            to be positioned *and* layered or the menu renders under the panels. */}
        <header className="glass-control relative z-30 mt-4 flex h-16 items-center gap-3 rounded-xl px-4">
          <Link href="/" className="flex items-center gap-2.5">
            <span className="grid size-8 place-items-center rounded-lg bg-primary text-primary-foreground">
              <AudioLines className="size-4" />
            </span>
            <span className="flex flex-col leading-tight">
              <span className="text-sm font-semibold">Mockbird</span>
              <span className="hidden text-[11px] text-muted-foreground sm:block">
                Voice studio — runs on your device
              </span>
            </span>
          </Link>

          <div className="ml-auto flex items-center gap-2">
            <EngineBadge phase={phase} loaded={engine.loaded} />
            {cloned.length > 0 && (
              <span className="hidden text-[11px] text-muted-foreground sm:inline">
                {cloned.length} voice{cloned.length === 1 ? "" : "s"}
              </span>
            )}
            <ThemeToggle />
          </div>
        </header>

        {/* bounded-main caps this column at the viewable height minus the
            header and footer, so a flex-1 child inside it fills exactly the
            remaining space and can never grow past the fold. */}
        <main className="bounded-main flex min-h-0 flex-1 flex-col items-center pt-6">
          <nav
            aria-label="Sections"
            className="glass-control mb-4 inline-flex w-full max-w-md items-center gap-1 rounded-xl p-1"
          >
            {TABS.map((tab) => {
              const active = pathname === tab.href;
              const Icon = tab.icon;
              return (
                <Link
                  key={tab.href}
                  href={tab.href}
                  aria-current={active ? "page" : undefined}
                  title={tab.hint}
                  className={cn(
                    "flex flex-1 items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-all",
                    active
                      ? "bg-foreground/10 text-foreground shadow-sm"
                      : "text-muted-foreground hover:bg-foreground/5 hover:text-foreground",
                  )}
                >
                  <Icon className="size-4" />
                  {tab.label}
                </Link>
              );
            })}
          </nav>

          <div className="flex min-h-0 w-full flex-1 flex-col">{children}</div>
        </main>

        <footer className="mt-8 text-center text-[11px] leading-relaxed text-muted-foreground">
          Everything runs on your device — your recordings never leave this browser. Powered by{" "}
          <a
            className="underline underline-offset-2"
            href="https://github.com/OpenMOSS/MOSS-TTS-Nano"
            target="_blank"
            rel="noreferrer"
          >
            MOSS-TTS-Nano
          </a>
          .
        </footer>
      </div>
    </div>
  );
}

function EngineBadge({ phase, loaded }: { phase: string; loaded: boolean }) {
  const label = loaded
    ? "Ready"
    : phase === "downloading"
      ? "Downloading"
      : phase === "loading"
        ? "Starting"
        : phase === "error" || phase === "unsupported"
          ? "Needs attention"
          : "Not set up";

  const tone = loaded
    ? "bg-success"
    : phase === "downloading" || phase === "loading"
      ? "bg-warning animate-engine-pulse"
      : phase === "error" || phase === "unsupported"
        ? "bg-destructive"
        : "bg-muted-foreground/40";

  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px]"
      data-engine-phase={phase}
      data-engine-loaded={loaded ? "ready" : "0"}
      data-testid="engine-badge"
    >
      <span className={cn("size-1.5 rounded-full", tone)} />
      <span className="hidden sm:inline">{label}</span>
    </span>
  );
}
