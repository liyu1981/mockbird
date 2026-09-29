"use client";

import { cn } from "cn";
import { useAtomValue } from "jotai";
import { AudioLines, Cpu, Download, Settings2, Sparkles } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { formatBytes } from "@/lib/model/modelManifest";
import { capabilitiesAtom, engineAtom, enginePhaseAtom } from "@/state/modelAtoms";
import { clonedVoicesAtom, selectedVoiceAtom } from "@/state/voiceAtoms";

const NAV = [
  { href: "/studio", label: "Studio", icon: AudioLines, hint: "Write text and speak it" },
  { href: "/voices", label: "Voices", icon: Sparkles, hint: "Clone a voice from a sample" },
  { href: "/model", label: "Model", icon: Download, hint: "Weights, storage, threads" },
  { href: "/settings", label: "Settings", icon: Settings2, hint: "Runtime details" },
] as const;

const PHASE_LABEL: Record<string, string> = {
  idle: "Engine idle",
  downloading: "Downloading weights",
  loading: "Loading engine",
  ready: "Engine ready",
  error: "Engine error",
  unsupported: "Unsupported browser",
};

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const phase = useAtomValue(enginePhaseAtom);
  const engine = useAtomValue(engineAtom);
  const capabilities = useAtomValue(capabilitiesAtom);
  const clonedVoices = useAtomValue(clonedVoicesAtom);
  const selectedVoice = useAtomValue(selectedVoiceAtom);

  const usedBytes = capabilities?.cachedFiles.reduce((sum, file) => sum + file.size, 0) ?? 0;

  return (
    <div
      className="flex min-h-screen w-full bg-background"
      // Handy for e2e/smoke tests and for debugging the engine lifecycle.
      data-engine-phase={phase}
      data-engine-files={engine.files.length}
      data-opfs={capabilities?.opfs ? "yes" : "no"}
      data-cross-origin-isolated={capabilities?.crossOriginIsolated ? "yes" : "no"}
    >
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r bg-sidebar px-3 py-4 md:flex">
        <Link href="/studio" className="mb-6 flex items-center gap-2 px-2">
          <span className="grid size-8 place-items-center rounded-lg bg-primary text-primary-foreground">
            <AudioLines className="size-4" />
          </span>
          <span className="flex flex-col leading-tight">
            <span className="text-sm font-semibold">Mockbird</span>
            <span className="text-[11px] text-muted-foreground">local voice cloning</span>
          </span>
        </Link>

        <nav className="flex flex-col gap-1">
          {NAV.map((item) => {
            const active = pathname === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition-colors",
                  active
                    ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                    : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground",
                )}
              >
                <Icon className="size-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="mt-auto space-y-2 px-1">
          <EngineStatusPill />
          <div className="flex items-center justify-between text-[11px] text-muted-foreground">
            <span className="inline-flex items-center gap-1">
              <Cpu className="size-3" />
              {engine.threads} thread{engine.threads === 1 ? "" : "s"}
            </span>
            <span>{usedBytes > 0 ? formatBytes(usedBytes) : "no weights"}</span>
          </div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b bg-background/85 px-4 backdrop-blur md:px-6">
          <div className="flex items-center gap-2 md:hidden">
            <AudioLines className="size-4 text-primary" />
            <span className="text-sm font-semibold">Mockbird</span>
          </div>
          <nav className="flex items-center gap-1 md:hidden">
            {NAV.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-label={item.label}
                  className={cn(
                    "grid size-8 place-items-center rounded-lg",
                    pathname === item.href
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:bg-accent/60",
                  )}
                >
                  <Icon className="size-4" />
                </Link>
              );
            })}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            {clonedVoices.length > 0 && (
              <Badge variant="secondary">{clonedVoices.length} cloned</Badge>
            )}
            {selectedVoice && (
              <Badge variant="outline" className="max-w-40 truncate">
                {selectedVoice.label}
              </Badge>
            )}
            <EngineStatusPill compact />
          </div>
        </header>

        {phase === "downloading" && <DownloadStrip />}

        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 md:px-6 md:py-8">
          {children}
        </main>

        <footer className="border-t px-4 py-4 text-[11px] text-muted-foreground md:px-6">
          Inference runs entirely in your browser with ONNX Runtime Web. Audio never leaves this
          device. Weights are cached in OPFS and fetched from the Hugging Face CDN.
          MOSS-TTS-Nano by{" "}
          <a
            className="underline underline-offset-2"
            href="https://github.com/OpenMOSS/MOSS-TTS-Nano"
            target="_blank"
            rel="noreferrer"
          >
            OpenMOSS
          </a>
          .
        </footer>
      </div>
    </div>
  );
}

function DownloadStrip() {
  return (
    <div className="h-1 w-full bg-muted">
      <div className="h-1 w-full animate-pulse bg-primary/70" />
    </div>
  );
}

function EngineStatusPill({ compact = false }: { compact?: boolean }) {
  const phase = useAtomValue(enginePhaseAtom);
  const engine = useAtomValue(engineAtom);

  const tone =
    phase === "ready"
      ? "bg-emerald-500"
      : phase === "error" || phase === "unsupported"
        ? "bg-destructive"
        : phase === "idle"
          ? "bg-muted-foreground/50"
          : "bg-amber-500 animate-engine-pulse";

  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full border px-2 py-1 text-[11px]",
              compact && "hidden sm:inline-flex",
            )}
          />
        }
      >
        <span className={cn("size-1.5 rounded-full", tone)} />
        {PHASE_LABEL[phase] ?? phase}
        {engine.threads > 1 && phase === "ready" ? ` · ${engine.threads}t` : null}
      </TooltipTrigger>
      <TooltipContent>
        {engine.error ??
          (phase === "ready"
            ? `ONNX Runtime Web ${engine.ortVersion ?? ""} · ${engine.files.length} files cached`
            : "Weights are downloaded once, then served from your browser cache.")}
      </TooltipContent>
    </Tooltip>
  );
}
