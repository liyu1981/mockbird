"use client";

import { useAtom, useAtomValue } from "jotai";
import {
  AlertTriangle,
  Check,
  Download,
  HardDrive,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Trash2,
} from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { useEngine } from "@/components/engine/EngineProvider";
import { Details, Panel } from "@/components/layout/Panel";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { formatBytes, MODEL_GROUPS } from "@/lib/model/modelManifest";
import {
  capabilitiesAtom,
  downloadActiveAtom,
  downloadSummaryAtom,
  engineAtom,
  engineErrorAtom,
  engineLogAtom,
  enginePhaseAtom,
  modelProgressAtom,
  profileSnapshotAtom,
  profilingEnabledAtom,
  threadsAtom,
} from "@/state/modelAtoms";

/** One-time setup, progress, and the things people actually need. */
export default function SettingsPage() {
  const { client, downloadModels } = useEngine();
  const phase = useAtomValue(enginePhaseAtom);
  const engine = useAtomValue(engineAtom);
  const capabilities = useAtomValue(capabilitiesAtom);
  const progress = useAtomValue(modelProgressAtom);
  const summary = useAtomValue(downloadSummaryAtom);
  const downloading = useAtomValue(downloadActiveAtom);
  const engineError = useAtomValue(engineErrorAtom);
  const [threads, setThreads] = useAtom(threadsAtom);
  const [profiling, setProfiling] = useAtom(profilingEnabledAtom);
  const [advanced, setAdvanced] = useState(false);

  const ready = engine.loaded;
  const totalSize = progress.totalBytes;
  const percent = summary.total > 0 ? (summary.received / summary.total) * 100 : 0;

  return (
    <div className="space-y-4">
      {/* ---------- 1. The one thing a first-time user needs ---------- */}
      <Panel
        title="Speech model"
        description="Mockbird needs the MOSS-TTS-Nano model to make any sound. It downloads once and then lives in this browser."
        actions={
          ready ? (
            <Badge className="gap-1">
              <Check className="size-3" /> Ready
            </Badge>
          ) : phase === "downloading" || phase === "loading" ? (
            <Badge variant="secondary" className="gap-1">
              <Loader2 className="size-3 animate-spin" />
              {phase === "downloading" ? "Downloading" : "Starting"}
            </Badge>
          ) : null
        }
      >
        <div className="space-y-4">
          {ready ? (
            <div className="flex flex-wrap items-center gap-4 rounded-lg border border-border/60 bg-background/40 p-4">
              <p className="min-w-0 flex-1 text-sm text-muted-foreground">
                Everything is downloaded ({formatBytes(totalSize)}) and works offline. Nothing
                needs the network any more.
              </p>
              <Button variant="outline" onClick={() => client?.loadEngine(threads)} size="sm">
                <RefreshCw className="size-3.5" /> Restart engine
              </Button>
            </div>
          ) : (
            <>
              <div className="rounded-lg border border-border/60 bg-background/40 p-4">
                <p className="text-sm">
                  <strong className="font-medium">
                    {downloading
                      ? "Downloading…"
                      : phase === "loading"
                        ? "Starting the engine…"
                        : `About ${formatBytes(763_000_000)} to download`}
                  </strong>
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  {downloading
                    ? `${formatBytes(summary.received)} of ${formatBytes(summary.total)} — keep this tab open, it resumes if the connection drops.`
                    : "A one-time download. After this Mockbird works offline, and your recordings stay on this device."}
                </p>

                {downloading && (
                  <div className="mt-3 space-y-1.5">
                    <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-primary transition-[width] duration-300"
                        style={{ width: `${Math.min(100, percent)}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>{percent.toFixed(0)}%</span>
                      <span>
                        {summary.speed > 1024 ? `${formatBytes(summary.speed)}/s` : ""}{" "}
                        {summary.etaSec ? `· about ${Math.ceil(summary.etaSec)}s left` : ""}
                      </span>
                    </div>
                  </div>
                )}

                {summary.failed.length > 0 && (
                  <p className="mt-3 text-sm text-destructive">{summary.failed[0]}</p>
                )}
              </div>

              <div className="flex flex-wrap gap-2">
                <Button
                  size="lg"
                  onClick={() => downloadModels(MODEL_GROUPS.map((group) => group.id))}
                  disabled={downloading || !capabilities?.opfs}
                >
                  {downloading ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Download className="size-4" />
                  )}
                  {downloading ? "Downloading…" : "Download the model"}
                </Button>
                {downloading && (
                  <Button variant="outline" size="lg" onClick={() => client?.cancelDownload()}>
                    Cancel
                  </Button>
                )}
              </div>

              {!capabilities?.opfs && (
                <p className="flex items-start gap-2 text-sm text-destructive">
                  <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                  This browser cannot store the model (no OPFS support). Chrome or Edge is the
                  safest choice.
                </p>
              )}
            </>
          )}

          {engineError && phase !== "ready" && (
            <p className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
              <AlertTriangle className="mt-0.5 size-4 shrink-0" />
              {engineError}
            </p>
          )}
        </div>
      </Panel>

      {/* ---------- 2. Storage & privacy ---------- */}
      <Panel title="Storage and privacy" description="What is stored, and where.">
        <div className="space-y-3 text-sm">
          <Row icon={<HardDrive className="size-4" />}>
            <span className="font-medium">Model files</span>
            <span className="text-muted-foreground">
              {ready || progress.received > 0
                ? `${formatBytes(progress.received)} used`
                : "not downloaded yet"}
              {capabilities?.quota
                ? ` · ${formatBytes(capabilities.quota)} available in this browser`
                : ""}
            </span>
          </Row>
          <Separator />
          <Row icon={<ShieldCheck className="size-4" />}>
            <span className="font-medium">Your recordings</span>
            <span className="text-muted-foreground">
              Encoded to audio codes on this device and never uploaded anywhere.
            </span>
          </Row>
          <Separator />
          <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
            <p className="text-muted-foreground">
              Want to start over? Clearing the model frees up space but keeps your cloned
              voices.
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                if (confirm("Delete the downloaded model? Your voices are kept.")) {
                  client?.clearModelCache();
                  toast.success("Model files cleared.");
                }
              }}
            >
              <Trash2 className="size-3.5" /> Free up space
            </Button>
          </div>
        </div>
      </Panel>

      {/* ---------- 3. Everything technical, opt-in ---------- */}
      <div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setAdvanced((value) => !value)}
          className="text-muted-foreground"
        >
          {advanced ? "Hide" : "Show"} advanced & troubleshooting
        </Button>
      </div>

      {advanced && (
        <div className="space-y-4">
          <Panel title="Performance" description="Only needed on slow machines.">
            <div className="space-y-5">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="threads">
                    Processing threads
                    <span className="ml-2 font-normal text-muted-foreground">
                      higher is faster but uses more CPU
                    </span>
                  </Label>
                  <span className="font-mono text-sm">{threads}</span>
                </div>
                <input
                  id="threads"
                  type="range"
                  min={1}
                  max={Math.min(8, (capabilities?.hardwareConcurrency ?? 4) - 1 || 1)}
                  value={threads}
                  onChange={(event) => setThreads(Number(event.target.value))}
                  className="w-full accent-[var(--primary)]"
                />
              </div>
              <div className="flex items-center justify-between">
                <Label htmlFor="profiling">
                  Collect timing data
                  <span className="ml-2 font-normal text-muted-foreground">
                    shown below after a generation
                  </span>
                </Label>
                <Switch id="profiling" checked={profiling} onCheckedChange={setProfiling} />
              </div>
            </div>
          </Panel>

          <Details
            summary="Browser support report"
            description="What Mockbird detected about this browser."
          >
            <dl className="grid grid-cols-2 gap-y-1.5 text-xs sm:grid-cols-3">
              {[
                ["Worker report", capabilities ? "received" : "none"],
                [
                  "OPFS",
                  capabilities ? (capabilities.opfs ? "available" : "missing") : "unknown",
                ],
                ["Probe error", capabilities?.error ?? "none"],
                [
                  "Secure context",
                  typeof isSecureContext === "undefined"
                    ? "unknown"
                    : isSecureContext
                      ? "yes"
                      : "no",
                ],
                ["Cross-origin isolated", capabilities?.crossOriginIsolated ? "yes" : "no"],
                [
                  "SharedArrayBuffer",
                  capabilities?.sharedArrayBuffer ? "available" : "missing",
                ],
                ["ONNX Runtime", engine.ortVersion ?? "—"],
                ["Engine", ready ? "ready" : "not loaded"],
                ["Voice cloning", engine.cloningReady ? "ready" : "needs encoder"],
                ["Hardware threads", String(capabilities?.hardwareConcurrency ?? "—")],
                [
                  "Device memory",
                  capabilities?.deviceMemoryGb
                    ? `${capabilities.deviceMemoryGb} GB`
                    : "unknown",
                ],
              ].map(([label, value]) => (
                <div key={label} className="contents">
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd className="truncate font-mono">{value}</dd>
                </div>
              ))}
            </dl>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => void client?.probe().catch(() => undefined)}
              >
                <RefreshCw className="size-3.5" /> Re-check browser
              </Button>
            </div>
          </Details>

          <Details summary="Downloaded files" description="Raw file list from the cache.">
            <ul className="max-h-64 space-y-1 overflow-y-auto font-mono text-[11px]">
              {(capabilities?.cachedFiles ?? []).map((file) => (
                <li key={file.path} className="flex justify-between gap-3">
                  <span className="truncate">{file.path.split("/").pop()}</span>
                  <span className="shrink-0 text-muted-foreground">
                    {formatBytes(file.size)}
                  </span>
                </li>
              ))}
              {(capabilities?.missingFiles ?? []).map((path) => (
                <li key={path} className="flex justify-between gap-3 text-muted-foreground/60">
                  <span className="truncate line-through">{path.split("/").pop()}</span>
                  <span className="shrink-0">missing</span>
                </li>
              ))}
            </ul>
          </Details>

          {capabilities && <TimingReport />}

          <Details summary="Engine log" defaultOpen={Boolean(engineError)}>
            <LogView />
          </Details>
        </div>
      )}
    </div>
  );
}

function Row({ icon, children }: { icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-muted-foreground">
      <span className="text-foreground">{icon}</span>
      {children}
    </div>
  );
}

function TimingReport() {
  const profile = useAtomValue(profileSnapshotAtom);
  if (!profile?.enabled) return null;
  const timings = Object.entries(profile.timings ?? {}).sort(([, a], [, b]) => b - a);
  if (timings.length === 0) return null;
  return (
    <Details
      summary="Last generation timings"
      description="Enable timing data above to fill this in."
    >
      <dl className="grid grid-cols-2 gap-y-1 text-xs sm:grid-cols-3">
        {timings.slice(0, 12).map(([name, ms]) => (
          <div key={name} className="contents">
            <dt className="truncate font-mono text-muted-foreground">{name}</dt>
            <dd className="font-mono">{ms.toFixed(1)} ms</dd>
          </div>
        ))}
      </dl>
    </Details>
  );
}

function LogView() {
  const log = useAtomValue(engineLogAtom);
  if (log.length === 0)
    return <p className="text-sm text-muted-foreground">Nothing logged yet.</p>;
  return (
    <ul className="max-h-64 space-y-0.5 overflow-y-auto font-mono text-[11px]">
      {log
        .slice()
        .reverse()
        .map((entry) => (
          <li key={`${entry.at}-${entry.message.slice(0, 24)}`} className="flex gap-2">
            <span className="shrink-0 text-muted-foreground/70">
              {new Date(entry.at).toLocaleTimeString()}
            </span>
            <span
              className={
                entry.level === "error"
                  ? "text-destructive"
                  : entry.level === "warn"
                    ? "text-warning"
                    : "text-muted-foreground"
              }
            >
              {entry.message}
            </span>
          </li>
        ))}
    </ul>
  );
}
