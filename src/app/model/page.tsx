"use client";

import { useAtom, useAtomValue } from "jotai";
import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Download,
  HardDrive,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Trash2,
  Zap,
} from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { useEngine } from "@/components/engine/EngineProvider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import {
  formatBytes,
  MODEL_GROUPS,
  type ModelGroupId,
  totalSizeForGroups,
  usingMirror,
} from "@/lib/model/modelManifest";
import {
  capabilitiesAtom,
  downloadActiveAtom,
  engineAtom,
  engineErrorAtom,
  enginePhaseAtom,
  modelProgressAtom,
  profilingEnabledAtom,
  threadsAtom,
} from "@/state/modelAtoms";

export default function ModelPage() {
  const { downloadModels, client, loadEngine } = useEngine();
  const capabilities = useAtomValue(capabilitiesAtom);
  const engine = useAtomValue(engineAtom);
  const phase = useAtomValue(enginePhaseAtom);
  const engineError = useAtomValue(engineErrorAtom);
  const progress = useAtomValue(modelProgressAtom);
  const downloading = useAtomValue(downloadActiveAtom);
  const [threads, setThreads] = useAtom(threadsAtom);
  const [profiling, setProfiling] = useAtom(profilingEnabledAtom);
  // Default to everything: cloning is the headline feature, and the encoder is
  // only ~45 MB on top of the 720 MB the decoder needs anyway.
  const [groups, setGroups] = useState<ModelGroupId[]>(() =>
    MODEL_GROUPS.map((group) => group.id),
  );

  const selectedSize = totalSizeForGroups(groups);
  const everythingSize = totalSizeForGroups(MODEL_GROUPS.map((group) => group.id));
  const cached = capabilities?.cachedFiles ?? [];
  const cachedBytes = cached.reduce((sum, file) => sum + file.size, 0);

  const maxThreads = Math.min(
    8,
    capabilities?.hardwareConcurrency ? Math.max(1, capabilities.hardwareConcurrency - 1) : 4,
  );

  const toggleGroup = (id: ModelGroupId) => {
    setGroups((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id],
    );
  };

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Model lab</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          MOSS-TTS-Nano is a 0.1B-parameter multilingual TTS model. Its ONNX weights total{" "}
          <strong className="text-foreground">763 MB</strong>; they are downloaded once from the
          Hugging Face CDN and then cached in your browser&apos;s private file system. Nothing
          is uploaded, and synthesis works offline afterwards.
        </p>
      </header>

      {!capabilities?.opfs && (
        <Card className="border-destructive/40 bg-destructive/5">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-destructive">
              <AlertTriangle className="size-4" /> Browser not supported
            </CardTitle>
            <CardDescription>
              Mockbird needs the Origin Private File System to cache ~763 MB of weights. Chrome
              or Edge 108+ is the reference target; Safari and Firefox work without persistent
              caching.
            </CardDescription>
          </CardHeader>
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">1 · Choose components</CardTitle>
            <CardDescription>
              The decoder is always required. The cloning encoder is only needed if you want to
              clone your own voice — built-in voices ship with pre-computed audio codes.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {MODEL_GROUPS.map((group) => {
              const selected = groups.includes(group.id);
              return (
                <button
                  key={group.id}
                  type="button"
                  onClick={() => toggleGroup(group.id)}
                  disabled={group.required}
                  className={`w-full rounded-lg border p-3 text-left transition-colors ${
                    selected
                      ? "border-primary/60 bg-primary/5"
                      : "border-border hover:bg-accent/50"
                  } ${group.required ? "cursor-default" : ""}`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-sm font-medium">{group.label}</span>
                    {group.required ? (
                      <Badge variant="secondary">required</Badge>
                    ) : selected ? (
                      <Badge>selected</Badge>
                    ) : (
                      <Badge variant="outline">optional</Badge>
                    )}
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">{group.description}</p>
                  <p className="mt-1 text-[11px] text-muted-foreground/80">
                    {formatBytes(totalSizeForGroups([group.id]))}
                  </p>
                </button>
              );
            })}
          </CardContent>
          <CardFooter className="gap-2">
            <Button
              onClick={() => downloadModels(groups)}
              disabled={downloading || !capabilities?.opfs}
            >
              {downloading ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Download className="size-4" />
              )}
              {downloading ? "Downloading…" : cachedBytes > 0 ? "Update / resume" : "Download"}
            </Button>
            {downloading && (
              <Button variant="ghost" onClick={() => client?.cancelDownload()}>
                Cancel
              </Button>
            )}
            <span className="ml-auto text-xs text-muted-foreground">
              {formatBytes(selectedSize)} selected
            </span>
          </CardFooter>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-base">2 · Progress</CardTitle>
            <CardDescription>
              {downloading
                ? "Keep this tab open. Downloads resume automatically if the connection drops."
                : `Expected download: ${formatBytes(everythingSize)} — once, then cached in this browser forever.`}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {downloading && (
              <div className="space-y-1.5">
                <Progress value={progress.percent} />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>
                    {formatBytes(progress.received)} / {formatBytes(progress.totalBytes)}
                  </span>
                  <span>{progress.percent.toFixed(1)}%</span>
                </div>
              </div>
            )}

            <ul className="divide-y rounded-lg border">
              {progress.files.map((file) => (
                <li key={file.path} className="flex items-center gap-3 px-3 py-2 text-xs">
                  <span className="w-4 shrink-0">
                    {file.phase === "done" ? (
                      <CheckCircle2 className="size-3.5 text-emerald-500" />
                    ) : file.phase === "downloading" ? (
                      <Loader2 className="size-3.5 animate-spin text-primary" />
                    ) : file.phase === "error" ? (
                      <AlertTriangle className="size-3.5 text-destructive" />
                    ) : (
                      <span className="block size-1.5 rounded-full bg-muted-foreground/40" />
                    )}
                  </span>
                  <span className="min-w-0 flex-1 truncate font-mono text-[11px]">
                    {file.name}
                  </span>
                  {file.group === "codec-encode" && (
                    <Badge variant="outline" className="text-[10px]">
                      cloning
                    </Badge>
                  )}
                  <span className="w-20 text-right text-muted-foreground">
                    {file.phase === "done"
                      ? file.fromCache
                        ? "cached"
                        : "done"
                      : file.phase === "downloading"
                        ? `${formatBytes(file.received)}`
                        : file.sizeLabel}
                  </span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">3 · Runtime</CardTitle>
            <CardDescription>
              WASM threads require a cross-origin isolated page (Mockbird sends COOP/COEP
              headers for this).
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between gap-4">
              <Label htmlFor="threads" className="flex items-center gap-2">
                <Cpu className="size-4 text-muted-foreground" />
                ORT threads
              </Label>
              <div className="flex items-center gap-2">
                <span className="w-6 text-right font-mono text-sm">{threads}</span>
                <input
                  id="threads"
                  type="range"
                  min={1}
                  max={maxThreads}
                  value={threads}
                  onChange={(event) => setThreads(Number(event.target.value))}
                  className="w-32 accent-[var(--primary)]"
                />
              </div>
            </div>
            <div className="flex items-center justify-between gap-4">
              <Label htmlFor="profiling" className="flex items-center gap-2">
                <Zap className="size-4 text-muted-foreground" />
                Runtime profiling log
              </Label>
              <Switch id="profiling" checked={profiling} onCheckedChange={setProfiling} />
            </div>
            <Separator />
            <dl className="grid grid-cols-2 gap-y-1.5 text-xs">
              <dt className="text-muted-foreground">Phase</dt>
              <dd className="text-right font-mono">{phase}</dd>
              <dt className="text-muted-foreground">Cached files</dt>
              <dd className="text-right font-mono">{cached.length}</dd>
              <dt className="text-muted-foreground">ONNX Runtime</dt>
              <dd className="text-right font-mono">{engine.ortVersion ?? "—"}</dd>
              <dt className="text-muted-foreground">OPFS</dt>
              <dd className="text-right font-mono">
                {capabilities ? (capabilities.opfs ? "available" : "missing") : "checking…"}
              </dd>
              <dt className="text-muted-foreground">Cross-origin isolated</dt>
              <dd className="text-right font-mono">
                {capabilities?.crossOriginIsolated ? "yes" : "no"}
              </dd>
              <dt className="text-muted-foreground">Voice cloning</dt>
              <dd className="text-right font-mono">
                {engine.cloningReady ? "ready" : "encoder missing"}
              </dd>
            </dl>
            {engineError && (
              <p className="rounded-lg border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive">
                {engineError}
              </p>
            )}
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" size="sm" onClick={loadEngine}>
                <RefreshCw className="size-4" /> Reload engine
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => void client?.probe().catch(() => undefined)}
              >
                <RefreshCw className="size-4" /> Re-check browser
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Storage & privacy</CardTitle>
            <CardDescription>
              Weights and voice samples live in this browser profile only.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-start gap-2 text-xs text-muted-foreground">
              <HardDrive className="mt-0.5 size-4 shrink-0" />
              <span>
                {capabilities?.usage != null
                  ? `${formatBytes(capabilities.usage)} used`
                  : "usage unknown"}
                {capabilities?.quota != null
                  ? ` of about ${formatBytes(capabilities.quota)} available`
                  : ""}
                {capabilities?.persist
                  ? " · persistent storage granted"
                  : " · storage may be evicted under pressure"}
              </span>
            </div>
            <div className="flex items-start gap-2 text-xs text-muted-foreground">
              <ShieldCheck className="mt-0.5 size-4 shrink-0" />
              <span>
                Reference recordings are encoded to audio codes on-device and never uploaded.{" "}
                {usingMirror()
                  ? "Weights come from your own mirror."
                  : "Weights come from huggingface.co."}
              </span>
            </div>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => {
                if (confirm("Delete all cached model weights? Voices are kept.")) {
                  client?.clearModelCache();
                  toast.success("Model cache cleared.");
                }
              }}
            >
              <Trash2 className="size-4" /> Clear weight cache
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
