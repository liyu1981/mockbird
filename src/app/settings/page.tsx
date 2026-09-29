"use client";

import { useAtomValue } from "jotai";
import { Activity, Cpu, HardDrive, Info, Layers, Zap } from "lucide-react";
import { Fragment } from "react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { formatBytes, usingMirror } from "@/lib/model/modelManifest";
import {
  capabilitiesAtom,
  engineAtom,
  engineErrorAtom,
  engineLogAtom,
  profileSnapshotAtom,
} from "@/state/modelAtoms";

export default function SettingsPage() {
  const capabilities = useAtomValue(capabilitiesAtom);
  const engine = useAtomValue(engineAtom);
  const engineError = useAtomValue(engineErrorAtom);
  const log = useAtomValue(engineLogAtom);
  const profile = useAtomValue(profileSnapshotAtom);

  const rows: [string, string][] = [
    ["Engine phase", engine.loaded ? "ready" : "not loaded"],
    ["Worker report", capabilities ? "received" : "none (worker did not answer)"],
    [
      "OPFS (navigator.storage.getDirectory)",
      capabilities ? (capabilities.opfs ? "available" : "MISSING") : "unknown",
    ],
    ["Probe error", capabilities?.error ?? (capabilities ? "none" : "worker never reported")],
    [
      "Secure context",
      typeof isSecureContext === "undefined" ? "unknown" : isSecureContext ? "yes" : "no",
    ],
    ["ONNX Runtime Web", engine.ortVersion ?? "—"],
    ["Threads", String(engine.threads)],
    ["Cached files", String(engine.files.length)],
    [
      "Weights size",
      formatBytes(capabilities?.cachedFiles.reduce((sum, f) => sum + f.size, 0) ?? 0),
    ],
    ["Storage quota", capabilities?.quota ? formatBytes(capabilities.quota) : "unknown"],
    ["Storage used", capabilities?.usage != null ? formatBytes(capabilities.usage) : "unknown"],
    ["Persistent storage", capabilities?.persist ? "granted" : "best effort"],
    ["Cross-origin isolated", capabilities?.crossOriginIsolated ? "yes" : "no"],
    ["SharedArrayBuffer", capabilities?.sharedArrayBuffer ? "available" : "missing"],
    ["Hardware threads", String(capabilities?.hardwareConcurrency ?? "unknown")],
    [
      "Device memory",
      capabilities?.deviceMemoryGb ? `${capabilities.deviceMemoryGb} GB` : "unknown",
    ],
    ["Weight source", usingMirror() ? "custom mirror" : "huggingface.co"],
  ];

  return (
    <div className="space-y-4">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="text-sm text-muted-foreground">
          Runtime diagnostics. Everything here is read from the engine worker.
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Cpu className="size-4 text-muted-foreground" /> Runtime
            </CardTitle>
            <CardDescription>
              ONNX Runtime Web with the WASM (SIMD + threads) execution provider.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="divide-y text-xs">
              {rows.map(([label, value]) => (
                <div key={label} className="flex items-center justify-between gap-4 py-1.5">
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd className="truncate font-mono">{value}</dd>
                </div>
              ))}
            </dl>
            {engineError && (
              <>
                <Separator className="my-3" />
                <p className="rounded-lg border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive">
                  {engineError}
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <HardDrive className="size-4 text-muted-foreground" /> Cached weights
            </CardTitle>
            <CardDescription>Stored in the Origin Private File System.</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-64">
              <ul className="space-y-1 pr-3 font-mono text-[11px]">
                {(capabilities?.cachedFiles ?? []).map((file) => (
                  <li key={file.path} className="flex justify-between gap-3">
                    <span className="truncate">{file.path.split("/").pop()}</span>
                    <span className="shrink-0 text-muted-foreground">
                      {formatBytes(file.size)}
                    </span>
                  </li>
                ))}
                {(capabilities?.missingFiles ?? []).map((path) => (
                  <li
                    key={path}
                    className="flex justify-between gap-3 text-muted-foreground/60"
                  >
                    <span className="truncate line-through">{path.split("/").pop()}</span>
                    <span className="shrink-0">missing</span>
                  </li>
                ))}
              </ul>
              <ScrollBar orientation="vertical" />
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Activity className="size-4 text-muted-foreground" /> Engine log
          </CardTitle>
          <CardDescription>
            Session creation, tensor shapes and timing hints from the vendored runtime.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {log.length === 0 ? (
            <p className="text-sm text-muted-foreground">No log output yet.</p>
          ) : (
            <ScrollArea className="h-64">
              <ul className="space-y-0.5 pr-3 font-mono text-[11px]">
                {log
                  .slice()
                  .reverse()
                  .map((entry) => (
                    <li
                      key={`${entry.at}-${entry.message.slice(0, 24)}`}
                      className="flex gap-2"
                    >
                      <span className="shrink-0 text-muted-foreground/70">
                        {new Date(entry.at).toLocaleTimeString()}
                      </span>
                      <span
                        className={
                          entry.level === "error"
                            ? "text-destructive"
                            : entry.level === "warn"
                              ? "text-amber-600"
                              : "text-muted-foreground"
                        }
                      >
                        {entry.message}
                      </span>
                    </li>
                  ))}
              </ul>
              <ScrollBar orientation="vertical" />
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      {profile?.enabled && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Zap className="size-4 text-muted-foreground" /> Generation profile
            </CardTitle>
            <CardDescription>
              Timings from the last synthesis run (enable the toggle on the Model tab to collect
              them).
            </CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-y-1.5 text-xs">
              {Object.entries(profile.timings ?? {})
                .sort(([, a], [, b]) => b - a)
                .slice(0, 14)
                .map(([name, ms]) => (
                  <Fragment key={name}>
                    <dt className="truncate font-mono text-muted-foreground">{name}</dt>
                    <dd className="text-right font-mono">{ms.toFixed(1)} ms</dd>
                  </Fragment>
                ))}
              {Object.entries(profile.counters ?? {}).map(([name, count]) => (
                <Fragment key={name}>
                  <dt className="truncate font-mono text-muted-foreground">{name}</dt>
                  <dd className="text-right font-mono">{count}</dd>
                </Fragment>
              ))}
            </dl>
            {Object.keys(profile.timings ?? {}).length === 0 &&
              Object.keys(profile.counters ?? {}).length === 0 && (
                <p className="text-sm text-muted-foreground">
                  No profile data yet — generate something first.
                </p>
              )}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Info className="size-4 text-muted-foreground" /> About
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <p>
            Mockbird runs{" "}
            <a
              className="underline underline-offset-2"
              href="https://github.com/OpenMOSS/MOSS-TTS-Nano"
              target="_blank"
              rel="noreferrer"
            >
              MOSS-TTS-Nano
            </a>{" "}
            (0.1B parameters, 48 kHz stereo, 20 languages) entirely in your browser using the
            official ONNX export and ONNX Runtime Web.
          </p>
          <div className="flex flex-wrap gap-1.5">
            <Badge variant="secondary">
              <Layers className="size-3" /> OPFS weights cache
            </Badge>
            <Badge variant="secondary">Cross-origin isolated</Badge>
            <Badge variant="secondary">AudioWorklet streaming</Badge>
            <Badge variant="secondary">No server inference</Badge>
          </div>
          <p className="text-xs">
            Please only clone voices you have permission to use. The reference recordings never
            leave this device.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
