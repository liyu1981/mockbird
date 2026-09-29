"use client";

import { atom } from "jotai";

import type { DownloadState } from "@/lib/model/downloader";
import { formatBytes, MODEL_FILES } from "@/lib/model/modelManifest";
import type { CapabilitiesPayload, EngineState, ProfileSnapshot } from "@/lib/worker/protocol";

export type EnginePhase =
  | "unsupported"
  | "idle"
  | "downloading"
  | "loading"
  | "ready"
  | "error";

export type DownloadSummary = {
  state: DownloadState;
  received: number;
  total: number;
  speed: number;
  etaSec: number | null;
  active: boolean;
  failed: string[];
};

const emptyEngine: EngineState = {
  loaded: false,
  threads: 1,
  files: [],
  cloningReady: false,
  error: null,
  ortVersion: null,
};

export const capabilitiesAtom = atom<CapabilitiesPayload | null>(null);
export const engineAtom = atom<EngineState>(emptyEngine);
export const enginePhaseAtom = atom<EnginePhase>("idle");
export const engineErrorAtom = atom<string | null>(null);
export const downloadStateAtom = atom<DownloadState>({});
export const downloadActiveAtom = atom(false);
export const engineLogAtom = atom<{ level: string; message: string; at: number }[]>([]);
export const profileSnapshotAtom = atom<ProfileSnapshot | null>(null);

/** User preference: ORT worker threads (capped by hardware at load time). */
export const threadsAtom = atom<number>(4);
export const profilingEnabledAtom = atom(false);

export const downloadSummaryAtom = atom<DownloadSummary>((get) => {
  const state = get(downloadStateAtom);
  const active = get(downloadActiveAtom);
  const entries = Object.values(state);
  const received = entries.reduce((sum, item) => sum + item.received, 0);
  const total = entries.reduce((sum, item) => sum + item.total, 0);
  const speed = entries.reduce((sum, item) => sum + item.speed, 0);
  const failed = entries
    .filter((item) => item.phase === "error" && item.error)
    .map((item) => `${item.path.split("/").pop()}: ${item.error}`);
  const remaining = Math.max(0, total - received);
  return {
    state,
    received,
    total,
    speed,
    etaSec: speed > 1024 ? remaining / speed : null,
    active,
    failed,
  };
});

export const modelProgressAtom = atom((get) => {
  const summary = get(downloadSummaryAtom);
  const files = MODEL_FILES.map((file) => {
    const progress = summary.state[file.path];
    return {
      path: file.path,
      name: file.path.split("/").pop() as string,
      group: file.group,
      size: file.size,
      sizeLabel: formatBytes(file.size),
      received: progress?.received ?? 0,
      phase: progress?.phase ?? "idle",
      speed: progress?.speed ?? 0,
      fromCache: progress?.fromCache ?? false,
      error: progress?.error,
    };
  });
  const done = files.filter((file) => file.phase === "done").length;
  return {
    files,
    done,
    total: files.length,
    received: summary.received,
    totalBytes: summary.total,
    percent: summary.total > 0 ? Math.min(100, (summary.received / summary.total) * 100) : 0,
  };
});

/** True when every required group is cached. */
export const synthesisReadyAtom = atom((get) => {
  const engine = get(engineAtom);
  return engine.loaded;
});

export const cloningReadyAtom = atom((get) => get(engineAtom).cloningReady);
