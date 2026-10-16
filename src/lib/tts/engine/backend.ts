/**
 * Backend selection.
 *
 * `ort` — the vendored onnxruntime-web runtime (default, and what the app ships).
 * `wgsl` — the WebGPU engine under `src/lib/tts/engine` (experimental).
 *
 * Set `NEXT_PUBLIC_ENGINE=wgsl` before `pnpm dev` to switch. The flag is read
 * through `process.env.NEXT_PUBLIC_*`, which Next inlines into every client
 * chunk including the worker, so the worker sees the same value as the page.
 *
 * There is deliberately no automatic fallback: if the WGSL backend throws, the
 * error surfaces rather than being silently served by ORT — a silent fallback
 * would make GPU/CPU differences invisible exactly when we are trying to measure
 * them.
 */

export type EngineBackend = "ort" | "wgsl";

export const ENGINE_BACKEND: EngineBackend =
  process.env.NEXT_PUBLIC_ENGINE === "wgsl" ? "wgsl" : "ort";

export const isWgpuBackend = (): boolean => ENGINE_BACKEND === "wgsl";
