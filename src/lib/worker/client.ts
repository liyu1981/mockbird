"use client";

/**
 * Promise/event wrapper around the TTS worker. Components never talk to
 * `postMessage` directly; they go through this client and the jotai atoms that
 * the engine provider feeds.
 */

import type { DownloadState } from "@/lib/model/downloader";
import type { ModelGroupId } from "@/lib/model/modelManifest";
import type { StoredVoiceMeta } from "@/lib/model/opfsStore";
import type {
  CapabilitiesPayload,
  ClonedVoiceResult,
  EngineState,
  GenerationStats,
  Request,
  Response,
  SynthParams,
} from "./protocol";

type Listener = (message: Response) => void;

export class TtsWorkerClient {
  private worker: Worker;
  private listeners = new Set<Listener>();
  /**
   * Messages the worker posts while booting (engine-state, logs) can arrive
   * before React has subscribed. Buffer them so the first subscriber still sees
   * the current engine state instead of an empty app.
   */
  private backlog: Response[] = [];
  private inflight = new Map<
    string,
    { resolve: (value: Response) => void; reject: (error: Error) => void }
  >();
  private counter = 0;

  constructor() {
    this.worker = new Worker(new URL("../../workers/tts.worker.ts", import.meta.url), {
      type: "module",
      name: "mockbird-tts",
    });
    this.worker.addEventListener("message", (event: MessageEvent<Response>) => {
      const message = event.data;
      const key = correlationKey(message);
      if (key) {
        const pending = this.inflight.get(key);
        if (pending) {
          this.inflight.delete(key);
          pending.resolve(message);
        }
      }
      this.bufferOrEmit(message);
    });
    this.worker.addEventListener("error", (event) => {
      const error = new Error(event.message || "The TTS worker crashed.");
      for (const pending of this.inflight.values()) pending.reject(error);
      this.inflight.clear();
      // Keep boot-time failures visible: a module error in the worker happens
      // before React subscribes, and would otherwise be lost.
      this.bufferOrEmit({ type: "fatal", message: error.message });
    });
    this.worker.addEventListener("messageerror", (event) => {
      this.bufferOrEmit({
        type: "fatal",
        message: `Unserialisable message from the TTS worker: ${String(event)}`,
      });
    });
  }

  private emit(message: Response) {
    for (const listener of this.listeners) listener(message);
  }

  /** Buffer boot-time messages until the first subscriber shows up. */
  private bufferOrEmit(message: Response) {
    if (this.listeners.size === 0) {
      this.backlog.push(message);
      // The backlog only exists to cover boot, so keep it small.
      if (this.backlog.length > 50) this.backlog.shift();
      return;
    }
    this.emit(message);
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    if (this.backlog.length > 0) {
      const pending = this.backlog;
      this.backlog = [];
      for (const message of pending) listener(message);
    }
    return () => this.listeners.delete(listener);
  }

  private send(request: Request, transfer?: Transferable[]): void {
    this.worker.postMessage(request, transfer ?? []);
  }

  private request<R extends Response>(
    request: Request & { correlationId?: string },
    transfer?: Transferable[],
  ): Promise<R> {
    this.counter += 1;
    const correlationId = `req-${Date.now()}-${this.counter}`;
    return new Promise<R>((resolve, reject) => {
      this.inflight.set(correlationId, {
        resolve: resolve as (value: Response) => void,
        reject,
      });
      this.send({ ...request, correlationId } as Request, transfer);
    });
  }

  // --- queries ------------------------------------------------------------

  probe(): Promise<Extract<Response, { type: "probe-result" }>> {
    return this.request({ type: "probe" });
  }

  listVoices(): Promise<Extract<Response, { type: "voices" }>> {
    return this.request({ type: "list-voices" });
  }

  listBuiltinVoices(): Promise<Extract<Response, { type: "builtin-voices" }>> {
    return this.request({ type: "list-builtin-voices" });
  }

  // --- model cache --------------------------------------------------------

  downloadModels(groups: ModelGroupId[]): void {
    this.send({ type: "download-models", groups });
  }

  cancelDownload(): void {
    this.send({ type: "cancel-download" });
  }

  deleteModelFile(path: string): void {
    this.send({ type: "delete-model-file", path });
  }

  clearModelCache(): void {
    this.send({ type: "clear-model-cache" });
  }

  // --- engine -------------------------------------------------------------

  loadEngine(threads: number): void {
    this.send({ type: "load-engine", threads });
  }

  unloadEngine(threads: number): void {
    this.send({ type: "unload-engine", threads });
  }

  // --- voices -------------------------------------------------------------

  createVoice(
    name: string,
    source: StoredVoiceMeta["source"],
    channels: Float32Array[],
    sampleRate: number,
  ): Promise<Extract<Response, { type: "voice-created" }>> {
    return this.request(
      {
        type: "create-voice",
        name,
        source,
        channels,
        sampleRate,
      },
      channels.map((channel) => channel.buffer as ArrayBuffer),
    );
  }

  deleteVoice(id: string): void {
    this.send({ type: "delete-voice", id });
  }

  renameVoice(id: string, name: string): void {
    this.send({ type: "rename-voice", id, name });
  }

  importVoice(payload: string): void {
    this.send({ type: "import-voice", payload });
  }

  // --- synthesis ----------------------------------------------------------

  synthesize(args: {
    requestId: string;
    text: string;
    voiceId: string | null;
    builtinVoice: string | null;
    params: SynthParams;
  }): void {
    this.send({ type: "synthesize", ...args });
  }

  setProfiling(enabled: boolean): void {
    this.send({ type: "set-profiling", enabled });
  }

  cancelGeneration(requestId: string): void {
    this.send({ type: "cancel-generation", requestId });
  }

  countTokens(text: string): void {
    this.send({ type: "count-tokens", text });
  }

  terminate(): void {
    this.worker.terminate();
  }
}

function correlationKey(message: Response): string | null {
  const withId = message as { correlationId?: string };
  return withId.correlationId ?? null;
}

export type {
  CapabilitiesPayload,
  ClonedVoiceResult,
  DownloadState,
  EngineState,
  GenerationStats,
};
