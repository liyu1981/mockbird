/**
 * SentencePiece client for the tokenizer worker.
 *
 * The tokenizer itself is still the vendored `public/tokenizer.worker.js`
 * (SentencePiece compiled to wasm-in-JS) — it is the one component with no
 * Apache-2.0 JS equivalent, so it stays a documented licence exception
 * (PLAN.md §10.1) even after ONNX Runtime is gone. Everything else in the
 * engine is ported from Apache-2.0 sources.
 *
 * Protocol (stable, defined by that worker):
 *   -> { source, type: "request", requestId, action, modelKey, text|tokenizerBase64 }
 *   <- { source, type: "response", requestId, ok, data | error }
 */

const SOURCE = "nano-reader-tokenizer-sandbox";

type Pending = { resolve: (value: unknown) => void; reject: (error: Error) => void };

let worker: Worker | null = null;
const pending = new Map<string, Pending>();
let counter = 0;
let readyPromise: Promise<Worker> | null = null;

function ensureWorker(): Promise<Worker> {
  if (worker) return Promise.resolve(worker);
  if (readyPromise) return readyPromise;
  readyPromise = new Promise<Worker>((resolve, reject) => {
    let spawned: Worker;
    try {
      spawned = new Worker("/tokenizer.worker.js", {
        type: "module",
        name: "mockbird-tokenizer",
      });
    } catch (error) {
      reject(
        new Error(`Failed to start tokenizer worker: ${(error as Error)?.message ?? error}`),
      );
      return;
    }
    spawned.addEventListener("message", (event: MessageEvent) => {
      const payload = event.data as {
        source?: string;
        type?: string;
        requestId?: string;
        ok?: boolean;
        data?: unknown;
        error?: string;
      };
      if (!payload || payload.source !== SOURCE || payload.type !== "response") return;
      const entry = payload.requestId ? pending.get(payload.requestId) : undefined;
      if (!entry) return;
      pending.delete(payload.requestId as string);
      if (payload.ok) entry.resolve(payload.data);
      else entry.reject(new Error(payload.error || "Tokenizer request failed"));
    });
    spawned.addEventListener(
      "error",
      (event) => reject(new Error(event.message || "Tokenizer worker failed to load.")),
      { once: true },
    );
    worker = spawned;
    resolve(spawned);
  }).finally(() => {
    readyPromise = null;
  });
  return readyPromise;
}

async function request(action: string, body: Record<string, unknown>): Promise<unknown> {
  const tokenizerWorker = await ensureWorker();
  counter += 1;
  const requestId = `tokenizer-${Date.now()}-${counter}`;
  return new Promise((resolve, reject) => {
    pending.set(requestId, { resolve, reject });
    tokenizerWorker.postMessage({
      source: SOURCE,
      type: "request",
      requestId,
      action,
      ...body,
    });
  });
}

export type TokenizerClient = {
  load(bytes: Uint8Array): Promise<string>;
  encode(modelKey: string, text: string): Promise<number[]>;
  loadedKey(): string | null;
};

export function createTokenizer(): TokenizerClient {
  let modelKey: string | null = null;
  return {
    async load(bytes: Uint8Array) {
      // 0.47 MB per load, once per session — the model is cached inside the worker
      let binary = "";
      const chunk = 32768;
      for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
      }
      modelKey = `moss::${bytes.length}`;
      await request("loadTokenizer", { modelKey, tokenizerBase64: btoa(binary) });
      return modelKey;
    },
    async encode(key: string, text: string) {
      const ids = await request("encodeText", { modelKey: key, text: String(text || "") });
      return Array.isArray(ids) ? (ids as number[]) : [];
    },
    loadedKey: () => modelKey,
  };
}
