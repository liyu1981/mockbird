/// <reference lib="webworker" />

/**
 * MOSS-TTS-Nano engine worker.
 *
 * Owns the vendored browser runtime, the OPFS model cache and the tokenizer
 * worker. The main thread never touches ORT, so a 670 MB weight decode or a
 * multi-second decode loop can never block rendering or audio playback.
 */

import { env as ortEnv } from "onnxruntime-web/wasm";
import { type DownloadEvent, downloadModelFiles } from "@/lib/model/downloader";
import {
  filesForGroup,
  isCloningReady,
  isSynthesisReady,
  MODEL_FILES,
  type ModelFile,
  type ModelGroupId,
  REQUIRED_GROUPS,
} from "@/lib/model/modelManifest";
import {
  clearModelCache,
  deleteModelFile,
  deleteVoice,
  hasModelFile,
  isOpfsAvailable,
  listCachedModelFiles,
  listVoices,
  modelFileSizeOnDisk,
  type OpfsCapabilities,
  probeCapabilities,
  readModelFileBuffer,
  readModelFileText,
  readVoiceCodes,
  readVoiceMeta,
  recordModelFile,
  type StoredVoiceMeta,
  writeVoice,
} from "@/lib/model/opfsStore";
import { createBrowserOnnxTtsRuntime } from "@/lib/tts/vendor/mossTtsRuntime";
import type {
  BuiltinVoice,
  ClonedVoiceResult,
  EngineState,
  GenerationStats,
  Request,
  Response,
  SynthParams,
} from "@/lib/worker/protocol";
import { DEFAULT_SYNTH_PARAMS } from "@/lib/worker/protocol";

declare const self: DedicatedWorkerGlobalScope;

const MODEL_ROOT = "models";

const runtime = createBrowserOnnxTtsRuntime({
  logger: (message: string) => {
    post({ type: "engine-log", level: "info", message: String(message) });
  },
  /**
   * OPFS-backed asset reader. The vendored runtime resolves every graph, weight
   * sidecar and metadata JSON through this interface, so nothing has to be
   * reachable over HTTP.
   */
  assetReader: {
    async exists({ relativePath }: { rootPath: string; relativePath: string }) {
      return hasModelFile(normalizeAssetPath(relativePath));
    },
    async readText({ relativePath }: { rootPath: string; relativePath: string }) {
      return readModelFileText(normalizeAssetPath(relativePath));
    },
    async readJson({ relativePath }: { rootPath: string; relativePath: string }) {
      return JSON.parse(await readModelFileText(normalizeAssetPath(relativePath)));
    },
    async readBuffer({ relativePath }: { rootPath: string; relativePath: string }) {
      return readModelFileBuffer(normalizeAssetPath(relativePath));
    },
  },
});

function normalizeAssetPath(relativePath: string): string {
  const segments: string[] = [];
  for (const raw of String(relativePath || "")
    .replace(/\\/g, "/")
    .split("/")) {
    if (!raw || raw === ".") continue;
    if (raw === "..") {
      segments.pop();
      continue;
    }
    segments.push(raw);
  }
  return segments.join("/");
}

let currentCorrelationId: string | null = null;

function post(message: Response, transfer?: Transferable[]) {
  if (currentCorrelationId) {
    self.postMessage({ ...message, correlationId: currentCorrelationId }, transfer ?? []);
    return;
  }
  self.postMessage(message, transfer ?? []);
}

const state: {
  engine: EngineState;
  downloadAbort: AbortController | null;
  cancelled: Set<string>;
  activeRequest: string | null;
  builtinVoices: BuiltinVoice[] | null;
  tokenCounter: ((text: string) => Promise<number>) | null;
  /**
   * Last prompt audio codes, keyed by `cloned:<id>` / `builtin:<name>`. A long
   * text is spoken as one `synthesize` request per chunk, so without this the
   * reference codes would be re-read from OPFS and re-materialised every chunk.
   */
  promptAudioCodesCache: { key: string; codes: number[][] } | null;
} = {
  engine: {
    loaded: false,
    threads: 1,
    files: [],
    cloningReady: false,
    error: null,
    ortVersion: null,
  },
  downloadAbort: null,
  cancelled: new Set(),
  activeRequest: null,
  builtinVoices: null,
  tokenCounter: null,
  promptAudioCodesCache: null,
};

function publishEngineState() {
  post({ type: "engine-state", state: { ...state.engine } });
}

async function refreshEngineState() {
  const files = await listCachedModelFiles();
  const present = new Set(files);
  state.engine = {
    ...state.engine,
    files,
    cloningReady: isCloningReady(present),
    loaded: isSynthesisReady(present),
  };
  publishEngineState();
}

async function downloadGroups(groups: ModelGroupId[]) {
  const files = groups.flatMap((group) => filesForGroup(group));
  if (files.length === 0) return;

  const controller = new AbortController();
  state.downloadAbort = controller;

  const onEvent = (event: DownloadEvent) => {
    if (event.type === "progress") {
      post({ type: "download-progress", state: event.state });
    } else if (event.type === "done") {
      post({ type: "download-done", state: event.state });
    } else if (event.type === "error") {
      post({ type: "download-error", message: event.message });
    }
  };

  try {
    await downloadModelFiles({ files, signal: controller.signal, onEvent });
    // Keep the index honest if a mirror served different bytes than recorded.
    for (const file of files) {
      const size = await modelFileSizeOnDisk(file.path);
      if (size > 0) {
        await recordModelFile(file.path, { size, etag: null, updatedAt: Date.now() });
      }
    }
  } catch (error) {
    if ((error as DOMException)?.name !== "AbortError") {
      post({
        type: "download-error",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  } finally {
    state.downloadAbort = null;
    await refreshEngineState();
  }
}

async function loadEngine(threads: number) {
  const capped = Math.max(1, Math.min(threads, 8));
  state.engine.error = null;
  publishEngineState();
  try {
    await runtime.configure({ modelPath: MODEL_ROOT, threadCount: capped });
    await runtime.ensureSynthesisLoaded();
    await runtime.warmup();
    // Loading SentencePiece here costs ~1 s; doing it during the (already slow)
    // engine load keeps the first generation's time-to-first-audio low.
    await runtime.ensureTokenizerLoaded();
    state.engine = {
      ...state.engine,
      loaded: true,
      threads: capped,
      error: null,
      ortVersion: readOrtVersion(),
    };
  } catch (error) {
    state.engine = {
      ...state.engine,
      loaded: false,
      error: error instanceof Error ? error.message : String(error),
    };
    post({
      type: "engine-log",
      level: "error",
      message: `Engine load failed: ${state.engine.error}`,
    });
  }
  publishEngineState();
}

function readOrtVersion(): string | null {
  // Same module instance as the vendored runtime's import, so its version is
  // the one actually running the sessions.
  return ortEnv.versions?.web ?? null;
}

/**
 * The manifest is loaded by `ensureManifestLoaded()` (which resolves it through
 * a candidate-path fallback, since it lives in a sub-directory of the model
 * root), so read it from the runtime instead of re-reading a fixed path.
 */
async function getManifest(): Promise<{
  builtin_voices?: {
    voice: string;
    display_name: string;
    group: string;
    audio_file: string;
    prompt_audio_codes?: number[][];
  }[];
}> {
  await runtime.ensureManifestLoaded();
  const manifest = runtime.manifest as unknown as {
    builtin_voices?: {
      voice: string;
      display_name: string;
      group: string;
      audio_file: string;
      prompt_audio_codes?: number[][];
    }[];
  };
  if (!manifest) throw new Error("Manifest not loaded.");
  return manifest;
}

async function builtinVoices(): Promise<BuiltinVoice[]> {
  if (state.builtinVoices) return state.builtinVoices;
  const manifest = await getManifest();
  state.builtinVoices = (manifest.builtin_voices ?? []).map((voice) => ({
    voice: voice.voice,
    display_name: voice.display_name,
    group: voice.group,
    audio_file: voice.audio_file,
    frames: voice.prompt_audio_codes?.length ?? 0,
  }));
  return state.builtinVoices;
}

async function builtinVoiceCodes(name: string): Promise<number[][]> {
  const manifest = await getManifest();
  const match = (manifest.builtin_voices ?? []).find((voice) => voice.voice === name);
  if (!match?.prompt_audio_codes) {
    throw new Error(`Built-in voice not found: ${name}`);
  }
  return match.prompt_audio_codes;
}

async function createVoice(
  name: string,
  source: StoredVoiceMeta["source"],
  channels: Float32Array[],
  sampleRate: number,
): Promise<ClonedVoiceResult> {
  await runtime.configure({
    modelPath: MODEL_ROOT,
    threadCount: state.engine.threads,
  });
  await runtime.ensureCodecEncodeLoaded();

  const channelCount = 2;
  const length = channels[0]?.length ?? 0;
  const waveform = new Float32Array(channelCount * length);
  for (let channel = 0; channel < channelCount; channel += 1) {
    const source_ = channels[Math.min(channel, channels.length - 1)];
    if (source_) waveform.set(source_.subarray(0, length), channel * length);
  }

  const codes = await runtime.encodeReferenceAudioFromWaveform(waveform, length);
  const meta: StoredVoiceMeta = {
    id: `v_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
    name: name.trim() || "Untitled voice",
    createdAt: Date.now(),
    durationSec: length / sampleRate,
    sampleRate,
    channels: channelCount,
    numQuantizers: codes[0]?.length ?? 16,
    frames: codes.length,
    source,
  };
  const flat = new Uint16Array(codes.length * (codes[0]?.length ?? 16));
  codes.forEach((row, index) => {
    flat.set(row, index * (codes[0]?.length ?? 16));
  });
  await writeVoice(meta, flat);
  return { meta, codes: flat };
}

/**
 * The vendored runtime reads sampling/length knobs from the manifest, so the
 * Studio sliders are applied by overriding `generation_defaults` for the request.
 */
async function applyGenerationDefaults(params: SynthParams): Promise<void> {
  await runtime.ensureManifestLoaded();
  const manifest = runtime.manifest;
  if (!manifest?.generation_defaults) return;
  manifest.generation_defaults = {
    ...manifest.generation_defaults,
    max_new_frames: params.maxNewFrames,
    do_sample: params.doSample,
    sample_mode: params.sampleMode,
    text_temperature: params.textTemperature,
    text_top_p: params.textTopP,
    text_top_k: params.textTopK,
    audio_temperature: params.audioTemperature,
    audio_top_p: params.audioTopP,
    audio_top_k: params.audioTopK,
    audio_repetition_penalty: params.audioRepetitionPenalty,
  };
}

/**
 * Splits user text into the pieces the model will actually speak.
 *
 * The runtime normalises first (numbers, units, punctuation) and then packs
 * whole sentences into chunks of at most `maxTokens` tokens, so chunk
 * boundaries always land on sentence edges — that is what makes a long text
 * sound like one speaker rather than a queue of short sentences.
 */
async function analyzeText(
  text: string,
  maxTokens: number,
  enableNormalizeTtsText: boolean,
  enableWeTextProcessing: boolean,
): Promise<{ tokens: number; chunks: string[] }> {
  await runtime.ensureTokenizerLoaded();
  const prepared = runtime.prepareSynthesisText(String(text ?? ""), {
    enableNormalizeTtsText,
    enableWeTextProcessing,
  });
  const normalized = prepared.text?.trim() ?? "";
  const tokens = normalized ? await runtime.countTextTokens(normalized) : 0;
  const chunks = normalized ? await runtime.splitVoiceCloneText(normalized, maxTokens) : [];
  return { tokens, chunks: chunks.length > 0 ? chunks : normalized ? [normalized] : [] };
}

async function resolvePromptAudioCodes(
  request: Extract<Request, { type: "synthesize" }>,
): Promise<number[][]> {
  if (request.voiceId) {
    const cache = state.promptAudioCodesCache;
    if (cache?.key === `cloned:${request.voiceId}`) return cache.codes;
    const meta = await readVoiceMeta(request.voiceId);
    if (!meta) throw new Error("Voice not found — it may have been deleted.");
    const flat = await readVoiceCodes(request.voiceId);
    const width = meta.numQuantizers || 16;
    const codes: number[][] = [];
    for (let index = 0; index * width < flat.length; index += 1) {
      codes.push(Array.from(flat.subarray(index * width, (index + 1) * width)));
    }
    state.promptAudioCodesCache = { key: `cloned:${request.voiceId}`, codes };
    return codes;
  }
  if (request.builtinVoice) {
    const cache = state.promptAudioCodesCache;
    if (cache?.key === `builtin:${request.builtinVoice}`) return cache.codes;
    const codes = await builtinVoiceCodes(request.builtinVoice);
    state.promptAudioCodesCache = { key: `builtin:${request.builtinVoice}`, codes };
    return codes;
  }
  throw new Error("Pick a voice before generating.");
}

async function synthesize(request: Extract<Request, { type: "synthesize" }>) {
  const { requestId } = request;
  const chunkIndex = Number.isInteger(request.chunkIndex) ? request.chunkIndex : 0;
  const chunkCount =
    Number.isInteger(request.chunkCount) && request.chunkCount > 0 ? request.chunkCount : 1;
  const params: SynthParams = { ...DEFAULT_SYNTH_PARAMS, ...request.params };
  const startedAt = Date.now();
  const stats: GenerationStats = {
    startedAt,
    firstAudioMs: null,
    textChunks: 1,
    generatedFrames: 0,
    promptAudioFrames: 0,
    rtf: null,
  };

  state.activeRequest = requestId;
  post({ type: "generation-started", requestId, chunkIndex, chunkCount });

  try {
    await applyGenerationDefaults(params);
    const promptAudioCodes = await resolvePromptAudioCodes(request);
    stats.promptAudioFrames = promptAudioCodes.length;

    const result = await runtime.synthesizeVoiceClone({
      text: request.text,
      promptAudioCodes,
      doSample: params.doSample,
      sampleMode: params.sampleMode,
      streaming: true,
      voiceCloneMaxTextTokens: params.voiceCloneMaxTextTokens,
      enableNormalizeTtsText: params.enableNormalizeTtsText,
      enableWeTextProcessing: params.enableWeTextProcessing,
      isCancelled: () => state.cancelled.has(requestId),
      onAudioChunk: async (chunk: {
        channels: number;
        sampleRate: number;
        chunkData: Float32Array[];
        isPause: boolean;
      }) => {
        if (state.cancelled.has(requestId)) return;
        if (stats.firstAudioMs === null) {
          stats.firstAudioMs = Date.now() - startedAt;
        }
        const channels = chunk.chunkData.map((data) => Float32Array.from(data));
        try {
          post(
            {
              type: "generation-chunk",
              chunk: {
                requestId,
                chunkIndex,
                chunkCount,
                channels,
                sampleRate: chunk.sampleRate,
                isPause: Boolean(chunk.isPause),
              },
            },
            channels.map((data) => data.buffer as ArrayBuffer),
          );
        } catch {
          // Transferable already detached: fall back to a structured clone.
          post({
            type: "generation-chunk",
            chunk: {
              requestId,
              chunkIndex,
              chunkCount,
              channels,
              sampleRate: chunk.sampleRate,
              isPause: Boolean(chunk.isPause),
            },
          });
        }
      },
      onPreparedText: async (prepared: { text: string; textChunks?: string[] }) => {
        stats.textChunks = prepared.textChunks?.length ?? 1;
      },
    });

    const textChunks: string[] = (result as { textChunks?: string[] })?.textChunks ?? [];
    stats.textChunks = textChunks.length || stats.textChunks;
    const profile = runtime.getProfileSnapshot?.() ?? null;
    if (
      profile?.counters &&
      typeof profile.counters["generation.generated_frames"] === "number"
    ) {
      stats.generatedFrames = profile.counters["generation.generated_frames"];
    }
    const elapsed = (Date.now() - startedAt) / 1000;
    stats.rtf = elapsed > 0 ? elapsed : null;

    state.cancelled.delete(requestId);
    post({ type: "generation-stats", requestId, stats });
    post({ type: "generation-done", requestId, chunkIndex, chunkCount, stats });
    post({ type: "profile", snapshot: runtime.getProfileSnapshot() as never });
  } catch (error) {
    const cancelled = state.cancelled.has(requestId);
    state.cancelled.delete(requestId);
    if (cancelled) {
      post({ type: "generation-done", requestId, chunkIndex, chunkCount, stats });
    } else {
      post({
        type: "generation-error",
        requestId,
        chunkIndex,
        chunkCount,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  } finally {
    if (state.activeRequest === requestId) state.activeRequest = null;
  }
}

async function handle(request: Request): Promise<void> {
  switch (request.type) {
    case "probe": {
      // A probe must always answer: if it throws, the UI has no way to tell
      // "this browser lacks OPFS" apart from "the probe broke", and used to
      // report the first as the second.
      const nav = self.navigator as Navigator & { deviceMemory?: number };
      const capabilities = await probeCapabilities().catch(
        (error): OpfsCapabilities & { probeError?: string } => ({
          opfs: isOpfsAvailable(),
          persist: false,
          crossOriginIsolated:
            typeof crossOriginIsolated === "boolean" ? crossOriginIsolated : false,
          sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
          quota: null,
          usage: null,
          probeError: error instanceof Error ? error.message : String(error),
        }),
      );

      const cachedFiles: { path: string; size: number }[] = [];
      let probeError: string | null =
        "probeError" in capabilities ? (capabilities.probeError ?? null) : null;
      try {
        // Republish the engine state so a late subscriber always learns what is
        // already cached (the worker boots before React subscribes).
        await refreshEngineState();
        for (const path of await listCachedModelFiles()) {
          cachedFiles.push({ path, size: await modelFileSizeOnDisk(path) });
        }
      } catch (error) {
        probeError = error instanceof Error ? error.message : String(error);
      }

      const present = new Set(cachedFiles.map((file) => file.path));
      post({
        type: "probe-result",
        payload: {
          opfs: capabilities.opfs,
          persist: capabilities.persist,
          crossOriginIsolated: capabilities.crossOriginIsolated,
          sharedArrayBuffer: capabilities.sharedArrayBuffer,
          quota: capabilities.quota,
          usage: capabilities.usage,
          cachedFiles,
          missingFiles: MODEL_FILES.filter((file: ModelFile) => !present.has(file.path)).map(
            (file) => file.path,
          ),
          hardwareConcurrency: self.navigator.hardwareConcurrency ?? 4,
          deviceMemoryGb: nav.deviceMemory ?? null,
          error: probeError,
        },
      });
      return;
    }
    case "list-voices": {
      post({ type: "voices", voices: await listVoices() });
      return;
    }
    case "download-models": {
      const groups = request.groups.length > 0 ? request.groups : [...REQUIRED_GROUPS];
      await downloadGroups(groups);
      return;
    }
    case "cancel-download": {
      state.downloadAbort?.abort();
      state.downloadAbort = null;
      return;
    }
    case "delete-model-file": {
      await deleteModelFile(request.path);
      await refreshEngineState();
      return;
    }
    case "clear-model-cache": {
      await clearModelCache();
      state.engine.loaded = false;
      await refreshEngineState();
      return;
    }
    case "load-engine": {
      await loadEngine(request.threads);
      return;
    }
    case "unload-engine": {
      state.engine = {
        loaded: false,
        threads: request.threads,
        files: state.engine.files,
        cloningReady: state.engine.cloningReady,
        error: null,
        ortVersion: null,
      };
      publishEngineState();
      return;
    }
    case "list-builtin-voices": {
      try {
        post({ type: "builtin-voices", voices: await builtinVoices() });
      } catch (error) {
        post({
          type: "engine-log",
          level: "error",
          message: `Built-in voices unavailable: ${
            error instanceof Error ? error.message : String(error)
          }`,
        });
      }
      return;
    }
    case "create-voice": {
      try {
        const result = await createVoice(
          request.name,
          request.source,
          request.channels,
          request.sampleRate,
        );
        post({ type: "voice-created", result }, [result.codes.buffer]);
        post({ type: "voices", voices: await listVoices() });
      } catch (error) {
        post({
          type: "fatal",
          message: `Voice cloning failed: ${
            error instanceof Error ? error.message : String(error)
          }`,
        });
      }
      return;
    }
    case "delete-voice": {
      await deleteVoice(request.id);
      post({ type: "voice-deleted", id: request.id });
      post({ type: "voices", voices: await listVoices() });
      return;
    }
    case "rename-voice": {
      const meta = await readVoiceMeta(request.id);
      if (meta) {
        const codes = await readVoiceCodes(request.id);
        await writeVoice({ ...meta, name: request.name }, codes);
        post({ type: "voices", voices: await listVoices() });
      }
      return;
    }
    case "import-voice": {
      try {
        const parsed = JSON.parse(request.payload) as {
          meta: StoredVoiceMeta;
          codes: string;
        };
        const codes = Uint16Array.from(atob(parsed.codes), (char) => char.charCodeAt(0));
        const meta: StoredVoiceMeta = {
          ...parsed.meta,
          id: `v_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
          createdAt: Date.now(),
          source: "import",
        };
        await writeVoice(meta, codes);
        post({ type: "voices", voices: await listVoices() });
      } catch (error) {
        post({
          type: "fatal",
          message: `Import failed: ${error instanceof Error ? error.message : String(error)}`,
        });
      }
      return;
    }
    case "synthesize": {
      await synthesize(request);
      return;
    }
    case "set-profiling": {
      runtime.setProfilingEnabled(request.enabled);
      post({ type: "profile", snapshot: runtime.getProfileSnapshot() as never });
      return;
    }
    case "cancel-generation": {
      state.cancelled.add(request.requestId);
      return;
    }
    case "analyze-text": {
      try {
        const analysis = await analyzeText(
          request.text,
          request.maxTokens,
          request.enableNormalizeTtsText,
          request.enableWeTextProcessing,
        );
        post({ type: "text-analysis", text: request.text, ...analysis });
      } catch (error) {
        // The composer still needs *something* to show when the tokenizer is
        // unavailable; estimate rather than failing.
        post({
          type: "text-analysis",
          text: request.text,
          tokens: Math.max(1, Math.round(request.text.length / 2)),
          chunks: request.text.trim() ? [request.text.trim()] : [],
        });
        post({
          type: "engine-log",
          level: "warn",
          message: `Text analysis failed: ${
            error instanceof Error ? error.message : String(error)
          }`,
        });
      }
      return;
    }
    default: {
      post({ type: "fatal", message: `Unknown request: ${JSON.stringify(request)}` });
    }
  }
}

self.addEventListener("message", (event: MessageEvent<Request>) => {
  const request = event.data;
  currentCorrelationId = request.correlationId ?? null;
  handle(request)
    .catch((error) => {
      post({
        type: "fatal",
        message: error instanceof Error ? error.message : String(error),
      });
    })
    .finally(() => {
      currentCorrelationId = null;
    });
});

void refreshEngineState();
