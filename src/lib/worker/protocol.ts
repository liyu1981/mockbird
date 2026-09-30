/**
 * Typed RPC protocol between the main thread and the MOSS-TTS worker.
 *
 * Everything heavy (weight download, ORT session creation, the decode loop) runs
 * in the worker; the main thread only sends intent and receives PCM chunks.
 */

import type { DownloadState } from "@/lib/model/downloader";
import type { ModelFile, ModelGroupId } from "@/lib/model/modelManifest";
import type { StoredVoiceMeta } from "@/lib/model/opfsStore";

export type SynthParams = {
  sampleMode: "fixed" | "greedy";
  doSample: boolean;
  textTemperature: number;
  textTopP: number;
  textTopK: number;
  audioTemperature: number;
  audioTopP: number;
  audioTopK: number;
  audioRepetitionPenalty: number;
  maxNewFrames: number;
  voiceCloneMaxTextTokens: number;
  enableNormalizeTtsText: boolean;
  enableWeTextProcessing: boolean;
  seed: number | null;
};

export const DEFAULT_SYNTH_PARAMS: SynthParams = {
  sampleMode: "fixed",
  doSample: true,
  textTemperature: 1,
  textTopP: 1,
  textTopK: 50,
  audioTemperature: 0.8,
  audioTopP: 0.95,
  audioTopK: 25,
  audioRepetitionPenalty: 1.2,
  maxNewFrames: 375,
  voiceCloneMaxTextTokens: 75,
  enableNormalizeTtsText: true,
  enableWeTextProcessing: false,
  seed: null,
};

export type ProfileSnapshot = {
  enabled: boolean;
  counters: Record<string, number>;
  timings: Record<string, number>;
} | null;

export type BuiltinVoice = {
  voice: string;
  display_name: string;
  group: string;
  audio_file: string;
  frames: number;
};

export type AudioChunkMessage = {
  requestId: string;
  /** 0-based index of the text chunk being spoken; -1 when unknown. */
  chunkIndex: number;
  chunkCount: number;
  /** Channel-major PCM (left, right) at 48 kHz. */
  channels: Float32Array[];
  sampleRate: number;
  isPause: boolean;
};

export type GenerationStats = {
  startedAt: number;
  firstAudioMs: number | null;
  textChunks: number;
  generatedFrames: number;
  promptAudioFrames: number;
  rtf: number | null;
};

export type ClonedVoiceResult = {
  meta: StoredVoiceMeta;
  codes: Uint16Array;
};

// --- main -> worker -------------------------------------------------------

export type RequestBody =
  | { type: "probe" }
  | { type: "list-voices" }
  | { type: "download-models"; groups: ModelGroupId[] }
  | { type: "cancel-download" }
  | { type: "delete-model-file"; path: string }
  | { type: "clear-model-cache" }
  | { type: "load-engine"; threads: number }
  | { type: "unload-engine"; threads: number }
  | { type: "list-builtin-voices" }
  | {
      type: "create-voice";
      name: string;
      source: StoredVoiceMeta["source"];
      /** Channel-major 48 kHz PCM, transferred to the worker. */
      channels: Float32Array[];
      sampleRate: number;
    }
  | { type: "delete-voice"; id: string }
  | { type: "rename-voice"; id: string; name: string }
  | { type: "import-voice"; payload: string }
  | {
      type: "synthesize";
      requestId: string;
      text: string;
      voiceId: string | null;
      builtinVoice: string | null;
      params: SynthParams;
      /** 0-based position of this text chunk inside the user's run. */
      chunkIndex: number;
      chunkCount: number;
    }
  | { type: "cancel-generation"; requestId: string }
  | { type: "set-profiling"; enabled: boolean }
  | {
      type: "analyze-text";
      text: string;
      maxTokens: number;
      enableNormalizeTtsText: boolean;
      enableWeTextProcessing: boolean;
    };

/** Every request may carry a correlation id; the worker echoes it on replies. */
export type Request = RequestBody & { correlationId?: string };

// --- worker -> main -------------------------------------------------------

export type EngineState = {
  loaded: boolean;
  threads: number;
  files: string[];
  cloningReady: boolean;
  error: string | null;
  ortVersion: string | null;
};

export type Response =
  | { type: "probe-result"; payload: CapabilitiesPayload }
  | { type: "voices"; voices: StoredVoiceMeta[] }
  | { type: "download-progress"; state: DownloadState }
  | { type: "download-done"; state: DownloadState }
  | { type: "download-error"; message: string }
  | { type: "engine-state"; state: EngineState }
  | { type: "engine-log"; level: "info" | "warn" | "error"; message: string }
  | { type: "builtin-voices"; voices: BuiltinVoice[] }
  | { type: "voice-created"; result: ClonedVoiceResult }
  | { type: "voice-deleted"; id: string }
  | { type: "generation-started"; requestId: string; chunkIndex: number; chunkCount: number }
  | { type: "generation-chunk"; chunk: AudioChunkMessage }
  | { type: "generation-stats"; requestId: string; stats: Partial<GenerationStats> }
  | {
      type: "generation-done";
      requestId: string;
      chunkIndex: number;
      chunkCount: number;
      stats: GenerationStats;
    }
  | {
      type: "generation-error";
      requestId: string;
      chunkIndex: number;
      chunkCount: number;
      message: string;
    }
  | { type: "text-analysis"; text: string; tokens: number; chunks: string[] }
  | { type: "profile"; snapshot: ProfileSnapshot | null }
  | { type: "fatal"; message: string };

export type CapabilitiesPayload = {
  opfs: boolean;
  persist: boolean;
  crossOriginIsolated: boolean;
  sharedArrayBuffer: boolean;
  quota: number | null;
  usage: number | null;
  cachedFiles: { path: string; size: number }[];
  missingFiles: ModelFile["path"][];
  hardwareConcurrency: number;
  deviceMemoryGb: number | null;
  /** Set when the probe itself failed; `opfs` then only reflects feature detection. */
  error: string | null;
};

export type WorkerEvent = {
  data: Response;
};

export const isResponse = (value: unknown): value is Response =>
  typeof value === "object" && value !== null && "type" in value;
