"use client";

/**
 * The single owner of the TTS worker for the whole app.
 *
 * It lives in the root layout (so navigating between /model, /voices and /studio
 * never reloads the 670 MB session or interrupts playback) and translates worker
 * messages into jotai atoms. Components read atoms and call `useEngine()`.
 *
 * Long texts are spoken **chunk by chunk**: the whole input is split into
 * sentence-aligned pieces first, then each piece is a separate worker request
 * whose audio is appended to the same player as it streams in. The run is paced
 * so the decoder never outruns the speaker, and everything that was produced is
 * kept for a single WAV download.
 */

import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef } from "react";
import { toast } from "sonner";
import { type StreamPlayer, useStreamPlayer } from "@/audio/useStreamPlayer";
import type { ModelGroupId } from "@/lib/model/modelManifest";
import { interChunkGapSeconds } from "@/lib/tts/chunking";
import { TtsWorkerClient } from "@/lib/worker/client";
import type { GenerationStats } from "@/lib/worker/protocol";
import { hydrateAtomsFromStorage } from "@/state/atomWithStorage";
import {
  capabilitiesAtom,
  downloadActiveAtom,
  downloadStateAtom,
  engineAtom,
  engineErrorAtom,
  engineLogAtom,
  enginePhaseAtom,
  profileSnapshotAtom,
  profilingEnabledAtom,
  synthesisReadyAtom,
  threadsAtom,
} from "@/state/modelAtoms";
import {
  EMPTY_GENERATION_STATS,
  generationAtom,
  historyAtom,
  IDLE_GENERATION_STATE,
  STORAGE_KEYS,
  synthParamsAtom,
  textChunksAtom,
  textTokensAtom,
} from "@/state/studioAtoms";
import {
  builtinVoicesAtom,
  clonedVoicesAtom,
  selectedVoiceAtom,
  type VoiceSelection,
  voicesLoadingAtom,
} from "@/state/voiceAtoms";

export type EngineApi = {
  client: TtsWorkerClient | null;
  player: StreamPlayer;
  ready: boolean;
  downloadModels: (groups: ModelGroupId[]) => void;
  loadEngine: () => void;
  /** Token count + chunk plan for the composer, as the user types. */
  analyzeText: (text: string) => void;
  generate: (args: { text: string; voice: VoiceSelection }) => void;
  cancel: () => void;
};

const SAMPLE_RATE = 48_000;
/**
 * Playback lead the scheduler aims for. The decoder is several times faster than
 * realtime, so without this the ring buffer would swallow the whole text and
 * start dropping audio for anything longer than a few sentences.
 */
const MIN_LEAD_SECONDS = 2;
const MAX_LEAD_SECONDS = 40;

type ChunkOutcome = { ok: true; stats: GenerationStats } | { ok: false; message: string };

type Run = {
  id: string;
  cancelled: boolean;
  /** Settles the chunk currently in flight, if any. */
  settle: ((outcome: ChunkOutcome) => void) | null;
};

let sharedClient: TtsWorkerClient | null = null;

function getClient(): TtsWorkerClient | null {
  if (typeof window === "undefined") return null;
  sharedClient ??= new TtsWorkerClient();
  return sharedClient;
}

const EngineContext = createContext<EngineApi | null>(null);

export function EngineProvider({ children }: { children: React.ReactNode }) {
  const client = useMemo(() => getClient(), []);
  const player = useStreamPlayer(true);

  const setCapabilities = useSetAtom(capabilitiesAtom);
  const setEngine = useSetAtom(engineAtom);
  const setEnginePhase = useSetAtom(enginePhaseAtom);
  const setEngineError = useSetAtom(engineErrorAtom);
  const setDownloadState = useSetAtom(downloadStateAtom);
  const setDownloadActive = useSetAtom(downloadActiveAtom);
  const pushLog = useSetAtom(engineLogAtom);
  const setClonedVoices = useSetAtom(clonedVoicesAtom);
  const setBuiltinVoices = useSetAtom(builtinVoicesAtom);
  const setVoicesLoading = useSetAtom(voicesLoadingAtom);
  const setSelectedVoice = useSetAtom(selectedVoiceAtom);
  const setGeneration = useSetAtom(generationAtom);
  const setTokenCount = useSetAtom(textTokensAtom);
  const setTextChunks = useSetAtom(textChunksAtom);
  const setProfile = useSetAtom(profileSnapshotAtom);
  const [threads] = useAtom(threadsAtom);
  const [profiling] = useAtom(profilingEnabledAtom);
  const [params] = useAtom(synthParamsAtom);

  const engine = useAtomValue(engineAtom);
  const capabilities = useAtomValue(capabilitiesAtom);
  const ready = useAtomValue(synthesisReadyAtom);

  /** The run currently speaking, if any. Replaced by every new Speak press. */
  const runRef = useRef<Run | null>(null);
  /** Request id of the chunk in flight, so late chunks from a stopped run drop. */
  const activeRequestRef = useRef<string | null>(null);
  /** Monotonic counter so a slow text analysis cannot overwrite a newer one. */
  const analysisSeqRef = useRef(0);
  /** Text the last analysis request was for, echoed back by the worker. */
  const analysisTextRef = useRef("");
  /** Serializes runs: the ORT sessions must never run two syntheses at once. */
  const runChainRef = useRef<Promise<void>>(Promise.resolve());
  const disposedRef = useRef(false);

  useEffect(() => {
    disposedRef.current = false;
    return () => {
      disposedRef.current = true;
    };
  }, []);

  const threadsRef = useRef(threads);
  const paramsRef = useRef(params);
  const playerRef = useRef(player);

  // Keep the "latest value" refs in sync after render (the React Compiler lint
  // rules forbid writing refs during render).
  useEffect(() => {
    threadsRef.current = threads;
    paramsRef.current = params;
    playerRef.current = player;
  }, [params, player, threads]);

  // Adopt persisted settings/history after mount (SSR-safe, single jotai copy).
  useEffect(() => {
    hydrateAtomsFromStorage([
      { atom: synthParamsAtom, key: STORAGE_KEYS.synthParams },
      { atom: historyAtom, key: STORAGE_KEYS.history },
    ]);
  }, []);

  // --- worker wiring -------------------------------------------------------
  useEffect(() => {
    if (!client) return;

    const log = (level: "info" | "warn" | "error", message: string) => {
      pushLog((prev) => [...prev, { level, message, at: Date.now() }].slice(-200));
      if (level === "error") setEngineError(message);
    };

    const unsubscribe = client.subscribe((message) => {
      switch (message.type) {
        case "probe-result": {
          setCapabilities(message.payload);
          if (message.payload.error) {
            setEnginePhase("error");
            setEngineError(`Browser check failed: ${message.payload.error}`);
          } else if (!message.payload.opfs) {
            setEnginePhase("unsupported");
            setEngineError(
              "This browser exposes no Origin Private File System (navigator.storage.getDirectory is missing), so the ~763 MB of weights cannot be cached. Chrome/Edge 108+, Firefox 111+ and Safari 17+ all support it — if that is what you are on, this is a real bug: the Settings tab shows what the engine worker reported.",
            );
          }
          break;
        }
        case "voices": {
          setClonedVoices(message.voices);
          setVoicesLoading(false);
          break;
        }
        case "builtin-voices": {
          setBuiltinVoices(message.voices);
          setVoicesLoading(false);
          break;
        }
        case "download-progress": {
          setDownloadState(message.state);
          setDownloadActive(true);
          setEnginePhase("downloading");
          break;
        }
        case "download-done": {
          setDownloadState(message.state);
          setDownloadActive(false);
          setEnginePhase("loading");
          break;
        }
        case "download-error": {
          setDownloadActive(false);
          setEnginePhase("error");
          setEngineError(message.message);
          log("error", `Download failed: ${message.message}`);
          break;
        }
        case "engine-state": {
          setEngine(message.state);
          if (message.state.error) {
            setEnginePhase("error");
            setEngineError(message.state.error);
          } else if (message.state.loaded) {
            setEnginePhase("ready");
            setEngineError(null);
            void client.listVoices();
            void client.listBuiltinVoices();
          } else if (message.state.files.length > 0) {
            setEnginePhase((prev) => (prev === "downloading" ? prev : "idle"));
          }
          break;
        }
        case "engine-log": {
          log(message.level, message.message);
          break;
        }
        case "voice-created": {
          toast.success(`Voice “${message.result.meta.name}” is ready to use.`);
          break;
        }
        case "fatal": {
          log("error", message.message);
          runRef.current?.settle?.({ ok: false, message: message.message });
          setGeneration((prev) => ({
            ...prev,
            status: "error",
            error: message.message,
            streaming: false,
          }));
          break;
        }
        case "generation-started": {
          // The player is flushed once per *run* (see `generate`), never per
          // chunk, otherwise every chunk would wipe the audio before it.
          setGeneration((prev) => ({
            ...prev,
            streaming: true,
            chunkIndex: message.chunkIndex,
          }));
          break;
        }
        case "generation-chunk": {
          if (message.chunk.requestId !== activeRequestRef.current) break;
          // Read the frame count BEFORE append(): append() transfers these
          // buffers to the AudioWorklet, which detaches them and zeroes
          // `.length`.
          const frames = message.chunk.channels[0]?.length ?? 0;
          playerRef.current.append(message.chunk.channels, message.chunk.sampleRate);
          setGeneration((prev) => ({
            ...prev,
            samples: prev.samples + frames,
            streaming: true,
          }));
          break;
        }
        case "generation-stats": {
          // Per-chunk stats; the run aggregates them once every chunk is done.
          setGeneration((prev) => ({
            ...prev,
            stats: { ...EMPTY_GENERATION_STATS, ...prev.stats, ...message.stats },
          }));
          break;
        }
        case "generation-done": {
          activeRequestRef.current = null;
          runRef.current?.settle?.({ ok: true, stats: message.stats });
          break;
        }
        case "generation-error": {
          activeRequestRef.current = null;
          runRef.current?.settle?.({ ok: false, message: message.message });
          break;
        }
        case "text-analysis": {
          if (message.text !== analysisTextRef.current) break;
          setTokenCount(message.tokens);
          setTextChunks(message.chunks);
          break;
        }
        case "profile": {
          setProfile(message.snapshot);
          break;
        }
        default:
          break;
      }
    });

    void client.probe().catch(() => undefined);
    void client.listVoices().catch(() => undefined);

    return unsubscribe;
  }, [
    client,
    pushLog,
    setBuiltinVoices,
    setCapabilities,
    setClonedVoices,
    setDownloadActive,
    setDownloadState,
    setEngine,
    setEngineError,
    setEnginePhase,
    setGeneration,
    setProfile,
    setTokenCount,
    setVoicesLoading,
  ]);

  // The worker boots asynchronously; if it never answers, say so instead of
  // showing the "browser not supported" card (which is what a null capabilities
  // used to do — the real cause is usually a worker that failed to load).
  useEffect(() => {
    if (capabilities) return;
    const timer = window.setTimeout(() => {
      if (client) {
        void client.probe().catch(() => undefined);
      }
      setEnginePhase("error");
      setEngineError(
        "The engine worker did not report its capabilities within 15s. This usually means the worker bundle failed to load — try a hard reload (Ctrl/Cmd+Shift+R), and check the Settings tab for the log.",
      );
    }, 15_000);
    return () => window.clearTimeout(timer);
    // Runs once: a later probe result clears `capabilities` and cancels this.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, capabilities, setEngineError, setEnginePhase]);

  // Load ORT sessions as soon as the required weights are cached.
  useEffect(() => {
    if (!client) return;
    if (engine.loaded) return;
    if (engine.files.length === 0) return;
    if (capabilities && !capabilities.opfs) return;
    setEnginePhase("loading");
    client.loadEngine(threadsRef.current);
  }, [client, engine.loaded, engine.files.length, capabilities, setEnginePhase]);

  // Keep the profiling flag in sync (it is read by the vendored runtime).
  useEffect(() => {
    if (!client) return;
    client.setProfiling(profiling);
  }, [client, profiling]);

  // Keep the worker in sync when the user changes the thread budget.
  useEffect(() => {
    if (!client || !engine.loaded) return;
    client.loadEngine(threads);
  }, [client, engine.loaded, threads]);

  // --- actions -------------------------------------------------------------
  const downloadModels = useCallback(
    (groups: ModelGroupId[]) => {
      if (!client) return;
      runRef.current = null;
      activeRequestRef.current = null;
      playerRef.current.stop();
      playerRef.current.flush();
      setGeneration({ ...IDLE_GENERATION_STATE });
      client.downloadModels(groups);
    },
    [client, setGeneration],
  );

  const loadEngine = useCallback(() => {
    client?.loadEngine(threadsRef.current);
  }, [client]);

  const analyzeText = useCallback(
    (text: string) => {
      if (!client) return;
      const params = paramsRef.current;
      const trimmed = text.trim();
      analysisTextRef.current = text;
      analysisSeqRef.current += 1;
      const seq = analysisSeqRef.current;
      if (!trimmed) {
        setTokenCount(0);
        setTextChunks([]);
        return;
      }
      void client
        .analyzeText({
          text,
          maxTokens: params.voiceCloneMaxTextTokens,
          enableNormalizeTtsText: params.enableNormalizeTtsText,
          enableWeTextProcessing: params.enableWeTextProcessing,
        })
        .then((result) => {
          if (analysisSeqRef.current !== seq || analysisTextRef.current !== text) return;
          // The worker is the trust boundary here: never let a malformed reply
          // put `undefined` into an atom the UI iterates over.
          setTokenCount(Number.isFinite(result?.tokens) ? result.tokens : 0);
          setTextChunks(Array.isArray(result?.chunks) ? result.chunks : []);
        })
        .catch(() => undefined);
    },
    [client, setTextChunks, setTokenCount],
  );

  const generate = useCallback(
    ({ text, voice }: { text: string; voice: VoiceSelection }) => {
      if (!client) return;
      if (!text.trim()) {
        toast.error("Type something to say first.");
        return;
      }
      if (!ready) {
        toast.error("The engine is still loading. Check the Model tab.");
        return;
      }

      // A new press supersedes whatever was speaking. The previous run is asked
      // to stop and this one is queued behind it, so the ORT sessions are never
      // driven by two requests at the same time.
      const previous = runRef.current;
      if (previous && !previous.cancelled) {
        previous.cancelled = true;
        if (activeRequestRef.current) client.cancelGeneration(activeRequestRef.current);
      }
      activeRequestRef.current = null;
      setSelectedVoice(voice);

      const run: Run = {
        id: `gen-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        cancelled: false,
        settle: null,
      };
      runRef.current = run;
      const params = paramsRef.current;

      /** Still the run the user is waiting on (not superseded, not unmounted). */
      const isCurrent = () => runRef.current === run && !disposedRef.current;
      /** ... and nobody asked it to stop. */
      const halted = () => !isCurrent() || run.cancelled;

      const speakChunk = (chunkText: string, index: number, count: number) => {
        const requestId = `${run.id}#${index}`;
        activeRequestRef.current = requestId;
        return new Promise<ChunkOutcome>((resolve) => {
          run.settle = (outcome) => {
            run.settle = null;
            resolve(outcome);
          };
          client.synthesize({
            requestId,
            text: chunkText,
            voiceId: voice.kind === "cloned" ? voice.id : null,
            builtinVoice: voice.kind === "builtin" ? voice.id : null,
            params,
            chunkIndex: index,
            chunkCount: count,
          });
        });
      };

      /** Waits until the player has drained to a safe lead before the next chunk. */
      const waitForLead = async (leadSeconds: number) => {
        const deadlineFrames = leadSeconds * SAMPLE_RATE;
        // Hard cap: the wait depends on the worklet's fill reports. If they
        // stop arriving (old cached worklet, suspended context) the run must
        // continue anyway rather than sit on "Working…" forever.
        const startedAt = Date.now();
        const maxWaitMs = Math.max(4_000, leadSeconds * 2_000);
        while (!halted() && playerRef.current.getBufferedFrames() > deadlineFrames) {
          if (Date.now() - startedAt > maxWaitMs) break;
          await new Promise((resolve) => window.setTimeout(resolve, 120));
        }
      };

      /** Everything decoded so far in this run, in seconds. */
      const runAudioSeconds = () => playerRef.current.totalFrames / SAMPLE_RATE;

      const start = async () => {
        // Flush once per run: everything spoken by this run accumulates in the
        // player so it can be exported as a single clip.
        if (!isCurrent()) return;
        playerRef.current.stop();
        playerRef.current.flush();
        setGeneration({
          ...IDLE_GENERATION_STATE,
          runId: run.id,
          requestId: run.id,
          status: "starting",
        });
        const startedAt = Date.now();

        const analysis = await Promise.race([
          client
            .analyzeText({
              text,
              maxTokens: params.voiceCloneMaxTextTokens,
              enableNormalizeTtsText: params.enableNormalizeTtsText,
              enableWeTextProcessing: params.enableWeTextProcessing,
            })
            .catch(() => null),
          // A dead worker must not leave the button spinning forever; fall back
          // to speaking the text as one piece.
          new Promise<null>((resolve) => window.setTimeout(() => resolve(null), 10_000)),
        ]);
        if (halted()) {
          // Cancelled while queued: nothing was generated for this run.
          if (runRef.current === run) {
            setGeneration((prev) => ({ ...prev, status: "idle", streaming: false }));
          }
          return;
        }

        const usable = (Array.isArray(analysis?.chunks) ? analysis.chunks : []).filter(
          (chunk) => Boolean(chunk?.trim()),
        );
        const chunks = usable.length > 0 ? usable : [text.trim()];
        setGeneration((prev) => ({ ...prev, chunks }));

        let firstAudioMs: number | null = null;
        let promptAudioFrames = 0;
        let generatedFrames = 0;
        let leadSeconds = MIN_LEAD_SECONDS;

        const publishStats = (chunksDone: number) => {
          const audioSeconds = runAudioSeconds();
          setGeneration((prev) => ({
            ...prev,
            chunksDone,
            streaming: false,
            stats: {
              startedAt,
              firstAudioMs,
              textChunks: chunks.length,
              generatedFrames,
              promptAudioFrames,
              rtf: audioSeconds > 0 ? (Date.now() - startedAt) / 1000 / audioSeconds : null,
            },
          }));
        };

        for (let index = 0; index < chunks.length; index += 1) {
          if (halted()) {
            // Stopped by the user: keep whatever was decoded so it can be saved.
            if (isCurrent()) {
              publishStats(Math.min(index, chunks.length));
              setGeneration((prev) => ({ ...prev, status: "done" }));
            }
            return;
          }

          if (index > 0) {
            // Let the speaker catch up, then breathe between two chunks so the
            // result sounds like one passage rather than a queue of clips.
            await waitForLead(leadSeconds);
            if (halted()) continue;
            playerRef.current.appendSilence(interChunkGapSeconds(chunks[index]));
          }

          const chunkStartedAt = Date.now();
          setGeneration((prev) => ({
            ...prev,
            status: index === 0 ? "starting" : "running",
            chunkIndex: index,
            streaming: false,
          }));

          const outcome = await speakChunk(chunks[index], index, chunks.length);
          if (!isCurrent()) return;

          if (run.cancelled) {
            publishStats(Math.min(index + 1, chunks.length));
            setGeneration((prev) => ({ ...prev, status: "done" }));
            return;
          }

          // Aim for enough buffered audio to cover the next chunk's generation
          // time, so playback never underruns between two chunks.
          const chunkSeconds = (Date.now() - chunkStartedAt) / 1000;
          leadSeconds = Math.min(
            MAX_LEAD_SECONDS,
            Math.max(MIN_LEAD_SECONDS, chunkSeconds * 1.5),
          );

          if (outcome.ok) {
            firstAudioMs ??= outcome.stats.firstAudioMs;
            promptAudioFrames = outcome.stats.promptAudioFrames;
            generatedFrames += outcome.stats.generatedFrames;
          }
          publishStats(index + 1);

          if (!outcome.ok) {
            setGeneration((prev) => ({
              ...prev,
              status: "error",
              error:
                chunks.length > 1
                  ? `Chunk ${index + 1} of ${chunks.length} failed: ${outcome.message}`
                  : outcome.message,
            }));
            return;
          }
        }

        if (!isCurrent()) return;
        setGeneration((prev) => ({ ...prev, status: "done", streaming: false }));
      };

      // Serialize runs: the worker handles one synthesis at a time.
      const chained = runChainRef.current.then(start, start);
      runChainRef.current = chained.then(
        () => undefined,
        () => undefined,
      );
      void chained;
    },
    [client, ready, setGeneration, setSelectedVoice],
  );

  const cancel = useCallback(() => {
    const run = runRef.current;
    if (!run || run.cancelled) return;
    run.cancelled = true;
    const requestId = activeRequestRef.current;
    if (requestId) client?.cancelGeneration(requestId);
    setGeneration((prev) => (prev.status === "done" ? prev : { ...prev, status: "stopping" }));
  }, [client, setGeneration]);

  const api = useMemo<EngineApi>(
    () => ({
      client,
      player,
      ready,
      downloadModels,
      loadEngine,
      analyzeText,
      generate,
      cancel,
    }),
    [client, player, ready, downloadModels, loadEngine, analyzeText, generate, cancel],
  );

  // Expose the engine through a React context so pages can call `useEngine()`.
  return <EngineContext.Provider value={api}>{children}</EngineContext.Provider>;
}

export function useEngine(): EngineApi {
  const value = useContext(EngineContext);
  if (!value) {
    throw new Error("useEngine must be used inside <EngineProvider>.");
  }
  return value;
}

export type { VoiceSelection };
