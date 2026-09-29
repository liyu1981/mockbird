"use client";

/**
 * The single owner of the TTS worker for the whole app.
 *
 * It lives in the root layout (so navigating between /model, /voices and /studio
 * never reloads the 670 MB session or interrupts playback) and translates worker
 * messages into jotai atoms. Components read atoms and call `useEngine()`.
 */

import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { usePathname } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef } from "react";
import { toast } from "sonner";
import { type StreamPlayer, useStreamPlayer } from "@/audio/useStreamPlayer";
import type { ModelGroupId } from "@/lib/model/modelManifest";
import { TtsWorkerClient } from "@/lib/worker/client";
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
  generationAtom,
  historyAtom,
  STORAGE_KEYS,
  synthParamsAtom,
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
  countTokens: (text: string) => void;
  generate: (args: { text: string; voice: VoiceSelection }) => void;
  cancel: () => void;
};

let sharedClient: TtsWorkerClient | null = null;

function getClient(): TtsWorkerClient | null {
  if (typeof window === "undefined") return null;
  sharedClient ??= new TtsWorkerClient();
  return sharedClient;
}

const EngineContext = createContext<EngineApi | null>(null);

export function EngineProvider({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
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
  const setProfile = useSetAtom(profileSnapshotAtom);
  const [threads] = useAtom(threadsAtom);
  const [profiling] = useAtom(profilingEnabledAtom);
  const [params] = useAtom(synthParamsAtom);

  const engine = useAtomValue(engineAtom);
  const capabilities = useAtomValue(capabilitiesAtom);
  const ready = useAtomValue(synthesisReadyAtom);

  const activeRequestRef = useRef<string | null>(null);

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

  // A new page means a new context for "which voice am I using".
  useEffect(() => {
    setSelectedVoice(null);
  }, [pathname, setSelectedVoice]);

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
          setGeneration((prev) => ({
            ...prev,
            status: "error",
            error: message.message,
            streaming: false,
          }));
          break;
        }
        case "generation-started": {
          playerRef.current.flush();
          setGeneration({
            requestId: message.requestId,
            status: "running",
            error: null,
            stats: null,
            samples: 0,
            streaming: true,
          });
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
          }));
          break;
        }
        case "generation-stats": {
          setGeneration((prev) => ({ ...prev, stats: message.stats as never }));
          break;
        }
        case "generation-done": {
          setGeneration((prev) => ({
            ...prev,
            status: "done",
            streaming: false,
            stats: (message.stats as never) ?? prev.stats,
          }));
          activeRequestRef.current = null;
          break;
        }
        case "generation-error": {
          setGeneration((prev) => ({
            ...prev,
            status: "error",
            streaming: false,
            error: message.message,
          }));
          activeRequestRef.current = null;
          break;
        }
        case "token-count": {
          setTokenCount(message.tokens);
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
      playerRef.current.stop();
      playerRef.current.flush();
      setGeneration({
        requestId: null,
        status: "idle",
        error: null,
        stats: null,
        samples: 0,
        streaming: false,
      });
      client.downloadModels(groups);
    },
    [client, setGeneration],
  );

  const loadEngine = useCallback(() => {
    client?.loadEngine(threadsRef.current);
  }, [client]);

  const countTokens = useCallback(
    (text: string) => {
      if (!client || !text.trim()) return;
      client.countTokens(text);
    },
    [client],
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
      const requestId = `gen-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      activeRequestRef.current = requestId;
      setSelectedVoice(voice);
      setGeneration({
        requestId,
        status: "starting",
        error: null,
        stats: null,
        samples: 0,
        streaming: false,
      });
      client.synthesize({
        requestId,
        text,
        voiceId: voice.kind === "cloned" ? voice.id : null,
        builtinVoice: voice.kind === "builtin" ? voice.id : null,
        params: paramsRef.current,
      });
    },
    [client, ready, setGeneration, setSelectedVoice],
  );

  const cancel = useCallback(() => {
    if (!client) return;
    const requestId = activeRequestRef.current;
    if (!requestId) return;
    client.cancelGeneration(requestId);
    setGeneration((prev) => ({ ...prev, status: "stopping" }));
  }, [client, setGeneration]);

  const api = useMemo<EngineApi>(
    () => ({
      client,
      player,
      ready,
      downloadModels,
      loadEngine,
      countTokens,
      generate,
      cancel,
    }),
    [client, player, ready, downloadModels, loadEngine, countTokens, generate, cancel],
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
