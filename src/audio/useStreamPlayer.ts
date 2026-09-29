"use client";

/**
 * Streaming playback for generated audio.
 *
 * The worker streams 48 kHz stereo PCM chunks; they are pushed into an
 * AudioWorklet ring buffer and played while generation continues. The full
 * signal is also accumulated in a `Float32Array` so the Studio can offer a WAV
 * download and a waveform without a second pass.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { encodeWav } from "@/lib/tts/audio";

const SAMPLE_RATE = 48_000;
/** Peak buckets computed per chunk, so the waveform is O(1) per update. */
const PEAKS_PER_CHUNK = 256;

export type StreamPlayerState = {
  ready: boolean;
  playing: boolean;
  /** Frames (per channel) currently buffered in the worklet. */
  bufferedFrames: number;
  totalFrames: number;
  overflowed: boolean;
  error: string | null;
};

export type StreamPlayer = StreamPlayerState & {
  /** Monotonic peak envelope of everything generated so far. */
  peaks: Float32Array;
  append: (channels: Float32Array[], sampleRate: number) => void;
  flush: () => void;
  pause: () => void;
  resume: () => void;
  stop: () => void;
  /** Full generation as a WAV blob (null until audio exists). */
  toWav: () => Blob | null;
};

function chunkPeaks(left: Float32Array, right: Float32Array): Float32Array {
  const peaks = new Float32Array(PEAKS_PER_CHUNK);
  const perBucket = Math.max(1, Math.floor(left.length / PEAKS_PER_CHUNK));
  for (let bucket = 0; bucket < PEAKS_PER_CHUNK; bucket += 1) {
    const start = bucket * perBucket;
    const end = Math.min(left.length, start + perBucket);
    let peak = 0;
    for (let index = start; index < end; index += 1) {
      const value = Math.abs((left[index] + (right[index] ?? left[index])) * 0.5);
      if (value > peak) peak = value;
    }
    peaks[bucket] = peak;
  }
  return peaks;
}

export function useStreamPlayer(enabled: boolean): StreamPlayer {
  const contextRef = useRef<AudioContext | null>(null);
  const nodeRef = useRef<AudioWorkletNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const channelsRef = useRef<[Float32Array[], Float32Array[]]>([[], []]);
  const totalFramesRef = useRef(0);
  const [peakVersion, setPeakVersion] = useState(0);
  const peakChunksRef = useRef<Float32Array[]>([]);
  const [state, setState] = useState<StreamPlayerState>({
    ready: false,
    playing: false,
    bufferedFrames: 0,
    totalFrames: 0,
    overflowed: false,
    error: null,
  });

  // --- setup ---------------------------------------------------------------
  useEffect(() => {
    if (!enabled) return;
    let disposed = false;

    const setup = async () => {
      try {
        const context = new AudioContext({
          sampleRate: SAMPLE_RATE,
          latencyHint: "interactive",
        });
        if (disposed) {
          await context.close();
          return;
        }
        // Served from public/ so webpack (dev) and Turbopack (Vercel) behave the
        // same: `new URL(..., import.meta.url)` makes webpack treat the .ts file as
        // a raw asset and the browser then rejects its MIME type.
        await context.audioWorklet.addModule("/stream-player.worklet.js");
        if (disposed) {
          await context.close();
          return;
        }
        const audioNode = new AudioWorkletNode(context, "mockbird-stream-player", {
          numberOfInputs: 0,
          numberOfOutputs: 1,
          outputChannelCount: [2],
        });
        const gain = context.createGain();
        gain.gain.value = 0.9;
        audioNode.connect(gain).connect(context.destination);
        audioNode.port.onmessage = (event: MessageEvent) => {
          const data = event.data as {
            type: string;
            frames?: number;
            paused?: boolean;
            dropped?: number;
          };
          if (data.type === "buffered" || data.type === "state") {
            setState((prev) => ({
              ...prev,
              ready: true,
              playing: data.paused === true ? false : prev.playing,
              bufferedFrames: data.frames ?? prev.bufferedFrames,
            }));
          } else if (data.type === "overflow") {
            setState((prev) => ({ ...prev, overflowed: true }));
          } else if (data.type === "stopped") {
            setState((prev) => ({ ...prev, playing: false, bufferedFrames: 0 }));
          } else if (data.type === "drained") {
            setState((prev) => ({ ...prev, playing: false }));
          }
        };
        contextRef.current = context;
        nodeRef.current = audioNode;
        gainRef.current = gain;
        setState((prev) => ({ ...prev, ready: true, error: null }));
      } catch (error) {
        setState((prev) => ({
          ...prev,
          error: `Audio playback unavailable: ${
            error instanceof Error ? error.message : String(error)
          }`,
        }));
      }
    };

    void setup();

    return () => {
      disposed = true;
      nodeRef.current?.disconnect();
      gainRef.current?.disconnect();
      void contextRef.current?.close().catch(() => undefined);
      nodeRef.current = null;
      gainRef.current = null;
      contextRef.current = null;
    };
  }, [enabled]);

  // --- controls ------------------------------------------------------------
  const append = useCallback((channels: Float32Array[], sampleRate: number) => {
    const left = channels[0] ?? new Float32Array(0);
    const right = channels[1] ?? channels[0] ?? new Float32Array(0);
    const [leftChunks, rightChunks] = channelsRef.current;
    peakChunksRef.current = [...peakChunksRef.current, chunkPeaks(left, right)];
    leftChunks.push(left);
    rightChunks.push(right);
    const total = totalFramesRef.current + left.length;
    totalFramesRef.current = total;
    // Deliberately NOT transferring: these arrays are also the source buffer for
    // `toWav()`, and transferring detaches them (leaving a zero-length tail and a
    // throwing `.set()`). The structured clone here is ~30 KB per chunk, which is
    // nothing next to the audio itself.
    nodeRef.current?.port.postMessage({ type: "append", left, right, sampleRate });
    setPeakVersion((version) => version + 1);
    setState((prev) => ({
      ...prev,
      totalFrames: total,
      playing: true,
    }));
  }, []);

  const flush = useCallback(() => {
    channelsRef.current = [[], []];
    totalFramesRef.current = 0;
    peakChunksRef.current = [];
    nodeRef.current?.port.postMessage({ type: "flush" });
    setPeakVersion((version) => version + 1);
    setState((prev) => ({ ...prev, totalFrames: 0, bufferedFrames: 0, overflowed: false }));
  }, []);

  const pause = useCallback(() => {
    nodeRef.current?.port.postMessage({ type: "pause" });
    setState((prev) => ({ ...prev, playing: false }));
  }, []);

  const resume = useCallback(() => {
    void contextRef.current?.resume();
    nodeRef.current?.port.postMessage({ type: "resume" });
    setState((prev) => ({ ...prev, playing: true }));
  }, []);

  const stop = useCallback(() => {
    nodeRef.current?.port.postMessage({ type: "stop" });
    setState((prev) => ({ ...prev, playing: false, bufferedFrames: 0 }));
  }, []);

  const toWav = useCallback((): Blob | null => {
    const [leftChunks, rightChunks] = channelsRef.current;
    const total = totalFramesRef.current;
    if (total === 0) return null;
    const left = new Float32Array(total);
    const right = new Float32Array(total);
    let offset = 0;
    for (let index = 0; index < leftChunks.length; index += 1) {
      const chunk = leftChunks[index];
      left.set(chunk, offset);
      right.set(rightChunks[index] ?? chunk, offset);
      offset += chunk.length;
    }
    return encodeWav([left, right], SAMPLE_RATE);
  }, []);

  const peaks = useMemo(() => {
    const chunks = peakChunksRef.current;
    if (chunks.length === 0) return new Float32Array(0);
    const merged = new Float32Array(chunks.length * PEAKS_PER_CHUNK);
    for (let index = 0; index < chunks.length; index += 1) {
      merged.set(chunks[index], index * PEAKS_PER_CHUNK);
    }
    return merged;
    // peakVersion is a deliberate cache-busting dependency: the peaks live in a
    // ref, and the version counter is what tells React to recompute.
  }, [peakVersion]);

  return { ...state, peaks, append, flush, pause, resume, stop, toWav };
}
