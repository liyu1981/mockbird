"use client";

/**
 * Records a reference clip with `MediaRecorder`, shows a live input level, and
 * hands back 48 kHz stereo PCM ready for the codec encoder.
 */

import { Mic, Square, Trash2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { analyzePcm, decodeToModelPcm } from "@/lib/tts/audio";

const MIN_SECONDS = 2;
const MAX_SECONDS = 30;
const IDEAL_SECONDS = 12;

export type RecordingResult = {
  channels: Float32Array[];
  sampleRate: number;
  durationSec: number;
  analysis: { peak: number; rms: number; clipped: number };
  blob: Blob;
};

export function VoiceRecorder({
  onCaptured,
  disabled,
  onRecordingChange,
}: {
  onCaptured: (result: RecordingResult) => void;
  disabled?: boolean;
  /** Lets the page disable the "hear it first" preview while the mic is live. */
  onRecordingChange?: (recording: boolean) => void;
}) {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserFrameRef = useRef<number | null>(null);
  const startedAtRef = useRef(0);
  const stopTimerRef = useRef<number | null>(null);
  // Lets the max-duration timer stop the recording without `start` having to
  // depend on `stop` (which is declared after it).
  const stopRef = useRef<() => Promise<void>>(async () => {});

  const cleanup = useCallback(() => {
    if (analyserFrameRef.current !== null) {
      cancelAnimationFrame(analyserFrameRef.current);
      analyserFrameRef.current = null;
    }
    for (const track of streamRef.current?.getTracks() ?? []) track.stop();
    streamRef.current = null;
    void audioContextRef.current?.close().catch(() => undefined);
    audioContextRef.current = null;
  }, []);

  useEffect(() => cleanup, [cleanup]);

  // The parent passes a fresh closure every render, so keep the latest one in a
  // ref: an inline dependency would re-run this effect on every parent render and
  // keep reporting "not recording" while the microphone is live.
  const notifyRef = useRef(onRecordingChange);
  useEffect(() => {
    notifyRef.current = onRecordingChange;
  });

  // Never leave the parent thinking we are still recording (e.g. on unmount).
  useEffect(() => () => notifyRef.current?.(false), []);

  const stop = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder) return;
    if (stopTimerRef.current !== null) {
      window.clearInterval(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    const blob = await new Promise<Blob>((resolve) => {
      recorder.onstop = () => resolve(new Blob(chunksRef.current, { type: recorder.mimeType }));
      recorder.stop();
    });
    cleanup();
    setRecording(false);
    notifyRef.current?.(false);
    setLevel(0);

    const buffer = await blob.arrayBuffer();
    try {
      const pcm = await decodeToModelPcm(buffer);
      const analysis = analyzePcm(pcm.channels);
      onCaptured({ ...pcm, analysis, blob });
      if (pcm.durationSec < MIN_SECONDS) {
        toast.warning(
          `That clip is only ${pcm.durationSec.toFixed(1)}s. 8–15s of clean speech clones best.`,
        );
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, [cleanup, onCaptured]);

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      streamRef.current = stream;

      const context = new AudioContext();
      audioContextRef.current = context;
      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      const data = new Float32Array(analyser.fftSize);
      const tick = () => {
        analyser.getFloatTimeDomainData(data);
        let sum = 0;
        for (let index = 0; index < data.length; index += 1) sum += data[index] * data[index];
        setLevel(Math.min(1, Math.sqrt(sum / data.length) * 4));
        analyserFrameRef.current = requestAnimationFrame(tick);
      };
      tick();

      const mimeType = pickMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.start(250);
      recorderRef.current = recorder;
      startedAtRef.current = Date.now();
      setRecording(true);
      notifyRef.current?.(true);
      setSeconds(0);

      const timer = window.setInterval(() => {
        const elapsed = (Date.now() - startedAtRef.current) / 1000;
        setSeconds(elapsed);
        if (elapsed >= MAX_SECONDS) void stopRef.current();
      }, 100);
      stopTimerRef.current = timer;
    } catch (caught) {
      cleanup();
      setError(
        caught instanceof Error
          ? caught.message
          : "Microphone access was denied. You can upload a file instead.",
      );
    }
  }, [cleanup]);

  // Keep the timer able to stop the recording without `start` depending on
  // `stop` (declared below) — synced after render, per the React Compiler rules.
  useEffect(() => {
    stopRef.current = stop;
  }, [stop]);

  const progress = Math.min(100, (seconds / IDEAL_SECONDS) * 100);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        {!recording ? (
          <Button onClick={start} disabled={disabled} variant="default">
            <Mic className="size-4" /> Record sample
          </Button>
        ) : (
          <Button onClick={stop} variant="destructive">
            <Square className="size-4" /> Stop · {seconds.toFixed(1)}s
          </Button>
        )}
        {seconds > 0 && !recording && (
          <Button variant="ghost" size="sm" onClick={() => setSeconds(0)}>
            <Trash2 className="size-3.5" /> Reset
          </Button>
        )}
        <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
          <span>{IDEAL_SECONDS}s target</span>
          <span>·</span>
          <span>max {MAX_SECONDS}s</span>
        </div>
      </div>

      {recording && (
        <div className="space-y-1.5">
          <Progress value={progress} />
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-emerald-500 transition-[width] duration-75"
              style={{ width: `${Math.round(level * 100)}%` }}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Speak naturally, one sentence at a time. Avoid background noise and music.
          </p>
        </div>
      )}

      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

function pickMimeType(): string | null {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  for (const type of candidates) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return null;
}
