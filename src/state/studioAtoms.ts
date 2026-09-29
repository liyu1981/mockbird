"use client";

import { atom } from "jotai";
import {
  DEFAULT_SYNTH_PARAMS,
  type GenerationStats,
  type SynthParams,
} from "@/lib/worker/protocol";
import { atomWithStorage } from "@/state/atomWithStorage";

export const STORAGE_KEYS = {
  synthParams: "mockbird.synth-params",
  history: "mockbird.history",
} as const;

export const SAMPLE_TEXTS: { id: string; label: string; text: string }[] = [
  {
    id: "zh-intro",
    label: "中文 · 介绍",
    text: "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。",
  },
  {
    id: "en-intro",
    label: "English · intro",
    text: "Welcome to MOSS-TTS-Nano, a realtime multilingual speech model that runs entirely in your browser.",
  },
  {
    id: "zh-long",
    label: "中文 · 长文本",
    text: "语音克隆的目标，是在极短的时间内复刻一段声音 characteristics。我们希望任何人都能在浏览器里完成这件事：上传十秒录音，输入一句话，几秒之后听到属于自己的声音。所有推理都在本地完成，音频不会离开设备。",
  },
  {
    id: "en-long",
    label: "English · long form",
    text: "Voice cloning is the fastest way to get started. Record a short sample, pick it in the studio, and write anything you want to hear in that voice. Everything runs locally: the model weights are cached in your browser and your audio never leaves the device.",
  },
];

export const textAtom = atom("");
export const textTokensAtom = atom(0);

export const synthParamsAtom = atomWithStorage<SynthParams>(STORAGE_KEYS.synthParams, {
  ...DEFAULT_SYNTH_PARAMS,
});

export const advancedOpenAtom = atom(false);

export type GenerationStatus = "idle" | "starting" | "running" | "stopping" | "done" | "error";

export type GenerationState = {
  requestId: string | null;
  status: GenerationStatus;
  error: string | null;
  stats: GenerationStats | null;
  /** Total samples received so far (per channel). */
  samples: number;
  /** True while the decoder is still streaming audio. */
  streaming: boolean;
};

const idleGeneration: GenerationState = {
  requestId: null,
  status: "idle",
  error: null,
  stats: null,
  samples: 0,
  streaming: false,
};

export const generationAtom = atom<GenerationState>(idleGeneration);

export type HistoryEntry = {
  id: string;
  createdAt: number;
  text: string;
  voiceLabel: string;
  voiceKind: "builtin" | "cloned";
  params: SynthParams;
  durationSec: number;
  firstAudioMs: number | null;
};

export const historyAtom = atomWithStorage<HistoryEntry[]>(STORAGE_KEYS.history, []);

export const lastGeneratedBlobUrlAtom = atom<string | null>(null);

/** Rough realtime factor: 1.0 means the model generates as fast as it plays. */
export const realtimeFactorAtom = atom<number | null>((get) => {
  const { stats } = get(generationAtom);
  if (!stats || stats.rtf === null) return null;
  const audioSec = get(generationAtom).samples / 48_000;
  if (audioSec <= 0) return null;
  return stats.rtf / audioSec;
});
