"use client";

import { atom } from "jotai";

import type { StoredVoiceMeta } from "@/lib/model/opfsStore";
import type { BuiltinVoice } from "@/lib/worker/protocol";

export type VoiceSelection =
  | { kind: "builtin"; id: string; label: string; group: string }
  | { kind: "cloned"; id: string; label: string; group: string };

export const clonedVoicesAtom = atom<StoredVoiceMeta[]>([]);
export const builtinVoicesAtom = atom<BuiltinVoice[]>([]);
export const voicesLoadingAtom = atom(false);
export const selectedVoiceAtom = atom<VoiceSelection | null>(null);

export const draftNameAtom = atom("");
export const draftSourceAtom = atom<"recording" | "upload" | null>(null);
export const draftPcmAtom = atom<{
  channels: Float32Array[];
  sampleRate: number;
  durationSec: number;
} | null>(null);
export const draftAnalysisAtom = atom<{
  peak: number;
  rms: number;
  clipped: number;
} | null>(null);
export const cloningBusyAtom = atom(false);
export const cloningErrorAtom = atom<string | null>(null);

export const allVoicesAtom = atom<VoiceSelection[]>((get) => {
  const builtin = get(builtinVoicesAtom).map(
    (voice): VoiceSelection => ({
      kind: "builtin",
      id: voice.voice,
      label: voice.display_name,
      group: voice.group,
    }),
  );
  const cloned = get(clonedVoicesAtom).map(
    (voice): VoiceSelection => ({
      kind: "cloned",
      id: voice.id,
      label: voice.name,
      group: voice.source === "import" ? "Imported" : "Your recording",
    }),
  );
  return [...cloned, ...builtin];
});

export const selectedVoiceMetaAtom = atom<StoredVoiceMeta | null>((get) => {
  const selected = get(selectedVoiceAtom);
  if (selected?.kind !== "cloned") return null;
  return get(clonedVoicesAtom).find((voice) => voice.id === selected.id) ?? null;
});
