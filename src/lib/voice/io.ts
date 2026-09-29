/**
 * Voice import/export. Runs on the main thread (OPFS is available there too) so
 * the Studio and Voices pages can move voices around without the engine worker.
 */

import type { StoredVoiceMeta } from "@/lib/model/opfsStore";
import { listVoices, readVoiceCodes, readVoiceMeta, writeVoice } from "@/lib/model/opfsStore";

export const VOICE_FILE_FORMAT = "mockbird-voice@1";

export type VoiceFile = {
  format: typeof VOICE_FILE_FORMAT;
  meta: Omit<StoredVoiceMeta, "id" | "createdAt"> & {
    id?: string;
    createdAt?: number;
  };
  /** Base64 of the little-endian Uint16 code array. */
  codes: string;
};

export function codesToBase64(codes: Uint16Array): string {
  const bytes = new Uint8Array(codes.byteLength);
  new Uint8Array(codes.buffer, codes.byteOffset, codes.byteLength).forEach((value, index) => {
    bytes[index] = value;
  });
  // Chunked to stay well below the argument limit of String.fromCharCode.
  let binary = "";
  const chunk = 0x8000;
  for (let index = 0; index < bytes.length; index += chunk) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunk));
  }
  return btoa(binary);
}

export function base64ToCodes(base64: string): Uint16Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Uint16Array(bytes.buffer);
}

export async function readVoiceCodesForExport(id: string): Promise<Uint16Array> {
  return readVoiceCodes(id);
}

export async function exportVoiceFile(id: string): Promise<VoiceFile> {
  const meta = await readVoiceMeta(id);
  if (!meta) throw new Error("Voice not found.");
  const codes = await readVoiceCodes(id);
  return { format: VOICE_FILE_FORMAT, meta, codes: codesToBase64(codes) };
}

export async function importVoiceFile(payload: string): Promise<StoredVoiceMeta> {
  const parsed = JSON.parse(payload) as VoiceFile;
  if (parsed?.format !== VOICE_FILE_FORMAT) {
    throw new Error("Not a Mockbird voice file.");
  }
  const codes = base64ToCodes(parsed.codes);
  const meta: StoredVoiceMeta = {
    ...(parsed.meta as StoredVoiceMeta),
    id: `v_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
    createdAt: Date.now(),
    source: "import",
    frames: Math.floor(codes.length / (parsed.meta.numQuantizers || 16)),
  };
  await writeVoice(meta, codes);
  return meta;
}

export { listVoices };
