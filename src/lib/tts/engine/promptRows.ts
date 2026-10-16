/**
 * Prompt-row helpers shared by the CPU reference, the GPU kernel harness and the
 * tests, so a row means exactly the same thing everywhere:
 *
 *   column 0       the text token (e.g. `audio_assistant_slot_token_id` = 9)
 *   columns 1..16  the 16 audio codes; `audio_pad_token_id` (1024) means "empty"
 */

export const VOCAB_SIZE = 16384;
export const N_VQ = 16;
export const ROW_WIDTH = 17;
export const AUDIO_PAD_ID = 1024;
export const HIDDEN_SIZE = 768;
export const CODEBOOK_SIZE = 1024;

import type { BlockLocation } from "./weights";

export type { BlockLocation };

/** The single-row embedding, in JS. Mirrors `EMBED_ROWS`. */
export function embedRowFor(
  row: Int32Array | ArrayLike<number>,
  tables: { wte: BlockLocation; audio: BlockLocation[] },
  read: (block: BlockLocation, index: number) => number,
): Float32Array {
  const out = new Float32Array(HIDDEN_SIZE);
  const textId = row[0];
  for (let d = 0; d < HIDDEN_SIZE; d += 1) out[d] = read(tables.wte, textId * HIDDEN_SIZE + d);
  for (let k = 0; k < N_VQ; k += 1) {
    const id = row[k + 1];
    if (id === AUDIO_PAD_ID) continue;
    for (let d = 0; d < HIDDEN_SIZE; d += 1)
      out[d] += read(tables.audio[k], id * HIDDEN_SIZE + d);
  }
  return out;
}

/** Row fed back into the global transformer for one generated frame. */
export function frameRowFrom(assistantSlot: number, frame: number[]): Int32Array {
  const row = new Int32Array(ROW_WIDTH).fill(AUDIO_PAD_ID);
  row[0] = assistantSlot;
  for (let i = 0; i < Math.min(frame.length, N_VQ); i += 1) row[i + 1] = frame[i];
  return row;
}
