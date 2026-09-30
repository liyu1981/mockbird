/**
 * Long-form text helpers shared by the composer, the engine provider and the UI.
 *
 * The model itself only speaks ~75 tokens at a time (see `voice_clone_max_text_tokens`),
 * so anything longer has to be split into sentence-aligned pieces and spoken one
 * after another. These helpers describe what that pipeline looks like from the
 * main thread; the actual split is done by the vendored runtime in the worker,
 * because it owns the SentencePiece tokenizer.
 */

/**
 * Silence inserted between two spoken chunks, mirroring the upstream runtime's
 * inter-chunk pause so the result does not sound like a queue of clips.
 */
export const INTER_CHUNK_GAP_SHORT_SECONDS = 0.4;
export const INTER_CHUNK_GAP_LONG_SECONDS = 0.24;

export function interChunkGapSeconds(chunkText: string): number {
  const words = String(chunkText ?? "")
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
  return words <= 4 ? INTER_CHUNK_GAP_SHORT_SECONDS : INTER_CHUNK_GAP_LONG_SECONDS;
}

/** Rough spoken length of one chunk, used for the "≈ 42s" estimate in the UI. */
export function estimateSpeechSeconds(chunkText: string): number {
  const text = String(chunkText ?? "");
  const cjk = (text.match(/[\u3400-\u9fff\u3040-\u30ff]/g) ?? []).length;
  const words = text
    .replace(/[\u3400-\u9fff\u3040-\u30ff]/g, " ")
    .split(/\s+/)
    .filter(Boolean).length;
  // ~4.5 CJK characters or ~2.8 English words per second, with a floor so an
  // abbreviation ("OK.") does not estimate as 0.1s.
  return Math.max(0.8, cjk / 4.5 + words / 2.8);
}

/** Total estimate for a run of chunks, including the gaps between them. */
export function estimateRunSeconds(chunks: string[]): number {
  return chunks.reduce((total, chunk, index) => {
    const gap = index === 0 ? 0 : interChunkGapSeconds(chunk);
    return total + estimateSpeechSeconds(chunk) + gap;
  }, 0);
}

/** Label for the composer, e.g. "3 chunks · ≈ 21s". */
export function describeChunks(chunks: string[]): string {
  if (chunks.length === 0) return "";
  const seconds = estimateRunSeconds(chunks);
  const rounded = seconds < 90 ? `${Math.round(seconds)}s` : `${Math.round(seconds / 60)}m`;
  return chunks.length === 1
    ? `1 clip · ≈ ${rounded}`
    : `${chunks.length} chunks · ≈ ${rounded}`;
}
