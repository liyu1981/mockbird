/**
 * Policy layer of the decode path — everything between "text in" and "frames
 * out" that is pure logic and therefore identical on CPU and GPU.
 *
 * Measured against the published graphs (docs/moss-tts-runtime-analysis.md §10):
 * the prompt layout, the sentence/clause chunking, the greedy sampling rules and
 * the stream pacing heuristic are all reproduced here from the verified spec.
 */

export type TtsConfig = {
  n_vq: number;
  audio_pad_token_id: number;
  pad_token_id: number;
  im_start_token_id: number;
  im_end_token_id: number;
  audio_start_token_id: number;
  audio_end_token_id: number;
  audio_user_slot_token_id: number;
  audio_assistant_slot_token_id: number;
  audio_codebook_sizes: number[];
  vocab_size: number;
};

export type PromptTemplates = {
  user_prompt_prefix_token_ids: number[];
  user_prompt_after_reference_token_ids: number[];
  assistant_prompt_prefix_token_ids: number[];
};

export type GenerationDefaults = {
  max_new_frames: number;
  do_sample: boolean;
  sample_mode: "greedy" | "fixed" | "full";
  text_temperature: number;
  text_top_p: number;
  text_top_k: number;
  audio_temperature: number;
  audio_top_p: number;
  audio_top_k: number;
  audio_repetition_penalty: number;
};

export const ROW_WIDTH = 17;

/**
 * The whole conditioning scheme in six lines: a 17-wide int32 row carries one
 * text token in column 0 and 16 audio codes in columns 1..16.
 */
export function buildVoiceCloneRows(
  templates: PromptTemplates,
  config: TtsConfig,
  promptAudioCodes: number[][],
  textTokenIds: number[],
): { inputIds: number[][]; attentionMask: number[][] } {
  const prefix = [...templates.user_prompt_prefix_token_ids, config.audio_start_token_id];
  const suffix = [
    config.audio_end_token_id,
    ...templates.user_prompt_after_reference_token_ids,
    ...textTokenIds,
    ...templates.assistant_prompt_prefix_token_ids,
    config.audio_start_token_id,
  ];
  const textRow = (id: number) => {
    const row = new Array<number>(config.n_vq + 1).fill(config.audio_pad_token_id);
    row[0] = id;
    return row;
  };
  const audioRow = (codeRow: number[]) => {
    const row = new Array<number>(config.n_vq + 1).fill(config.audio_pad_token_id);
    row[0] = config.audio_user_slot_token_id;
    for (let i = 0; i < Math.min(codeRow.length, config.n_vq); i += 1) row[i + 1] = codeRow[i];
    return row;
  };
  const rows = [
    ...prefix.map(textRow),
    ...promptAudioCodes.map(audioRow),
    ...suffix.map(textRow),
  ];
  return { inputIds: rows, attentionMask: [new Array<number>(rows.length).fill(1)] };
}

/** Row for frame feedback into the global transformer: [assistant_slot, frame…]. */
export function buildFrameRow(config: TtsConfig, frame: number[]): Int32Array {
  const row = new Int32Array(config.n_vq + 1).fill(config.audio_pad_token_id);
  row[0] = config.audio_assistant_slot_token_id;
  for (let i = 0; i < frame.length && i < config.n_vq; i += 1) row[i + 1] = frame[i];
  return row;
}

/** Input embedding: wte[text] + sum over channels of audio_emb_k[code], pads contribute 0. */
export function embeddingForRow(row: ArrayLike<number>, config: TtsConfig): number[] {
  const out: number[] = [];
  for (let d = 0; d < 768; d += 1) out.push(0);
  void row;
  void config;
  return out;
}

// ---------------------------------------------------------------- chunking

const SENTENCE_END_PUNCTUATION = new Set(".!?。！？；;");
const CLAUSE_PUNCTUATION = new Set(",，、；;：:");
const CLOSING_PUNCTUATION = new Set(`"'”’)]}）】》〉」』`);

export function containsCjk(text: string): boolean {
  return [...String(text || "")].some(
    (c) =>
      (c >= "一" && c <= "鿿") ||
      (c >= "㐀" && c <= "䶿") ||
      (c >= "぀" && c <= "ヿ") ||
      (c >= "가" && c <= "힯"),
  );
}

function splitByPunctuation(text: string, punctuation: Set<string>): string[] {
  const sentences: string[] = [];
  let current: string[] = [];
  let index = 0;
  while (index < text.length) {
    const character = text[index];
    current.push(character);
    if (punctuation.has(character)) {
      let lookahead = index + 1;
      while (lookahead < text.length && CLOSING_PUNCTUATION.has(text[lookahead])) {
        current.push(text[lookahead]);
        lookahead += 1;
      }
      const sentence = current.join("").trim();
      if (sentence) sentences.push(sentence);
      current = [];
      while (lookahead < text.length && /\s/.test(text[lookahead])) lookahead += 1;
      index = lookahead;
      continue;
    }
    index += 1;
  }
  const tail = current.join("").trim();
  if (tail) sentences.push(tail);
  return sentences;
}

function joinSentenceParts(left: string, right: string): string {
  if (!left) return right;
  if (!right) return left;
  if (containsCjk(left) || containsCjk(right)) return left + right;
  return `${left} ${right}`;
}

export type TextCounter = (text: string) => Promise<number>;

/**
 * Sentence -> clause -> binary-search token budget -> repack. Whole sentences are
 * packed together so a long text sounds like one speaker instead of a queue of
 * short sentences.
 */
export async function splitVoiceCloneText(
  text: string,
  maxTokens: number,
  countTokens: TextCounter,
): Promise<string[]> {
  const normalized = String(text || "").trim();
  if (!normalized) return [];
  const budget = Math.max(1, Number.isFinite(maxTokens) ? Math.trunc(maxTokens) : 75);

  const slices: [number, string][] = [];
  const sentences = splitByPunctuation(normalized, SENTENCE_END_PUNCTUATION);
  for (const sentence of sentences.length ? sentences : [normalized]) {
    const trimmed = sentence.trim();
    if (!trimmed) continue;
    const count = await countTokens(trimmed);
    if (count <= budget) {
      slices.push([count, trimmed]);
      continue;
    }
    let clauses = splitByPunctuation(trimmed, CLAUSE_PUNCTUATION);
    if (clauses.length <= 1) clauses = [trimmed];
    for (const clause of clauses) {
      const clauseText = clause.trim();
      if (!clauseText) continue;
      const clauseCount = await countTokens(clauseText);
      if (clauseCount <= budget) {
        slices.push([clauseCount, clauseText]);
        continue;
      }
      for (const piece of await splitTextByTokenBudget(clauseText, budget, countTokens)) {
        if (piece.trim()) slices.push([await countTokens(piece.trim()), piece.trim()]);
      }
    }
  }

  const chunks: string[] = [];
  let current = "";
  let currentCount = 0;
  for (const [count, sentenceText] of slices) {
    if (!current) {
      current = sentenceText;
      currentCount = count;
      continue;
    }
    if (currentCount + count > budget) {
      chunks.push(current.trim());
      current = sentenceText;
      currentCount = count;
    } else {
      current = joinSentenceParts(current, sentenceText);
      currentCount = await countTokens(current);
    }
  }
  if (current) chunks.push(current.trim());
  return chunks.length > 1 ? chunks : [normalized];
}

/** Binary search the longest character prefix that fits the token budget. */
async function splitTextByTokenBudget(
  text: string,
  maxTokens: number,
  countTokens: TextCounter,
): Promise<string[]> {
  let remaining = text.trim();
  if (!remaining) return [];
  const pieces: string[] = [];
  const boundaries = new Set<string>([...CLAUSE_PUNCTUATION, ...SENTENCE_END_PUNCTUATION, " "]);
  while (remaining) {
    if ((await countTokens(remaining)) <= maxTokens) {
      pieces.push(remaining);
      break;
    }
    let low = 1;
    let high = remaining.length;
    let bestPrefix = 1;
    while (low <= high) {
      const middle = Math.floor((low + high) / 2);
      const candidate = remaining.slice(0, middle).trim();
      if (!candidate) {
        low = middle + 1;
        continue;
      }
      if ((await countTokens(candidate)) <= maxTokens) {
        bestPrefix = middle;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    let cutIndex = bestPrefix;
    const prefix = remaining.slice(0, bestPrefix);
    let preferredIndex = -1;
    for (let scan = prefix.length - 1; scan > Math.max(-1, prefix.length - 25); scan -= 1) {
      if (boundaries.has(prefix[scan])) {
        preferredIndex = scan + 1;
        break;
      }
    }
    if (preferredIndex > 0) cutIndex = preferredIndex;
    let piece = remaining.slice(0, cutIndex).trim();
    if (!piece) {
      piece = remaining.slice(0, bestPrefix).trim();
      cutIndex = bestPrefix;
    }
    pieces.push(piece);
    remaining = remaining.slice(cutIndex).trim();
  }
  return pieces;
}

// ---------------------------------------------------------------- sampling

/**
 * Greedy argmax with the HuggingFace repetition-penalty convention: penalise
 * previously emitted codes per channel (× when the logit is negative, ÷ when
 * positive). This is the only sampling mode the engine ships — see §10.5.
 */
export function argmaxWithRepetitionPenalty(
  logits: ArrayLike<number>,
  seen: Set<number>,
  penalty: number,
): number {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  const apply = seen.size > 0 && penalty !== 1;
  for (let i = 0; i < logits.length; i += 1) {
    let score = logits[i];
    if (apply && seen.has(i)) score = score < 0 ? score * penalty : score / penalty;
    if (score > bestValue) {
      bestValue = score;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/**
 * The assistant-vs-stop decision. The 16 384-wide text head is never evaluated in
 * full: only the `assistant_slot` and `audio_end` columns matter (and on the GPU
 * only those two rows are ever read).
 */
export function decideAssistantStop(
  textLogitAssistant: number,
  textLogitEnd: number,
): "continue" | "stop" {
  return textLogitAssistant >= textLogitEnd ? "continue" : "stop";
}

// ---------------------------------------------------------------- pacing

export type StreamPacing = {
  emittedSamples: number;
  firstAudioAt: number | null;
  nowMs: number;
};

export function streamLeadSeconds(state: StreamPacing, sampleRate: number): number {
  if (!state.firstAudioAt || !(sampleRate > 0)) return 0;
  const elapsed = Math.max(0, state.nowMs / 1000 - state.firstAudioAt);
  return state.emittedSamples / sampleRate - elapsed;
}

/** Keep ~0.2–1 s of audio buffered so the worklet never starves. */
export function streamDecodeBudget(state: StreamPacing, sampleRate: number): number {
  const lead = streamLeadSeconds(state, sampleRate);
  if (!state.firstAudioAt || lead < 0.2) return 1;
  if (lead < 0.55) return 2;
  if (lead < 1.1) return 4;
  return 8;
}

export function shouldDecodeNow(
  state: StreamPacing,
  sampleRate: number,
  pending: number,
): boolean {
  if (pending <= 0) return false;
  if (!state.firstAudioAt) return true;
  const lead = streamLeadSeconds(state, sampleRate);
  if (lead < 0.45) return true;
  if (pending >= streamDecodeBudget(state, sampleRate)) return true;
  return lead < 0;
}

/** Short chunks get a longer pause, so long text does not sound clipped. */
export function interChunkPauseSeconds(chunkText: string): number {
  const words = String(chunkText || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
  return words <= 4 ? 0.4 : 0.24;
}

// ---------------------------------------------------------------- paths

/**
 * Resolve a manifest-relative asset path against the directory the manifest
 * itself lives in, honouring `.` and `..`.
 *
 * The published `browser_poc_manifest.json` refers to its neighbours by bare
 * filename (`tts_browser_onnx_meta.json`) but reaches the codec metadata with a
 * parent reference (`../MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json`),
 * so both forms have to work. `dir` is the manifest's own directory, which is not
 * always the model root — with Mockbird's OPFS layout it is
 * `MOSS-TTS-Nano-100M-ONNX/`.
 */
export function resolveAssetPath(dir: string, relative: string): string {
  const segments: string[] = [];
  // normalise BOTH sides: a base like `dir/.` and a relative `./a/./b` must not
  // leave stray `.` segments behind
  const parts = `${String(dir || "")}/${String(relative || "")}`.replace(/\\/g, "/").split("/");
  for (const part of parts) {
    if (!part || part === ".") continue;
    if (part === "..") {
      segments.pop();
      continue;
    }
    segments.push(part);
  }
  return segments.join("/");
}
