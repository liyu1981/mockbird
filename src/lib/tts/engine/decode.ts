/**
 * The decode path: the global single-row transformer step and the 17-step local
 * frame head, with the token ids staying on the device.
 *
 * Why it is shaped like this
 * -------------------------
 * Per generated frame the local head runs 17 *sequential* transformer steps (one
 * continue/stop decision plus 16 audio-code draws), each a full d=768 layer. The
 * naive port would read 1024 logits back 16 times and re-upload the argmax — 17
 * CPU round-trips per frame. Instead:
 *
 *   - the argmax-with-penalty kernel writes each winner into the `tokens` buffer
 *     on the device and sets its bit in the per-channel `seen` mask,
 *   - the next step's embedding kernel reads that id straight out of `tokens`,
 *
 * so only two small reads leave the GPU per frame: the 1-element continue flag
 * after step 0, and the 16 token ids after step 16. That is the schedule in
 * §9.4 ("token ids never leave the GPU"), and it is what keeps a frame inside
 * its 80 ms budget.
 *
 * The global step reuses the *prefill* KV ring — one `GPUBuffer` shared with
 * `PrefillRunner` — so a frame attends over the 195 prompt rows plus every frame
 * appended since, which is exactly the tensor the batched prefill wrote. Slots
 * are `position % maxSlots`; both rings are sized so they never wrap (global:
 * prefill_rows + max_new_frames, local: 17).
 */

import {
  ADD_RESIDUAL,
  ARGMAX_PENALTY,
  ATTEND_RING,
  EMBED_ROW_LOCAL,
  GELU_TANH,
  GEMV,
  GEMV_NOB,
  LAYERNORM_1,
  ROPE_SPLIT_RING,
  TEXT_HEAD_DECIDE,
} from "./kernels-decode";
import type { LayerWeights } from "./prefill";
import type { BlockLocation, GpuWeights } from "./weights";

const HIDDEN = 768;
const QKV = 2304;
const FFN = 3072;
const HEADS = 12;
const LOCAL_SLOTS = 17;
const CODES = 1024;
/** 12 scalars + 16 table offsets = 28 u32, padded to 48 so the block stays aligned */
const UNIFORM_U32 = 48;

type Params = {
  n: number;
  k: number;
  wOff: number;
  bOff: number;
  dOff: number;
  sOff: number;
  aOff: number;
  cOff: number;
  p0: number;
  p1: number;
  p2: number;
  p3: number;
  /** the 16 audio-embedding tables, element offsets inside their chunk */
  audio?: number[];
};

/**
 * Float bits for a uniform field. The Params struct is u32-typed, so a
 * fractional value (the 1.2 repetition penalty) cannot survive `>>> 0` — it
 * would truncate to 1 and silently disable the penalty.
 */
const bitsOf = (value: number): number => {
  const scratch = new Float32Array(1);
  scratch[0] = value;
  return new Uint32Array(scratch.buffer)[0];
};

const params = (p: Partial<Params> & { n: number; k: number }): Params => ({
  wOff: 0,
  bOff: 0,
  dOff: 0,
  sOff: 0,
  aOff: 0,
  cOff: 0,
  p0: 0,
  p1: 0,
  p2: 0,
  p3: 0,
  ...p,
});

type KernelName =
  | "embedRowLocal"
  | "layerNorm1"
  | "gemv"
  | "gemvNoBias"
  | "geluTanh"
  | "ropeSplitRing"
  | "addResidual"
  | "argmaxPenalty"
  | "textHeadDecide"
  | "attendRing";

const KERNELS: Record<KernelName, string> = {
  embedRowLocal: EMBED_ROW_LOCAL,
  layerNorm1: LAYERNORM_1,
  gemv: GEMV,
  gemvNoBias: GEMV_NOB,
  geluTanh: GELU_TANH,
  ropeSplitRing: ROPE_SPLIT_RING,
  addResidual: ADD_RESIDUAL,
  argmaxPenalty: ARGMAX_PENALTY,
  textHeadDecide: TEXT_HEAD_DECIDE,
  attendRing: ATTEND_RING,
};

/** The local head's weights, resolved once from the local graph's table. */
export type LocalHeadWeights = {
  cAttn: BlockLocation;
  cAttnBias: BlockLocation;
  cProj: BlockLocation;
  cProjBias: BlockLocation;
  ln1W: BlockLocation;
  ln1B: BlockLocation;
  ln2W: BlockLocation;
  ln2B: BlockLocation;
  fcIn: BlockLocation;
  fcInBias: BlockLocation;
  fcOut: BlockLocation;
  fcOutBias: BlockLocation;
  lnFW: BlockLocation;
  lnFB: BlockLocation;
  /** text_lm_head [768, 16384] — only 2 of its columns are ever read */
  textHead: BlockLocation;
  /** 16 × audio_lm_heads[k] [768, 1024] */
  audioHeads: BlockLocation[];
};

export type LocalHeadConfig = {
  audioEndTokenId: number;
  audioAssistantSlotTokenId: number;
  repetitionPenalty: number;
};

/** Embedding tables the global and local heads share conceptually. */
export type EmbedTables = { wte: BlockLocation; audio: BlockLocation[] };

/**
 * One d=768 layer — the arithmetic is identical for the global transformer's 12
 * blocks and the local head's 1; only the weights and the KV ring differ.
 */
type StepLayer = {
  cAttn: BlockLocation;
  cAttnBias: BlockLocation;
  cProj: BlockLocation;
  cProjBias: BlockLocation;
  ln1W: BlockLocation;
  ln1B: BlockLocation;
  ln2W: BlockLocation;
  ln2B: BlockLocation;
  fcIn: BlockLocation;
  fcInBias: BlockLocation;
  fcOut: BlockLocation;
  fcOutBias: BlockLocation;
};

const asStepLayer = (w: LayerWeights | LocalHeadWeights): StepLayer => ({
  cAttn: w.cAttn,
  cAttnBias: w.cAttnBias,
  cProj: w.cProj,
  cProjBias: w.cProjBias,
  ln1W: w.ln1W,
  ln1B: w.ln1B,
  ln2W: w.ln2W,
  ln2B: w.ln2B,
  fcIn: w.fcIn,
  fcInBias: w.fcInBias,
  fcOut: w.fcOut,
  fcOutBias: w.fcOutBias,
});

export type DecodeBuffers = {
  x: GPUBuffer;
  norm: GPUBuffer;
  qkv: GPUBuffer;
  ffn: GPUBuffer;
  /** attention output (context) for the current row */
  ctx: GPUBuffer;
  /** GEMV scratch: the local head's ln_f output and the per-channel logits */
  proj: GPUBuffer;
  hidden: GPUBuffer;
  rowIds: GPUBuffer;
  localKv: GPUBuffer;
  tokens: GPUBuffer;
  seen: GPUBuffer;
  flag: GPUBuffer;
};

export class DecodeRunner {
  // NB: explicit fields, not parameter properties — Node's type stripping
  // (tests/gpu) rejects parameter properties.
  private readonly device: GPUDevice;
  /** the global transformer's weight pool */
  private readonly weights: GpuWeights;
  /**
   * The local head's OWN weight pool. Its `BlockLocation`s come from a separate
   * upload, so their chunk/offset pair only means anything against that pool:
   * binding a local block against the global chunks reads unrelated bytes and
   * produces a plausible-but-wrong hidden state.
   */
  private readonly localWeights: GpuWeights;
  private readonly local: LocalHeadWeights;
  private readonly localConfig: LocalHeadConfig;
  private readonly kv: GPUBuffer;
  private readonly maxSlots: number;
  readonly buffers: DecodeBuffers;
  private readonly pipelines = new Map<KernelName, GPUComputePipeline>();
  private readonly attendPipeline: GPUComputePipeline;
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly attendBindGroups = new Map<string, GPUBindGroup>();
  private readonly uniforms = new Map<string, GPUBuffer>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 1;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private pipelineLayout: GPUPipelineLayout | null = null;
  private attendBindGroupLayout: GPUBindGroupLayout | null = null;
  private attendPipelineLayout: GPUPipelineLayout | null = null;

  constructor(
    device: GPUDevice,
    weights: GpuWeights,
    localWeights: GpuWeights,
    kv: GPUBuffer,
    maxSlots: number,
    local: LocalHeadWeights,
    localConfig: LocalHeadConfig,
  ) {
    this.device = device;
    this.weights = weights;
    this.localWeights = localWeights;
    this.kv = kv;
    this.maxSlots = maxSlots;
    this.local = local;
    this.localConfig = localConfig;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (label: string, elements: number) =>
      device.createBuffer({ size: elements * 4, usage, label });
    this.buffers = {
      x: make("dec-x", HIDDEN),
      norm: make("dec-norm", HIDDEN),
      qkv: make("dec-qkv", QKV),
      ffn: make("dec-ffn", FFN),
      ctx: make("dec-ctx", HIDDEN),
      proj: make("dec-proj", FFN),
      hidden: make("dec-hidden", HIDDEN),
      rowIds: make("dec-row", 17),
      localKv: make("dec-local-kv", 2 * LOCAL_SLOTS * HIDDEN),
      tokens: make("dec-tokens", LOCAL_SLOTS),
      seen: make("dec-seen", 16 * 32),
      flag: make("dec-flag", 1),
    };
    this.attendPipeline = device.createComputePipeline({
      layout: this.makeAttendLayout(),
      compute: {
        module: device.createShaderModule({ code: KERNELS.attendRing, label: "attendRing" }),
        entryPoint: "main",
      },
      label: "attendRing",
    });
    this.clearSeen();
  }

  /** Zero the repetition-penalty seen-sets: a new utterance starts clean. */
  clearSeen(): void {
    this.device.queue.writeBuffer(this.buffers.seen, 0, new Uint32Array(16 * 32));
    this.device.queue.writeBuffer(this.buffers.flag, 0, new Float32Array([0]));
    this.device.queue.writeBuffer(this.buffers.tokens, 0, new Float32Array(LOCAL_SLOTS).buffer);
  }

  // ------------------------------------------------------------- plumbing
  private makeLayout(): GPUBindGroupLayout {
    if (this.bindGroupLayout) return this.bindGroupLayout;
    const v = GPUShaderStage.COMPUTE;
    const storage = (type: GPUBufferBindingType): GPUBufferBindingLayout => ({
      type,
      hasDynamicOffset: false,
      minBindingSize: 4,
    });
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: v, buffer: { type: "uniform" } },
        { binding: 1, visibility: v, buffer: storage("storage") },
        { binding: 2, visibility: v, buffer: storage("read-only-storage") },
        { binding: 3, visibility: v, buffer: storage("storage") },
        // the bias parameter (see kernels-decode DECODE_BINDINGS) and the u32
        // seen-mask the argmax kernel keeps its 1024-bit penalty history in
        { binding: 4, visibility: v, buffer: storage("read-only-storage") },
        { binding: 5, visibility: v, buffer: storage("storage") },
      ],
    });
    this.pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });
    return this.bindGroupLayout;
  }

  private makeAttendLayout(): GPUPipelineLayout {
    if (this.attendPipelineLayout) return this.attendPipelineLayout;
    const v = GPUShaderStage.COMPUTE;
    const storage = (type: GPUBufferBindingType): GPUBufferBindingLayout => ({
      type,
      hasDynamicOffset: false,
      minBindingSize: 4,
    });
    this.attendBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: v, buffer: { type: "uniform" } },
        { binding: 1, visibility: v, buffer: storage("storage") },
        { binding: 2, visibility: v, buffer: storage("read-only-storage") },
        { binding: 3, visibility: v, buffer: storage("storage") },
        { binding: 4, visibility: v, buffer: storage("storage") },
      ],
    });
    this.attendPipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.attendBindGroupLayout],
    });
    return this.attendPipelineLayout;
  }

  private pipeline(name: KernelName): GPUComputePipeline {
    const cached = this.pipelines.get(name);
    if (cached) return cached;
    this.makeLayout();
    const pipeline = this.device.createComputePipeline({
      layout: this.pipelineLayout as GPUPipelineLayout,
      compute: {
        module: this.device.createShaderModule({ code: KERNELS[name], label: name }),
        entryPoint: "main",
      },
      label: name,
    });
    this.pipelines.set(name, pipeline);
    return pipeline;
  }

  private bufferId(buffer: GPUBuffer): number {
    let id = this.bufferIds.get(buffer);
    if (id === undefined) {
      id = this.nextBufferId++;
      this.bufferIds.set(buffer, id);
    }
    return id;
  }

  private bind(
    name: string,
    uniforms: GPUBuffer,
    input: GPUBuffer,
    weights: GPUBuffer,
    output: GPUBuffer,
    bias: GPUBuffer,
    mask: GPUBuffer,
  ): GPUBindGroup {
    // the key must cover every binding: a bind group captures the buffers, so
    // reusing one across a dispatch that writes elsewhere computes on stale data
    const key = `${name}|${this.bufferId(uniforms)}|${this.bufferId(input)}|${this.bufferId(
      weights,
    )}|${this.bufferId(output)}|${this.bufferId(bias)}|${this.bufferId(mask)}`;
    const cached = this.bindGroups.get(key);
    if (cached) return cached;
    const group = this.device.createBindGroup({
      layout: this.makeLayout(),
      entries: [
        { binding: 0, resource: { buffer: uniforms } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: weights } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: bias } },
        { binding: 5, resource: { buffer: mask } },
      ],
    });
    this.bindGroups.set(key, group);
    return group;
  }

  private attendBind(
    uniforms: GPUBuffer,
    q: GPUBuffer,
    weights: GPUBuffer,
    ring: GPUBuffer,
    ctx: GPUBuffer,
  ): GPUBindGroup {
    const key = `att|${this.bufferId(uniforms)}|${this.bufferId(q)}|${this.bufferId(
      weights,
    )}|${this.bufferId(ring)}|${this.bufferId(ctx)}`;
    const cached = this.attendBindGroups.get(key);
    if (cached) return cached;
    const group = this.device.createBindGroup({
      layout: this.attendBindGroupLayout as GPUBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniforms } },
        { binding: 1, resource: { buffer: q } },
        { binding: 2, resource: { buffer: weights } },
        { binding: 3, resource: { buffer: ring } },
        { binding: 4, resource: { buffer: ctx } },
      ],
    });
    this.attendBindGroups.set(key, group);
    return group;
  }

  private uniform(slot: string): GPUBuffer {
    let buffer = this.uniforms.get(slot);
    if (!buffer) {
      buffer = this.device.createBuffer({
        size: UNIFORM_U32 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: `dec-params-${slot}`,
      });
      this.uniforms.set(slot, buffer);
    }
    return buffer;
  }

  private params(slot: string, p: Params): GPUBuffer {
    const buffer = this.uniform(slot);
    const array = new Uint32Array(UNIFORM_U32);
    array[0] = p.n >>> 0;
    array[1] = p.k >>> 0;
    array[2] = p.wOff >>> 0;
    array[3] = p.bOff >>> 0;
    array[4] = p.dOff >>> 0;
    array[5] = p.sOff >>> 0;
    array[6] = p.aOff >>> 0;
    array[7] = p.cOff >>> 0;
    array[8] = p.p0 >>> 0;
    array[9] = p.p1 >>> 0;
    array[10] = p.p2 >>> 0;
    array[11] = p.p3 >>> 0;
    if (p.audio) {
      for (let i = 0; i < Math.min(16, p.audio.length); i += 1)
        array[12 + i] = p.audio[i] >>> 0;
    }
    this.device.queue.writeBuffer(buffer, 0, array);
    return buffer;
  }

  private async readback(buffer: GPUBuffer, bytes: number): Promise<Float32Array> {
    const staging = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "decode-read" });
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0, bytes));
    staging.unmap();
    staging.destroy();
    return out;
  }

  // --------------------------------------------------------- the transformer
  /**
   * Encode one d=768 layer over a KV ring: LN1 → QKV → RoPE+cache → attention →
   * proj → residual → LN2 → MLP → residual, in that order per layer (running all
   * the attention first and all the MLPs after is one of the three bugs the CPU
   * reference caught, §10.5).
   *
   * `valid` is the number of live ring slots, i.e. `position + 1`.
   */
  private runLayer(
    pass: GPUComputePassEncoder,
    w: StepLayer,
    pool: GpuWeights,
    ring: GPUBuffer,
    keyBase: number,
    valueBase: number,
    valid: number,
    tag: string,
    stream: GPUBuffer,
  ): void {
    const b = this.buffers;
    const run = (
      name: KernelName,
      slot: string,
      p: Params,
      chunk: number,
      input: GPUBuffer,
      output: GPUBuffer,
      workgroups: number,
      biasChunk?: number,
    ) => {
      const uniforms = this.params(`${tag}:${slot}`, p);
      const weights = pool.chunks[chunk];
      const bias = pool.chunks[biasChunk ?? chunk];
      pass.setPipeline(this.pipeline(name));
      pass.setBindGroup(
        0,
        this.bind(name, uniforms, input, weights, output, bias, this.buffers.seen),
      );
      pass.dispatchWorkgroups(workgroups);
    };

    run(
      "layerNorm1",
      "ln1",
      params({ n: HIDDEN, k: 0, wOff: w.ln1W.offset, bOff: w.ln1B.offset }),
      w.ln1W.chunk,
      stream,
      b.norm,
      1,
      w.ln1B.chunk,
    );
    run(
      "gemv",
      "qkv",
      params({ n: QKV, k: HIDDEN, wOff: w.cAttn.offset, bOff: w.cAttnBias.offset }),
      w.cAttn.chunk,
      b.norm,
      b.qkv,
      QKV / 64,
      w.cAttnBias.chunk,
    );
    run(
      "ropeSplitRing",
      "rope",
      params({ n: HIDDEN, k: 0, sOff: keyBase, aOff: valueBase, p0: valid - 1 }),
      w.cAttn.chunk,
      b.qkv,
      ring,
      HEADS,
    );
    const attendUniforms = this.params(
      `${tag}:att`,
      params({
        n: HIDDEN,
        k: 0,
        sOff: keyBase,
        aOff: valueBase,
        p0: valid - 1,
        p1: valid,
      }),
    );
    pass.setPipeline(this.attendPipeline);
    pass.setBindGroup(
      0,
      this.attendBind(attendUniforms, b.qkv, pool.chunks[w.cAttn.chunk], ring, b.ctx),
    );
    pass.dispatchWorkgroups(HEADS);
    run(
      "gemv",
      "proj",
      params({ n: HIDDEN, k: HIDDEN, wOff: w.cProj.offset, bOff: w.cProjBias.offset }),
      w.cProj.chunk,
      b.ctx,
      b.proj,
      HIDDEN / 64,
      w.cProjBias.chunk,
    );
    run(
      "addResidual",
      "resid0",
      params({ n: HIDDEN, k: 0 }),
      w.cProj.chunk,
      b.proj,
      stream,
      Math.ceil(HIDDEN / 256),
    );
    run(
      "layerNorm1",
      "ln2",
      params({ n: HIDDEN, k: 0, wOff: w.ln2W.offset, bOff: w.ln2B.offset }),
      w.ln2W.chunk,
      stream,
      b.norm,
      1,
      w.ln2B.chunk,
    );
    run(
      "gemv",
      "fcin",
      params({ n: FFN, k: HIDDEN, wOff: w.fcIn.offset, bOff: w.fcInBias.offset }),
      w.fcIn.chunk,
      b.norm,
      b.ffn,
      FFN / 64,
      w.fcInBias.chunk,
    );
    run(
      "geluTanh",
      "gelu",
      params({ n: FFN, k: 0 }),
      w.fcIn.chunk,
      b.norm,
      b.ffn,
      Math.ceil(FFN / 256),
    );
    run(
      "gemv",
      "fcout",
      params({ n: HIDDEN, k: FFN, wOff: w.fcOut.offset, bOff: w.fcOutBias.offset }),
      w.fcOut.chunk,
      b.ffn,
      b.proj,
      HIDDEN / 64,
      w.fcOutBias.chunk,
    );
    run(
      "addResidual",
      "resid1",
      params({ n: HIDDEN, k: 0 }),
      w.fcOut.chunk,
      b.proj,
      stream,
      Math.ceil(HIDDEN / 256),
    );
  }

  /**
   * Encode the row embedding. The 16 audio tables are 3 MB each and the upload
   * rolls chunks of ~128 MB, so they land in several chunks and each needs its
   * own binding — the kernel therefore takes a *contiguous channel range*, which
   * works because the tables sit in sidecar order.
   */
  private embedRow(
    pass: GPUComputePassEncoder,
    mode: 0 | 1 | 2,
    embed: EmbedTables,
    rowBuffer: GPUBuffer,
    tag: string,
    opts: { channel?: number; textId?: number } = {},
  ): void {
    if (mode !== 0) {
      const chunk = mode === 1 ? embed.wte.chunk : embed.audio[opts.channel ?? 0].chunk;
      const offsets = embed.audio.map((block) => block.offset);
      const uniforms = this.params(
        `${tag}:embed`,
        params({
          n: HIDDEN,
          k: 0,
          wOff: embed.wte.offset,
          aOff: mode,
          p0: opts.channel ?? 0,
          // mode 1 embeds a text id, mode 2 reads the previous channel's token
          p1: mode === 1 ? (opts.textId ?? 0) : (opts.channel ?? 0),
          p2: 0,
          p3: 0,
          audio: offsets,
        }),
      );
      pass.setPipeline(this.pipeline("embedRowLocal"));
      pass.setBindGroup(
        0,
        this.bind(
          "embedRowLocal",
          uniforms,
          rowBuffer,
          this.weights.chunks[chunk],
          this.buffers.x,
          this.weights.chunks[chunk],
          this.buffers.seen,
        ),
      );
      pass.dispatchWorkgroups(Math.ceil(HIDDEN / 256));
      return;
    }
    // group the channels by the chunk their table landed in, keeping runs
    let from = 0;
    while (from < embed.audio.length) {
      const chunk = embed.audio[from].chunk;
      let to = from;
      while (to < embed.audio.length && embed.audio[to].chunk === chunk) to += 1;
      const uniforms = this.params(
        `${tag}:embed:${from}`,
        params({
          n: HIDDEN,
          k: 0,
          wOff: embed.wte.chunk === chunk ? embed.wte.offset : 0,
          aOff: 0,
          p1: from,
          p2: to - 1,
          p3: from === 0 ? 0 : 1,
          audio: embed.audio.map((block) => block.offset),
        }),
      );
      pass.setPipeline(this.pipeline("embedRowLocal"));
      pass.setBindGroup(
        0,
        this.bind(
          "embedRowLocal",
          uniforms,
          rowBuffer,
          this.weights.chunks[chunk],
          this.buffers.x,
          this.weights.chunks[chunk],
          this.buffers.seen,
        ),
      );
      pass.dispatchWorkgroups(Math.ceil(HIDDEN / 256));
      from = to;
    }
  }

  /**
   * One global decode step at `position`: embed the 17-wide row, run the 12
   * layers over the prefill's KV ring (appending this row's K/V), and return the
   * ln_f hidden state the local head consumes next.
   */
  async globalStep(
    row: Int32Array,
    position: number,
    layers: LayerWeights[],
    lnF: { w: BlockLocation; b: BlockLocation },
    embed: EmbedTables,
  ): Promise<Float32Array> {
    if (position >= this.maxSlots) {
      throw new Error(
        `decode position ${position} exceeds the KV ring (${this.maxSlots} slots); ` +
          `it must be sized prefill_rows + max_new_frames`,
      );
    }
    const b = this.buffers;
    // stage the row as f32 *values*: every id is < 2^24, so the round trip is
    // exact and the kernel can read it as a number instead of a bit pattern
    const staged = new Float32Array(17);
    for (let i = 0; i < 17; i += 1) staged[i] = row[i] ?? 1024;
    this.device.queue.writeBuffer(b.rowIds, 0, staged);

    const encoder = this.device.createCommandEncoder({ label: "global-step" });
    const pass = encoder.beginComputePass();
    this.embedRow(pass, 0, embed, b.rowIds, `g${position}`);
    for (let layer = 0; layer < layers.length; layer += 1) {
      const keyBase = layer * 2 * this.maxSlots * HIDDEN;
      this.runLayer(
        pass,
        asStepLayer(layers[layer]),
        this.weights,
        this.kv,
        keyBase,
        keyBase + this.maxSlots * HIDDEN,
        position + 1,
        `g${position}:l${layer}`,
        b.x,
      );
    }
    const uniforms = this.params(
      `g${position}:lnf`,
      params({
        n: HIDDEN,
        k: 0,
        wOff: lnF.w.offset,
        bOff: lnF.b.offset,
      }),
    );
    pass.setPipeline(this.pipeline("layerNorm1"));
    pass.setBindGroup(
      0,
      this.bind(
        "layerNorm1",
        uniforms,
        b.x,
        this.weights.chunks[lnF.w.chunk],
        b.hidden,
        this.weights.chunks[lnF.b.chunk],
        this.buffers.seen,
      ),
    );
    pass.dispatchWorkgroups(1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return this.readback(b.hidden, HIDDEN * 4);
  }

  /**
   * The local head: 17 sequential steps producing one 12.5 Hz frame of 16 codes,
   * or null when the text head says the utterance is over.
   *
   *   step 0    embed = the global hidden state  → argmax(text_head[:, {7, 9}])
   *   step 1    embed = wte[assistant_slot]      → argmax(audio_head_0)
   *   step 2+i  embed = audio_emb_i[tokens[i]]    → argmax(audio_head_{i+1})
   *
   * Each `argmax` is greedy with the HuggingFace repetition penalty over the
   * codes already emitted *on that channel in this utterance*.
   */
  async nextFrame(hidden: Float32Array, embed: EmbedTables): Promise<number[] | null> {
    const b = this.buffers;
    const local = asStepLayer(this.local);
    // step 0's embedding IS the global hidden state — no table read at all
    this.device.queue.writeBuffer(
      b.x,
      0,
      hidden.buffer as ArrayBuffer,
      hidden.byteOffset,
      hidden.byteLength,
    );

    const encoder = this.device.createCommandEncoder({ label: "local-head" });
    const pass = encoder.beginComputePass();

    for (let step = 0; step < LOCAL_SLOTS; step += 1) {
      if (step === 1) {
        this.embedRow(pass, 1, embed, b.tokens, "l1", {
          textId: this.localConfig.audioAssistantSlotTokenId,
        });
      } else if (step >= 2) {
        this.embedRow(pass, 2, embed, b.tokens, `l${step}`, { channel: step - 2 });
      }
      this.runLayer(
        pass,
        local,
        this.localWeights,
        b.localKv,
        0,
        LOCAL_SLOTS * HIDDEN,
        step + 1,
        `l${step}`,
        b.x,
      );
      // the local head's ln_f lands in proj, which the head matmul then reads
      const lnUniforms = this.params(
        `l${step}:lnf`,
        params({
          n: HIDDEN,
          k: 0,
          wOff: this.local.lnFW.offset,
          bOff: this.local.lnFB.offset,
        }),
      );
      pass.setPipeline(this.pipeline("layerNorm1"));
      pass.setBindGroup(
        0,
        this.bind(
          "layerNorm1",
          lnUniforms,
          b.x,
          this.localWeights.chunks[this.local.lnFW.chunk],
          b.proj,
          this.localWeights.chunks[this.local.lnFB.chunk],
          b.seen,
        ),
      );
      pass.dispatchWorkgroups(1);

      if (step === 0) {
        // only two columns of the 16 384-wide text head are ever read
        const headUniforms = this.params(
          "l0:texthead",
          params({
            n: HIDDEN,
            k: 0,
            wOff: this.local.textHead.offset,
            p0: this.localConfig.audioEndTokenId,
            p1: this.localConfig.audioAssistantSlotTokenId,
          }),
        );
        pass.setPipeline(this.pipeline("textHeadDecide"));
        pass.setBindGroup(
          0,
          this.bind(
            "textHeadDecide",
            headUniforms,
            b.proj,
            this.localWeights.chunks[this.local.textHead.chunk],
            b.flag,
            this.localWeights.chunks[this.local.textHead.chunk],
            b.seen,
          ),
        );
        pass.dispatchWorkgroups(1);
        continue;
      }

      const channel = step - 1;
      const head = this.local.audioHeads[channel];
      // the audio heads are a bare MatMul in the graph — no bias
      const headUniforms = this.params(
        `l${step}:audiohead`,
        params({
          n: CODES,
          k: HIDDEN,
          wOff: head.offset,
        }),
      );
      pass.setPipeline(this.pipeline("gemvNoBias"));
      pass.setBindGroup(
        0,
        this.bind(
          "gemvNoBias",
          headUniforms,
          b.proj,
          this.localWeights.chunks[head.chunk],
          b.ffn,
          this.localWeights.chunks[head.chunk],
          b.seen,
        ),
      );
      pass.dispatchWorkgroups(Math.ceil(CODES / 64));
      const argmaxUniforms = this.params(
        `l${step}:argmax`,
        params({
          n: CODES,
          k: 0,
          cOff: channel * 32,
          // float bits: the penalty is 1.2, and `1.2 >>> 0` would be 1
          p0: bitsOf(this.localConfig.repetitionPenalty),
          p1: channel,
          // the token lands where the *next* step embeds it from
          p2: channel,
        }),
      );
      pass.setPipeline(this.pipeline("argmaxPenalty"));
      pass.setBindGroup(
        0,
        this.bind(
          "argmaxPenalty",
          argmaxUniforms,
          b.tokens,
          this.localWeights.chunks[head.chunk],
          b.ffn,
          this.localWeights.chunks[head.chunk],
          b.seen,
        ),
      );
      pass.dispatchWorkgroups(1);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    // the only two reads that leave the GPU this frame
    const flag = await this.readback(b.flag, 4);
    if (flag[0] < 0.5) return null;
    const tokens = await this.readback(b.tokens, 16 * 4);
    const frame: number[] = [];
    for (let i = 0; i < 16; i += 1) frame.push(tokens[i]);
    return frame;
  }

  /** number of slots in the global KV ring */
  get maxSlotCount(): number {
    return this.maxSlots;
  }

  /** Debug probe: the fused QKV row of the last dispatched step. */
  async debugQkv(): Promise<Float32Array> {
    return this.readback(this.buffers.qkv, QKV * 4);
  }

  /** Debug probe: the residual stream / ln_f output of the last local step. */
  async debugBuffers(): Promise<{
    x: Float32Array;
    proj: Float32Array;
    norm: Float32Array;
    qkv: Float32Array;
  }> {
    return {
      x: await this.readback(this.buffers.x, HIDDEN * 4),
      proj: await this.readback(this.buffers.proj, HIDDEN * 4),
      norm: await this.readback(this.buffers.norm, HIDDEN * 4),
      qkv: await this.readback(this.buffers.qkv, QKV * 4),
    };
  }

  /**
   * Run ONE local-head step and stop — used by the conformance test to pin a
   * wrong step exactly instead of inferring it from a mismatched token.
   */
  async nextFrameSteps(
    hidden: Float32Array,
    embed: EmbedTables,
    maxStep: number,
  ): Promise<Float32Array> {
    const b = this.buffers;
    const local = asStepLayer(this.local);
    this.device.queue.writeBuffer(
      b.x,
      0,
      hidden.buffer as ArrayBuffer,
      hidden.byteOffset,
      hidden.byteLength,
    );
    const encoder = this.device.createCommandEncoder({ label: "local-head-partial" });
    const pass = encoder.beginComputePass();
    for (let step = 0; step <= maxStep; step += 1) {
      if (step === 1) {
        this.embedRow(pass, 1, embed, b.tokens, "p1", {
          textId: this.localConfig.audioAssistantSlotTokenId,
        });
      } else if (step >= 2) {
        this.embedRow(pass, 2, embed, b.tokens, `p${step}`, { channel: step - 2 });
      }
      this.runLayer(
        pass,
        local,
        this.localWeights,
        b.localKv,
        0,
        LOCAL_SLOTS * HIDDEN,
        step + 1,
        `p${step}`,
        b.x,
      );
      const lnUniforms = this.params(
        `p${step}:lnf`,
        params({
          n: HIDDEN,
          k: 0,
          wOff: this.local.lnFW.offset,
          bOff: this.local.lnFB.offset,
        }),
      );
      pass.setPipeline(this.pipeline("layerNorm1"));
      pass.setBindGroup(
        0,
        this.bind(
          "layerNorm1",
          lnUniforms,
          b.x,
          this.localWeights.chunks[this.local.lnFW.chunk],
          b.proj,
          this.localWeights.chunks[this.local.lnFB.chunk],
          b.seen,
        ),
      );
      pass.dispatchWorkgroups(1);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return this.readback(b.proj, HIDDEN * 4);
  }

  /** Debug probe: one layer's K/V ring, up to `slots` slots. */
  async debugKvLayer(
    layer: number,
    slots: number,
  ): Promise<{ key: Float32Array; value: Float32Array }> {
    return { key: new Float32Array(0), value: new Float32Array(0) };
  }

  dispose(): void {
    for (const buffer of Object.values(this.buffers)) buffer.destroy();
    for (const buffer of this.uniforms.values()) buffer.destroy();
  }
}
