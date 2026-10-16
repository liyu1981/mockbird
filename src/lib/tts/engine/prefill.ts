/**
 * Prefill: the 195-row prompt through 12 global transformer layers in one pass.
 *
 * Per layer the dispatch sequence is
 *   layerNormRows -> gemmNt(qkv) -> ropeSplitCache -> attentionScores
 *   -> attentionSoftmax -> attentionApply -> gemmNt(c_proj) -> addRows
 *   -> layerNormRows -> gemmNt(fc_in) -> geluRows -> gemmNt(fc_out) -> addRows
 *
 * and the last row's hidden state is returned as the state the first decode step
 * consumes. Everything is driven from the block table built by the weight upload,
 * so no kernel needs to know an absolute buffer address.
 */

import { BATCH_KERNELS, type BatchKernelName } from "./kernels-batch";
import { type BlockLocation, blocksByShape, type GpuWeights } from "./weights";

const HIDDEN = 768;
const QKV = 2304;
const FFN = 3072;
const HEADS = 12;
const HEAD_DIM = 64;
/** uniform block: 12 scalars + 16 audio-table offsets = 112 B -> 256 B */
const UNIFORM_FLOATS = 64;
const AUDIO_OFFSET_SLOT = 10; // a0, immediately after aux3 (index 9)

type Bindings = {
  uniforms: GPUBuffer;
  input: GPUBuffer;
  weights: GPUBuffer;
  output: GPUBuffer;
  /** when set, binding 2 uses this buffer instead of a weight chunk */
  binding2?: GPUBuffer;
};

export type PrefillBuffers = {
  /** input ids [S, 17] */
  ids: GPUBuffer;
  /** residual stream [S, 768] */
  x: GPUBuffer;
  /** fused QKV [S, 2304] */
  qkv: GPUBuffer;
  /** layernorm outputs [S, 3072] spare */
  norm: GPUBuffer;
  /** pre-softmax scores + row maxima/sums [12, S, S] + 2 * [12, S] */
  scores: GPUBuffer;
  /** softmax probabilities [12, S, S] — separate buffer: a writable binding
   *  cannot alias another writable binding in the same bind group */
  probs: GPUBuffer;
  /** attention context [S, 768] */
  context: GPUBuffer;
  /** MLP activation [S, 3072] */
  ffn: GPUBuffer;
  /** [S, 768] scratch for projection outputs */
  proj: GPUBuffer;
  /** KV cache: [(layer*2 + isV) * maxSlots * 768 + slot*768 + head*64 + dim] */
  kv: GPUBuffer;
  /** host-visible copy of the final hidden [S, 768] */
  hidden: GPUBuffer;
};

export type LayerWeights = {
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
};

export function resolveLayerWeights(
  weights: GpuWeights,
  table: LayerWeightSource,
): LayerWeights[] {
  const cAttn = blocksByShape(weights, "768x2304");
  const cProj = blocksByShape(weights, "768x768");
  const fcIn = blocksByShape(weights, "768x3072");
  const fcOut = blocksByShape(weights, "3072x768");
  const count = Math.min(cAttn.length, cProj.length, fcIn.length, fcOut.length);
  const layers: LayerWeights[] = [];
  for (let i = 0; i < count; i += 1) {
    const p = `core.model.transformer.h.${i}.`;
    layers.push({
      cAttn: cAttn[i].block,
      cAttnBias: table.get(`${p}attn.c_attn.bias`),
      cProj: cProj[i].block,
      cProjBias: table.get(`${p}attn.c_proj.bias`),
      ln1W: table.get(`${p}ln_1.weight`),
      ln1B: table.get(`${p}ln_1.bias`),
      ln2W: table.get(`${p}ln_2.weight`),
      ln2B: table.get(`${p}ln_2.bias`),
      fcIn: fcIn[i].block,
      fcInBias: table.get(`${p}mlp.fc_in.bias`),
      fcOut: fcOut[i].block,
      fcOutBias: table.get(`${p}mlp.fc_out.bias`),
      lnFW: table.get("core.model.transformer.ln_f.weight"),
      lnFB: table.get("core.model.transformer.ln_f.bias"),
    });
  }
  return layers;
}

export interface LayerWeightSource {
  get(name: string): BlockLocation;
}

export class PrefillRunner {
  private readonly pipelines = new Map<BatchKernelName, GPUComputePipeline>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly uniformBuffers = new Map<string, GPUBuffer>();
  private uniformSnapshots = new Map<string, Uint32Array>();
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private pipelineLayout: GPUPipelineLayout | null = null;
  private readonly maxSlots: number;

  // NB: no TypeScript parameter properties here — Node's type stripping (used by
  // tests/gpu) rejects them, so the fields are declared and assigned explicitly.
  private readonly device: GPUDevice;
  private readonly weights: GpuWeights;

  constructor(device: GPUDevice, weights: GpuWeights, maxSlots: number) {
    this.device = device;
    this.weights = weights;
    this.maxSlots = maxSlots;
    const S = GPUBufferUsage.STORAGE;
    const CD = GPUBufferUsage.COPY_DST;
    const CS = GPUBufferUsage.COPY_SRC;
    const usage = S | CD | CS;
    const rows = 256;
    const scores = HEADS * rows * rows + 3 * HEADS * rows;
    this.buffers = {
      ids: device.createBuffer({ size: rows * 17 * 4, usage, label: "prefill-ids" }),
      x: device.createBuffer({ size: rows * HIDDEN * 4, usage, label: "prefill-x" }),
      qkv: device.createBuffer({ size: rows * QKV * 4, usage, label: "prefill-qkv" }),
      norm: device.createBuffer({ size: rows * FFN * 4, usage, label: "prefill-norm" }),
      scores: device.createBuffer({ size: scores * 4, usage, label: "prefill-scores" }),
      probs: device.createBuffer({
        size: HEADS * rows * rows * 4,
        usage,
        label: "prefill-probs",
      }),
      context: device.createBuffer({
        size: rows * HIDDEN * 4,
        usage,
        label: "prefill-context",
      }),
      ffn: device.createBuffer({ size: rows * FFN * 4, usage, label: "prefill-ffn" }),
      proj: device.createBuffer({ size: rows * HIDDEN * 4, usage, label: "prefill-proj" }),
      kv: device.createBuffer({
        size: 12 * 2 * maxSlots * HIDDEN * 4,
        usage,
        label: "prefill-kv",
      }),
      hidden: device.createBuffer({
        size: rows * HIDDEN * 4,
        usage,
        label: "prefill-hidden",
      }),
    };
  }

  readonly buffers: PrefillBuffers;

  /**
   * One explicit bind group layout for every kernel: params (uniform), input,
   * weights (read-only storage), output (storage). `layout: "auto"` derives the
   * layout per shader and *drops* bindings a kernel does not read, which makes a
   * shared 4-entry bind group invalid for the kernels that ignore `weights`.
   */
  private layout(): GPUBindGroupLayout {
    if (this.bindGroupLayout) return this.bindGroupLayout;
    const storage = (type: GPUBufferBindingType): GPUBufferBindingLayout => ({
      type,
      hasDynamicOffset: false,
      minBindingSize: 4,
    });
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        {
          // binding 1 is writable: ropeSplitCache updates the roped Q in place
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: storage("storage"),
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: storage("read-only-storage"),
        },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: storage("storage") },
        // the bias/LayerNorm-bias parameter gets its own binding: a weight
        // block and its bias routinely land in different chunks (see GEMM_NT)
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: storage("read-only-storage"),
        },
      ],
    });
    this.pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });
    return this.bindGroupLayout;
  }

  private pipeline(name: BatchKernelName): GPUComputePipeline {
    const cached = this.pipelines.get(name);
    if (cached) return cached;
    // ensure the shared layout/pipeline-layout exist before the first pipeline
    this.layout();
    const pipeline = this.device.createComputePipeline({
      layout: this.pipelineLayout as GPUPipelineLayout,
      compute: {
        module: this.device.createShaderModule({ code: BATCH_KERNELS[name], label: name }),
        entryPoint: "main",
      },
      label: name,
    });
    this.pipelines.set(name, pipeline);
    return pipeline;
  }

  private bind(key: string, bias: GPUBuffer, buffers: Bindings): GPUBindGroup {
    // the key MUST cover every binding, not just (name, chunk): a bind group
    // captures the uniform/input/output buffers as well, and reusing it across
    // dispatches that write to different buffers silently computes on the wrong
    // data
    const cacheKey = `${key}|${this.bufferId(buffers.uniforms)}|${this.bufferId(buffers.input)}|${this.bufferId(buffers.weights)}|${this.bufferId(buffers.output)}|${this.bufferId(bias)}`;
    const cached = this.bindGroups.get(cacheKey);
    if (cached) return cached;
    const group = this.device.createBindGroup({
      layout: this.layout(),
      entries: [
        { binding: 0, resource: { buffer: buffers.uniforms } },
        { binding: 1, resource: { buffer: buffers.input } },
        { binding: 2, resource: { buffer: buffers.binding2 ?? buffers.weights } },
        { binding: 3, resource: { buffer: buffers.output } },
        { binding: 4, resource: { buffer: bias } },
      ],
    });
    this.bindGroups.set(cacheKey, group);
    return group;
  }

  private ids = new WeakMap<GPUBuffer, number>();
  private nextId = 1;
  private bufferId(buffer: GPUBuffer): number {
    let id = this.ids.get(buffer);
    if (id === undefined) {
      id = this.nextId++;
      this.ids.set(buffer, id);
    }
    return id;
  }

  /** Uniform slots are keyed by call site so reuse is explicit and bounded. */
  private uniform(slot: string): GPUBuffer {
    const cached = this.uniformBuffers.get(slot);
    if (cached) return cached;
    const buffer = this.device.createBuffer({
      size: UNIFORM_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: `prefill-params-${slot}`,
    });
    this.uniformBuffers.set(slot, buffer);
    return buffer;
  }

  private writeParams(
    slot: string,
    values: Record<string, number>,
    audioOffsets?: number[],
  ): GPUBuffer {
    const buffer = this.uniform(slot);
    const array = new Uint32Array(UNIFORM_FLOATS);
    for (const [key, value] of Object.entries(values)) {
      const index = (
        {
          rows: 0,
          cols: 1,
          depth: 2,
          wOff: 3,
          bOff: 4,
          yOff: 5,
          xOff: 6,
          aux: 7,
          aux2: 8,
          aux3: 9,
        } as Record<string, number>
      )[key];
      if (index !== undefined) array[index] = value >>> 0;
    }
    if (audioOffsets) {
      for (let i = 0; i < Math.min(16, audioOffsets.length); i += 1) {
        array[AUDIO_OFFSET_SLOT + i] = audioOffsets[i] >>> 0;
      }
    }
    this.uniformSnapshots.set(slot, array.slice());
    this.device.queue.writeBuffer(buffer, 0, array);
    return buffer;
  }

  /** Debug probe: what a call site actually asked for (validation only). */
  debugUniform(slot: string): Uint32Array | undefined {
    return this.uniformSnapshots.get(slot)?.slice();
  }

  /**
   * Run the prefill for `rows` prompt rows (ids already staged in `buffers.ids`).
   * Returns the full [rows, 768] hidden state.
   */
  /**
   * `layerCount` exists so the conformance test can run a single layer and
   * compare its KV cache against the reference — the fastest way to localise a
   * wrong kernel (the cache proves embed/LN/QKV/RoPE in one shot).
   */
  async run(
    rows: number,
    layers: LayerWeights[],
    embed: { wte: BlockLocation; audio: BlockLocation[] },
    maxSlots: number,
    layerCount = layers.length,
  ): Promise<Float32Array> {
    const b = this.buffers;
    const encoder = this.device.createCommandEncoder({ label: "prefill" });
    const pass = encoder.beginComputePass();

    // `binding2` defaults to the weight chunk, but the attention kernels read the
    // KV cache there instead
    const dispatch = (
      name: BatchKernelName,
      slot: string,
      params: Record<string, number>,
      weightChunk: number,
      workgroups: [number, number],
      input: GPUBuffer,
      output: GPUBuffer,
      binding2?: GPUBuffer,
      biasChunk?: number,
      audioOffsets?: number[],
    ) => {
      const pipeline = this.pipeline(name);
      const uniforms = this.writeParams(slot, params, audioOffsets);
      const weights = binding2 ?? this.weights.chunks[weightChunk];
      // kernels that read p.bOff need the bias's own chunk; the rest can point
      // at any bound buffer
      const bias = this.weights.chunks[biasChunk ?? weightChunk];
      pass.setPipeline(pipeline);
      pass.setBindGroup(
        0,
        this.bind(name, bias, {
          uniforms,
          input,
          weights,
          output,
          binding2,
        }),
      );
      pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, 1);
    };

    // --- embedding
    dispatch(
      "embedRows",
      "embed",
      { rows, cols: HIDDEN, wOff: embed.wte.offset, yOff: 0 },
      embed.wte.chunk,
      [rows, 1],
      b.ids,
      b.x,
      undefined,
      undefined, // no bias term; embedRows reads nothing from binding 4
      embed.audio.map((block) => block.offset),
    );

    for (let layer = 0; layer < Math.min(layerCount, layers.length); layer += 1) {
      const w = layers[layer];
      // kv layout: [(layer * 2 + isV) * maxSlots * 768 + slot * 768 + head * 64 + dim]
      const layerBase = layer * 2 * maxSlots * HIDDEN;
      const kvKeyBase = layerBase;
      const kvValueBase = layerBase + maxSlots * HIDDEN;

      // LN1
      dispatch(
        "layerNormRows",
        `ln1:${layer}`,
        { rows, cols: HIDDEN, xOff: 0, yOff: 0, wOff: w.ln1W.offset, bOff: w.ln1B.offset },
        w.ln1W.chunk,
        [rows, 1],
        b.x,
        b.norm,
        undefined,
        w.ln1B.chunk,
      );
      // QKV projection
      dispatch(
        "gemmNt",
        `qkv:${layer}`,
        {
          rows,
          cols: QKV,
          depth: HIDDEN,
          xOff: 0,
          yOff: 0,
          wOff: w.cAttn.offset,
          bOff: w.cAttnBias.offset,
        },
        w.cAttn.chunk,
        [Math.ceil(QKV / 64), rows],
        b.norm,
        b.qkv,
        undefined,
        w.cAttnBias.chunk,
      );
      // RoPE + split + KV cache write
      dispatch(
        "ropeSplitCache",
        `rope:${layer}`,
        { rows, xOff: kvKeyBase, aux: kvValueBase },
        w.cAttn.chunk,
        [rows * HEADS, 1],
        b.qkv,
        b.kv,
      );
      // attention
      dispatch(
        "attentionScores",
        `scores:${layer}`,
        { rows, aux: 0, xOff: kvKeyBase, aux2: HEADS * rows * rows },
        w.cAttn.chunk,
        [HEADS, Math.ceil(rows / 64)],
        b.qkv,
        b.scores,
        b.kv,
      );
      dispatch(
        "attentionSoftmax",
        `softmax:${layer}`,
        { rows, aux: 0, aux2: HEADS * rows * rows, aux3: HEADS * rows * rows + HEADS * rows },
        w.cAttn.chunk,
        [HEADS * rows, 1],
        b.scores,
        b.probs,
      );
      dispatch(
        "attentionApply",
        `apply:${layer}`,
        { rows, xOff: kvValueBase, aux: 0, yOff: 0 },
        w.cAttn.chunk,
        [HEADS, 1],
        b.probs,
        b.context,
        b.kv,
      );
      // output projection + residual
      dispatch(
        "gemmNt",
        `proj:${layer}`,
        {
          rows,
          cols: HIDDEN,
          depth: HIDDEN,
          xOff: 0,
          yOff: 0,
          wOff: w.cProj.offset,
          bOff: w.cProjBias.offset,
        },
        w.cProj.chunk,
        [HIDDEN / 64, rows],
        b.context,
        b.proj,
        undefined,
        w.cProjBias.chunk,
      );
      dispatch(
        "addRows",
        `resid0:${layer}`,
        { rows, cols: HIDDEN, xOff: 0, yOff: 0 },
        w.cProj.chunk,
        [Math.ceil((rows * HIDDEN) / 256), 1],
        b.proj,
        b.x,
      );
      // LN2
      dispatch(
        "layerNormRows",
        `ln2:${layer}`,
        { rows, cols: HIDDEN, xOff: 0, yOff: 0, wOff: w.ln2W.offset, bOff: w.ln2B.offset },
        w.ln2W.chunk,
        [rows, 1],
        b.x,
        b.norm,
        undefined,
        w.ln2B.chunk,
      );
      // MLP
      dispatch(
        "gemmNt",
        `fcin:${layer}`,
        {
          rows,
          cols: FFN,
          depth: HIDDEN,
          xOff: 0,
          yOff: 0,
          wOff: w.fcIn.offset,
          bOff: w.fcInBias.offset,
        },
        w.fcIn.chunk,
        [Math.ceil(FFN / 64), rows],
        b.norm,
        b.ffn,
        undefined,
        w.fcInBias.chunk,
      );
      // GELU works in place on binding 3 and never reads binding 1, so any
      // distinct buffer satisfies the layout (aliasing a writable binding with
      // itself is rejected)
      dispatch(
        "geluRows",
        `gelu:${layer}`,
        { rows, cols: FFN, yOff: 0 },
        w.fcIn.chunk,
        [Math.ceil((rows * FFN) / 256), 1],
        b.norm,
        b.ffn,
      );
      dispatch(
        "gemmNt",
        `fcout:${layer}`,
        {
          rows,
          cols: HIDDEN,
          depth: FFN,
          xOff: 0,
          yOff: 0,
          wOff: w.fcOut.offset,
          bOff: w.fcOutBias.offset,
        },
        w.fcOut.chunk,
        [HIDDEN / 64, rows],
        b.ffn,
        b.proj,
        undefined,
        w.fcOutBias.chunk,
      );
      dispatch(
        "addRows",
        `resid1:${layer}`,
        { rows, cols: HIDDEN, xOff: 0, yOff: 0 },
        w.fcOut.chunk,
        [Math.ceil((rows * HIDDEN) / 256), 1],
        b.proj,
        b.x,
      );
    }

    // final layernorm
    dispatch(
      "layerNormRows",
      "lnf",
      {
        rows,
        cols: HIDDEN,
        xOff: 0,
        yOff: 0,
        wOff: layers[0].lnFW.offset,
        bOff: layers[0].lnFB.offset,
      },
      layers[0].lnFW.chunk,
      [rows, 1],
      b.x,
      b.hidden,
      undefined,
      layers[0].lnFB.chunk,
    );

    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    const bytes = rows * HIDDEN * 4;
    const readback = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const copy = this.device.createCommandEncoder();
    copy.copyBufferToBuffer(b.hidden, 0, readback, 0, bytes);
    this.device.queue.submit([copy.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const hidden = new Float32Array(readback.getMappedRange().slice(0, bytes));
    readback.unmap();
    readback.destroy();
    return hidden;
  }

  /**
   * Write a sentinel into the first slots of the KV cache. Used by warm-up to tell
   * "the kernel wrote nothing" apart from "the read-back looked in the wrong
   * place" — two failures with one message otherwise.
   */
  async poisonKv(rows: number): Promise<void> {
    const sentinel = new Float32Array(HIDDEN * rows).fill(-12345.5);
    this.device.queue.writeBuffer(
      this.buffers.kv,
      0,
      sentinel.buffer,
      sentinel.byteOffset,
      sentinel.byteLength,
    );
    await this.device.queue.onSubmittedWorkDone();
  }

  /** Debug probe: read an activation buffer back (validation only). */
  async debugRead(name: keyof PrefillBuffers, elements: number): Promise<Float32Array> {
    const source = this.buffers[name];
    const size = elements * 4;
    const buffer = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, buffer, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(buffer.getMappedRange().slice(0, size));
    buffer.unmap();
    buffer.destroy();
    return out;
  }

  /** Stage prompt ids [rows, 17] (u32) into the input buffer. */
  stageIds(ids: Uint32Array): void {
    this.device.queue.writeBuffer(
      this.buffers.ids,
      0,
      ids.buffer,
      ids.byteOffset,
      ids.byteLength,
    );
  }

  dispose(): void {
    for (const buffer of Object.values(this.buffers)) buffer.destroy();
  }

  /**
   * Read back one layer's K/V cache for `rows` slots. Used by the conformance
   * test: the cache is where the rotary convention and the head layout live, and a
   * wrong sign there still yields a plausible hidden state while silently breaking
   * every decode step.
   */
  async readLayerCache(
    layer: number,
    rows: number,
  ): Promise<{ key: Float32Array; value: Float32Array }> {
    // kv layout: [(layer * 2 + isV) * maxSlots * 768 + slot * 768 + head * 64 + dim]
    // read the whole layer block through debugRead (a proven single path) and slice
    const perLayer = 2 * this.maxSlots * HIDDEN;
    const block = await this.debugRead("kv", perLayer * (layer + 1));
    const from = layer * perLayer;
    // slots are contiguous from the layer base, so slot 0 starts at `from`
    const slice = (base: number) => block.slice(base, base + rows * HIDDEN);
    return { key: slice(from), value: slice(from + this.maxSlots * HIDDEN) };
  }
}

export const PREFILL_SHAPES = { HIDDEN, QKV, FFN, HEADS, HEAD_DIM };
