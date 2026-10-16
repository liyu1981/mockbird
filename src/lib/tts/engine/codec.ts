/**
 * The codec decoder: 16 RVQ codes per 12.5 Hz frame → 3840 stereo samples.
 *
 * The whole thing, in the order the graph runs it (§10.6/§10.7):
 *
 *   RVQ front    16 × codebook gather → [512] → W_out → [768] → interleaved ×4
 *   stage 1      4 layers, d=256, ring C=500,  in 192 → 256,  out 256 → 768
 *   stage 2      2 layers, ring C=1696, in 384 → 256,  out 256 → 768
 *   stage 3      2 layers, ring C=2480, in 384 → 256,  out 256 → 768
 *   stage 4      4 layers, ring C=3248, in 384 → 256,  out 256 → 240
 *   head         audio[c][t·120+j] = latent[t][2j+c]
 *
 * with an interleaved ×2 sub-pixel upsample between the stages — `new[2t+r, m] =
 * old[t, 2m+r]`, verified maxErr 0 — and, per layer,
 *
 *   x += layer_scale_1 · out_proj(attention(LN1(x)))
 *   x += layer_scale_2 · ffn.2(exact_gelu(ffn.0(LN2(x))))
 *
 * Attention runs against a **wrapping** ring keyed by absolute position:
 * `valid_j = stored_j ≥ 0 ∧ 0 ≤ cur − stored_j < C`, which is the shift-append
 * the graph spells as concat + mask. That is what makes a long utterance correct
 * — stage 1 keeps only the last 500 of the 4·frames positions it has seen.
 *
 * Everything is dispatched per position: a batched GEMM over the stage's ≤32
 * positions would cut the dispatch count by ~30×, but attention still has to run
 * per position, and the dispatch overhead (~1.7k per frame) is well inside the
 * 80 ms budget. Correctness against the golden PCM first.
 */

import {
  ATTEND_RING_CODEC,
  CODEC_HEAD,
  GELU_ERF,
  GEMV_NOB,
  LAYER_SCALE_ADD,
  LAYERNORM_1,
  ROPE_SPLIT_CODEC,
  RVQ_ACCUM,
  RVQ_PROJECT,
  UPSAMPLE2,
} from "./kernels-decode";
import type { BlockLocation, GpuWeights } from "./weights";

const D = 256;
const CODEC_HEADS = 4;
const N_VQ = 16;
/** 48 kHz ÷ 12.5 Hz = 3840 samples per frame per channel */
const SAMPLES_PER_FRAME = 3840;
/** one frame of input codes → 4 positions of 192 at the stage-1 entry */
const RVQ_POSITIONS = 4;
const RVQ_WIDTH = 192;

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
  | "rvqAccum"
  | "rvqProject"
  | "layerNorm1"
  | "gemvNoBias"
  | "geluErf"
  | "ropeSplitCodec"
  | "attendCodec"
  | "layerScaleAdd"
  | "upsample2"
  | "codecHead";

const KERNELS: Record<KernelName, string> = {
  rvqAccum: RVQ_ACCUM,
  rvqProject: RVQ_PROJECT,
  layerNorm1: LAYERNORM_1,
  gemvNoBias: GEMV_NOB,
  geluErf: GELU_ERF,
  ropeSplitCodec: ROPE_SPLIT_CODEC,
  attendCodec: ATTEND_RING_CODEC,
  layerScaleAdd: LAYER_SCALE_ADD,
  upsample2: UPSAMPLE2,
  codecHead: CODEC_HEAD,
};

/** Per-layer codec weights, in graph order. */
export type CodecLayerWeights = {
  inProj: BlockLocation;
  outProj: BlockLocation;
  ffnIn: BlockLocation;
  ffnOut: BlockLocation;
  norm1W: BlockLocation;
  norm1B: BlockLocation;
  norm2W: BlockLocation;
  norm2B: BlockLocation;
  scale1: BlockLocation;
  scale2: BlockLocation;
};

export type CodecStage = {
  layers: CodecLayerWeights[];
  /** ring context in positions */
  context: number;
  /** input projection [width, 256] — only stage 1 has 192, the rest 384 */
  inputProj: BlockLocation;
  /** output projection [256, 768] for stages 1–3, [256, 240] for stage 4 */
  outputProj: BlockLocation;
  /** positions this stage produces per input frame: 4, 8, 16, 32 */
  positions: number;
};

export type CodecWeights = {
  stages: CodecStage[];
  /** the 16 [512,8] out_proj convolutions, in quantiser order */
  quantProj: BlockLocation;
  /** [768,512] */
  quantOutput: BlockLocation;
  quantOutputBias: BlockLocation;
  /** per-quantiser [512] biases */
  quantBias: BlockLocation[];
  /** per-quantiser [1024,8] codebooks */
  codebook: BlockLocation[];
};

/** Ring layout, derived from the stage list (contexts and layer counts). */
type RingSlot = {
  layer: number;
  stage: number;
  keyBase: number;
  valueBase: number;
  slotBase: number;
};

export class CodecRunner {
  private readonly device: GPUDevice;
  private readonly weights: GpuWeights;
  private readonly w: CodecWeights;
  private readonly stages: CodecStage[];
  private readonly slots: RingSlot[];
  private readonly ringElements: number;
  private readonly slotCount: number;
  /** absolute position written next, per stage */
  private readonly stagePositions: number[];
  /** index into `slots` of each stage's first layer (slots are global, stage-major) */
  private readonly stageSlotOffset: number[];

  private readonly ring: GPUBuffer;
  private readonly slotBuffer: GPUBuffer;
  private readonly codes: GPUBuffer;
  private readonly accum: GPUBuffer;
  private readonly stage0: GPUBuffer;
  private readonly wide: GPUBuffer;
  private readonly narrow: GPUBuffer;
  private readonly stream: GPUBuffer;
  private readonly norm: GPUBuffer;
  private readonly qkv: GPUBuffer;
  private readonly ctx: GPUBuffer;
  private readonly ffn: GPUBuffer;
  private readonly proj: GPUBuffer;
  private readonly pcm: GPUBuffer;
  private readonly pcmCapacityFrames: number;

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

  constructor(device: GPUDevice, weights: GpuWeights, w: CodecWeights) {
    this.device = device;
    this.weights = weights;
    this.w = w;
    this.stages = w.stages;

    // ring: per layer 2 × context × 256, laid out in stage/layer order
    this.slots = [];
    let keyBase = 0;
    let slotBase = 0;
    let layer = 0;
    for (let s = 0; s < this.stages.length; s += 1) {
      for (let i = 0; i < this.stages[s].layers.length; i += 1) {
        const context = this.stages[s].context;
        this.slots.push({
          layer,
          stage: s,
          keyBase,
          valueBase: keyBase + context * D,
          slotBase,
        });
        keyBase += 2 * context * D;
        slotBase += context;
        layer += 1;
      }
    }
    this.ringElements = keyBase;
    this.slotCount = slotBase;
    this.stagePositions = this.stages.map(() => 0);
    this.stageSlotOffset = [];
    {
      let acc = 0;
      for (const stage of this.stages) {
        this.stageSlotOffset.push(acc);
        acc += stage.layers.length;
      }
    }

    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (label: string, elements: number) =>
      device.createBuffer({ size: elements * 4, usage, label });
    this.ring = make("codec-ring", this.ringElements);
    this.slotBuffer = device.createBuffer({
      size: this.slotCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: "codec-slots",
    });
    // codes: 16 f32 values followed by 32 u32 descriptors (16 bias + 16
    // codebook offsets) — the sidecar interleaves those, so they can't be
    // derived from one offset
    this.codes = device.createBuffer({
      size: (16 + 32) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: "codec-codes",
    });
    this.accum = make("codec-accum", 512);
    this.stage0 = make("codec-stage0", RVQ_POSITIONS * RVQ_WIDTH);
    this.wide = make("codec-wide", 32 * 768);
    this.narrow = make("codec-narrow", 32 * 384);
    this.stream = make("codec-stream", 32 * D);
    this.norm = make("codec-norm", 32 * D);
    this.qkv = make("codec-qkv", 32 * 3 * D);
    this.ctx = make("codec-ctx", 32 * D);
    this.ffn = make("codec-ffn", 32 * 1024);
    this.proj = make("codec-proj", 32 * D);
    this.pcmCapacityFrames = 64;
    this.pcm = make("codec-pcm", this.pcmCapacityFrames * 2 * SAMPLES_PER_FRAME);

    this.attendPipeline = device.createComputePipeline({
      layout: this.makeAttendLayout(),
      compute: {
        module: device.createShaderModule({ code: KERNELS.attendCodec, label: "attendCodec" }),
        entryPoint: "main",
      },
      label: "attendCodec",
    });
    this.resetRings();
    this.uploadDescriptors();
  }

  /** A fresh utterance: every ring slot reads as "empty" (position −1). */
  resetRings(): void {
    this.device.queue.writeBuffer(this.slotBuffer, 0, new Int32Array(this.slotCount).fill(-1));
    for (let s = 0; s < this.stagePositions.length; s += 1) this.stagePositions[s] = 0;
  }

  /**
   * Copy the 32 per-quantiser descriptor offsets next to the codes, so the RVQ
   * kernels can reach the biases and codebooks (which alternate in the sidecar).
   */
  private uploadDescriptors(): void {
    const data = new Uint32Array(32);
    for (let k = 0; k < N_VQ; k += 1) {
      data[k] = this.w.quantBias[k].offset;
      data[16 + k] = this.w.codebook[k].offset;
    }
    this.device.queue.writeBuffer(this.codes, 16 * 4, data);
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
        // the bias/LayerNorm-bias parameter gets its own binding; see the note
        // on kernels-decode's DECODE_BINDINGS
        { binding: 4, visibility: v, buffer: storage("read-only-storage") },
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
    // binding 4 is the context/output slot, which both the rope kernel (it parks
    // the roped q) and the attention kernel (the result) write, so it must be
    // declared `storage`, not `read-only-storage`
    this.attendBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: v, buffer: { type: "uniform" } },
        { binding: 1, visibility: v, buffer: storage("storage") },
        { binding: 2, visibility: v, buffer: storage("read-only-storage") },
        { binding: 3, visibility: v, buffer: storage("storage") },
        { binding: 4, visibility: v, buffer: storage("storage") },
        { binding: 5, visibility: v, buffer: storage("storage") },
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
    // ropeSplitCodec binds the ring, the context and the slot array — the same
    // six bindings the attention uses, not the four-binding default layout
    const six = name === "ropeSplitCodec";
    if (!six) this.makeLayout();
    const layout = six ? this.makeAttendLayout() : (this.pipelineLayout as GPUPipelineLayout);
    const pipeline = this.device.createComputePipeline({
      layout,
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
  ): GPUBindGroup {
    const key = `${name}|${this.bufferId(uniforms)}|${this.bufferId(input)}|${this.bufferId(
      weights,
    )}|${this.bufferId(output)}|${this.bufferId(bias)}`;
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
    slots: GPUBuffer,
  ): GPUBindGroup {
    const key = `att|${this.bufferId(uniforms)}|${this.bufferId(q)}|${this.bufferId(
      weights,
    )}|${this.bufferId(ring)}|${this.bufferId(ctx)}|${this.bufferId(slots)}`;
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
        { binding: 5, resource: { buffer: slots } },
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
        label: `codec-params-${slot}`,
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
    this.device.queue.writeBuffer(buffer, 0, array);
    return buffer;
  }

  // -------------------------------------------------------------- the frame
  /**
   * One frame: stage the codes, then run RVQ → 4 stages → head. The PCM lands in
   * `this.pcm` at frame `frameIndex`; the caller copies it out.
   */
  private encodeFrame(pass: GPUComputePassEncoder, frameIndex: number): void {
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
      const uniforms = this.params(`${frameIndex}:${slot}`, p);
      const weights = this.weights.chunks[chunk];
      const bias = this.weights.chunks[biasChunk ?? chunk];
      pass.setPipeline(this.pipeline(name));
      pass.setBindGroup(0, this.bind(name, uniforms, input, weights, output, bias));
      pass.dispatchWorkgroups(workgroups);
    };

    // ---- RVQ front: [16 codes] -> acc[512] -> g[768] -> stage0[4, 192]
    run(
      "rvqAccum",
      "rvq-acc",
      params({ n: 512, k: 0, wOff: this.w.quantProj.offset, dOff: 0, sOff: 0 }),
      this.w.quantProj.chunk,
      this.codes,
      this.accum,
      512 / 64,
    );
    run(
      "rvqProject",
      "rvq-out",
      params({
        n: 768,
        k: 0,
        aOff: this.w.quantOutput.offset,
        cOff: this.w.quantOutputBias.offset,
        dOff: 0,
        sOff: 0,
      }),
      this.w.quantOutput.chunk,
      this.accum,
      this.stage0,
      768 / 64,
      this.w.quantOutputBias.chunk,
    );

    // ---- the four stages
    // stage 1 reads the RVQ front (192 wide); stages 2..4 read the previous
    // stage's upsampled output (384 wide)
    let stageInput = this.stage0;
    let stageInputWidth = RVQ_WIDTH;
    for (let s = 0; s < this.stages.length; s += 1) {
      const stage = this.stages[s];
      const n = stage.positions;
      const positionBase = this.stagePositions[s];
      const outWidth = stage.outputProj.dims[1] ?? 768;
      // The stage-entry projection runs ONCE per stage (graph node
      // `/input_proj/MatMul` -> `/norm1`): it maps the stage input into the
      // d=256 residual stream every layer then reads and updates. Running it
      // inside the layer loop (as this used to) clobbered the residual each
      // layer and left the transformer layers with nothing to add into.
      const inProj = stage.inputProj;
      for (let i = 0; i < n; i += 1) {
        run(
          "gemvNoBias",
          `s${s}:inproj:${i}`,
          params({
            n: D,
            k: stageInputWidth,
            wOff: inProj.offset,
            sOff: i * stageInputWidth,
            dOff: i * D,
          }),
          inProj.chunk,
          stageInput,
          this.stream,
          Math.ceil(D / 64),
        );
      }

      let layerIndex = 0;
      for (const layer of stage.layers) {
        const slot = this.slots[this.stageSlotOffset[s] + layerIndex];
        const context = stage.context;
        for (let i = 0; i < n; i += 1) {
          const xOff = i * D;
          run(
            "layerNorm1",
            `s${s}:l${layerIndex}:ln1:${i}`,
            params({
              n: D,
              k: 0,
              wOff: layer.norm1W.offset,
              bOff: layer.norm1B.offset,
              sOff: xOff,
            }),
            layer.norm1W.chunk,
            this.stream,
            this.norm,
            1,
            layer.norm1B.chunk,
          );
          run(
            "gemvNoBias",
            `s${s}:l${layerIndex}:qkv:${i}`,
            params({ n: 3 * D, k: D, wOff: layer.inProj.offset, sOff: 0, dOff: 0 }),
            layer.inProj.chunk,
            this.norm,
            this.qkv,
            Math.ceil((3 * D) / 64),
          );
          const position = positionBase + i;
          const weightsChunk = this.weights.chunks[layer.inProj.chunk];

          // rope the row in place and write k/v + the absolute position into the
          // layer's ring slot (rotate on write, exactly like the graph)
          const ropeUniforms = this.params(
            `${frameIndex}:s${s}:l${layerIndex}:rope:${i}`,
            params({
              n: D,
              k: 0,
              dOff: 0,
              sOff: slot.slotBase,
              aOff: slot.valueBase,
              cOff: slot.keyBase,
              p0: position,
              p1: context,
            }),
          );
          pass.setPipeline(this.pipeline("ropeSplitCodec"));
          pass.setBindGroup(
            0,
            this.attendBind(
              ropeUniforms,
              this.qkv,
              weightsChunk,
              this.ring,
              this.ctx,
              this.slotBuffer,
            ),
          );
          pass.dispatchWorkgroups(CODEC_HEADS);

          const attnUniforms = this.params(
            `${frameIndex}:s${s}:l${layerIndex}:att:${i}`,
            params({
              n: D,
              k: 0,
              dOff: 0,
              sOff: slot.slotBase,
              aOff: slot.valueBase,
              cOff: slot.keyBase,
              p0: position,
              p1: context,
            }),
          );
          pass.setPipeline(this.attendPipeline);
          pass.setBindGroup(
            0,
            this.attendBind(
              attnUniforms,
              this.qkv,
              weightsChunk,
              this.ring,
              this.ctx,
              this.slotBuffer,
            ),
          );
          pass.dispatchWorkgroups(CODEC_HEADS);

          run(
            "gemvNoBias",
            `s${s}:l${layerIndex}:outproj:${i}`,
            params({ n: D, k: D, wOff: layer.outProj.offset, sOff: 0 }),
            layer.outProj.chunk,
            this.ctx,
            this.proj,
            Math.ceil(D / 64),
          );
          run(
            "layerScaleAdd",
            `s${s}:l${layerIndex}:ls1:${i}`,
            params({ n: D, k: 0, dOff: xOff, sOff: 0, wOff: layer.scale1.offset }),
            layer.scale1.chunk,
            this.proj,
            this.stream,
            Math.ceil(D / 256),
          );

          run(
            "layerNorm1",
            `s${s}:l${layerIndex}:ln2:${i}`,
            params({
              n: D,
              k: 0,
              wOff: layer.norm2W.offset,
              bOff: layer.norm2B.offset,
              sOff: xOff,
            }),
            layer.norm2W.chunk,
            this.stream,
            this.norm,
            1,
            layer.norm2B.chunk,
          );
          run(
            "gemvNoBias",
            `s${s}:l${layerIndex}:ffn0:${i}`,
            params({ n: 1024, k: D, wOff: layer.ffnIn.offset, sOff: 0, dOff: 0 }),
            layer.ffnIn.chunk,
            this.norm,
            this.ffn,
            1024 / 64,
          );
          run(
            "geluErf",
            `s${s}:l${layerIndex}:gelu:${i}`,
            params({ n: 1024, k: 0, dOff: 0 }),
            layer.ffnIn.chunk,
            this.norm,
            this.ffn,
            Math.ceil(1024 / 256),
          );
          run(
            "gemvNoBias",
            `s${s}:l${layerIndex}:ffn2:${i}`,
            params({ n: D, k: 1024, wOff: layer.ffnOut.offset, sOff: 0 }),
            layer.ffnOut.chunk,
            this.ffn,
            this.proj,
            Math.ceil(D / 64),
          );
          run(
            "layerScaleAdd",
            `s${s}:l${layerIndex}:ls2:${i}`,
            params({ n: D, k: 0, dOff: xOff, sOff: 0, wOff: layer.scale2.offset }),
            layer.scale2.chunk,
            this.proj,
            this.stream,
            Math.ceil(D / 256),
          );
        }
        layerIndex += 1;
      }

      // the stage's output projection, then the ×2 sub-pixel upsample (after the
      // last stage there is none: its 240-wide output is the latent)
      for (let i = 0; i < n; i += 1) {
        run(
          "gemvNoBias",
          `s${s}:outproj:${i}`,
          params({
            n: outWidth,
            k: D,
            wOff: stage.outputProj.offset,
            sOff: i * D,
            dOff: i * outWidth,
          }),
          stage.outputProj.chunk,
          this.stream,
          this.wide,
          Math.ceil(outWidth / 64),
        );
      }
      this.stagePositions[s] = positionBase + n;
      if (s < this.stages.length - 1) {
        run(
          "upsample2",
          `s${s}:up`,
          params({ n, k: 0, p0: 768 / 2, sOff: 0, dOff: 0 }),
          stage.outputProj.chunk,
          this.wide,
          this.narrow,
          Math.ceil((n * 768) / 256),
        );
        stageInput = this.narrow;
        stageInputWidth = 384;
      }
    }

    // ---- head: latent [n, 240] -> interleaved stereo PCM
    run(
      "codecHead",
      "head",
      params({
        n: this.stages[this.stages.length - 1].positions,
        k: 0,
        sOff: 0,
        dOff: frameIndex * 2 * SAMPLES_PER_FRAME,
      }),
      this.stages[this.stages.length - 1].outputProj.chunk,
      this.wide,
      this.pcm,
      Math.ceil((2 * 120 * this.stages[this.stages.length - 1].positions) / 256),
    );
  }

  /**
   * Decode `frames` (16 codes each) into 48 kHz stereo PCM, one Float32Array per
   * channel, continuing whatever is already in the ring caches.
   */
  async decodeFrames(frames: number[][]): Promise<Float32Array[]> {
    if (!frames.length) return [];
    const channels: Float32Array[] = [
      new Float32Array(frames.length * SAMPLES_PER_FRAME),
      new Float32Array(frames.length * SAMPLES_PER_FRAME),
    ];
    // the PCM staging buffer holds a batch at a time; the ring state carries over
    const batch = Math.min(frames.length, this.pcmCapacityFrames);
    for (let start = 0; start < frames.length; start += batch) {
      const slice = frames.slice(start, start + batch);
      for (let i = 0; i < slice.length; i += 1) {
        const codes = new Float32Array(N_VQ);
        for (let k = 0; k < N_VQ; k += 1) codes[k] = slice[i][k] ?? 0;
        this.device.queue.writeBuffer(this.codes, 0, codes);
        const encoder = this.device.createCommandEncoder({ label: `codec-frame-${i}` });
        const pass = encoder.beginComputePass();
        this.encodeFrame(pass, i);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
      }
      const bytes = slice.length * 2 * SAMPLES_PER_FRAME * 4;
      const staging = this.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const copy = this.device.createCommandEncoder();
      copy.copyBufferToBuffer(this.pcm, 0, staging, 0, bytes);
      this.device.queue.submit([copy.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const raw = new Float32Array(staging.getMappedRange().slice(0, bytes));
      staging.unmap();
      staging.destroy();
      const perFrame = 2 * SAMPLES_PER_FRAME;
      for (let c = 0; c < 2; c += 1) {
        for (let i = 0; i < slice.length; i += 1) {
          const from = i * perFrame + c * SAMPLES_PER_FRAME;
          channels[c].set(
            raw.subarray(from, from + SAMPLES_PER_FRAME),
            (start + i) * SAMPLES_PER_FRAME,
          );
        }
      }
    }
    return channels;
  }

  dispose(): void {
    for (const buffer of [
      this.ring,
      this.slotBuffer,
      this.codes,
      this.accum,
      this.stage0,
      this.wide,
      this.narrow,
      this.stream,
      this.norm,
      this.qkv,
      this.ctx,
      this.ffn,
      this.proj,
      this.pcm,
    ]) {
      buffer.destroy();
    }
    for (const buffer of this.uniforms.values()) buffer.destroy();
  }
}
