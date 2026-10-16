/**
 * Weight upload.
 *
 * The ONNX sidecars are bigger than the maximum binding size we can get, so the
 * weights are copied once into a set of chunk-sized GPU buffers. The copy is
 * unavoidable in any case — OPFS bytes live on the CPU and must cross to the GPU
 * — so we use it to place every named tensor on a block boundary, which is what
 * lets a kernel read a whole weight block (e.g. one layer's `c_attn`) from a
 * single chunk with no cross-chunk addressing in WGSL.
 *
 * Zero-copy is preserved where it actually mattered: *parsing* never copies or
 * reorders (§10.2 — the loader hands out views into the sidecar), so the upload
 * is the one and only pass over the bytes.
 */

import { alignUp, type GpuDeviceInfo } from "./device";
import type { WeightTable } from "./onnxWeights";

export type BlockLocation = {
  /** index into `GpuWeights.chunks` */
  chunk: number;
  /** element offset inside that chunk */
  offset: number;
  dims: number[];
  elements: number;
  /** byte offset in the original sidecar — this is the layer ordering key */
  sidecarOffset: number;
};

export type GpuWeights = {
  chunks: GPUBuffer[];
  /** tensor name -> where it landed on the GPU */
  blocks: Map<string, BlockLocation>;
  bytesUploaded: number;
  chunkBytes: number;
  tensorCount: number;
  dispose(): void;
};

/**
 * Upload every tensor of a parsed table, packing whole tensors into
 * `chunkBytes`-sized buffers at block-aligned offsets.
 */
export function uploadWeights(
  device: GPUDevice,
  table: WeightTable,
  info: GpuDeviceInfo,
): GpuWeights {
  const chunkBytes = alignUp(info.chunkBytes);
  // everything below counts ELEMENTS (f32), never bytes — mixing the two is how
  // the rollover check ends up four times too late
  const chunkElements = chunkBytes / 4;
  const chunks: GPUBuffer[] = [];
  const blocks = new Map<string, BlockLocation>();
  let chunkIndex = -1;
  let usedElements = 0;
  let chunk: GPUBuffer | null = null;

  const openChunk = (): GPUBuffer => {
    const created = device.createBuffer({
      size: chunkBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: `moss-weights-${chunks.length}`,
    });
    chunks.push(created);
    chunk = created;
    chunkIndex += 1;
    usedElements = 0;
    return created;
  };

  // sidecar order == the export's layer-major order, which is what
  // `blocksByShape` relies on to recover per-layer matrices.
  const entries = table.names().map((name) => table.get(name));
  entries.sort((a, b) => a.offset - b.offset || a.name.localeCompare(b.name));

  for (const tensor of entries) {
    const elements = tensor.data.length;
    if (elements > chunkElements) {
      throw new Error(
        `tensor ${tensor.name} (${((elements * 4) / 1e6).toFixed(1)} MB) exceeds the ${(chunkBytes / 1e6).toFixed(1)} MB chunk size`,
      );
    }
    // open the next chunk FIRST, then take the offset: computing the offset
    // before rolling over leaves it stale and writes past the buffer end
    const target =
      chunk && alignUp(usedElements) + elements <= chunkElements ? chunk : openChunk();
    const offset = alignUp(usedElements);
    device.queue.writeBuffer(
      target,
      offset * 4,
      tensor.data.buffer,
      tensor.data.byteOffset,
      elements * 4,
    );
    blocks.set(tensor.name, {
      chunk: chunkIndex,
      offset,
      dims: tensor.dims,
      elements,
      sidecarOffset: tensor.offset,
    });
    usedElements = offset + elements;
  }

  return {
    chunks,
    blocks,
    // the last chunk is only partially filled; report what the chunks hold
    bytesUploaded: chunks.length * chunkBytes,
    chunkBytes,
    tensorCount: blocks.size,
    dispose() {
      for (const buffer of chunks) buffer.destroy();
      chunks.length = 0;
      blocks.clear();
    },
  };
}

/** Locate a block, or throw naming the tensor (a missing weight is a load bug). */
export function blockOf(weights: GpuWeights, name: string): BlockLocation {
  const block = weights.blocks.get(name);
  if (!block) throw new Error(`weight block not uploaded: ${name}`);
  return block;
}

/**
 * Anonymous `onnx::MatMul_*` initializers are identified by shape; their layer
 * order comes from the sidecar byte offset (the export is layer-major:
 * c_attn, c_proj, fc_in, fc_out) — NOT from the tensor name, whose numeric
 * suffixes do not sort lexicographically.
 */
export function blocksByShape(
  weights: GpuWeights,
  shape: string,
): { name: string; block: BlockLocation }[] {
  return [...weights.blocks.entries()]
    .map(([name, block]) => ({ name, block }))
    .filter((entry) => entry.block.dims.join("x") === shape)
    .sort((a, b) => a.block.sidecarOffset - b.block.sidecarOffset);
}

/** Sum of a block's element count — used by tests to prove nothing was dropped. */
export function totalElements(weights: GpuWeights): number {
  let total = 0;
  for (const block of weights.blocks.values()) total += block.elements;
  return total;
}
