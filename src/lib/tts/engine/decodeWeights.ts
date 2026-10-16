/**
 * Weight resolution for the parts of the decode path that are not the global
 * transformer: the local frame head and the codec decoder.
 *
 * Both graphs export their projections as anonymous `onnx::MatMul_*` /
 * `onnx::Conv_*` initializers, so they are identified by shape and ordered by
 * sidecar byte offset — the export is layer-major (§10.2). The codec's
 * `[256,768]` shape is shared by 12 attention inputs *and* 3 stage outputs, so
 * the stage/layer split is recovered from the known layer counts (4/2/2/4) and
 * the `[x, x, 1]` conv weights by their per-quantiser bias that precedes them.
 */

import type { CodecStage, CodecWeights } from "./codec";
import type { LocalHeadWeights } from "./decode";
import { type BlockLocation, blocksByShape, type GpuWeights } from "./weights";

type Lookup = { get(name: string): BlockLocation };

/**
 * The local head's single d=768 layer plus the two head matrices. The layer
 * weights are the graph's only [768,2304] / [768,768] / [768,3072] /
 * [3072,768] blocks, so shape alone identifies them.
 */
export function resolveLocalWeights(weights: GpuWeights, table: Lookup): LocalHeadWeights {
  const pick = (shape: string, index: number) => {
    const list = blocksByShape(weights, shape);
    if (!list[index]) throw new Error(`local head: no ${shape} block at index ${index}`);
    return list[index].block;
  };
  const nVq = 16;
  const audioHeads: BlockLocation[] = [];
  for (let k = 0; k < nVq; k += 1) audioHeads.push(pick("768x1024", k));
  return {
    cAttn: pick("768x2304", 0),
    cAttnBias: table.get("model.local_transformer.h.0.attn.c_attn.bias"),
    cProj: pick("768x768", 0),
    cProjBias: table.get("model.local_transformer.h.0.attn.c_proj.bias"),
    ln1W: table.get("model.local_transformer.h.0.ln_1.weight"),
    ln1B: table.get("model.local_transformer.h.0.ln_1.bias"),
    ln2W: table.get("model.local_transformer.h.0.ln_2.weight"),
    ln2B: table.get("model.local_transformer.h.0.ln_2.bias"),
    fcIn: pick("768x3072", 0),
    fcInBias: table.get("model.local_transformer.h.0.mlp.fc_in.bias"),
    fcOut: pick("3072x768", 0),
    fcOutBias: table.get("model.local_transformer.h.0.mlp.fc_out.bias"),
    lnFW: table.get("model.local_transformer.ln_f.weight"),
    lnFB: table.get("model.local_transformer.ln_f.bias"),
    textHead: pick("768x16384", 0),
    audioHeads,
  };
}

/** The codec stage layout, from the `codec_browser_onnx_meta.json` streaming block. */
export type CodecShape = {
  /** layers per stage, in stage order */
  layersPerStage: number[];
  /** ring context per stage, in stage order */
  contextPerStage: number[];
};

/**
 * Resolve the codec decoder's weights.
 *
 * The projection order in the sidecar is, per stage: for each layer
 * `[in_proj, out_proj, ffn.0, ffn.2]`, then one `output_proj`. With the layer
 * counts known that disambiguates the shared `[256,768]` shape, and it is the
 * only way to tell an attention input from a stage output.
 */
export function resolveCodecWeights(
  weights: GpuWeights,
  table: Lookup,
  shape: CodecShape,
): CodecWeights {
  const byShape = (dims: string) => blocksByShape(weights, dims);
  // the [256,768] shape is shared by every layer's attention input AND every
  // stage output, so they are resolved from ONE offset-ordered list and split
  // by the known layer counts
  const wide = byShape("256x768");
  const outProj = byShape("256x256");
  const ffnIn = byShape("256x1024");
  const ffnOut = byShape("1024x256");
  const inputProjWide = byShape("384x256");
  const stageOutput4 = byShape("256x240");
  if (!stageOutput4[0]) throw new Error("codec: no [256,240] stage-4 output projection");

  const expected = shape.layersPerStage.reduce((a, b) => a + b, 0);
  if (outProj.length !== expected) {
    throw new Error(
      `codec: expected ${expected} [256,256] out projections, found ${outProj.length}`,
    );
  }
  // walk the [256,768] run: per stage, layersPerStage attention inputs, then one
  // stage output. The LAST stage is the exception — its output projection is
  // [256,240] (the latent width), so it does not appear in this list at all.
  let cursor = 0;
  let layerCursor = 0;
  const stages: CodecStage[] = [];
  const last = shape.layersPerStage.length - 1;
  for (let s = 0; s < shape.layersPerStage.length; s += 1) {
    const count = shape.layersPerStage[s];
    const layers = [];
    for (let i = 0; i < count; i += 1) {
      const prefix = `model.decoder.${[1, 3, 5, 7][s]}.transformer.layers.${i}.`;
      layers.push({
        inProj: wide[cursor].block,
        outProj: outProj[layerCursor].block,
        ffnIn: ffnIn[layerCursor].block,
        ffnOut: ffnOut[layerCursor].block,
        norm1W: table.get(`${prefix}norm1.weight`),
        norm1B: table.get(`${prefix}norm1.bias`),
        norm2W: table.get(`${prefix}norm2.weight`),
        norm2B: table.get(`${prefix}norm2.bias`),
        scale1: table.get(`${prefix}layer_scale_1.scale`),
        scale2: table.get(`${prefix}layer_scale_2.scale`),
      });
      cursor += 1;
      layerCursor += 1;
    }
    stages.push({
      layers,
      context: shape.contextPerStage[s],
      // stage 1's input projection is [192,256] (the interleaved ×4 front), the
      // rest are [384,256]
      inputProj: s === 0 ? byShape("192x256")[0].block : inputProjWide[s - 1].block,
      // the last stage projects straight to the 240-wide latent
      outputProj: s === last ? stageOutput4[0].block : wide[cursor++].block,
      positions: 4 * 2 ** s,
    });
  }
  if (cursor !== wide.length) {
    throw new Error(
      `codec: consumed ${cursor} of ${wide.length} [256,768] blocks — the stage layout ` +
        `does not match the graph (layer counts per stage: ${shape.layersPerStage.join(",")})`,
    );
  }

  // RVQ front: 16 × [512,8,1] convolutions with the per-quantiser bias, and the
  // [768,512,1] output projection
  const convs = byShape("512x8x1");
  if (convs.length !== 16)
    throw new Error(`codec: expected 16 RVQ convolutions, found ${convs.length}`);
  const quantBias: BlockLocation[] = [];
  const codebook: BlockLocation[] = [];
  for (let k = 0; k < 16; k += 1) {
    quantBias.push(table.get(`model.quantizer.quantizers.${k}.out_proj.bias`));
    codebook.push(table.get(`model.quantizer.quantizers.${k}.codebook.weight`));
  }
  return {
    stages,
    quantProj: convs[0].block,
    quantOutput: byShape("768x512x1")[0].block,
    quantOutputBias: table.get("model.quantizer.output_proj.bias"),
    quantBias,
    codebook,
  };
}
