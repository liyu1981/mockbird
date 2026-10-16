/**
 * ONNX weight reader for the MOSS-TTS-Nano decode graphs.
 *
 * Replaces onnxruntime-web's session/external-data machinery: the engine needs
 * zero-copy views into the `.data` sidecars that Mockbird already holds in OPFS,
 * not a graph executor. Everything here was discovered by reading the published
 * graphs (docs/moss-tts-runtime-analysis.md §10.2) — the loader has to cope with
 * three distinct weight sources and non-aligned inline payloads.
 *
 *   (a) graph initializers      -> external bytes, `offset`/`length` in the sidecar
 *   (b) Constant node attributes -> inline `raw_data` inside the .onnx
 *   (c) Identity node aliases    -> the LayerNorm scales of every layer but 0
 */

const ELEM_SIZE: Record<number, number> = { 1: 4, 6: 4, 7: 8, 10: 2, 11: 8 };

type Field = readonly [number, number, number | Uint8Array];

function readVarint(buf: Uint8Array, i: number): [number, number] {
  let result = 0;
  let shift = 0;
  for (;;) {
    const byte = buf[i++];
    result += (byte & 0x7f) * 2 ** shift;
    if (!(byte & 0x80)) return [result, i];
    shift += 7;
  }
}

/** Iterate protobuf fields as [fieldNumber, wireType, value] tuples. */
function* fields(buf: Uint8Array): Generator<Field> {
  let i = 0;
  while (i < buf.length) {
    const [key, next] = readVarint(buf, i);
    i = next;
    const fieldNo = key >>> 3;
    const wire = key & 7;
    if (wire === 0) {
      const [v, j] = readVarint(buf, i);
      i = j;
      yield [fieldNo, wire, v];
    } else if (wire === 1) {
      yield [fieldNo, wire, buf.subarray(i, i + 8)];
      i += 8;
    } else if (wire === 2) {
      const [len, j] = readVarint(buf, i);
      i = j;
      yield [fieldNo, wire, buf.subarray(i, i + len)];
      i += len;
    } else if (wire === 5) {
      yield [fieldNo, wire, buf.subarray(i, i + 4)];
      i += 4;
    } else {
      throw new Error(`unsupported protobuf wire type ${wire}`);
    }
  }
}

export type Tensor = {
  name: string;
  dims: number[];
  data: Float32Array;
  source: "external" | "inline-constant" | "inline-raw" | "alias";
  /** byte offset inside the sidecar, or -1 for inline tensors */
  offset: number;
};

export type WeightTable = {
  /** resolve a tensor by graph name, following Identity aliases */
  get(name: string): Tensor;
  has(name: string): boolean;
  names(): string[];
  size: number;
  /** initializers + inline constants, ordered by sidecar offset */
  byShape(dims: string): Float32Array[];
};

const decoder = new TextDecoder();

function parseTensorProto(buf: Uint8Array) {
  const dims: number[] = [];
  const external: Record<string, string> = {};
  let name = "";
  let elemType = 0;
  let raw: Uint8Array | null = null;
  for (const [fn, wt, v] of fields(buf)) {
    if (fn === 1 && wt === 0) dims.push(v as number);
    else if (fn === 2) elemType = v as number;
    else if (fn === 8) name = decoder.decode(v as Uint8Array);
    else if (fn === 9) raw = v as Uint8Array;
    else if (fn === 13 && wt === 2) {
      let key = "";
      let value = "";
      for (const [a, , w] of fields(v as Uint8Array)) {
        if (a === 1) key = decoder.decode(w as Uint8Array);
        else if (a === 2) value = decoder.decode(w as Uint8Array);
      }
      external[key] = value;
    }
  }
  return { name, dims, elemType, raw, external };
}

/**
 * Wrap `byteLength` bytes at `byteOffset` as f32. Protobuf payloads are not
 * guaranteed 4-byte aligned, so anything not 4-byte aligned is copied
 * byte-wise first; the sidecar path stays zero-copy.
 */
function toFloat32(
  base: ArrayBufferLike,
  byteOffset: number,
  byteLength: number,
  forceCopy: boolean,
): Float32Array {
  const count = Math.floor(byteLength / 4);
  if (forceCopy || byteOffset % 4 !== 0) {
    const bytes = new Uint8Array(count * 4);
    bytes.set(new Uint8Array(base, byteOffset, count * 4));
    return new Float32Array(bytes.buffer);
  }
  return new Float32Array(base, byteOffset, count);
}

export function openOnnx(onnxBytes: Uint8Array, sidecar: Uint8Array): WeightTable {
  const table = new Map<string, Tensor>();
  const aliases = new Map<string, string>();

  let graph: Uint8Array | null = null;
  for (const [fn, , v] of fields(onnxBytes)) if (fn === 7) graph = v as Uint8Array;
  if (!graph) throw new Error("no GraphProto in model");

  const add = (t: Tensor) => {
    if (!table.has(t.name)) table.set(t.name, t);
  };

  for (const [fn, , gv] of fields(graph)) {
    const v = gv as Uint8Array;
    if (fn === 5) {
      // initializer
      const t = parseTensorProto(v);
      const size = ELEM_SIZE[t.elemType];
      if (!size || t.elemType !== 1) continue; // f32 only for now
      if (t.external.length) {
        const offset = Number(t.external.offset ?? 0);
        const length = Number(t.external.length);
        add({
          name: t.name,
          dims: t.dims,
          data: toFloat32(sidecar.buffer, sidecar.byteOffset + offset, length, false),
          source: "external",
          offset,
        });
      } else if (t.raw && t.raw.byteLength >= 4) {
        add({
          name: t.name,
          dims: t.dims,
          data: toFloat32(t.raw.buffer, t.raw.byteOffset, t.raw.byteLength, true),
          source: "inline-raw",
          offset: -1,
        });
      }
    } else if (fn === 1) {
      // node
      let opType = "";
      const ins: string[] = [];
      const outs: string[] = [];
      const attributes: { name: string; bytes: Uint8Array }[] = [];
      for (const [a, , w] of fields(v)) {
        const b = w as Uint8Array;
        if (a === 1) ins.push(decoder.decode(b));
        else if (a === 2) outs.push(decoder.decode(b));
        else if (a === 4) opType = decoder.decode(b);
        else if (a === 5) {
          let attrName = "";
          let payload: Uint8Array | null = null;
          for (const [x, , y] of fields(b)) {
            if (x === 1) attrName = decoder.decode(y as Uint8Array);
            else if (x === 5) payload = y as Uint8Array;
          }
          if (payload) attributes.push({ name: attrName, bytes: payload });
        }
      }
      if (opType === "Identity" && ins.length === 1 && outs.length === 1)
        aliases.set(outs[0], ins[0]);
      if (opType === "Constant") {
        for (const attr of attributes) {
          if (attr.name !== "value") continue;
          const t = parseTensorProto(attr.bytes);
          if (t.elemType !== 1 || !t.raw || t.raw.byteLength < 4) continue;
          const data = toFloat32(t.raw.buffer, t.raw.byteOffset, t.raw.byteLength, true);
          for (const out of outs)
            add({ name: out, dims: t.dims, data, source: "inline-constant", offset: -1 });
        }
      }
    }
  }

  // resolve Identity aliases (the LayerNorm scales of layers 1..11)
  const resolveName = (name: string) => {
    let current = name;
    for (let hops = 0; hops < 8; hops += 1) {
      if (table.has(current)) return current;
      const next = aliases.get(current);
      if (!next) break;
      current = next;
    }
    return current;
  };

  for (const [alias, target] of aliases) {
    if (table.has(alias)) continue;
    const resolved = resolveName(target);
    const t = table.get(resolved);
    if (t) add({ ...t, name: alias, source: "alias" });
  }

  const get = (name: string) => {
    const t = table.get(resolveName(name));
    if (!t) throw new Error(`missing parameter ${name}`);
    return t;
  };

  return {
    get,
    has: (name: string) => table.has(resolveName(name)),
    names: () => [...table.keys()],
    size: table.size,
    /**
     * Anonymous `onnx::MatMul_*` initializers are identified by SHAPE, and the
     * layer order comes from the sidecar byte offsets (the export is layer-major:
     * c_attn, c_proj, fc_in, fc_out). Returns the data views, not the tensors.
     */
    byShape(dims: string): Float32Array[] {
      return [...table.values()]
        .filter((t) => t.source === "external" && t.dims.join("x") === dims && t.offset >= 0)
        .sort((a, b) => a.offset - b.offset)
        .map((t) => t.data);
    },
  };
}
