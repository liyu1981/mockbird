"""Index-based ONNX graph surgery.

Node *names* in the MOSS traced graphs are not unique (the same /Concat_5,
/Reshape_1 … labels appear in several layers), so name-based output injection
silently binds to the wrong node. Everything here is addressed by topological
node index instead.

  list  <model.onnx> [--from N --to M]        structural listing with indices
  trace <model.onnx> --from N                 forward BFS from node N's outputs
  patch <src> <dst> idx:<nodeIndex>[:type] …  add graph outputs by node index
"""
import collections
import sys


def read_varint(b, i):
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, i
        s += 7


def parse(b):
    i, n, out = 0, len(b), []
    while i < n:
        key, i = read_varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = read_varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            ln, i = read_varint(b, i)
            v, i = b[i:i + ln], i + ln
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(wt)
        out.append((fn, wt, v))
    return out


def s(v):
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


class Graph:
    def __init__(self, path):
        data = open(path, "rb").read()
        self.top = parse(data)
        self.graph_index = next(i for i, (fn, _, _) in enumerate(self.top) if fn == 7)
        parts = parse(self.top[self.graph_index][2])
        self.parts = parts
        self.nodes = []
        self._index_nodes(parts)
        self._index_io(parts)
        self.producer = {}
        for n in self.nodes:
            for o in n["out"]:
                self.producer.setdefault(o, []).append(n)
        self.consumer = collections.defaultdict(list)
        for n in self.nodes:
            for i in n["in"]:
                self.consumer[i].append(n)

    def _index_nodes(self, parts):
        idx = 0
        for fn, _, v in parts:
            if fn != 1:
                continue
            d = collections.defaultdict(list)
            for a, _, w in parse(v):
                d[a].append(w)
            attrs = {}
            for ab in d.get(5, []):
                ad = collections.defaultdict(list)
                for a, _, w in parse(ab):
                    ad[a].append(w)
                an = s(ad[1][0]) if ad.get(1) else "?"
                def ints(field):
                    vals = []
                    for item in ad.get(field, []):
                        if isinstance(item, int):
                            vals.append(item)
                        else:
                            try:
                                for fn2, wt2, v2 in parse(item):
                                    if wt2 == 0:
                                        vals.append(v2)
                            except Exception:
                                pass
                    return vals
                for field, label in ((2, "i"), (3, "f"), (8, "ints")):
                    if ad.get(field):
                        vals = ints(field)
                        if field == 2:
                            attrs[an] = f"{label}:{vals[0] if vals else '?'}"
                        else:
                            attrs[an] = f"{label}:{vals}"
                        break
                else:
                    attrs[an] = "x"
            self.nodes.append({
                "i": idx,
                "op": s(d[4][0]) if d.get(4) else "",
                "name": s(d[3][0]) if d.get(3) else "",
                "in": [s(x) for x in d.get(1, [])],
                "out": [s(x) for x in d.get(2, [])],
                "attrs": attrs,
            })
            idx += 1

    def _index_io(self, parts):
        self.inputs, self.outputs = [], []
        for fn, _, v in parts:
            if fn not in (11, 12):
                continue
            d = collections.defaultdict(list)
            for a, _, w in parse(v):
                d[a].append(w)
            name = s(d[1][0]) if d.get(1) else ""
            (self.inputs if fn == 11 else self.outputs).append(name)


def cmd_list(g, args):
    lo = int(args[args.index("--from") + 1]) if "--from" in args else 0
    hi = int(args[args.index("--to") + 1]) if "--to" in args else len(g.nodes)
    for n in g.nodes[lo:hi]:
        extra = " ".join(f"{k}={v}" for k, v in n["attrs"].items() if k in ("axis", "perm", "to", "starts", "ends", "axes", "steps"))
        print(f'{n["i"]:5d} {n["op"]:14s} {n["name"][:38]:38s} {extra:28s} {n["in"][:3]} -> {n["out"]}')


def cmd_trace(g, args):
    start = int(args[args.index("--from") + 1])
    seen = set()
    frontier = [start]
    order = []
    while frontier:
        cur = frontier.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        order.append(cur)
        for o in g.nodes[cur]["out"]:
            for c in g.consumer.get(o, []):
                if c["i"] not in seen:
                    frontier.append(c["i"])
    for i in sorted(order):
        n = g.nodes[i]
        extra = " ".join(f"{k}={v}" for k, v in n["attrs"].items() if k in ("axis", "perm", "to"))
        print(f'{n["i"]:5d} {n["op"]:14s} {n["name"][:34]:34s} {extra:24s} {n["in"][:2]} -> {n["out"]}')


def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def tag(fn, wt):
    return varint((fn << 3) | wt)


def value_info(name, elem):
    tensor = tag(1, 0) + varint(elem)
    type_pb = tag(1, 2) + varint(len(tensor)) + tensor
    body = tag(1, 2) + varint(len(name)) + name.encode() + tag(2, 2) + varint(len(type_pb)) + type_pb
    return body


ELEM = {"f32": 1, "i32": 6, "i64": 7, "bool": 9, "f16": 10}
ORIGINAL_ELEM = {"audio_code_lengths": 6, "audio_lengths": 6, "attn_cached_positions_out_0": 6, "attn_offset_out_0": 6, "transformer_offset_out_0": 6}
ALIAS = {"@audio_codes": "audio_codes", "@audio_code_lengths": "audio_code_lengths"}


def _emit(out, a, wt, w):
    out += tag(a, wt)
    if wt == 0:
        out += varint(w)
    elif wt == 2:
        out += varint(len(w)) + w
    else:                       # wt 1 / 5: fixed-width, already raw bytes
        out += w
    return out


def _encode_node(nb, renames):
    """Re-emit a NodeProto renaming string fields (1=inputs, 2=outputs) and
    recursing into attributes so If/Loop branch graphs stay consistent."""
    items = []
    for a, wt, w in parse(nb):
        if a in (1, 2) and wt == 2:
            new = renames.get(w.decode())
            b = (new.encode() if new is not None else w)
            items.append((a, 2, b))
        elif a == 5 and wt == 2:
            items.append((a, 2, _encode_attr(w, renames)))
        elif a == 11 and wt == 2:
            items.append((a, 2, _encode_graph(w, renames)))
        else:
            items.append((a, wt, w))
    out = bytearray()
    for a, wt, w in items:
        _emit(out, a, wt, w)
    return bytes(out)


def _encode_attr(ab, renames):
    """Re-emit an AttributeProto, recursing into its subgraphs (field 11).
    Wire types are preserved, so scalar attributes survive untouched."""
    out = bytearray()
    for a, wt, w in parse(ab):
        # AttributeProto: g=6 (single GraphProto), graphs=11 (repeated)
        if a in (6, 11) and wt == 2:
            _emit(out, a, 2, _encode_graph(w, renames))
        else:
            _emit(out, a, wt, w)
    return bytes(out)


def _encode_graph(graph_bytes, renames):
    out = bytearray()
    for fn, wt, v in parse(graph_bytes):
        if fn == 1 and renames:
            v = _encode_node(v, renames)
        _emit(out, fn, wt, v)
    return bytes(out)


def cmd_patch(g, src, dst, specs):
    # these graphs instantiate the same submodule several times, so tensor NAMES
    # collide (4x decoder.8 in the codec). Address nodes by index and rename.
    renames, planned = {}, []
    for spec in specs:
        idx, _, t = spec.partition(":")
        old = g.nodes[int(idx)]["out"][0]
        new = f"__probe_{len(planned)}__"
        renames[old] = new
        planned.append((new, t))
    out = bytearray()
    for fn, wt, v in g.parts:
        if fn == 12:
            continue
        if fn == 1 and renames:
            v = _encode_node(v, renames)
        _emit(out, fn, wt, v)
    def orig_type(name):
        # int32 outputs: offsets, cached positions, lengths, code lengths
        if "offset" in name or "positions" in name or "length" in name:
            return 6
        return ORIGINAL_ELEM.get(name, 1)

    infos = [value_info(name, orig_type(name)) for name in g.outputs]
    added = []
    for spec, (name, t) in zip(specs, planned):
        infos.append(value_info(name, ELEM.get(t, 1)))
        added.append(f"{spec.split(':')[0]}->{name}")
    for body in infos:
        _emit(out, 12, 2, body)
    new_graph = bytes(out)
    model = bytearray()
    for i, (fn, wt, v) in enumerate(g.top):
        _emit(model, fn, wt, new_graph if i == g.graph_index else v)
    open(dst, "wb").write(bytes(model))
    print(f"patched {dst}: {len(infos)} outputs (+{len(added)}: {', '.join(added)})")


if __name__ == "__main__":
    cmd = sys.argv[1]
    g = Graph(sys.argv[2])
    if cmd == "list":
        cmd_list(g, sys.argv[3:])
    elif cmd == "trace":
        cmd_trace(g, sys.argv[3:])
    elif cmd == "patch":
        cmd_patch(g, sys.argv[2], sys.argv[3], sys.argv[4:])
