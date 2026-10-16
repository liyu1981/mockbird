/**
 * Node-side WebGPU bootstrap for the kernel harness.
 *
 * The engine itself runs in a Web Worker and uses `navigator.gpu`; in Node the
 * `webgpu` package (Dawn) exposes the same classes plus a `create()` factory, so
 * the harness gets an identical API without a browser. This is what makes
 * kernel-level CI possible for a runtime with no CPU fallback.
 */
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

export const gpu = create([]);

export async function requestDevice({ powerPreference = "high-performance" } = {}) {
  const adapter = await gpu.requestAdapter({ powerPreference });
  if (!adapter) throw new Error("no WebGPU adapter available");
  const device = await adapter.requestDevice();
  return { adapter, device };
}
