/**
 * WebGPU device acquisition for the TTS engine.
 *
 * The weight sidecars (440.8 MB global + 229.7 MB local + 44.2 MB codec) are far
 * larger than the *default* `maxStorageBufferBindingSize` of 128 MiB, and the
 * adapter we measured reports exactly that as its maximum — so the weights are
 * split into chunk-sized buffers and the kernels address them through a metadata
 * table (see `weights.ts`). Nothing here assumes one big binding works.
 */

export type GpuDeviceInfo = {
  adapterName: string;
  limits: Record<string, number>;
  features: string[];
  /** largest binding we can use, already min()'d against what we need */
  chunkBytes: number;
  /** max storage buffers bindable per stage */
  storageBuffersPerStage: number;
  /** true when the adapter offers f16 arithmetic (halves the bandwidth bill) */
  shaderF16: boolean;
};

/** Bytes we would like to bind in one shot; the adapter usually says no. */
const DESIRED_BINDING_BYTES = 512 * 1024 * 1024;
/** Weight-block alignment: f32, so 4 bytes — plus slack for a future f16 path. */
const BLOCK_ALIGNMENT = 256;

/**
 * `gpu` is injectable so the same code path runs in the browser (`navigator.gpu`)
 * and under the headless Dawn binding used by `tests/gpu`. Keeping it injected
 * rather than read from a global also means the engine never depends on where it
 * was constructed from.
 */
export async function createDevice(provider?: GPU): Promise<{
  device: GPUDevice;
  info: GpuDeviceInfo;
}> {
  const gpu =
    provider ??
    (globalThis.navigator as (Navigator & { gpu?: GPU }) | undefined)?.gpu ??
    undefined;
  if (!gpu) {
    throw new Error("WebGPU is not available in this browser/context (navigator.gpu missing)");
  }
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("no WebGPU adapter — the WGSL engine needs one");
  }

  // ask for what we can actually use; every limit is clamped to the adapter's max
  const maxBinding = Math.min(
    adapter.limits.maxStorageBufferBindingSize,
    DESIRED_BINDING_BYTES,
  );
  const maxBuffer = Math.min(
    adapter.limits.maxBufferSize,
    Math.max(maxBinding, DESIRED_BINDING_BYTES),
  );
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: maxBinding,
      maxBufferSize: maxBuffer,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
    },
  });

  const info: GpuDeviceInfo = {
    adapterName: describeAdapter(adapter),
    limits: {
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
      maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
    },
    features: [...adapter.features],
    chunkBytes: Math.min(device.limits.maxStorageBufferBindingSize, maxBuffer),
    storageBuffersPerStage: device.limits.maxStorageBuffersPerShaderStage,
    shaderF16: adapter.features.has("shader-f16"),
  };
  return { device, info };
}

function describeAdapter(adapter: GPUAdapter): string {
  const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
  if (!info) return "unknown adapter";
  const parts = [info.vendor, info.architecture, info.description, info.device].filter(Boolean);
  return parts.length ? parts.join(" / ") : "unnamed adapter";
}

export function alignUp(value: number, alignment = BLOCK_ALIGNMENT): number {
  return Math.ceil(value / alignment) * alignment;
}
