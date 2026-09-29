/**
 * OPFS (Origin Private File System) helpers.
 *
 * Layout:
 *   models/<repo-dir>/<file>          model weights + metadata
 *   models/.index.json                { [relativePath]: { size, etag, updatedAt } }
 *   voices/<id>/meta.json             VoiceMeta
 *   voices/<id>/codes.bin             Float-free Uint16 codes, frame-major [frames][nVq]
 *
 * Everything runs inside the TTS worker; `navigator.storage.getDirectory()` is
 * available in Workers in every browser that supports OPFS at all.
 */

export type ModelFileRecord = {
  size: number;
  etag: string | null;
  updatedAt: number;
};

export type OpfsCapabilities = {
  opfs: boolean;
  persist: boolean;
  crossOriginIsolated: boolean;
  sharedArrayBuffer: boolean;
  quota: number | null;
  usage: number | null;
};

const MODELS_DIR = "models";
const VOICES_DIR = "voices";
const INDEX_FILE = ".index.json";

export function isOpfsAvailable(): boolean {
  return (
    typeof navigator !== "undefined" && typeof navigator.storage?.getDirectory === "function"
  );
}

async function rootDir(): Promise<FileSystemDirectoryHandle> {
  if (!isOpfsAvailable()) {
    throw new Error(
      "The Origin Private File System is unavailable. Use a Chromium-based browser (Chrome/Edge 108+).",
    );
  }
  const root = await navigator.storage.getDirectory();
  if (typeof navigator.storage.persist === "function") {
    // Best effort: keeps the 763 MB of weights from being evicted.
    await navigator.storage.persist().catch(() => false);
  }
  return root;
}

async function getDir(
  path: string,
  create: boolean,
): Promise<FileSystemDirectoryHandle | null> {
  let handle = await rootDir();
  for (const segment of path.split("/").filter(Boolean)) {
    try {
      handle = await handle.getDirectoryHandle(segment, { create });
    } catch (error) {
      if ((error as DOMException)?.name === "NotFoundError") return null;
      throw error;
    }
  }
  return handle;
}

async function getFile(path: string, create: boolean): Promise<FileSystemFileHandle | null> {
  const segments = path.split("/").filter(Boolean);
  const fileName = segments.pop();
  if (!fileName) return null;
  const dir = await getDir(segments.join("/"), create);
  if (!dir) return null;
  try {
    return await dir.getFileHandle(fileName, { create });
  } catch (error) {
    if ((error as DOMException)?.name === "NotFoundError") return null;
    throw error;
  }
}

// ---------------------------------------------------------------------------
// model index
// ---------------------------------------------------------------------------

type ModelIndex = Record<string, ModelFileRecord>;

let indexCache: ModelIndex | null = null;

export async function readModelIndex(): Promise<ModelIndex> {
  if (indexCache) return indexCache;
  const handle = await getFile(`${MODELS_DIR}/${INDEX_FILE}`, false);
  if (!handle) {
    indexCache = {};
    return indexCache;
  }
  try {
    const file = await handle.getFile();
    indexCache = JSON.parse(await file.text()) as ModelIndex;
  } catch {
    indexCache = {};
  }
  return indexCache;
}

async function writeModelIndex(index: ModelIndex): Promise<void> {
  indexCache = index;
  const handle = await getFile(`${MODELS_DIR}/${INDEX_FILE}`, true);
  if (!handle) return;
  const writable = await handle.createWritable();
  await writable.write(JSON.stringify(index));
  await writable.close();
}

export async function recordModelFile(
  relativePath: string,
  record: ModelFileRecord,
): Promise<void> {
  const index = await readModelIndex();
  index[relativePath] = record;
  await writeModelIndex(index);
}

export async function forgetModelFile(relativePath: string): Promise<void> {
  await deleteModelFile(relativePath);
}

/** Relative paths of every model file currently in the cache. */
export async function listCachedModelFiles(): Promise<string[]> {
  const index = await readModelIndex();
  return Object.keys(index);
}

/**
 * Returns a `Uint8Array` (not an `ArrayBuffer`) to match the upstream runtime's
 * asset reader contract: it iterates `.length` when base64-encoding the
 * tokenizer model, and ORT accepts either.
 */
export async function readModelFileBuffer(relativePath: string): Promise<Uint8Array> {
  const handle = await getFile(`${MODELS_DIR}/${relativePath}`, false);
  if (!handle) {
    throw new Error(`Model file missing from cache: ${relativePath}`);
  }
  const file = await handle.getFile();
  return new Uint8Array(await file.arrayBuffer());
}

export async function hasModelFile(relativePath: string): Promise<boolean> {
  const handle = await getFile(`${MODELS_DIR}/${relativePath}`, false);
  return Boolean(handle);
}

export async function readModelFileText(relativePath: string): Promise<string> {
  const handle = await getFile(`${MODELS_DIR}/${relativePath}`, false);
  if (!handle) {
    throw new Error(`Model file missing from cache: ${relativePath}`);
  }
  const file = await handle.getFile();
  return file.text();
}

export type ModelFileWriter = {
  write: (chunk: Uint8Array) => Promise<void>;
  close: () => Promise<void>;
  abort: () => Promise<void>;
  /** Bytes already on disk when this writer was opened (resume support). */
  startingOffset: number;
};

/**
 * Opens a writer for a model file. With `append: true` the existing bytes are
 * kept and the stream is seeked to the end, which is what makes HTTP-range
 * resume work after a reload or a crash mid-download.
 */
export async function openModelFileWriter(
  relativePath: string,
  append: boolean,
): Promise<ModelFileWriter> {
  const existing = append ? await modelFileSizeOnDisk(relativePath) : 0;
  const handle = await getFile(`${MODELS_DIR}/${relativePath}`, true);
  if (!handle) throw new Error(`Cannot create model file: ${relativePath}`);
  const writable = await handle.createWritable({ keepExistingData: append });
  if (append && existing > 0) {
    await writable.seek(existing);
  }
  return {
    startingOffset: existing,
    write: async (chunk) => {
      // Copy into a fresh ArrayBuffer: FileSystemWriteChunkType rejects
      // ArrayBufferView<ArrayBufferLike> (e.g. views over SharedArrayBuffer).
      await writable.write(new Uint8Array(chunk));
    },
    close: async () => {
      await writable.close();
    },
    abort: async () => {
      await writable.abort().catch(() => undefined);
    },
  };
}

/** Byte length of a partially downloaded file (used to resume). */
export async function modelFileSizeOnDisk(relativePath: string): Promise<number> {
  const handle = await getFile(`${MODELS_DIR}/${relativePath}`, false);
  if (!handle) return 0;
  const file = await handle.getFile();
  return file.size;
}

export async function deleteModelFile(relativePath: string): Promise<void> {
  const segments = relativePath.split("/").filter(Boolean);
  const name = segments.pop();
  if (!name) return;
  const parent = await getDir(`${MODELS_DIR}/${segments.join("/")}`, false);
  if (parent) {
    await parent.removeEntry(name).catch(() => undefined);
  }
  const index = await readModelIndex();
  delete index[relativePath];
  await writeModelIndex(index);
}

export async function clearModelCache(): Promise<void> {
  const dir = await getDir(MODELS_DIR, false);
  if (!dir) return;
  for await (const [name] of dir as unknown as AsyncIterable<[string, FileSystemHandle]>) {
    await dir.removeEntry(name, { recursive: true }).catch(() => undefined);
  }
  indexCache = null;
}

// ---------------------------------------------------------------------------
// voices
// ---------------------------------------------------------------------------

export type StoredVoiceMeta = {
  id: string;
  name: string;
  createdAt: number;
  /** Seconds of reference audio. */
  durationSec: number;
  sampleRate: number;
  channels: number;
  numQuantizers: number;
  frames: number;
  source: "recording" | "upload" | "import";
  notes?: string;
};

export async function listVoices(): Promise<StoredVoiceMeta[]> {
  const dir = await getDir(VOICES_DIR, false);
  if (!dir) return [];
  const voices: StoredVoiceMeta[] = [];
  for await (const [, handle] of dir as unknown as AsyncIterable<[string, FileSystemHandle]>) {
    if (handle.kind !== "directory") continue;
    try {
      const metaHandle = await (handle as FileSystemDirectoryHandle).getFileHandle("meta.json");
      const meta = JSON.parse(await (await metaHandle.getFile()).text()) as StoredVoiceMeta;
      voices.push(meta);
    } catch {
      // Ignore half-written or corrupt voice folders.
    }
  }
  return voices.sort((a, b) => b.createdAt - a.createdAt);
}

export async function writeVoice(meta: StoredVoiceMeta, codes: Uint16Array): Promise<void> {
  const dir = await getDir(`${VOICES_DIR}/${meta.id}`, true);
  if (!dir) throw new Error("Cannot create voice directory.");

  const codesHandle = await dir.getFileHandle("codes.bin", { create: true });
  const codesWritable = await codesHandle.createWritable();
  await codesWritable.write(new Uint16Array(codes));
  await codesWritable.close();

  const metaHandle = await dir.getFileHandle("meta.json", { create: true });
  const metaWritable = await metaHandle.createWritable();
  await metaWritable.write(JSON.stringify(meta, null, 2));
  await metaWritable.close();
}

export async function readVoiceCodes(id: string): Promise<Uint16Array> {
  const dir = await getDir(`${VOICES_DIR}/${id}`, false);
  if (!dir) throw new Error(`Voice not found: ${id}`);
  const handle = await dir.getFileHandle("codes.bin");
  const buffer = await (await handle.getFile()).arrayBuffer();
  return new Uint16Array(buffer);
}

export async function readVoiceMeta(id: string): Promise<StoredVoiceMeta | null> {
  const dir = await getDir(`${VOICES_DIR}/${id}`, false);
  if (!dir) return null;
  try {
    const handle = await dir.getFileHandle("meta.json");
    return JSON.parse(await (await handle.getFile()).text()) as StoredVoiceMeta;
  } catch {
    return null;
  }
}

export async function deleteVoice(id: string): Promise<void> {
  const dir = await getDir(VOICES_DIR, false);
  if (!dir) return;
  await dir.removeEntry(id, { recursive: true }).catch(() => undefined);
}

// ---------------------------------------------------------------------------
// capabilities / quota
// ---------------------------------------------------------------------------

export async function probeCapabilities(): Promise<OpfsCapabilities> {
  let quota: number | null = null;
  let usage: number | null = null;
  try {
    const estimate = await navigator.storage?.estimate?.();
    quota = estimate?.quota ?? null;
    usage = estimate?.usage ?? null;
  } catch {
    /* ignore */
  }
  let persist = false;
  try {
    persist = (await navigator.storage?.persisted?.()) ?? false;
  } catch {
    /* ignore */
  }
  return {
    opfs: isOpfsAvailable(),
    persist,
    crossOriginIsolated:
      typeof crossOriginIsolated !== "undefined" ? crossOriginIsolated : false,
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    quota,
    usage,
  };
}
