import { fileUrl, type ModelFile } from "./modelManifest";
import {
  deleteModelFile,
  type ModelFileWriter,
  modelFileSizeOnDisk,
  openModelFileWriter,
  readModelIndex,
  recordModelFile,
} from "./opfsStore";

export type DownloadPhase = "idle" | "checking" | "downloading" | "done" | "error";

export type FileProgress = {
  path: string;
  group: ModelFile["group"];
  phase: DownloadPhase;
  received: number;
  total: number;
  /** Bytes per second over a sliding window. */
  speed: number;
  error?: string;
  /** True when the bytes came from the OPFS cache instead of the network. */
  fromCache: boolean;
};

export type DownloadState = Record<string, FileProgress>;

export type DownloadEvent =
  | { type: "progress"; state: DownloadState }
  | { type: "done"; state: DownloadState }
  | { type: "error"; message: string };

const MAX_ATTEMPTS = 4;
const MAX_PARALLEL = 3;

function emptyProgress(file: ModelFile): FileProgress {
  return {
    path: file.path,
    group: file.group,
    phase: "idle",
    received: 0,
    total: file.size,
    speed: 0,
    fromCache: false,
  };
}

class SpeedMeter {
  private samples: { at: number; bytes: number }[] = [];

  add(bytes: number, stamp: number) {
    this.samples.push({ at: stamp, bytes });
    const cutoff = stamp - 4000;
    while (this.samples.length > 1 && this.samples[0].at < cutoff) {
      this.samples.shift();
    }
  }

  value(): number {
    if (this.samples.length < 2) return 0;
    const first = this.samples[0];
    const last = this.samples[this.samples.length - 1];
    const seconds = (last.at - first.at) / 1000;
    if (seconds <= 0) return 0;
    const bytes = last.bytes - first.bytes;
    return bytes / seconds;
  }

  reset(bytes: number, at: number) {
    this.samples = [{ at, bytes }];
  }
}

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

export type DownloadOptions = {
  files: ModelFile[];
  signal?: AbortSignal;
  onEvent: (event: DownloadEvent) => void;
};

/**
 * Downloads model files into OPFS with HTTP-range resume, per-file progress and
 * bounded parallelism. Files already recorded in the OPFS index (matching
 * `content-length`) are skipped, which is what makes the second visit instant.
 */
export async function downloadModelFiles({
  files,
  signal,
  onEvent,
}: DownloadOptions): Promise<DownloadState> {
  const index = await readModelIndex();
  const state: DownloadState = {};

  const emit = (event: DownloadEvent) => onEvent(event);

  for (const file of files) {
    state[file.path] = emptyProgress(file);
  }

  const queue: ModelFile[] = [];
  for (const file of files) {
    const progress = state[file.path];
    progress.phase = "checking";
    const cached = index[file.path];
    const onDisk = cached ? await modelFileSizeOnDisk(file.path) : 0;
    if (cached && onDisk > 0 && onDisk === cached.size) {
      progress.phase = "done";
      progress.received = onDisk;
      progress.total = onDisk;
      progress.fromCache = true;
      continue;
    }
    queue.push(file);
  }
  emit({ type: "progress", state: structuredClone(state) });

  const failures: string[] = [];

  const runOne = async (file: ModelFile): Promise<void> => {
    const progress = state[file.path];
    const meter = new SpeedMeter();

    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
      if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
      const resumeFrom = await modelFileSizeOnDisk(file.path);
      meter.reset(resumeFrom, now());
      progress.phase = "downloading";
      progress.received = resumeFrom;
      progress.error = undefined;

      let writer: ModelFileWriter | null = null;
      try {
        const headers: Record<string, string> = {};
        if (resumeFrom > 0) headers.Range = `bytes=${resumeFrom}-`;
        const response = await fetch(fileUrl(file), {
          headers,
          // The HF CDN answers with `access-control-allow-origin: *`, which is
          // also what lets the request pass our `require-corp` policy.
          mode: "cors",
          credentials: "omit",
          cache: "no-store",
          signal,
        });

        if (response.status === 416) {
          // Range not satisfiable: the partial file is bogus, start over.
          await deleteModelFile(file.path);
          continue;
        }
        if (!response.ok && response.status !== 206) {
          throw new Error(`HTTP ${response.status} ${response.statusText}`);
        }

        const contentLength = Number(response.headers.get("content-length") ?? 0);
        const etag = response.headers.get("etag");
        if (response.status === 200) {
          // Server ignored the range request: restart from zero.
          await deleteModelFile(file.path);
          progress.received = 0;
          meter.reset(0, now());
        } else if (resumeFrom > 0) {
          progress.received = resumeFrom;
        }
        if (contentLength > 0) {
          progress.total = progress.received + contentLength;
        }

        writer = await openModelFileWriter(
          file.path,
          resumeFrom > 0 && response.status === 206,
        );
        progress.total = Math.max(progress.total, writer.startingOffset + contentLength);

        if (!response.body) throw new Error("Response has no body (streaming unsupported).");
        const reader = response.body.getReader();
        let received = progress.received;
        let lastEmit = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          if (!value) continue;
          await writer.write(value);
          received += value.byteLength;
          progress.received = received;
          meter.add(received, now());
          const stamp = now();
          if (stamp - lastEmit > 250) {
            lastEmit = stamp;
            progress.speed = meter.value();
            emit({ type: "progress", state: structuredClone(state) });
          }
        }

        await writer.close();
        writer = null;

        const written = await modelFileSizeOnDisk(file.path);
        await recordModelFile(file.path, {
          size: written,
          etag,
          updatedAt: Date.now(),
        });
        progress.received = written;
        progress.total = written;
        progress.phase = "done";
        progress.speed = 0;
        emit({ type: "progress", state: structuredClone(state) });
        return;
      } catch (error) {
        if (writer) await writer.abort();
        if ((error as DOMException)?.name === "AbortError") throw error;
        const message = error instanceof Error ? error.message : String(error);
        progress.error = message;
        if (attempt === MAX_ATTEMPTS) {
          progress.phase = "error";
          failures.push(`${file.path}: ${message}`);
          emit({ type: "progress", state: structuredClone(state) });
          return;
        }
        // Linear backoff, then resume from whatever landed on disk.
        await new Promise((resolve) => setTimeout(resolve, attempt * 1200));
      }
    }
  };

  const runners: Promise<void>[] = [];
  let cursor = 0;
  const pump = async () => {
    for (;;) {
      if (cursor >= queue.length) return;
      const file = queue[cursor++];
      await runOne(file);
    }
  };
  for (let i = 0; i < Math.min(MAX_PARALLEL, queue.length); i += 1) {
    runners.push(pump());
  }

  try {
    await Promise.all(runners);
  } catch (error) {
    if ((error as DOMException)?.name === "AbortError") throw error;
    throw error;
  }

  emit({ type: "progress", state: structuredClone(state) });
  if (failures.length > 0) {
    const message = failures.join("; ");
    emit({ type: "error", message });
    throw new Error(message);
  }
  emit({ type: "done", state: structuredClone(state) });
  return state;
}
