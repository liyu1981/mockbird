/**
 * Audio helpers that need a DOM `AudioContext`, therefore they live on the main
 * thread. The worker receives plain channel-major PCM.
 */

export const TARGET_SAMPLE_RATE = 48_000;
export const TARGET_CHANNELS = 2;

/**
 * Decodes any audio/video file the browser understands and returns 48 kHz stereo
 * channel-major PCM, which is exactly what `moss_audio_tokenizer_encode.onnx`
 * expects.
 */
export async function decodeToModelPcm(
  data: ArrayBuffer,
): Promise<{ channels: Float32Array[]; sampleRate: number; durationSec: number }> {
  // decodeAudioData detaches the buffer, so always hand it a copy.
  const decodeContext = new AudioContext();
  let decoded: AudioBuffer;
  try {
    decoded = await decodeContext.decodeAudioData(data.slice(0));
  } catch (error) {
    throw new Error(
      `Could not decode this audio file (${error instanceof Error ? error.message : String(error)}). Try WAV, MP3, M4A, OGG or WebM.`,
    );
  } finally {
    await decodeContext.close().catch(() => undefined);
  }

  const frameCount = Math.max(1, Math.ceil(decoded.duration * TARGET_SAMPLE_RATE));
  const needsResample =
    decoded.sampleRate !== TARGET_SAMPLE_RATE || decoded.numberOfChannels !== TARGET_CHANNELS;

  let rendered = decoded;
  if (needsResample) {
    // OfflineAudioContext does the resampling (and mono→stereo upmix) natively.
    const offline = new OfflineAudioContext(TARGET_CHANNELS, frameCount, TARGET_SAMPLE_RATE);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start(0);
    rendered = await offline.startRendering();
  }

  const length = rendered.length;
  const channels: Float32Array[] = [];
  for (let channel = 0; channel < TARGET_CHANNELS; channel += 1) {
    const data = rendered.getChannelData(Math.min(channel, rendered.numberOfChannels - 1));
    // Copy: the AudioBuffer may be reused/GC'd after this call.
    channels.push(new Float32Array(data.subarray(0, length)));
  }

  return {
    channels,
    sampleRate: TARGET_SAMPLE_RATE,
    durationSec: rendered.length / TARGET_SAMPLE_RATE,
  };
}

/** Peak/RMS analysis used for the "recording quality" hints. */
export function analyzePcm(channels: Float32Array[]): {
  peak: number;
  rms: number;
  clipped: number;
} {
  const left = channels[0] ?? new Float32Array(0);
  let peak = 0;
  let sumSquares = 0;
  let clipped = 0;
  for (let index = 0; index < left.length; index += 1) {
    const value = Math.abs(left[index]);
    if (value > peak) peak = value;
    if (value > 0.985) clipped += 1;
    sumSquares += value * value;
  }
  return {
    peak,
    rms: Math.sqrt(sumSquares / Math.max(1, left.length)),
    clipped: clipped / Math.max(1, left.length),
  };
}

/** Encodes channel-major float PCM as a 16-bit stereo WAV file. */
export function encodeWav(channels: Float32Array[], sampleRate: number): Blob {
  const channelCount = Math.max(1, channels.length);
  const frameCount = channels[0]?.length ?? 0;
  const bytesPerSample = 2;
  const blockAlign = channelCount * bytesPerSample;
  const dataSize = frameCount * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, text: string) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // format = PCM
  view.setUint16(22, channelCount, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let frame = 0; frame < frameCount; frame += 1) {
    for (let channel = 0; channel < channelCount; channel += 1) {
      const sample = Math.max(-1, Math.min(1, channels[channel][frame] ?? 0));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }

  return new Blob([buffer], { type: "audio/wav" });
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** Down-mixes a channel-major buffer to a mono envelope for the waveform view. */
export function toWaveformPeaks(channels: Float32Array[], buckets: number): Float32Array {
  const left = channels[0] ?? new Float32Array(0);
  const right = channels[1] ?? left;
  const peaks = new Float32Array(buckets);
  const perBucket = Math.max(1, Math.floor(left.length / buckets));
  for (let bucket = 0; bucket < buckets; bucket += 1) {
    const start = bucket * perBucket;
    const end = Math.min(left.length, start + perBucket);
    let peak = 0;
    for (let index = start; index < end; index += 1) {
      const value = Math.abs((left[index] + right[index]) * 0.5);
      if (value > peak) peak = value;
    }
    peaks[bucket] = peak;
  }
  return peaks;
}
