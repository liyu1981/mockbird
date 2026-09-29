/**
 * Mockbird streaming playback worklet.
 *
 * Loaded as a plain module from /public (see docs/adr/0001-bundler-parity.md):
 * webpack 5 emits unknown extensions as raw assets, which the browser refuses to
 * execute as a module script, and Turbopack handles it differently. Serving it
 * from public/ makes dev (webpack) and production (Turbopack) behave identically.
 *
 * This file is the source of truth (it is not generated). It is plain JavaScript
 * because TypeScript's DOM/WebWorker libs do not declare the AudioWorklet
 * globals, so a .ts version would need ambient declarations anyway.
 */
/**
 * Streaming playback worklet.
 *
 * The decoder emits variable-size 48 kHz stereo chunks while generation is still
 * running, so we keep a lock-free-ish ring buffer per channel and drain it in
 * 128-frame render quanta. That gives gapless playback with a fixed callback
 * cost, and lets the UI show "buffered / playing" without touching the main
 * thread's audio graph on every chunk.
 */

/**
 * @typedef {{ type: "append", left: Float32Array, right: Float32Array, sampleRate: number }
 *   | { type: "flush" } | { type: "pause" } | { type: "resume" } | { type: "stop" }} CommandMessage
 */

const CAPACITY = 48_000 * 30; // 30 seconds of headroom

class StreamPlayerProcessor extends AudioWorkletProcessor {
  left = new Float32Array(CAPACITY);
  right = new Float32Array(CAPACITY);
  writeIndex = 0;
  readIndex = 0;
  paused = false;
  sampleRate = 48000;
  draining = false;

  constructor() {
    super();
    /** @param {MessageEvent<CommandMessage>} event */
    this.port.onmessage = (event) => {
      const message = event.data;
      switch (message.type) {
        case "append": {
          if (message.sampleRate && message.sampleRate !== this.sampleRate) {
            this.sampleRate = message.sampleRate;
          }
          this.push(message.left, message.right);
          this.port.postMessage({
            type: "buffered",
            frames: this.available(),
            paused: this.paused,
          });
          break;
        }
        case "flush": {
          this.writeIndex = 0;
          this.readIndex = 0;
          this.port.postMessage({ type: "buffered", frames: 0, paused: this.paused });
          break;
        }
        case "pause": {
          this.paused = true;
          this.port.postMessage({ type: "state", paused: true });
          break;
        }
        case "resume": {
          this.paused = false;
          this.port.postMessage({ type: "state", paused: false });
          break;
        }
        case "stop": {
          this.writeIndex = 0;
          this.readIndex = 0;
          this.paused = false;
          this.port.postMessage({ type: "stopped" });
          break;
        }
        default:
          break;
      }
    };
  }

  available() {
    return this.writeIndex - this.readIndex;
  }

  push(left, right) {
    const length = left.length;
    if (length >= CAPACITY) {
      // Keep only the tail; the decoder outran playback by more than 30 s.
      const offset = length - CAPACITY;
      this.left.set(left.subarray(offset));
      this.right.set(right.subarray(offset));
      this.readIndex = 0;
      this.writeIndex = CAPACITY;
      return;
    }
    if (this.available() + length > CAPACITY) {
      // Drop the oldest audio and report it, so the UI can show a gap notice.
      const overflow = this.available() + length - CAPACITY;
      this.readIndex += overflow;
      this.port.postMessage({ type: "overflow", dropped: overflow });
    }
    for (let index = 0; index < length; index += 1) {
      this.left[this.writeIndex] = left[index];
      this.right[this.writeIndex] = right[index];
      this.writeIndex = (this.writeIndex + 1) % CAPACITY;
    }
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    if (!output || output.length === 0) return true;
    const leftOut = output[0];
    const rightOut = output[1] ?? output[0];
    const frames = leftOut.length;

    if (this.draining) {
      this.readIndex = this.writeIndex;
      this.draining = false;
    }

    if (this.paused) {
      leftOut.fill(0);
      rightOut.fill(0);
      return true;
    }

    // Underrun: emit silence but stay alive for the next chunk.
    if (this.available() === 0) {
      leftOut.fill(0);
      rightOut.fill(0);
      this.port.postMessage({ type: "underrun" });
      return true;
    }

    const take = Math.min(frames, this.available());
    for (let index = 0; index < take; index += 1) {
      leftOut[index] = this.left[this.readIndex];
      rightOut[index] = this.right[this.readIndex];
      this.readIndex = (this.readIndex + 1) % CAPACITY;
    }
    for (let index = take; index < frames; index += 1) {
      leftOut[index] = 0;
      rightOut[index] = 0;
    }
    if (this.available() === 0) {
      this.port.postMessage({ type: "drained" });
    }
    return true;
  }

  flushQueue() {
    this.draining = true;
  }
}

registerProcessor("mockbird-stream-player", StreamPlayerProcessor);
