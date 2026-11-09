import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

const read = (path) => readFileSync(new URL(path, import.meta.url), "utf8");
const context = { ArrayBuffer, DataView, Uint8Array, BigInt, JSON, Math };
runInNewContext(read("../web/assets/i18n.js"), context);
runInNewContext(read("../web/assets/capture.js"), context);
runInNewContext(read("../web/assets/captions.js"), context);
const { FrameQueue, encodeFrame } = context.YataCapture;
const { CaptionBoard } = context.YataCaptions;

const frame = (seq, sampleStart, samples = 4096) => ({ seq, sampleStart, pcm: new ArrayBuffer(samples * 4) });

test("frames are encoded as uint32 seq, uint64 sample start and float32 payload", () => {
  const pcm = new Float32Array([0.5, -0.25]).buffer;
  const packet = encodeFrame(7, 2 ** 33, pcm);
  const view = new DataView(packet);
  assert.equal(packet.byteLength, 12 + 8);
  assert.equal(view.getUint32(0, true), 7);
  assert.equal(view.getBigUint64(4, true), 2n ** 33n);
  assert.equal(view.getFloat32(12, true), 0.5);
});

test("queue keeps at most 60 s and sends one gap marker before the next frame", () => {
  const queue = new FrameQueue(48_000);
  for (let seq = 0; seq < 800; seq++) queue.push(frame(seq, seq * 4096));
  const span = queue.frames.at(-1).sampleStart + 4096 - queue.frames[0].sampleStart;
  assert.ok(span <= 48_000 * 60);
  assert.ok(queue.gap && queue.gap.start === 0 && queue.gap.end === queue.frames[0].sampleStart);
  const sent = queue.take(0);
  const marker = JSON.parse(sent[0]);
  assert.equal(marker.type, "audio_gap");
  assert.equal(marker.end_sample, marker.next_sample);
  assert.equal(typeof sent[1], "object");
  assert.equal(sent.filter((item) => typeof item === "string").length, 1);
  assert.equal(queue.gap, null);
});

test("socket backpressure limits what is taken and undelivered audio is reported", () => {
  const queue = new FrameQueue(16_000);
  for (let seq = 0; seq < 10; seq++) queue.push(frame(seq, seq * 4096));
  assert.equal(queue.take(1_048_576).length, 0);
  const taken = queue.take(1_048_576 - 3 * (4096 * 4 + 12));
  assert.equal(taken.length, 3);
  assert.deepEqual({ ...queue.undelivered() }, { start: 3 * 4096, end: 10 * 4096 });
  queue.take(0);
  assert.equal(queue.undelivered(), null);
});

test("the caption board keeps two finals and replaces a partial with its final", () => {
  const board = new CaptionBoard(2);
  board.addFinal({ segment_id: "a", utterance_id: "u1", text: "一つ目", capture_start_ms: 0, language: "ja" });
  board.setPartial({ utterance_id: "u2", text: "二つ", language: "ja" });
  assert.deepEqual([...board.lines().map((line) => line.kind)], ["final", "partial"]);
  board.addFinal({ segment_id: "b", utterance_id: "u2", text: "二つ目", capture_start_ms: 1000 });
  board.addFinal({ segment_id: "b", utterance_id: "u2", text: "二つ目", capture_start_ms: 1000 });
  board.addFinal({ segment_id: "c", utterance_id: "u3", text: "三つ目", capture_start_ms: 2000 });
  assert.deepEqual([...board.lines().map((line) => line.text)], ["二つ目", "三つ目"]);
  board.addTranslation("c", "third");
  assert.equal(board.lines()[1].translation, "third");
  board.setPartial({ utterance_id: "u4", text: "四" });
  board.clear("u4");
  assert.equal(board.lines().length, 2);
});

test("the session clock continues a resumed session, and a shared surface reports at most an http(s) origin", () => {
  const { clockMs, describeSurface } = context.YataCapture;
  assert.equal(clockMs({ clock_offset_ms: 5000, clock_started_utc: "2026-11-04T00:00:00Z" }, Date.parse("2026-11-04T00:00:02Z")), 7000);
  assert.equal(clockMs({ clock_offset_ms: 5000, clock_started_utc: null }, Date.parse("2026-11-04T09:00:00Z")), 5000);
  const stream = (handle) => ({ getVideoTracks: () => [{ getCaptureHandle: () => handle, getSettings: () => ({ displaySurface: "browser" }) }] });
  assert.deepEqual({ ...describeSurface(stream({ origin: "https://www.youtube.com", handle: "" })) }, { url: "https://www.youtube.com", surface: "browser" });
  assert.equal(describeSurface(stream(null)).url, "");
  assert.equal(describeSurface(stream({ origin: "chrome-extension://abcdef" })).url, "");
  assert.equal(describeSurface({ getVideoTracks: () => [] }).url, "");
});
