import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

const read = (path) => readFileSync(new URL(path, import.meta.url), "utf8");
const context = {};
// The interface text comes from i18n.js; without a browser it is Japanese.
runInNewContext(read("../web/assets/i18n.js"), context);
runInNewContext(read("../web/assets/audio-diagnostics.js"), context);
const describe = (diagnostics, client, extra = {}) => context.YataAudioDiagnostics.describe({
  status: "capturing_audio", source_kind: "share", audio_diagnostics: diagnostics, ...extra,
}, client);
const server = {
  sample_rate: 48_000, server_socket_state: "open", asr_state: "ready", asr_engine: "faster-whisper",
  server_frames: 96, server_bytes: 1_572_864, server_accepted_samples: 393_216, server_duplicate_frames: 0,
};
const client = { capture_frames: 96, sent_frames: 96, sent_bytes: 1_572_864, queue_frames: 0, socket_buffered_bytes: 0,
  sample_rate: 48_000, track_state: "live", context_state: "running", socket_state: "open", reconnects: 0 };
const allText = (value) => [value.stage, ...value.lines].join("\n");

test("before any utterance the view explains that silence is not transcribed", () => {
  const view = describe(server, client);
  assert.equal(view.warning, false);
  assert.match(view.stage, /発話の区切りを待っています/);
  assert.match(allText(view), /取得 96フレーム、送信 96フレーム/);
  assert.match(allText(view), /エンジン faster-whisper/);
});

test("recognized, failed and dropped utterances are distinguished", () => {
  const ok = describe({ ...server, asr_utterances: 3, asr_successes: 3, asr_accepted: 2, asr_suppressed: 1, asr_last_result: "recognized" }, client);
  assert.match(ok.stage, /字幕と根拠として採用/);
  assert.match(allText(ok), /定型文として除外 1件/);
  const failed = describe({ ...server, asr_utterances: 1, asr_errors: 1, asr_last_result: "error", asr_last_error: "RuntimeError" }, client);
  assert.equal(failed.warning, true);
  assert.match(failed.stage, /失敗/);
  assert.match(allText(failed), /ASR 最終エラー: RuntimeError/);
  const dropped = describe({ ...server, asr_utterances: 4, asr_accepted: 4, asr_dropped_utterances: 2 }, client);
  assert.equal(dropped.warning, true);
  assert.match(allText(dropped), /追い付かず破棄 2件/);
});

test("summary failure stays separate from transcription", () => {
  const view = describe({ ...server, asr_utterances: 2, asr_accepted: 2, summary_attempts: 1, summary_errors: 1, summary_state: "error" }, client);
  assert.equal(view.warning, true);
  assert.match(view.stage, /採用しています/);
  assert.match(allText(view), /要約: 投入 1回、成功 0回、失敗 1回/);
});

test("ended share, missing client stats and reconnects are described", () => {
  assert.match(describe(server, { ...client, track_state: "ended" }).stage, /共有した音声が終了/);
  assert.match(describe({}, null).stage, /タブの音声を共有/);
  assert.match(describe({ ...server, server_socket_state: "reconnecting" }, client).stage, /再確立中/);
  assert.match(describe(server, client, { status: "ended" }).stage, /終了しています/);
});

test("bad counters and content never leak into the view", () => {
  const view = describe({ server_frames: 20, asr_errors: -1, asr_accepted: NaN, asr_last_error: "\nRuntimeError\n" + "x".repeat(400) }, null,
    { source_title: "PRIVATE_SOURCE_TITLE", evidence: [{ text: "PRIVATE_TRANSCRIPT" }] });
  assert.doesNotMatch(allText(view), /PRIVATE_|NaN|undefined|-1件/);
  assert.ok(view.lines.find((line) => line.startsWith("ASR 最終エラー:")).length <= 312);
});

test("HTML loads helpers before the app", () => {
  const html = read("../web/index.html");
  const at = (name) => html.indexOf(`/assets/${name}`);
  for (const helper of ["gaps.js", "audio-diagnostics.js", "capture.js", "captions.js", "topics.js", "waveform.js", "hud.js"]) {
    assert.ok(at(helper) > 0 && at(helper) < at("app.js"), helper);
  }
  assert.doesNotMatch(html, /pair-code|chrome:\/\/extensions/);
});
