(() => {
  "use strict";

  const t = (key, params) => globalThis.YataI18n.t(key, params);

  // This view only describes counters and states; it never reads transcript content.
  function describe(session, client) {
    const d = session.audio_diagnostics || {};
    const c = client || {};
    const count = (source, key) => Number.isFinite(source[key]) && source[key] >= 0 ? source[key] : 0;
    const unreported = t("diag.unreported");
    const stamp = (key) => typeof d[key] === "string" ? d[key] : unreported;
    const error = (key) => typeof d[key] === "string" ? d[key].replace(/\s+/g, " ").slice(0, 300) : "";
    const lastResult = t("diag.result." + (["recognized", "empty", "error"].includes(d.asr_last_result) ? d.asr_last_result : "none"));
    let stage;
    let warning = false;

    if (["ended", "cancelled"].includes(session.status)) stage = t("diag.stage.ended");
    else if (session.status === "reconnecting" || d.server_socket_state === "reconnecting") stage = t("diag.stage.reconnecting");
    else if (d.asr_state === "unavailable") { stage = t("diag.stage.asrUnavailable"); warning = true; }
    else if (c.track_state === "ended") { stage = t("diag.stage.trackEnded"); warning = true; }
    else if (c.context_state && !["running", "none"].includes(c.context_state)) { stage = t("diag.stage.contextStopped"); warning = true; }
    else if (!client && count(d, "server_frames") === 0) stage = t("diag.stage.noAudio");
    else if (client && count(c, "capture_frames") === 0) stage = t("diag.stage.noFrames");
    else if (d.asr_state === "loading") stage = t("diag.stage.loading");
    else if (count(d, "server_frames") === 0) stage = t("diag.stage.notReached");
    else if (d.asr_last_result === "error") { stage = t("diag.stage.asrFailed"); warning = true; }
    else if (count(d, "asr_utterances") === 0) stage = t("diag.stage.waiting");
    else if (count(d, "asr_accepted") === 0) stage = t("diag.stage.noneAccepted");
    else stage = t("diag.stage.working");
    if (count(d, "asr_dropped_utterances") > 0) warning = true;

    const lines = [];
    if (client) {
      lines.push(t("diag.client", {
        captured: count(c, "capture_frames"), sent: count(c, "sent_frames"), sentBytes: count(c, "sent_bytes"), queued: count(c, "queue_frames"),
        buffered: count(c, "socket_buffered_bytes"), reconnects: count(c, "reconnects"),
      }));
    }
    lines.push(
      t("diag.server", {
        frames: count(d, "server_frames"), bytes: count(d, "server_bytes"), duplicates: count(d, "server_duplicate_frames"),
        samples: count(d, "server_accepted_samples"),
      }),
      t("diag.asr", {
        utterances: count(d, "asr_utterances"), successes: count(d, "asr_successes"), errors: count(d, "asr_errors"), accepted: count(d, "asr_accepted"),
        suppressed: count(d, "asr_suppressed"), partials: count(d, "asr_partials"), queued: count(d, "asr_queue_utterances"),
        dropped: count(d, "asr_dropped_utterances"), last: lastResult, ms: count(d, "asr_last_duration_ms"),
      }),
      t("diag.summary", {
        attempts: count(d, "summary_attempts"), successes: count(d, "summary_successes"), errors: count(d, "summary_errors"),
        state: d.summary_state || t("diag.summaryNone"),
      }),
      t("diag.times", { server: stamp("server_last_frame_at_utc"), asr: stamp("asr_last_at_utc") }),
      t("diag.state", {
        rate: count(d, "sample_rate") || count(c, "sample_rate"), engine: d.asr_engine || t("diag.notConnected"), track: c.track_state || unreported,
        context: c.context_state || unreported, socket: d.server_socket_state || c.socket_state || unreported, asr: d.asr_state || t("diag.waiting"),
      }),
    );
    for (const [key, label] of [["asr_last_error", "diag.error.asr"], ["ingest_last_error", "diag.error.ingest"], ["summary_last_error", "diag.error.summary"]]) {
      if (error(key)) lines.push(t(label) + ": " + error(key));
    }
    if (d.summary_state === "error") warning = true;
    return { stage, warning, lines };
  }

  globalThis.YataAudioDiagnostics = { describe };
})();
