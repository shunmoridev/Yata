(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const t = (key, params) => YataI18n.t(key, params);
  const body = document.body;
  const state = {
    session: null, cursor: 0, evidence: new Map(), topics: new Map(), digest: null, gaps: new YataGapLedger(),
    captions: new YataCaptions.CaptionBoard(2), capture: null, clientStats: null, pip: null, poll: null, busy: false, polling: false,
    speakers: new Map(), lastSpeaker: null, filter: "all", search: "", showTranslation: true,
    rows: new Map(), lastReason: null, speakingTimer: null, models: false,
    sessions: [], sourceKind: "share", armed: null, armTimer: null, sessionQuery: "", listSignature: "",
    asrLimits: [600, 1500], slackReady: false,
    view: "session", setup: null, setupTimer: null, setupTicks: 0, setupLoading: false, setupSignature: "",
    planDraft: null, llmDraft: null, translationDraft: null, setupBrief: null, asrReady: true,
  };
  // The soft theme speaks in words; the HUD keeps its tactical codes (STANDBY, LIVE, IDLE, ...).
  const code = (word) => (globalThis.YataTheme?.get?.() === "hud" ? word : t("code." + word));
  const STATUS = {
    created: ["ARMED", "starting"], starting_audio: ["SYNC", "starting"], capturing_audio: ["LIVE", "live"], reconnecting: ["RELINK", "reconnecting"],
    draining: ["FINAL", "draining"], ended: ["END", "ended"], cancelled: ["ABORT", "cancelled"], error: ["ERROR", "error"],
  };

  const SOURCES = {
    share: {
      label: t("source.share.label"), icon: "i-tab", fields: [],
      description: t("source.share.description"),
      launch: t("source.share.launch"),
      capture: (options) => new YataCapture.Capture(options),
    },
  };
  const source = () => SOURCES[state.sourceKind];
  const CAPTIONS_ONLY = t("setup.model.none");
  const NO_SUMMARY_NOTES = { captions: t("setup.note.captions"), translation: t("setup.note.translation") };
  const FEATURES = new Set(String(document.documentElement?.dataset?.features || "").split(/\s+/).filter(Boolean));
  const feature = (name) => FEATURES.has(name);

  const api = async (path, options) => {
    const opts = options || {};
    const payloadBody = opts.body;
    const response = await fetch(path, {
      method: opts.method || "GET",
      headers: Object.assign({}, payloadBody ? { "Content-Type": "application/json" } : {}),
      body: payloadBody ? JSON.stringify(payloadBody) : undefined,
      cache: "no-store",
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || "HTTP " + response.status);
    return payload;
  };

  function toast(text, tone) {
    if (!text) return;
    const holder = $("session-message");
    const item = document.createElement("div");
    item.className = "toast " + (tone || "info");
    item.textContent = text;
    holder.append(item);
    while (holder.children.length > 4) holder.children[0].remove();
    if (typeof setTimeout === "function") {
      setTimeout(() => {
        item.classList.add("leaving");
        setTimeout(() => item.remove?.(), 320);
      }, tone === "error" ? 9000 : 4500);
    }
  }
  const message = (text, error) => toast(text, error ? "error" : "info");

  const pad2 = (value) => String(value).padStart(2, "0");
  const formatTime = (ms) => {
    if (!Number.isFinite(ms)) return "--:--:--";
    const s = Math.max(0, Math.floor(ms / 1000));
    return pad2(Math.floor(s / 3600)) + ":" + pad2(Math.floor(s / 60) % 60) + ":" + pad2(s % 60);
  };
  const segmentTime = (item) => item.media_start_ms != null ? item.media_start_ms : item.capture_start_ms;
  const terminal = () => !state.session || ["ended", "cancelled", "error"].includes(state.session.status);
  const elapsedMs = () => state.session ? YataCapture.clockMs(state.session, Date.now()) : 0;

  const waveform = new YataWaveform.Waveform($("waveform"), {
    onLevel: (level) => { const bar = $("input-meter").querySelector?.("i"); if (bar && bar.style) bar.style.width = Math.round(level * 100) + "%"; },
  });
  waveform.start();

  [["gauge-asr", "ASR"], ["gauge-queue", "QUEUE"], ["gauge-llm", "LLM"], ["gauge-link", "LINK"]].forEach(([id, word]) => {
    const name = $(id).querySelector?.("em");
    if (name) name.textContent = code(word);
  });

  // The canvas cannot read CSS variables by itself: take the theme's speaker colours from the page.
  function themeColor(name, fallback) {
    try {
      const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return /^#[0-9a-f]{6}$/i.test(value) ? value : fallback;
    } catch {
      return fallback;
    }
  }

  function fillLanguageSelects() {
    document.querySelectorAll(".language-select").forEach((select) => {
      select.replaceChildren();
      if (select.dataset.includeAuto) select.add(new Option("AUTO", "auto"));
      if (select.dataset.includeNone) select.add(new Option("—", ""));
      YataCaptions.LANGUAGES.forEach(([code, label]) => select.add(new Option(code.toUpperCase() + " · " + label, code)));
    });
    $("asr-language").value = "auto";
    $("output-language").value = (navigator.language || "ja").slice(0, 2) === "en" ? "en" : "ja";
    $("caption-language").value = "";
  }

  function light(id, value, tip) {
    const node = $(id);
    node.dataset.state = value;
    if (tip) {
      node.dataset.tip = tip;
      node.setAttribute?.("aria-label", tip);
    }
  }

  async function loadHealth() {
    try {
      const data = await api("/api/v1/health");
      // The LLM is optional: without it sessions run captions only, so a missing one is a warning, not an error.
      const llm = data.llm || { connected: false, models: [], reason: "" };
      light("sys-llm", llm.connected ? "ok" : "warn", llm.connected ? "LLM: " + llm.label
        : t("health.llmMissing", { reason: llm.reason || t("health.notConnected") }));
      // ASR time per utterance is judged against what the device can do: a CPU takes seconds, a GPU a tenth.
      state.asrLimits = data.asr.device === "cpu" ? [3000, 6000] : [600, 1500];
      light("sys-asr", data.asr.weights_present ? "ok" : "bad", "ASR: " + (data.asr.description || data.asr.engine)
        + (data.asr.weights_present ? "" : " · " + data.asr.reason));
      state.asrReady = Boolean(data.asr.weights_present);
      state.setupBrief = data.setup || null;
      state.translator = data.translator?.id ? data.translator : null;
      const plan = data.setup?.plan || null;
      const speakers = data.speakers || { enabled: false, reason: "" };
      light("sys-spk", speakers.enabled ? "ok" : "off", t("health.speakers", { reason: speakers.reason || t(speakers.enabled ? "common.on" : "common.off") }));
      const slack = feature("slack") && data.slack.enabled;
      $("sys-slack").hidden = !feature("slack");
      light("sys-slack", slack ? "ok" : "off", "Slack: " + (data.slack.reason || ""));
      const select = $("model-select");
      select.replaceChildren();
      llm.models.forEach((model) => select.add(new Option(llm.label ? t("setup.model.option", { model: model.label, backend: llm.label }) : model.label, model.id)));
      select.add(new Option(CAPTIONS_ONLY, ""));
      select.value = plan && plan.summaries === false ? "" : llm.models[0]?.id || "";
      select.disabled = false;
      state.models = llm.models.length > 0;
      $("slack-delivery").disabled = !slack;
      if (!slack) $("slack-delivery").checked = false;
      $("slack-slot").hidden = !slack;
      state.slackReady = slack;
      applySummaryChoice();
      if (!state.models && !(plan && plan.summaries === false)) {
        message(t("health.noModels", { reason: llm.reason || t("health.noModelReason") }));
      }
      renderPrepareLink();
      if (data.setup?.busy) watchSetup();
      if (data.setup && !state.session && state.view === "session" && !setupReady()) openPrepare();
    } catch (error) {
      light("sys-llm", "bad", error.message);
      message(t("health.apiDown", { error: error.message }), true);
    }
    updateControls();
  }

  function setMode() {
    const mode = state.view === "prepare" || state.view === "settings" ? state.view : !state.session ? "idle" : terminal() ? "ended" : "live";
    ["idle", "live", "ended", "prepare", "settings"].forEach((name) => body.classList.toggle("mode-" + name, name === mode));
  }

  function resetView() {
    state.cursor = 0;
    state.evidence.clear();
    state.topics.clear();
    state.digest = null;
    state.gaps = new YataGapLedger();
    state.captions = new YataCaptions.CaptionBoard(2);
    state.clientStats = null;
    state.speakers.clear();
    state.lastSpeaker = null;
    state.filter = "all";
    state.rows.clear();
    state.lastReason = null;
    ["transcript-list", "gap-list"].forEach((id) => $(id).replaceChildren());
    $("segment-count").textContent = "0";
    renderFeedEmpty();
    renderTopics();
    renderCaptions();
    renderRadar();
    renderGaps();
    renderSpeakerFilter();
    renderTelemetry();
  }

  function setSession(session) {
    const isNewSession = !state.session || state.session.id !== session.id;
    if (isNewSession) resetView();
    state.session = session;
    $("active-title").textContent = YataSessions.title(session);
    $("active-meta").textContent = sessionMeta(session);
    $("mission-clock").textContent = formatTime(elapsedMs());
    const link = $("active-source");
    link.href = session.source_url || "#";
    link.hidden = !session.source_url;
    $("readout-lang").textContent = !session.asr_language || session.asr_language === "auto" ? code("AUTO") : session.asr_language.toUpperCase();
    const rate = session.audio_diagnostics?.sample_rate || state.clientStats?.sample_rate;
    $("readout-rate").textContent = rate ? (rate / 1000).toFixed(1) + " kHz" : "-- kHz";
    $("feed-translate").hidden = !session.caption_language;
    showMemo(session, isNewSession);
    if (!state.capture && session.live_caption) state.captions.setPartial(session.live_caption);
    if (!state.capture && !session.live_caption) state.captions.clear();
    updateStatus(session);
    renderDiagnostics();
    renderTelemetry();
    renderCaptions();
    if (!state.poll) state.poll = window.setInterval(pollEvents, 900);
  }

  function sessionMeta(session) {
    let host = "";
    try { host = session.source_url ? new URL(session.source_url).host : ""; } catch { host = ""; }
    // Translations by PLaMo say so: its license asks that its output be marked as such.
    const use = session.summaries === false ? t(session.translator ? "meta.captionsTranslationPlamo" : "meta.captionsOnly")
      : session.translator ? t("meta.plamo") : "";
    return [t("meta.started", { time: YataSessions.startedLabel(session.started_at_utc) }), YataSessions.sourceLabel(session.source_kind), host, use]
      .filter(Boolean).join(" · ");
  }

  // A translation the translator cannot do yet holds the start back: the server would refuse the session only
  // after the share picker.
  function applySummaryChoice() {
    const noSummaries = $("model-select").value === "";
    const translator = noSummaries ? state.translator : null;
    const caption = $("caption-language");
    caption.disabled = noSummaries && !translator;
    Array.from(caption.options || caption.children).forEach((option) => {
      option.disabled = Boolean(translator && option.value && !translator.languages.includes(option.value));
    });
    if (caption.disabled || (translator && caption.value && !translator.languages.includes(caption.value))) caption.value = "";
    $("slack-delivery").disabled = noSummaries || !state.slackReady;
    if (noSummaries) $("slack-delivery").checked = false;
    const note = $("summary-note");
    const waits = translationWaits();
    note.hidden = !noSummaries;
    note.dataset.tone = waits ? "warn" : "";
    note.textContent = waits
      ? t("setup.note.translationMissing", { missing: (translator.missing || []).join(t("prep.fileSeparator")) || t("setup.note.translatorParts") })
      : NO_SUMMARY_NOTES[translator?.available ? "translation" : "captions"];
  }

  function translationWaits() {
    return $("model-select").value === "" && Boolean($("caption-language").value) && !state.translator?.available;
  }

  function updateStatus(session) {
    state.session = Object.assign(state.session || {}, session);
    const [label, tone] = STATUS[session.status] || [String(session.status || "").toUpperCase(), "starting"];
    const badge = $("session-status");
    badge.textContent = STATUS[session.status] ? code(label) : label;
    badge.dataset.state = tone;
    if (session.status_reason && session.status_reason !== state.lastReason) {
      state.lastReason = session.status_reason;
      message(session.status_reason, session.status === "error");
    }
    setMode();
    updateControls();
  }

  function updateControls() {
    const done = terminal();
    const idle = !state.session;
    $("launch").disabled = state.busy || Boolean(state.capture) || !idle || !state.asrReady || translationWaits();
    $("resume-session").disabled = state.busy || Boolean(state.capture) || idle || !done || !state.asrReady
      || (state.session.summaries !== false && !state.models);
    $("new-session").disabled = state.busy || Boolean(state.capture);
    $("open-prepare").disabled = state.busy || Boolean(state.capture);
    $("open-settings").disabled = state.busy || Boolean(state.capture);
    ["summarize-now", "stop-session", "cancel-session"].forEach((id) => { $(id).disabled = state.busy || done; });
    $("export-session").disabled = idle;
    $("memo-text").disabled = idle;
    $("memo-time").disabled = idle;
    $("start-share").disabled = state.busy || done || Boolean(state.capture);
    body.classList.toggle("capturing", Boolean(state.capture));
    body.classList.toggle("summaries-off", state.session?.summaries === false);
    $("open-pip").hidden = !("documentPictureInPicture" in window);
    renderSessions();
  }

  function renderSources() {
    const holder = $("source-options");
    holder.replaceChildren();
    Object.entries(SOURCES).forEach(([kind, item]) => {
      const tile = document.createElement("button");
      tile.className = "source-option";
      tile.setAttribute("role", "radio");
      tile.setAttribute("aria-checked", String(kind === state.sourceKind));
      tile.dataset.source = kind;
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
      use.setAttribute("href", "#" + item.icon);
      svg.append(use);
      const name = document.createElement("strong");
      name.textContent = item.label;
      const note = document.createElement("small");
      note.textContent = item.description;
      tile.append(svg, name, note);
      tile.addEventListener("click", () => {
        state.sourceKind = kind;
        renderSources();
      });
      holder.append(tile);
    });
    const current = source();
    $("slot-title").hidden = !current.fields.includes("title");
    $("slot-url").hidden = !current.fields.includes("url");
    $("launch-title").textContent = current.launch;
    $("launch-label").textContent = current.launch;
  }

  function setSpeaking(caption) {
    body.classList.add("speaking");
    waveform.setSpeech(true);
    clearTimeout(state.speakingTimer);
    state.speakingTimer = setTimeout(() => {
      body.classList.remove("speaking");
      waveform.setSpeech(false);
    }, 1400);
    if (caption && caption.speaker) tintFor(caption.speaker);
  }

  function tintFor(speaker) {
    const tone = YataCaptions.speakerTone(speaker);
    const main = themeColor("--tone-" + (tone || 1), "#0eb2d4");
    waveform.setTint(main, themeColor(tone === 5 ? "--accent-2" : "--accent", "#6152f3"));
    const chip = $("live-speaker");
    chip.hidden = !speaker;
    if (chip.textContent !== speaker) {
      chip.textContent = speaker;
      chip.className = "speaker-chip tone-" + tone;
    }
  }

  function renderCaptions() {
    const lines = state.captions.lines();
    const translate = Boolean(state.session?.caption_language);
    YataCaptions.render($("caption-lines"), lines, translate);
    if (state.pip) YataCaptions.render(state.pip.box, lines, translate);
  }

  // The feed is built incrementally so rows animate only when they arrive.
  function renderFeedEmpty() {
    const holder = $("transcript-list");
    if (state.rows.size) return;
    const ghost = document.createElement("div");
    ghost.className = "transcript-empty";
    for (let i = 0; i < 4; i++) ghost.append(document.createElement("i"));
    holder.replaceChildren(ghost);
  }

  function feedRow(item) {
    const tone = YataCaptions.speakerTone(item.speaker);
    const row = document.createElement("article");
    row.className = "feed-row" + (tone ? " tone-" + tone : "");
    row.dataset.speaker = item.speaker || "";
    const avatar = document.createElement("span");
    avatar.className = "avatar";
    avatar.textContent = item.speaker || "—";
    const main = document.createElement("div");
    const meta = document.createElement("div");
    meta.className = "row-meta";
    const time = document.createElement("time");
    time.textContent = formatTime(segmentTime(item));
    meta.append(time);
    if (item.language) {
      const lang = document.createElement("span");
      lang.className = "lang";
      lang.textContent = item.language;
      meta.append(lang);
    }
    const text = document.createElement("div");
    text.className = "row-text";
    text.textContent = item.text;
    const translation = document.createElement("div");
    translation.className = "row-translation";
    translation.hidden = true;
    main.append(meta, text, translation);
    row.append(avatar, main);
    return { row, translation, text: item.text };
  }

  function addEvidence(item) {
    if (state.rows.has(item.id)) return;
    const holder = $("transcript-list");
    if (!state.rows.size) holder.replaceChildren();
    const entry = feedRow(item);
    state.rows.set(item.id, entry);
    holder.insertBefore(entry.row, holder.firstChild || null);
    const translated = state.captions.translations.get(item.id);
    if (translated) setRowTranslation(item.id, translated);
    // Keep the DOM bounded on long streams; evidence itself stays in memory.
    if (state.rows.size > 400) {
      const oldest = state.rows.keys().next().value;
      state.rows.get(oldest).row.remove?.();
      state.rows.delete(oldest);
    }
    filterRow(entry);
    $("segment-count").textContent = String(state.evidence.size);
  }

  function setRowTranslation(id, text) {
    const entry = state.rows.get(id);
    if (!entry) return;
    entry.translation.textContent = text;
    entry.translation.hidden = false;
  }

  function filterRow(entry) {
    const speaker = entry.row.dataset.speaker;
    const query = state.search.toLowerCase();
    const matches = (state.filter === "all" || speaker === state.filter)
      && (!query || (entry.text + " " + entry.translation.textContent).toLowerCase().includes(query));
    entry.row.hidden = !matches;
  }

  function applyFilters() {
    state.rows.forEach(filterRow);
    renderSpeakerFilter();
  }

  function renderSpeakerFilter() {
    const holder = $("speaker-filter");
    holder.replaceChildren();
    if (state.speakers.size < 2) return;
    ["all", ...[...state.speakers.keys()].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))].forEach((label) => {
      const button = document.createElement("button");
      const tone = label === "all" ? 0 : YataCaptions.speakerTone(label);
      button.className = (state.filter === label ? "active " : "") + (tone ? "tone-" + tone : "");
      button.textContent = label === "all" ? code("ALL") : label;
      button.setAttribute?.("aria-pressed", String(state.filter === label));
      button.addEventListener("click", () => {
        state.filter = label;
        applyFilters();
      });
      holder.append(button);
    });
  }

  function renderTopics() {
    const topics = [...state.topics.values()].sort((a, b) => a.index - b.index);
    $("topic-count").textContent = String(topics.length);
    YataTopics.renderTopics($("topics-list"), topics, state.evidence, formatTime, feature("slack") && state.session?.slack_delivery ? redeliver : null);
    YataTopics.renderDigest($("digest"), state.digest, state.topics);
    renderMap();
  }

  function renderRadar() {
    YataHud.radar($("speaker-radar"), state.speakers, state.lastSpeaker);
  }

  function renderMap() {
    const topics = [...state.topics.values()].sort((a, b) => a.index - b.index);
    const gaps = state.gaps.entries();
    const now = terminal() ? Math.max(elapsedMs(), ...topics.map((topic) => topic.end_ms || 0)) : elapsedMs();
    YataHud.sessionMap($("session-map"), topics, gaps, now, (topicId) => {
      const card = document.querySelector?.(`[data-topic-id="${topicId}"]`);
      if (!card) return;
      card.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
      card.classList.remove("flash");
      void card.offsetWidth;
      card.classList.add("flash");
    });
  }

  function renderTelemetry() {
    const d = state.session?.audio_diagnostics || {};
    const c = state.clientStats || {};
    const asrMs = Number(d.asr_last_duration_ms);
    const [warnMs, badMs] = state.asrLimits;
    YataHud.gauge($("gauge-asr"), Number.isFinite(asrMs) ? asrMs / badMs : 0, Number.isFinite(asrMs) ? String(Math.round(asrMs)) : "--",
      !Number.isFinite(asrMs) ? "off" : asrMs < warnMs ? "ok" : asrMs < badMs ? "warn" : "bad");
    const queue = Number(d.asr_queue_utterances) || 0;
    YataHud.gauge($("gauge-queue"), queue / 8, String(queue), queue <= 1 ? "ok" : queue <= 4 ? "warn" : "bad");
    const noSummaries = state.session?.summaries === false;
    const llm = noSummaries ? (state.session?.translator ? d.translator_state || "" : "") : d.summary_state || "";
    YataHud.gauge($("gauge-llm"), llm === "busy" ? 0.7 : llm === "ready" ? 1 : llm === "error" ? 1 : 0,
      code(llm === "busy" ? "BUSY" : llm === "error" ? "ERR" : llm === "ready" ? "OK" : noSummaries && !state.session?.translator ? "OFF" : "IDLE"),
      llm === "busy" ? "busy" : llm === "error" ? "bad" : llm ? "ok" : "off");
    const link = d.server_socket_state || c.socket_state || "none";
    const linkTone = { open: "ok", reconnecting: "warn", connecting: "busy", closed: "off", none: "off" }[link] || "off";
    YataHud.gauge($("gauge-link"), link === "open" ? 1 : link === "reconnecting" ? 0.5 : 0, link === "connecting" ? "…" : code({ open: "OPEN", reconnecting: "RELINK" }[link] || "OFF"), linkTone);
    if (d.asr_state) light("sys-asr", { loading: "busy", busy: "busy", ready: "ok", unavailable: "bad" }[d.asr_state] || "ok");
    if (llm && !noSummaries) light("sys-llm", llm === "busy" ? "busy" : llm === "error" ? "warn" : "ok");
  }

  async function redeliver(topicId) {
    try {
      const result = await api("/api/v1/sessions/" + state.session.id + "/topics/" + topicId + "/deliver", { method: "POST" });
      state.topics.set(result.topic.id, result.topic);
      renderTopics();
    } catch (error) {
      message(t("toast.redeliverFailed", { error: error.message }), true);
    }
  }

  function renderGaps() {
    const holder = $("gap-list");
    holder.replaceChildren();
    const entries = state.gaps.entries();
    entries.forEach((gap) => {
      const row = document.createElement("div");
      row.className = "gap-item";
      const when = new Date(gap.lastSeen).toLocaleTimeString(YataI18n.locale, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
      row.textContent = gap.reason + " ×" + gap.count + " · " + when;
      holder.append(row);
    });
    const total = entries.reduce((sum, gap) => sum + gap.count, 0);
    $("gap-count").hidden = !total;
    $("gap-count").textContent = String(total);
    renderMap();
  }

  function renderDiagnostics() {
    if (!state.session) return;
    const diagnostic = YataAudioDiagnostics.describe(state.session, state.clientStats);
    const holder = $("audio-diagnostics-body");
    holder.replaceChildren();
    [diagnostic.stage, ...diagnostic.lines].forEach((line, index) => {
      if (index) holder.append(document.createElement("br"));
      const text = document.createElement("span");
      text.textContent = line;
      if (index === 0) text.classList.toggle("error-text", diagnostic.warning);
      holder.append(text);
    });
  }

  function handleEvent(event) {
    state.cursor = Math.max(state.cursor, event.cursor);
    if (event.type === "evidence_segment") {
      const item = event.data;
      state.evidence.set(item.id, item);
      if (item.speaker) {
        state.speakers.set(item.speaker, (state.speakers.get(item.speaker) || 0) + 1);
        state.lastSpeaker = item.speaker;
        renderRadar();
        renderSpeakerFilter();
      }
      if (item.source_type === "asr") state.captions.addFinal({ ...item, segment_id: item.id });
      addEvidence(item);
      renderCaptions();
    } else if (event.type === "topic") {
      const previous = state.topics.get(event.data.id);
      state.topics.set(event.data.id, event.data);
      renderTopics();
      if (previous && previous.status === "open" && event.data.status !== "open") toast("T" + pad2(event.data.index) + " " + (event.data.title || ""), "ok");
    } else if (event.type === "session_digest") {
      state.digest = event.data;
      renderTopics();
    } else if (event.type === "digest_delivery") {
      if (event.data.state === "failed") message(t("toast.digestSlackFailed", { reason: event.data.reason }), true);
    } else if (event.type === "caption_translation") {
      state.captions.addTranslation(event.data.segment_id, event.data.text);
      setRowTranslation(event.data.segment_id, event.data.text);
      renderCaptions();
    } else if (event.type === "session_state") {
      updateStatus(event.data);
    } else if (event.type === "analysis_error" || event.type === "ingest_error") {
      message((event.data.task ? event.data.task + ": " : "") + (event.data.message || t("toast.processingFailed")), true);
    } else if (event.type === "gap") {
      if (state.gaps.add(event)) renderGaps();
    }
  }

  async function pollEvents() {
    if (!state.session || state.polling) return;
    state.polling = true;
    try {
      const sessionId = state.session.id;
      const response = await api("/api/v1/sessions/" + sessionId + "/events?after=" + state.cursor);
      if (state.session?.id !== sessionId) return;
      setSession(response.session);
      response.events.forEach(handleEvent);
      if (!terminal()) renderMap();
      if (terminal() && !response.events.length && state.poll) {
        window.clearInterval(state.poll);
        state.poll = null;
      }
    } catch (error) {
      message(t("toast.pollFailed", { error: error.message }), true);
    } finally {
      state.polling = false;
    }
  }

  async function withBusy(work) {
    state.busy = true;
    updateControls();
    try {
      await work();
    } finally {
      state.busy = false;
      updateControls();
    }
  }

  async function createSession(reported) {
    const typed = (field, id) => source().fields.includes(field) ? $(id).value.trim() : "";
    const result = await api("/api/v1/sessions", {
      method: "POST",
      body: {
        source_kind: state.sourceKind,
        source_url: typed("url", "source-url") || reported.url || "",
        source_title: typed("title", "source-title"),
        selected_model: $("model-select").value,
        summaries: $("model-select").value !== "",
        asr_language: $("asr-language").value,
        output_language: $("output-language").value,
        caption_language: $("caption-language").disabled ? "" : $("caption-language").value,
        slack_delivery: feature("slack") && Boolean($("slack-delivery").checked),
      },
    });
    state.session = null;
    setSession(result.session);
    return result.session;
  }

  async function resumeSession(reported) {
    if (!state.session || !terminal()) throw new Error(t("toast.resumeEndedOnly"));
    const result = await api("/api/v1/sessions/" + state.session.id + "/resume", { method: "POST", body: { source_url: reported.url || "" } });
    setSession(result.session);
    return result.session;
  }

  function makeCapture() {
    const capture = source().capture({
      onState: (status, reason) => {
        if (status === "stopped") {
          if (state.capture === capture) state.capture = null;
          waveform.attach(null);
          updateControls();
          return;
        }
        if (reason && status !== "capturing_audio" && status !== "starting_audio") message(reason, status === "error");
      },
      onCaption: (caption) => {
        if (caption.type === "caption_partial") {
          state.captions.setPartial(caption);
          setSpeaking(caption);
        } else if (caption.type === "caption_final") {
          state.captions.addFinal(caption);
          setSpeaking(caption);
        } else {
          state.captions.clear(caption.utterance_id);
        }
        renderCaptions();
      },
      onGap: (gap) => {
        if (capture.sessionId) api("/api/v1/sessions/" + capture.sessionId + "/gap", { method: "POST", body: gap }).catch(() => {});
      },
      onStats: (stats) => {
        state.clientStats = stats;
        renderDiagnostics();
        renderTelemetry();
      },
    });
    return capture;
  }

  async function beginCapture(capture, session) {
    state.capture = capture;
    updateControls();
    await capture.begin(session);
    waveform.attach(capture.analyser);
    updateControls();
  }

  function shareFailed(capture, error) {
    if (state.capture === capture) state.capture = null;
    capture.stop().catch(() => {});
    waveform.attach(null);
    message(error.name === "NotAllowedError" ? t("toast.shareCancelled") : error.message, error.name !== "NotAllowedError");
    updateControls();
  }

  // The tab is picked before the session exists: the picker needs this click's user activation.
  async function share(start) {
    const capture = makeCapture();
    try {
      const reported = await capture.open();
      await beginCapture(capture, await start(reported));
    } catch (error) {
      shareFailed(capture, error);
    }
    refreshSessions();
  }

  async function startShare() {
    if (!state.session || state.capture || terminal()) return;
    const capture = makeCapture();
    try {
      await capture.open();
      await beginCapture(capture, state.session);
    } catch (error) {
      shareFailed(capture, error);
    }
  }

  async function refreshSessions() {
    try {
      state.sessions = (await api("/api/v1/sessions")).sessions || [];
    } catch (error) {
      return;  // the list is secondary; the health check reports a dead API
    }
    renderSessions();
  }

  function renderSessions() {
    const current = state.session;
    let rows = state.sessions.map((row) => current && row.id === current.id
      ? Object.assign({}, row, { display_title: current.display_title, status: current.status, duration_ms: elapsedMs(),
          topic_count: state.topics.size, segment_count: Math.max(row.segment_count || 0, state.evidence.size) })
      : row);
    if (current && !rows.some((row) => row.id === current.id)) {
      rows = [Object.assign({}, current, { updated_at_utc: new Date().toISOString(), duration_ms: elapsedMs(), topic_count: state.topics.size,
        segment_count: state.evidence.size })].concat(rows);
    }
    const options = { currentId: current?.id, locked: Boolean(state.capture) || state.busy, armedId: state.armed, query: state.sessionQuery };
    // Polling calls this every second; rebuild the list only when something on it changed.
    const signature = JSON.stringify([options, rows.map((row) => [row.id, row.display_title, row.status, row.topic_count,
      ["ended", "cancelled", "error"].includes(row.status) ? row.duration_ms : 0, row.updated_at_utc?.slice(0, 10)])]);
    if (signature === state.listSignature) return;
    state.listSignature = signature;
    $("session-total").textContent = String(rows.length);
    YataSessions.renderList($("session-list"), rows, Object.assign(options, { formatTime, onOpen: openSession, onDelete: requestDelete }));
  }

  function showSidebar(open) {
    body.classList.toggle("sidebar-open", open);
    $("toggle-sidebar").setAttribute("aria-expanded", String(open));
  }

  async function openSession(id) {
    if (state.capture || state.busy) return;
    showSidebar(false);
    if (state.session?.id === id) return;
    flushMemo();
    await withBusy(async () => {
      try {
        showSnapshot(await api("/api/v1/sessions/" + id));
      } catch (error) {
        message(t("toast.openFailed", { error: error.message }), true);
      }
    });
  }

  function showSnapshot(snapshot) {
    state.view = "session";
    state.session = null;
    setSession(snapshot.session);
    state.cursor = snapshot.cursor;
    Object.entries(snapshot.translations || {}).forEach(([id, text]) => state.captions.addTranslation(id, text));
    snapshot.evidence.forEach((item) => {
      state.evidence.set(item.id, item);
      if (item.speaker) {
        state.speakers.set(item.speaker, (state.speakers.get(item.speaker) || 0) + 1);
        state.lastSpeaker = item.speaker;
      }
    });
    // The feed keeps its newest 400 rows in the DOM, as during a live session.
    snapshot.evidence.slice(-400).forEach(addEvidence);
    snapshot.evidence.slice(-2).forEach((item) => state.captions.addFinal({ ...item, segment_id: item.id }));
    snapshot.topics.forEach((topic) => state.topics.set(topic.id, topic));
    state.digest = snapshot.digest;
    (snapshot.gaps || []).forEach((gap) => state.gaps.add({ cursor: gap.cursor, type: "gap", data: gap }));
    renderTopics();
    renderCaptions();
    renderRadar();
    renderSpeakerFilter();
    renderGaps();
    updateControls();
  }

  // Before the first health report nothing is known, so nothing is held back yet.
  function setupReady() {
    const brief = state.setupBrief;
    return !brief || (state.asrReady && Boolean(brief.plan) && (brief.attention || []).length === 0);
  }

  function newSession() {
    if (state.capture || state.busy) return;
    if (!setupReady()) {
      message(t("toast.setupFirst"));
      openPrepare();
      return;
    }
    leaveSession();
  }

  function leaveSession() {
    flushMemo();
    showSidebar(false);
    state.view = "session";
    state.session = null;
    if (state.poll) {
      window.clearInterval(state.poll);
      state.poll = null;
    }
    resetView();
    $("active-title").textContent = t("nav.newSession");
    $("active-meta").textContent = "";
    $("active-source").hidden = true;
    $("session-status").textContent = code("STANDBY");
    $("session-status").dataset.state = "idle";
    $("mission-clock").textContent = formatTime(0);
    setMode();
    updateControls();
    renderSessions();
  }

  function openPrepare() {
    if (state.capture || state.busy) return;
    leaveSession();
    state.view = "prepare";
    state.setupSignature = "";
    $("active-title").textContent = t("prepare.title");
    $("active-meta").textContent = t("prepare.meta");
    $("session-status").textContent = code("SETUP");
    setMode();
    updateControls();
    if (state.setup) renderPrepare();
    loadSetup();
    watchSetup();
  }

  function openSettings() {
    if (state.capture || state.busy) return;
    leaveSession();
    state.view = "settings";
    $("active-title").textContent = t("settings.title");
    $("active-meta").textContent = t("settings.meta");
    renderLanguageChoice();
    $("session-status").textContent = code("SETUP");
    setMode();
    updateControls();
  }

  const setupShown = () => state.view === "prepare";

  async function loadSetup() {
    try {
      showSetup((await api("/api/v1/setup")).setup);
    } catch (error) {
      if (setupShown()) message(t("toast.setupReadFailed", { error: error.message }), true);
    }
  }

  const runningJobs = (setup) => Object.values(setup?.jobs || {}).filter((job) => job.state === "running");

  function showSetup(setup) {
    if (!setup || !setup.asr) return;
    const before = state.setup;
    state.setup = setup;
    noticeJobs(before, setup);
    if (setupShown()) renderPrepare();
    renderPrepareLink();
  }

  function targetLabel(setup, job) {
    const builtin = setup.llm?.builtin;
    const translator = setup.translator;
    const models = (setup.asr?.models || []).concat(setup.speakers?.model ? [setup.speakers.model] : [],
      builtin?.runtime ? [builtin.runtime] : [], builtin?.models || [], translator?.models || (translator?.model ? [translator.model] : []));
    return job.target === "foundry" ? "Foundry Local" : models.find((model) => model.id === job.target)?.label || job.target;
  }


  function noticeJobs(before, after) {
    if (!before) return;
    let changed = false;
    Object.entries(after.jobs || {}).forEach(([component, job]) => {
      const previous = before.jobs?.[component];
      if (!previous || previous.state !== "running" || job.state === "running" || previous.target !== job.target) return;
      changed = true;
      const name = targetLabel(after, job);
      const done = ["pull", "test", "start"].includes(job.kind) ? "job.done." + job.kind : "job.done.default";
      if (job.state === "done") toast(t(done, { name }) + (job.note ? " " + job.note : ""), "ok");
      else if (job.state === "cancelled") message(t("job.cancelled", { name }));
      else message(t("job.failed", { name, error: job.error?.message || t("job.failedDefault") }), true);
    });
    if (changed) loadHealth();
  }

  function watchSetup() {
    if (state.setupTimer) return;
    state.setupTicks = 0;
    state.setupTimer = window.setInterval(() => {
      const busy = runningJobs(state.setup).length > 0;
      if (!setupShown() && !busy && state.setup) {
        window.clearInterval(state.setupTimer);
        state.setupTimer = null;
        return;
      }
      state.setupTicks += 1;
      if (state.setupLoading || (!busy && state.setupTicks % 5 !== 0)) return;
      state.setupLoading = true;
      loadSetup().finally(() => { state.setupLoading = false; });
    }, 1000);
  }

  const setupHandlers = {
    chooseAsr: (id) => setupCall("/api/v1/setup/asr", "PUT", { model: id }, true),
    download: (id) => setupCall("/api/v1/setup/models/" + id + "/download", "POST"),
    verify: (id) => setupCall("/api/v1/setup/models/" + id + "/verify", "POST"),
    pull: (name) => setupCall("/api/v1/setup/llm/pull", "POST", { model: name }),
    testLlm: (name, backend) => setupCall("/api/v1/setup/llm/test", "POST", { model: name, backend: backend || "ollama" }),
    chooseLlm: (backend) => saveBackend(backend),
    chooseTranslation: (choice) => saveTranslation(choice),
    saveOpenai: (values) => setupCall("/api/v1/setup/llm/openai", "PUT", values, true),
    saveTranslatorPath: (path) => setupCall("/api/v1/setup/translator/path", "PUT", { path }, true),
    foundryStart: () => setupCall("/api/v1/setup/llm/foundry/start", "POST"),
    foundryDownload: (name) => setupCall("/api/v1/setup/llm/foundry/download", "POST", { model: name }),
    cancel: (component) => setupCall("/api/v1/setup/jobs/" + component + "/cancel", "POST"),
    refresh: () => loadSetup().then(loadHealth),
  };

  async function setupCall(path, method, payload, refreshHealth) {
    try {
      const result = await api(path, { method, body: payload });
      showSetup(result.setup);
      await ensurePlan();
      watchSetup();
      if (refreshHealth) await loadHealth();
    } catch (error) {
      message(error.message, true);
      await loadSetup();
    }
  }

  // The use (and summary LLM) shown before the user chose one is what suits this PC; acting on the page keeps it.
  async function ensurePlan() {
    if (state.setup && !state.setup.plan) await savePlan(YataPrepare.planOf(state.setup));
    const llm = state.setup?.llm;
    if (llm && !llm.chosen && YataPrepare.planOf(state.setup).summaries) await saveBackend(YataPrepare.backendOf(state.setup));
  }

  async function saveBackend(backend) {
    state.llmDraft = backend;
    renderPrepare();
    try {
      const result = await api("/api/v1/setup/llm", { method: "PUT", body: { backend } });
      state.llmDraft = null;
      showSetup(result.setup);
      await loadHealth();
    } catch (error) {
      state.llmDraft = null;
      message(error.message, true);
      renderPrepare();
    }
  }

  async function saveTranslation(choice) {
    if (choice.engine) state.translationDraft = choice.engine;
    renderPrepare();
    try {
      const result = await api("/api/v1/setup/translator", { method: "PUT", body: choice });
      state.translationDraft = null;
      showSetup(result.setup);
      await loadHealth();
    } catch (error) {
      state.translationDraft = null;
      message(error.message, true);
      renderPrepare();
    }
  }

  async function savePlan(plan) {
    state.planDraft = plan;
    renderPrepare();
    try {
      const result = await api("/api/v1/setup/plan", { method: "PUT", body: plan });
      state.planDraft = null;
      showSetup(result.setup);
      await loadHealth();
    } catch (error) {
      state.planDraft = null;
      message(error.message, true);
    }
  }

  function renderPrepare() {
    const setup = state.setup;
    if (!setup) return;
    const plan = state.planDraft || YataPrepare.planOf(setup);
    // Free memory and disk change on every report; redraw only when something shown as state changed.
    const drafts = { llm: state.llmDraft, translation: state.translationDraft };
    const signature = JSON.stringify([setup, plan, drafts, state.view],
      (key, value) => (["available", "free", "elapsed_ms"].includes(key) ? undefined : value));
    if (signature === state.setupSignature) return;
    state.setupSignature = signature;
    $("prepare-machine").replaceChildren(...YataPrepare.machine(setup));
    $("plan-options").replaceChildren(...YataPrepare.planTiles(setup, plan, (choice) => savePlan({ ...choice, speakers: plan.speakers })));
    $("plan-speakers").checked = plan.speakers;
    const advice = [setup.recommended.plan_reason, setup.plan ? "" : t("prepare.planUnchosen")]
      .filter(Boolean).join(" ");
    $("plan-advice").textContent = advice;
    $("plan-advice").hidden = !advice;
    $("prepare-list").replaceChildren(...YataPrepare.requirements(setup, plan, setupHandlers, drafts));
    const storage = setup.storage;
    const input = $("storage-path");
    if (document.activeElement !== input) input.value = storage.path;
    const locked = storage.source === "env";
    input.disabled = locked;
    $("storage-save").disabled = locked;
    $("storage-reset").disabled = locked || storage.source === "default";
    $("storage-note").textContent = t(locked ? "prepare.storage.env" : storage.source === "default" ? "prepare.storage.default" : "prepare.storage.chosen")
      + " " + t("prepare.storage.reuse");
    const ready = YataPrepare.readiness(setup, plan, drafts);
    $("prepare-summary").textContent = ready.missing.length ? t("prepare.missing", { items: ready.missing.join(t("common.listSeparator")) }) : t("prepare.ready");
    $("prepare-done").disabled = ready.missing.length > 0;
  }

  function renderPrepareLink() {
    const link = $("open-prepare");
    const busy = runningJobs(state.setup).length > 0 || Boolean(state.setupBrief?.busy && !state.setup);
    const attention = !state.asrReady || (state.setupBrief?.attention || []).length > 0;
    link.dataset.attention = busy ? "busy" : attention ? "true" : "false";
    link.dataset.tip = t(busy ? "prepare.link.busy" : attention ? "prepare.link.attention" : "prepare.title");
  }

  const memo = { timer: null, saving: false, sessionId: null };

  function setMemoState(text) {
    $("memo-state").textContent = text;
  }

  function showMemo(session, isNewSession) {
    const box = $("memo-text");
    if (!isNewSession && (memo.timer != null || memo.saving || document.activeElement === box)) return;
    memo.sessionId = session.id;
    if (box.value !== (session.notes || "")) box.value = session.notes || "";
    if (isNewSession) setMemoState("");
  }

  function scheduleMemo() {
    clearTimeout(memo.timer);
    setMemoState(t("memo.editing"));
    memo.timer = setTimeout(saveMemo, 800);
  }

  async function saveMemo() {
    clearTimeout(memo.timer);
    memo.timer = null;
    const id = memo.sessionId;
    if (!id) return;
    const notes = $("memo-text").value;
    memo.saving = true;
    setMemoState(t("memo.saving"));
    try {
      await api("/api/v1/sessions/" + id + "/notes", { method: "PUT", body: { notes } });
      if (state.session?.id === id) {
        state.session.notes = notes;
        setMemoState(t("memo.saved"));
      }
    } catch (error) {
      setMemoState(t("memo.saveFailed"));
      message(t("toast.memoFailed", { error: error.message }), true);
    } finally {
      memo.saving = false;
    }
  }

  function flushMemo() {
    if (memo.timer != null) saveMemo();
  }

  function insertMemoTime() {
    const box = $("memo-text");
    const stamp = "[" + formatTime(elapsedMs()) + "] ";
    const start = box.selectionStart ?? box.value.length;
    const end = box.selectionEnd ?? start;
    box.value = box.value.slice(0, start) + stamp + box.value.slice(end);
    box.selectionStart = box.selectionEnd = start + stamp.length;
    box.focus?.();
    scheduleMemo();
  }

  async function exportSession() {
    const id = state.session?.id;
    if (!id) return;
    if (memo.timer != null) await saveMemo();
    const link = document.createElement("a");
    link.href = "/api/v1/sessions/" + id + "/export.md";
    link.download = "";
    document.body.append(link);
    link.click();
    link.remove();
  }

  const SIDE_WIDTH = "yata.sideWidth";

  function setSideWidth(px, save) {
    const grid = $("session-view");
    const resizer = $("column-resizer");
    if (px == null) {
      grid.classList.remove("sized");
      grid.style.removeProperty?.("--side-width");
      resizer.removeAttribute?.("aria-valuenow");
      try { localStorage.removeItem(SIDE_WIDTH); } catch { /* storage off: only this page forgets */ }
      return;
    }
    // The CSS keeps at least 360 px for the transcript; this keeps the side column from 300 px to the window.
    const value = Math.round(Math.max(300, Math.min(px, window.innerWidth || px)));
    grid.style.setProperty("--side-width", value + "px");
    grid.classList.add("sized");
    resizer.setAttribute("aria-valuenow", String(value));
    if (save) {
      try { localStorage.setItem(SIDE_WIDTH, String(value)); } catch { /* storage off: the width lasts this page */ }
    }
  }

  function sideWidth() {
    return $("column-side").getBoundingClientRect?.().width || 360;
  }

  function startResize(event) {
    if (event.button !== undefined && event.button !== 0) return;
    const resizer = $("column-resizer");
    const right = $("session-view").getBoundingClientRect?.().right || 0;
    resizer.setPointerCapture?.(event.pointerId);
    resizer.classList.add("dragging");
    body.classList.add("resizing");
    const move = (moved) => setSideWidth(right - moved.clientX - 8, false);
    const end = () => {
      resizer.classList.remove("dragging");
      body.classList.remove("resizing");
      resizer.removeEventListener("pointermove", move);
      resizer.removeEventListener("pointerup", end);
      resizer.removeEventListener("pointercancel", end);
      setSideWidth(sideWidth(), true);
    };
    resizer.addEventListener("pointermove", move);
    resizer.addEventListener("pointerup", end);
    resizer.addEventListener("pointercancel", end);
    event.preventDefault?.();
  }

  function keyResize(event) {
    const step = event.shiftKey ? 96 : 32;
    if (event.key === "ArrowLeft") setSideWidth(sideWidth() + step, true);
    else if (event.key === "ArrowRight") setSideWidth(sideWidth() - step, true);
    else if (event.key === "Home" || event.key === "Escape") setSideWidth(null);
    else return;
    event.preventDefault?.();
  }

  const LANGUAGE_NAMES = { ja: "日本語", en: "English" };

  function renderLanguageChoice() {
    const select = $("ui-language");
    if (select.options?.length || select.children?.length) {
      select.value = YataI18n.choice || "auto";
      return;
    }
    const browser = YataI18n.browserLocale(navigator) || YataI18n.FALLBACK;
    select.add(new Option(t("settings.languageAuto", { name: LANGUAGE_NAMES[browser] }), "auto"));
    YataI18n.SUPPORTED.forEach((code) => select.add(new Option(LANGUAGE_NAMES[code] || code, code)));
    select.value = YataI18n.choice || "auto";
  }

  async function saveLanguage(event) {
    const value = event.target.value || "auto";
    if (value === (YataI18n.choice || "auto")) return;
    event.target.disabled = true;
    message(t("settings.languageSaving"));
    try {
      await api("/api/v1/setup/locale", { method: "PUT", body: { locale: value === "auto" ? "" : value } });
      // Every label, list and card is drawn again in the new language; the settings cannot open while capturing.
      window.location?.reload?.();
    } catch (error) {
      event.target.disabled = false;
      event.target.value = YataI18n.choice || "auto";
      message(t("settings.languageFailed", { error: error.message }), true);
    }
  }

  function requestDelete(id) {
    clearTimeout(state.armTimer);
    if (state.armed !== id) {
      state.armed = id;
      state.armTimer = setTimeout(() => { state.armed = null; renderSessions(); }, 3000);
      renderSessions();
      return;
    }
    state.armed = null;
    deleteSession(id);
  }

  async function deleteSession(id) {
    try {
      await api("/api/v1/sessions/" + id, { method: "DELETE" });
      if (state.session?.id === id) newSession();
    } catch (error) {
      message(t("toast.deleteFailed", { error: error.message }), true);
    }
    await refreshSessions();
  }

  async function openPip() {
    if (state.pip) { state.pip.window.focus(); return; }
    const pip = await documentPictureInPicture.requestWindow({ width: 680, height: 190 });
    const link = pip.document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/assets/style.css";
    pip.document.head.append(link);
    pip.document.title = "Yata";
    pip.document.documentElement.dataset.theme = YataTheme.get();
    pip.document.body.className = "pip-body";
    const box = pip.document.createElement("div");
    box.className = "caption-lines pip-captions";
    pip.document.body.append(box);
    state.pip = { window: pip, box };
    pip.addEventListener("pagehide", () => { state.pip = null; });
    renderCaptions();
  }

  $("launch").addEventListener("click", () => withBusy(() => share(createSession)));
  $("resume-session").addEventListener("click", () => withBusy(() => share(resumeSession)));
  $("start-share").addEventListener("click", startShare);
  $("new-session").addEventListener("click", newSession);
  $("open-prepare").addEventListener("click", openPrepare);
  $("export-session").addEventListener("click", exportSession);
  $("memo-text").addEventListener("input", scheduleMemo);
  $("memo-text").addEventListener("blur", flushMemo);
  $("memo-time").addEventListener("click", insertMemoTime);
  $("column-resizer").addEventListener("pointerdown", startResize);
  $("column-resizer").addEventListener("keydown", keyResize);
  $("column-resizer").addEventListener("dblclick", () => setSideWidth(null));
  try {
    const saved = Number(localStorage.getItem(SIDE_WIDTH));
    if (saved > 0) setSideWidth(saved, false);
  } catch { /* storage off: the default width */ }
  $("open-settings").addEventListener("click", openSettings);
  $("settings-done").addEventListener("click", newSession);
  $("ui-language").addEventListener("change", saveLanguage);
  $("prepare-done").addEventListener("click", async () => {
    await ensurePlan();
    newSession();
  });
  $("plan-speakers").addEventListener("change", (event) => {
    const plan = state.planDraft || (state.setup ? YataPrepare.planOf(state.setup) : { summaries: false, translation: false });
    savePlan({ summaries: plan.summaries, translation: Boolean(plan.translation), speakers: Boolean(event.target.checked) });
  });
  $("storage-save").addEventListener("click", () => setupCall("/api/v1/setup/storage", "PUT", { path: $("storage-path").value }, true));
  $("storage-reset").addEventListener("click", () => setupCall("/api/v1/setup/storage", "PUT", { path: "" }, true));
  $("toggle-sidebar").addEventListener("click", () => showSidebar(!body.classList.contains("sidebar-open")));
  $("close-sidebar").addEventListener("click", () => showSidebar(false));
  $("sidebar-scrim").addEventListener("click", () => showSidebar(false));
  $("session-search").addEventListener("input", (event) => {
    state.sessionQuery = event.target.value || "";
    renderSessions();
  });
  $("theme-switch").hidden = !YataTheme.switchable;
  function showTheme(theme) {
    YataTheme.THEMES.forEach((name) => $("theme-" + name).setAttribute("aria-pressed", String(name === theme)));
    if (state.lastSpeaker) tintFor(state.lastSpeaker);
    else waveform.setTint(themeColor("--accent-2", "#0eb2d4"), themeColor("--accent", "#6152f3"));
    if (state.pip) state.pip.window.document.documentElement.dataset.theme = theme;
  }
  YataTheme.THEMES.forEach((name) => $("theme-" + name).addEventListener("click", () => YataTheme.set(name)));
  YataTheme.onChange(showTheme);
  $("open-pip").addEventListener("click", () => {
    openPip().catch((error) => message(t("toast.pipFailed", { error: error.message }), true));
  });
  $("summarize-now").addEventListener("click", () => withBusy(async () => {
    const button = $("summarize-now");
    button.classList.add("spinning");
    try {
      await api("/api/v1/sessions/" + state.session.id + "/summarize", { method: "POST" });
      await pollEvents();
    } catch (error) {
      message(error.message, true);
    } finally {
      button.classList.remove("spinning");
    }
  }));
  $("stop-session").addEventListener("click", () => withBusy(async () => {
    try {
      if (state.capture) await state.capture.finish();
      const result = await api("/api/v1/sessions/" + state.session.id + "/stop", { method: "POST" });
      updateStatus(result.session);
      await pollEvents();
      if (result.summary_error) message(t("toast.finalSummaryFailed", { error: result.summary_error }), true);
      else toast("DEBRIEF READY", "ok");
    } catch (error) {
      message(error.message, true);
    }
    refreshSessions();
  }));
  $("cancel-session").addEventListener("click", () => withBusy(async () => {
    try {
      if (state.capture) await state.capture.stop();
      const result = await api("/api/v1/sessions/" + state.session.id + "/cancel", { method: "POST" });
      updateStatus(result.session);
    } catch (error) {
      message(error.message, true);
    }
    refreshSessions();
  }));
  $("toggle-diagnostics").addEventListener("click", () => {
    const drawer = $("audio-diagnostics");
    drawer.hidden = !drawer.hidden;
    $("toggle-diagnostics").setAttribute("aria-expanded", String(!drawer.hidden));
    renderDiagnostics();
  });
  const summaryChanged = () => { applySummaryChoice(); updateControls(); };
  $("model-select").addEventListener("change", summaryChanged);
  $("caption-language").addEventListener("change", summaryChanged);
  $("feed-search").addEventListener("input", (event) => {
    state.search = event.target.value || "";
    applyFilters();
  });
  $("feed-translate").addEventListener("click", () => {
    state.showTranslation = !state.showTranslation;
    $("feed-translate").setAttribute("aria-pressed", String(!state.showTranslation));
    $("transcript-list").classList.toggle("translate-off", !state.showTranslation);
  });
  if (typeof setInterval === "function") {
    setInterval(() => {
      if (state.session && !terminal()) $("mission-clock").textContent = formatTime(elapsedMs());
    }, 1000);
  }

  setMode();
  fillLanguageSelects();
  renderSources();
  resetView();
  showTheme(YataTheme.get());
  loadHealth();
  refreshSessions();
})();
