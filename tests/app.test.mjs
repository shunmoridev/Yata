import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

const read = (path) => readFileSync(new URL(path, import.meta.url), "utf8");
const llmHealth = (models, over = {}) => ({ backend: "ollama", label: "Ollama", chosen: true, connected: models.length > 0,
  models: models.map((name) => ({ id: name, label: name })), reason: null, ...over });

class Element {
  constructor(doc, tag = "div") {
    this.ownerDocument = doc;
    this.tagName = tag;
    this.children = [];
    this.parent = null;
    this.ownText = "";
    this.value = "";
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.open = false;
    this.dataset = {};
    this.attributes = {};
    // CSSOM custom properties (the page sets --side-width this way) land on the same object.
    this.style = Object.defineProperties({}, {
      setProperty: { value(name, value) { this[name] = value; } },
      removeProperty: { value(name) { delete this[name]; } },
    });
    this.handlers = new Map();
    this.classes = new Set();
    this.files = null;
    const classes = this.classes;
    this.classList = {
      add: (...names) => names.forEach((name) => classes.add(name)),
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      contains: (name) => classes.has(name),
      toggle: (name, on) => { const next = on === undefined ? !classes.has(name) : on; next ? classes.add(name) : classes.delete(name); return next; },
    };
  }
  set className(value) { this.classes.clear(); String(value).split(/\s+/).filter(Boolean).forEach((name) => this.classes.add(name)); }
  get className() { return [...this.classes].join(" "); }
  set textContent(value) { this.ownText = String(value); this.children.forEach((child) => { child.parent = null; }); this.children = []; }
  get textContent() { return this.ownText + this.children.map((child) => child.textContent || "").join(""); }
  get firstChild() { return this.children[0] || null; }
  adopt(child) { if (child.parent) child.parent.children = child.parent.children.filter((item) => item !== child); child.parent = this; return child; }
  replaceChildren(...children) { this.ownText = ""; this.children.forEach((child) => { child.parent = null; }); this.children = []; children.forEach((child) => this.append(child)); }
  append(...children) { children.forEach((child) => { const node = typeof child === "string" ? this.ownerDocument.createTextNode(child) : child; this.adopt(node); this.children.push(node); }); }
  insertBefore(child, reference) { this.adopt(child); const index = reference ? this.children.indexOf(reference) : -1; if (index < 0) this.children.push(child); else this.children.splice(index, 0, child); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((item) => item !== this); this.parent = null; }
  add(option) { this.append(option); if (this.children.length === 1) this.value = option.value; }
  setAttribute(name, value) { this.attributes[name] = String(value); if (name === "class") this.className = value; }
  getAttribute(name) { return this.attributes[name]; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(type, callback) { this.handlers.set(type, callback); }
  removeEventListener(type) { this.handlers.delete(type); }
  // A link the page clicks (the export download) is recorded instead of followed.
  click() { (this.ownerDocument.clicked ||= []).push(this); }
  querySelector(selector) { const name = selector.replace(/^[a-z]*\./, ""); return this.find((node) => node.classes?.has(name) || node.tagName === selector); }
  find(predicate) {
    for (const child of this.children) {
      if (!(child instanceof Element)) continue;
      if (predicate(child)) return child;
      const nested = child.find(predicate);
      if (nested) return nested;
    }
    return null;
  }
  findAll(predicate, out = []) {
    for (const child of this.children) {
      if (!(child instanceof Element)) continue;
      if (predicate(child)) out.push(child);
      child.findAll(predicate, out);
    }
    return out;
  }
}

function makeDocument() {
  const elements = new Map();
  const doc = {
    getElementById: (id) => {
      if (!elements.has(id)) elements.set(id, new Element(doc));
      return elements.get(id);
    },
    createElement: (tag) => new Element(doc, tag),
    createElementNS: (_ns, tag) => new Element(doc, tag),
    createTextNode: (text) => ({ textContent: String(text) }),
    // The page asks for its language selects; the static markup's translations (i18n.js) are tested on index.html itself.
    querySelectorAll: (selector) => selector !== ".language-select" ? [] : ["asr-language", "output-language", "caption-language"].map((id) => {
      const select = doc.getElementById(id);
      if (id === "asr-language") select.dataset.includeAuto = "true";
      if (id === "caption-language") select.dataset.includeNone = "true";
      return select;
    }),
    querySelector: () => null,
  };
  return doc;
}

function harness(options = {}) {
  const doc = makeDocument();
  const $ = doc.getElementById;
  doc.body = new Element(doc, "body");
  // The ASR chip as index.html ships it: a value and a load bar.
  $("gauge-asr").append(doc.createElement("b"), doc.createElement("i"));
  $("gauge-llm").append(doc.createElement("b"));
  doc.documentElement = new Element(doc, "html");
  doc.documentElement.dataset.features = (options.features || []).join(" ");
  doc.documentElement.dataset.locale = options.locale || "";
  const stored = new Map(Object.entries(options.storage || {}));
  let poll;
  let response = null;
  let reloads = 0;
  const requests = [];
  const captures = [];
  const session = {
    id: "s1", source_kind: "share", active_input: "shared_audio", status: "capturing_audio", source_title: "", display_title: "",
    title_source: "auto", source_url: "", started_at_utc: "2026-11-04T00:00:00Z", caption_language: "ja", output_language: "ja",
    asr_language: "auto", clock_offset_ms: 0, clock_started_utc: "2026-11-04T00:00:00Z",
    audio_diagnostics: { asr_engine: "faster-whisper", sample_rate: 48000, asr_last_duration_ms: 120, server_socket_state: "open", summary_state: "busy" },
    live_caption: null,
  };
  const replies = {
    "GET /api/v1/health": () => ({ llm: llmHealth(["qwen3.6:35b-a3b", "qwen3:8b"]),
      asr: { engine: "faster-whisper", weights_present: true, loaded: false },
      speakers: { enabled: true, reason: "on" }, slack: options.slack || { enabled: false, reason: "未設定" } }),
    "GET /api/v1/sessions": () => ({ sessions: options.sessions || [] }),
    "POST /api/v1/sessions": () => ({ session }),
    ...(options.replies || {}),
  };
  const sandbox = {
    document: doc,
    window: { setInterval(callback) { poll = callback; return 1; }, clearInterval() {}, location: { reload() { reloads++; } } },
    navigator: { language: (options.languages || ["ja-JP"])[0], languages: options.languages || ["ja-JP", "ja"] },
    fetch: async (path, init) => {
      requests.push({ path, init });
      const route = replies[(init?.method || "GET") + " " + path];
      const value = route ? route(init) : response;
      return { ok: !(value && value.status >= 400), status: value?.status || 200, json: async () => value };
    },
    localStorage: { getItem: (key) => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, String(value)), removeItem: (key) => stored.delete(key) },
    Date, Map, Set, JSON, Math, Number, String, Object, Promise, URL, setTimeout: () => 0, clearTimeout() {},
    Option: class { constructor(label, value) { this.textContent = label; this.value = value; } },
  };
  sandbox.globalThis = sandbox;
  for (const file of ["theme.js", "i18n.js", "gaps.js", "audio-diagnostics.js", "capture.js", "captions.js", "topics.js", "waveform.js", "hud.js", "sessions.js", "prepare.js"]) {
    runInNewContext(read("../web/assets/" + file), sandbox);
  }
  sandbox.YataCapture.Capture = class {
    constructor(callbacks) { this.callbacks = callbacks; this.begun = null; this.stopped = false; captures.push(this); }
    get analyser() { return null; }
    async open() {
      if (options.refuseShare) throw Object.assign(new Error("denied"), { name: "NotAllowedError" });
      return { url: options.reportedUrl || "", surface: "browser" };
    }
    async begin(value) { this.begun = value; this.sessionId = value.id; }
    async finish() { return true; }
    async stop() { this.stopped = true; this.callbacks.onState("stopped", ""); }
  };
  runInNewContext(read("../web/assets/app.js"), sandbox);
  return {
    $, doc, session, body: doc.body, requests, captures, sandbox, stored, reloads: () => reloads,
    click: (id) => $(id).handlers.get("click")(),
    launch: () => $("launch").handlers.get("click")(),
    poll: (value) => { response = value; return poll(); },
    sent: (method, path) => requests.filter((request) => (request.init?.method || "GET") === method && request.path === path),
  };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

test("system lights, the setup page and the launch control reflect health", async () => {
  const h = harness();
  await flush();
  assert.equal(h.$("model-select").value, "qwen3.6:35b-a3b");
  assert.equal(h.$("sys-llm").dataset.state, "ok");
  assert.equal(h.$("sys-asr").dataset.state, "ok");
  assert.equal(h.$("sys-spk").dataset.state, "ok");
  assert.equal(h.$("sys-slack").hidden, true);
  assert.equal(h.$("slack-slot").hidden, true);
  assert.equal(h.$("theme-switch").hidden, true);
  assert.equal(h.doc.documentElement.dataset.theme, "soft");
  assert.equal(h.$("launch").disabled, false);
  assert.ok(h.body.classes.has("mode-idle"));
  const tiles = h.$("source-options").children;
  assert.equal(tiles.length, 1);
  assert.equal(tiles[0].getAttribute("aria-checked"), "true");
  assert.match(tiles[0].textContent, /タブの音声/);
  assert.equal(h.$("slot-title").hidden, true);
  assert.equal(h.$("slot-url").hidden, true);
  assert.equal(h.$("launch-title").textContent, "タブの音声を共有して開始");
  assert.equal(h.$("resume-session").disabled, true);
});

test("a live session renders captions, translations, speakers, topics, digest and gauges", async () => {
  const h = harness();
  await flush();
  await h.launch();
  assert.ok(h.body.classes.has("mode-live"));
  assert.equal(h.$("session-status").textContent, "取り込み中");
  assert.equal(h.$("session-status").dataset.state, "live");

  const evidence = { id: "e1", source_type: "asr", text: "Pricing starts at $999.", language: "en", speaker: "S1", capture_start_ms: 5000, capture_end_ms: 8000 };
  const second = { id: "e2", source_type: "asr", text: "Questions?", language: "en", speaker: "S2", capture_start_ms: 9000, capture_end_ms: 9500 };
  const closed = {
    id: "t1", index: 1, status: "closed", title: "新製品の価格", start_ms: 5000, end_ms: 8000, segment_ids: ["e1"],
    close_reason: "llm_boundary", summary_state: "ready", progress: null,
    summary: {
      title: "新製品の価格", overview: "価格は999ドルから。", overview_needs_review: false, rejected: 1,
      points: [{ text: "価格は999ドルからです。", evidence_ids: ["e1"], needs_review: false }, { text: "", evidence_ids: ["e1"], needs_review: true }],
      key_facts: [{ label: "価格", value: "$999", evidence_id: "e1" }],
    },
  };
  const open = { id: "t2", index: 2, status: "open", title: "", start_ms: 9000, end_ms: 12000, segment_ids: ["e2"], summary_state: "none",
    progress: { title: "質疑", overview: "質疑応答が始まりました。", points: [], key_facts: [] } };
  await h.poll({ session: h.session, events: [
    { cursor: 1, type: "evidence_segment", data: evidence },
    { cursor: 2, type: "evidence_segment", data: second },
    { cursor: 3, type: "caption_translation", data: { segment_id: "e1", language: "ja", text: "価格は999ドルからです。" } },
    { cursor: 4, type: "topic", data: closed },
    { cursor: 5, type: "topic", data: open },
    { cursor: 6, type: "session_digest", data: { overview: "新製品の価格が発表されました。", highlights: [{ text: "999ドルから", topic_ids: ["t1"], needs_review: false }] } },
    { cursor: 7, type: "analysis_error", data: { task: "話題の要約", message: "Ollama down" } },
    { cursor: 8, type: "gap", data: { reason: "音声フレームが欠落しました。", capture_start_ms: 7000, capture_end_ms: 7100, observed_at_utc: "2026-11-04T00:00:07Z" } },
  ] });

  const feed = h.$("transcript-list");
  assert.equal(h.$("segment-count").textContent, "2");
  assert.match(feed.children[0].textContent, /^S2/);
  assert.match(feed.children[1].textContent, /S1.*en.*Pricing starts at \$999\.価格は999ドルからです。/);
  assert.equal(h.$("speaker-filter").children.length, 3);

  const captions = h.$("caption-lines").textContent;
  assert.match(captions, /S1Pricing starts at \$999\.価格は999ドルからです。/);
  assert.match(captions, /S2Questions\?/);

  const topics = h.$("topics-list");
  assert.equal(h.$("topic-count").textContent, "2");
  assert.ok(topics.children[0].classes.has("open"));
  assert.ok(topics.children[1].classes.has("closed"));
  assert.match(topics.textContent, /T02.*質疑応答が始まりました。/s);
  assert.match(topics.textContent, /T01.*新製品の価格.*\$999/s);
  assert.ok(topics.find((node) => node.classes.has("draft")));
  const hazard = topics.find((node) => node.classes.has("hazard"));
  assert.match(hazard.dataset.tip, /本文を保留/);
  assert.equal(h.$("digest").hidden, false);
  assert.match(h.$("digest").textContent, /DEBRIEF.*新製品の価格が発表されました。.*T01/s);

  assert.match(h.$("speaker-radar").textContent, /S1.*S2/s);
  const map = h.$("session-map");
  assert.equal(map.findAll((node) => node.classes.has("map-segment")).length, 2);
  assert.equal(map.findAll((node) => node.classes.has("map-gap")).length, 1);
  assert.equal(h.$("gap-count").textContent, "1");
  assert.equal(h.$("gauge-asr").dataset.tone, "ok");
  assert.equal(h.$("gauge-asr").querySelector("b").textContent, "120");
  assert.equal(h.$("gauge-asr").querySelector("i").style.transform, "scaleX(0.080)");
  assert.equal(h.$("gauge-llm").dataset.tone, "busy");
  assert.equal(h.$("gauge-link").dataset.tone, "ok");
  assert.equal(h.$("readout-rate").textContent, "48.0 kHz");
  assert.match(h.$("session-message").textContent, /話題の要約: Ollama down/);

  h.$("speaker-filter").children[2].handlers.get("click")();
  assert.equal(feed.children[0].hidden, false);
  assert.equal(feed.children[1].hidden, true);
});

test("the launch button picks the tab first; the title and URL are never typed for tab audio", async () => {
  const h = harness({ reportedUrl: "https://www.youtube.com" });
  await flush();
  h.$("source-title").value = "typed by hand";
  Object.assign(h.session, { caption_language: null, status: "created" });
  await h.launch();
  const [created] = h.sent("POST", "/api/v1/sessions");
  const payload = JSON.parse(created.init.body);
  assert.equal(payload.source_kind, "share");
  assert.equal(payload.source_url, "https://www.youtube.com", "the origin the shared page reported");
  assert.equal(payload.source_title, "");
  assert.equal(h.captures.length, 1);
  assert.equal(h.captures[0].begun.id, "s1");
  assert.ok(h.body.classes.has("mode-live"));
  assert.equal(h.$("feed-translate").hidden, true);
  assert.match(h.$("active-title").textContent, /^タブ音声 11\/04 \d\d:\d\d$/);
  assert.match(h.$("active-meta").textContent, /^11\/04 \d\d:\d\d 開始 · タブ音声$/);
  assert.equal(h.$("launch").disabled, true, "the setup page's button only starts new sessions");
  Object.assign(h.session, { display_title: "新製品の発表会" });
  await h.poll({ session: h.session, events: [] });
  assert.equal(h.$("active-title").textContent, "新製品の発表会");
  assert.match(h.$("session-list").textContent, /今日.*取り込み中/s);
  assert.equal(h.$("session-list").find((node) => node.classes.has("session-row")).classes.has("active"), true);
});

test("a cancelled share picker creates no session", async () => {
  const h = harness({ refuseShare: true });
  await flush();
  await h.launch();
  assert.equal(h.sent("POST", "/api/v1/sessions").length, 0);
  assert.ok(h.body.classes.has("mode-idle"));
  assert.match(h.$("session-message").textContent, /共有がキャンセルされました/);
  assert.equal(h.$("launch").disabled, false);
});

test("the sidebar opens a saved session, continues it from the session bar, and deletes it on a second click", async () => {
  const base = { source_kind: "share", active_input: "shared_audio", source_url: "", title_source: "auto", caption_language: null,
    output_language: "ja", asr_language: "auto", audio_diagnostics: {}, live_caption: null, started_at_utc: "2026-11-03T01:00:00Z" };
  const ended = { ...base, id: "s9", status: "ended", source_title: "新製品の発表会", display_title: "新製品の発表会",
    source_url: "https://www.youtube.com", clock_offset_ms: 3_725_000, clock_started_utc: null };
  const listed = [{ id: "s9", display_title: "新製品の発表会", title_source: "auto", source_kind: "share", status: "ended",
    started_at_utc: "2026-11-03T01:00:00Z", updated_at_utc: "2026-11-03T02:00:00Z", duration_ms: 3_725_000, topic_count: 1, segment_count: 2 }];
  const snapshot = {
    session: ended, cursor: 40,
    evidence: [
      { id: "e1", source_type: "asr", text: "新しいカメラです。", speaker: "S1", capture_start_ms: 1000, capture_end_ms: 4000 },
      { id: "e2", source_type: "asr", text: "価格は後日です。", speaker: "S2", capture_start_ms: 5000, capture_end_ms: 8000 },
    ],
    topics: [{ id: "t1", index: 1, status: "closed", title: "新製品", start_ms: 1000, end_ms: 8000, segment_ids: ["e1", "e2"],
      summary_state: "ready", summary: { title: "新製品", overview: "紹介。", points: [], key_facts: [] } }],
    translations: { e1: "A new camera." },
    digest: { overview: "新製品の紹介。", highlights: [] },
    gaps: [{ reason: "音声フレームが欠落しました。", capture_start_ms: 4000, capture_end_ms: 4500, observed_at_utc: "2026-11-03T01:00:05Z", cursor: 12 }],
  };
  const h = harness({ sessions: listed, replies: {
    "GET /api/v1/sessions/s9": () => snapshot,
    "POST /api/v1/sessions/s9/resume": () => ({ session: { ...ended, status: "created", clock_started_utc: new Date().toISOString() } }),
    "POST /api/v1/sessions/s9/cancel": () => ({ session: { ...ended, status: "cancelled" } }),
    "DELETE /api/v1/sessions/s9": () => { listed.length = 0; return { deleted: "s9" }; },
  } });
  await flush();
  assert.equal(h.$("session-total").textContent, "1");
  const row = () => h.$("session-list").find((node) => node.classes.has("session-row"));
  const part = (name) => row().find((node) => node.classes.has(name));
  assert.ok(h.$("session-list").find((node) => node.classes.has("session-group")));
  assert.match(row().textContent, /新製品の発表会\d\d:\d\d · 01:02:05 · 話題 1/);

  await part("session-open").handlers.get("click")();
  assert.ok(h.body.classes.has("mode-ended"));
  assert.equal(h.$("active-title").textContent, "新製品の発表会");
  assert.match(h.$("active-meta").textContent, / 開始 · タブ音声 · www\.youtube\.com$/);
  assert.equal(h.$("active-source").hidden, false);
  assert.equal(h.$("mission-clock").textContent, "01:02:05");
  assert.equal(h.$("segment-count").textContent, "2");
  assert.match(h.$("transcript-list").textContent, /新しいカメラです。A new camera\./);
  assert.equal(h.$("topic-count").textContent, "1");
  assert.equal(h.$("digest").hidden, false);
  assert.equal(h.$("gap-count").textContent, "1");
  assert.equal(h.$("resume-session").disabled, false);
  assert.ok(row().classes.has("active"));
  await h.poll({ session: ended, events: [] });
  assert.match(h.requests.at(-1).path, /\/api\/v1\/sessions\/s9\/events\?after=40$/);

  await h.click("resume-session");
  assert.equal(h.sent("POST", "/api/v1/sessions/s9/resume").length, 1);
  assert.equal(h.sent("POST", "/api/v1/sessions").length, 0);
  assert.equal(h.captures.at(-1).begun.id, "s9");
  assert.equal(h.captures.at(-1).begun.clock_offset_ms, 3_725_000);
  assert.ok(h.body.classes.has("mode-live"));
  assert.equal(h.$("segment-count").textContent, "2", "the earlier lines stay");
  assert.equal(h.$("new-session").disabled, true, "no switching while this page captures");

  assert.equal(part("session-delete").disabled, true);
  await h.click("cancel-session");
  assert.equal(part("session-delete").disabled, false);
  part("session-delete").handlers.get("click")();
  assert.equal(h.sent("DELETE", "/api/v1/sessions/s9").length, 0, "the first click only arms the button");
  assert.equal(part("session-delete").dataset.armed, "true");
  part("session-delete").handlers.get("click")();
  await flush();
  await flush();
  assert.equal(h.sent("DELETE", "/api/v1/sessions/s9").length, 1);
  assert.ok(h.body.classes.has("mode-idle"));
  assert.equal(h.$("active-title").textContent, "新しいセッション");
  assert.match(h.$("session-list").textContent, /保存されたセッションはありません/);
});

test("the sidebar filter narrows the list by name and the new-session button returns to the setup page", async () => {
  const at = "2026-11-03T01:00:00Z";
  const ended = { id: "s2", source_kind: "share", status: "ended", display_title: "", started_at_utc: at,
    clock_offset_ms: 1000, clock_started_utc: null, audio_diagnostics: {} };
  const h = harness({ replies: { "GET /api/v1/sessions/s2": () => ({ session: ended, cursor: 3, evidence: [], topics: [], translations: {}, digest: null, gaps: [] }) },
    sessions: [
      { id: "s2", source_kind: "share", status: "ended", display_title: "", started_at_utc: at, updated_at_utc: at, duration_ms: 1000 },
      { id: "s3", source_kind: "share", status: "cancelled", display_title: "決算説明会", started_at_utc: at, updated_at_utc: at, duration_ms: 5000 },
    ] });
  await flush();
  const rows = () => h.$("session-list").findAll((node) => node.classes.has("session-row"));
  assert.equal(rows().length, 2);
  assert.match(rows()[1].textContent, /決算説明会中断/);
  h.$("session-search").handlers.get("input")({ target: { value: "決算" } });
  assert.deepEqual(rows().map((row) => row.dataset.sessionId), ["s3"]);
  h.$("session-search").handlers.get("input")({ target: { value: "存在しない" } });
  assert.match(h.$("session-list").textContent, /一致するセッションはありません/);
  h.$("session-search").handlers.get("input")({ target: { value: "" } });

  await rows()[0].find((node) => node.classes.has("session-open")).handlers.get("click")();
  assert.ok(h.body.classes.has("mode-ended"));
  assert.match(h.$("active-title").textContent, /^タブ音声 11\/03/);
  h.click("new-session");
  assert.ok(h.body.classes.has("mode-idle"));
  assert.equal(h.$("session-status").textContent, "待機中");
  assert.equal(h.$("active-title").textContent, "新しいセッション");
  await h.launch();
  assert.equal(h.sent("POST", "/api/v1/sessions").length, 1, "the launch button creates a new session again");
});

test("with the theme switch off every page is soft, whatever was saved before", async () => {
  const h = harness({ storage: { "yata.theme": "hud" } });
  await flush();
  assert.equal(h.doc.documentElement.dataset.theme, "soft");
  assert.equal(h.$("theme-switch").hidden, true);
  assert.equal(h.sandbox.YataTheme.set("hud"), "soft", "the switch cannot be used while its toggle is off");
  assert.equal(h.stored.get("yata.theme"), "hud", "nothing is written either");
});

test("with the theme switch on, the choice applies, is remembered and restored", async () => {
  const h = harness({ features: ["theme_switch"] });
  await flush();
  assert.equal(h.$("theme-switch").hidden, false);
  assert.equal(h.doc.documentElement.dataset.theme, "soft");
  assert.equal(h.$("theme-soft").getAttribute("aria-pressed"), "true");
  h.click("theme-hud");
  assert.equal(h.doc.documentElement.dataset.theme, "hud");
  assert.equal(h.$("theme-hud").getAttribute("aria-pressed"), "true");
  assert.equal(h.$("theme-soft").getAttribute("aria-pressed"), "false");
  assert.equal(h.stored.get("yata.theme"), "hud");
  const again = harness({ features: ["theme_switch"], storage: { "yata.theme": "hud" } });
  await flush();
  assert.equal(again.doc.documentElement.dataset.theme, "hud");
  assert.equal(again.$("theme-hud").getAttribute("aria-pressed"), "true");
  await again.launch();
  assert.equal(again.$("session-status").textContent, "LIVE");
  const unknown = harness({ features: ["theme_switch"], storage: { "yata.theme": "neon" } });
  assert.equal(unknown.doc.documentElement.dataset.theme, "soft");
});

test("Slack shows only with its toggle on and a configured webhook", async () => {
  const ready = { enabled: true, reason: "セッションで選んだ場合だけ送ります。" };
  const off = harness({ slack: { enabled: false, reason: "Slack 連携は機能トグルでオフです（YATA_FEATURE_SLACK）。" } });
  await flush();
  assert.equal(off.$("sys-slack").hidden, true);
  assert.equal(off.$("slack-slot").hidden, true);
  off.$("slack-delivery").checked = true;
  await off.launch();
  assert.equal(JSON.parse(off.sent("POST", "/api/v1/sessions")[0].init.body).slack_delivery, false);

  const on = harness({ features: ["slack"], slack: ready });
  await flush();
  assert.equal(on.$("sys-slack").hidden, false);
  assert.equal(on.$("sys-slack").dataset.state, "ok");
  assert.equal(on.$("slack-slot").hidden, false);
  assert.equal(on.$("slack-delivery").disabled, false);
  on.$("slack-delivery").checked = true;
  await on.launch();
  assert.equal(JSON.parse(on.sent("POST", "/api/v1/sessions")[0].init.body).slack_delivery, true);

  const unconfigured = harness({ features: ["slack"] });
  await flush();
  assert.equal(unconfigured.$("sys-slack").hidden, false);
  assert.equal(unconfigured.$("sys-slack").dataset.state, "off");
  assert.equal(unconfigured.$("slack-slot").hidden, true);
});

test("without a summary model the setup page starts a captions-only session", async () => {
  const health = { llm: llmHealth([], { reason: "Ollama に接続できません。" }),
    asr: { engine: "faster-whisper", weights_present: true, loaded: false, device: "cpu", description: "faster-whisper faster-whisper-small · CPU int8 · light" },
    speakers: { enabled: false, reason: "" }, slack: { enabled: false, reason: "" } };
  const h = harness({ replies: { "GET /api/v1/health": () => health } });
  Object.assign(h.session, { summaries: false, selected_model: "", caption_language: null, status: "created" });
  h.session.audio_diagnostics = { ...h.session.audio_diagnostics, asr_last_duration_ms: 2250 };
  await flush();
  const select = h.$("model-select");
  assert.deepEqual(select.children.map((option) => [option.textContent, option.value]), [["使わない（要約なし）", ""]]);
  assert.equal(select.value, "");
  assert.equal(h.$("summary-note").hidden, false);
  assert.equal(h.$("caption-language").disabled, true);
  assert.match(h.$("summary-note").textContent, /字幕の翻訳を選ぶと、翻訳専用のモデル（PLaMo）で訳します（「この PC の準備」で準備が必要です）/);
  assert.equal(h.$("launch").disabled, false, "no LLM is needed to start");
  assert.equal(h.$("sys-llm").dataset.state, "warn");
  assert.match(h.$("sys-asr").dataset.tip, /small · CPU int8/);
  assert.match(h.$("session-message").textContent, /字幕のみ/);
  await h.launch();
  const body = JSON.parse(h.sent("POST", "/api/v1/sessions")[0].init.body);
  assert.deepEqual([body.summaries, body.selected_model, body.caption_language], [false, "", ""]);
  assert.ok(h.body.classes.has("summaries-off"));
  assert.match(h.$("active-meta").textContent, /字幕のみ$/);
  assert.match(h.$("session-list").textContent, /字幕のみ/);
  // 2.25 s per utterance is normal on CPU, so the ASR chip stays green there (on a GPU it would be red).
  assert.equal(h.$("gauge-asr").dataset.tone, "ok");
  assert.equal(h.$("gauge-llm").querySelector("b").textContent, "オフ");
});

test("choosing captions only turns translation and Slack off, and a model turns them back on", async () => {
  const h = harness({ features: ["slack"], slack: { enabled: true, reason: "ok" } });
  await flush();
  const select = h.$("model-select");
  assert.equal(select.value, "qwen3.6:35b-a3b");
  assert.equal(select.children.at(-1).value, "");
  assert.equal(h.$("summary-note").hidden, true);
  h.$("caption-language").value = "en";
  h.$("slack-delivery").checked = true;
  select.value = "";
  select.handlers.get("change")();
  assert.equal(h.$("caption-language").disabled, true);
  assert.equal(h.$("caption-language").value, "");
  assert.equal(h.$("slack-delivery").disabled, true);
  assert.equal(h.$("slack-delivery").checked, false);
  assert.equal(h.$("summary-note").hidden, false);
  select.value = "qwen3:8b";
  select.handlers.get("change")();
  assert.equal(h.$("caption-language").disabled, false);
  assert.equal(h.$("slack-delivery").disabled, false);
  assert.equal(h.$("summary-note").hidden, true);
});

test("topic cards show Slack delivery state and offer a retry only after a failure", () => {
  const doc = makeDocument();
  const sandbox = { globalThis: {} };
  sandbox.globalThis = sandbox;
  runInNewContext(read("../web/assets/i18n.js"), sandbox);
  runInNewContext(read("../web/assets/topics.js"), sandbox);
  const container = new Element(doc);
  const retried = [];
  const topic = (id, delivery_state) => ({ id, index: 1, status: "closed", title: "t", start_ms: 0, end_ms: 1000, segment_ids: [],
    summary_state: "ready", summary: { title: "t", overview: "", points: [], key_facts: [] }, delivery_state, delivery_reason: "HTTP 500" });
  sandbox.YataTopics.renderTopics(container, [topic("a", "sent"), topic("b", "failed")], new Map(), () => "00:00:00", (id) => retried.push(id));
  const deliveries = container.findAll((node) => node.classes.has("delivery"));
  assert.equal(deliveries.length, 2);
  const sent = deliveries.find((node) => node.classes.has("sent"));
  const failed = deliveries.find((node) => node.classes.has("failed"));
  assert.equal(sent.dataset.tip, "Slack に送信済み");
  assert.match(failed.dataset.tip, /Slack への送信に失敗（HTTP 500）/);
  assert.equal(failed.tagName, "button");
  failed.handlers.get("click")();
  assert.deepEqual([...retried], ["b"]);
  const quiet = new Element(doc);
  sandbox.YataTopics.renderTopics(quiet, [topic("c", "failed")], new Map(), () => "00:00:00", null);
  assert.equal(quiet.findAll((node) => node.classes.has("delivery")).length, 0);
});

test("partial captions keep their DOM node, animate only new characters and lock into the final", () => {
  const doc = makeDocument();
  const sandbox = { globalThis: {} };
  sandbox.globalThis = sandbox;
  runInNewContext(read("../web/assets/captions.js"), sandbox);
  const { CaptionBoard, render } = sandbox.YataCaptions;
  const container = new Element(doc);
  const board = new CaptionBoard(2);
  board.setPartial({ utterance_id: "u1", text: "We expect", speaker: "S1" });
  render(container, board.lines(), false);
  const row = container.children[0];
  assert.ok(row.classes.has("partial"));
  board.setPartial({ utterance_id: "u1", text: "We expect demand", speaker: "S1" });
  render(container, board.lines(), false);
  assert.equal(container.children[0], row, "the same node is updated in place");
  const animated = row.findAll((node) => node.classes.has("ch")).map((node) => node.textContent).join("");
  assert.equal(animated, " demand", "only the new tail rises in");
  board.addFinal({ segment_id: "e1", utterance_id: "u1", text: "We expect demand to remain strong.", speaker: "S1" });
  render(container, board.lines(), false);
  assert.equal(container.children[0], row, "the final takes over the partial's node");
  assert.ok(row.classes.has("final"));
  assert.ok(row.classes.has("locking"));
  assert.ok(row.classes.has("tone-1"));
});

const GB = 1024 ** 3;
const SMALL = 486_212_372;

function setupModel(id, label, extra) {
  return {
    id, label, summary: label + " の説明", target: id === "whisper-turbo" ? "gpu" : "cpu", size: id === "whisper-turbo" ? 1_622_000_000 : SMALL,
    folder: "C:\\Yata\\models\\" + label, location: "folder", source: "Hugging Face · test/" + label, page: "https://huggingface.co/test/" + label,
    license: "MIT", state: "missing", usable: false, missing: [], mismatched: [], remaining: SMALL, partial_bytes: 0,
    recommended: id === "whisper-small", check: null, job: null, active: id === "whisper-small", ...extra,
  };
}

function setupStatus(over = {}) {
  return {
    hardware: { cpu: { name: "Test CPU", logical: 4 }, memory: { total: 8 * GB, available: 4 * GB }, gpu: null },
    storage: { path: "C:\\Yata\\models", source: "default", free: 100 * GB },
    plan: null,
    recommended: { plan: "captions", plan_reason: "この PC では字幕のみを勧めます。", asr: "whisper-small",
      asr_reason: "CUDA で使える GPU がないため、CPU で動く軽量モデルを勧めます。", llm: null, llm_reason: "メモリが 16 GB 未満のため、要約は勧めません。" },
    asr: { locked: null, chosen: null, active: { engine: "faster-whisper", weights_present: false, description: "faster-whisper （モデル未設定）" },
      models: [setupModel("whisper-turbo", "whisper-large-v3-turbo"), setupModel("whisper-small", "whisper-small")] },
    vad: { state: "ready", label: "Silero VAD v6", detail: "faster-whisper に同梱されています。" },
    llm: setupLlm(),
    speakers: { package: false, command: "uv sync --inexact --extra speakers", enabled: false, env: false,
      model: setupModel("speaker-campplus", "CAM++", { active: undefined, recommended: true, size: 28_281_164, remaining: 28_281_164 }) },
    jobs: {},
    ...over,
  };
}

function setupLlm(over = {}) {
  return {
    backend: "ollama", source: "earlier", chosen: true, locked: null, recommended: "builtin",
    options: [{ id: "builtin", label: "内蔵（llama.cpp）", usable: false }, { id: "ollama", label: "Ollama", usable: false },
      { id: "openai", label: "OpenAI 互換 API", usable: false }],
    ollama: { state: "absent", version: null, install_url: "https://ollama.com/download", store: "C:\\Users\\u\\.ollama\\models", free: 100 * GB },
    models: [{ name: "qwen3:8b", size: 5_225_387_677, need: 7 * GB, summary: "最小", installed: false, recommended: true, state: "unknown", check: null, job: null }],
    builtin: { env: null, executable: false, device: null, devices: [], server: { running: false },
      runtime: setupModel("llama-cpp", "llama.cpp b11800（Vulkan）", { target: "", size: 36_859_398, remaining: 36_859_398, recommended: true, active: undefined }),
      models: [setupModel("builtin-qwen3-8b", "Qwen3-8B（Q4_K_M）", { target: "", size: 5_027_783_488, remaining: 5_027_783_488, need: 7 * GB, recommended: true, active: undefined })] },
    openai: { base_url: "", model: "", key: false, key_env: "YATA_OPENAI_API_KEY", locked: [], server_models: [], reason: null, check: null, job: null, state: "unset" },
    foundry: { installed: false, service: { state: "absent", reason: null }, models: [], job: null },
    ...over,
  };
}

const withModel = (setup, id, extra) => ({ ...setup, asr: { ...setup.asr, models: setup.asr.models.map((model) => model.id === id ? { ...model, ...extra } : model) } });

function healthWith(over) {
  return () => ({
    llm: llmHealth(["qwen3:8b"]), asr: { engine: "faster-whisper", weights_present: true, loaded: false },
    speakers: { enabled: false, reason: "" }, slack: { enabled: false, reason: "" }, setup: { plan: null, attention: [], busy: false }, ...over,
  });
}

const buttons = (node, pattern) => node.findAll((item) => item.tagName === "button" && pattern.test(item.textContent));
const cardsOf = (h) => h.$("prepare-list").children.map((card) => card.dataset.req || card.tagName);

test("without a captions model the setup page opens first, recommends a model, and the launch waits", async () => {
  let setup = setupStatus();
  const h = harness({ replies: {
    "GET /api/v1/health": healthWith({ asr: { engine: "faster-whisper", weights_present: false, reason: "未設定" }, setup: { plan: null, attention: ["asr"], busy: false } }),
    "GET /api/v1/setup": () => ({ setup }),
    "PUT /api/v1/setup/plan": (init) => { setup = { ...setup, plan: JSON.parse(init.body) }; return { setup }; },
    "POST /api/v1/setup/models/whisper-small/download": () => {
      setup = withModel(setup, "whisper-small", { job: { component: "asr", target: "whisper-small", kind: "download", state: "running", phase: "download", done: 0, total: SMALL, rate: 0 } });
      setup.jobs = { asr: setup.asr.models[1].job };
      return { setup };
    },
  } });
  await flush();
  await flush();
  assert.ok(h.body.classes.has("mode-prepare"));
  assert.equal(h.$("active-title").textContent, "この PC の準備");
  assert.equal(h.$("launch").disabled, true);
  assert.equal(h.$("open-prepare").dataset.attention, "true");
  assert.deepEqual(cardsOf(h), ["asr", "vad", "details"]);
  assert.match(h.$("prepare-machine").textContent, /Test CPU（4 スレッド）/);
  assert.equal(h.$("plan-options").children.find((tile) => tile.getAttribute("aria-checked") === "true").dataset.plan, "captions");
  const asr = h.$("prepare-list").children[0];
  assert.match(asr.textContent, /CPU で動く軽量モデル/);
  assert.match(asr.textContent, /whisper-smallおすすめCPU 向け未取得/);
  assert.match(asr.textContent, /C:\\Yata\\models\\whisper-small/);
  assert.match(asr.textContent, /Hugging Face · test\/whisper-small · MIT/);
  assert.match(h.$("prepare-summary").textContent, /字幕のモデル/);
  assert.equal(h.$("prepare-done").disabled, true);
  const start = buttons(asr, /準備する（464 MB）/)[1] || buttons(asr, /準備する（464 MB）/)[0];
  await start.handlers.get("click")();
  await flush();
  assert.equal(h.sent("POST", "/api/v1/setup/models/whisper-small/download").length, 1);
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/plan")[0].init.body), { summaries: false, translation: false, speakers: false });
  setup = withModel(setup, "whisper-small", { job: { ...setup.asr.models[1].job, done: SMALL / 2, rate: 12 * 1024 * 1024, file: "model.bin" } });
  setup.jobs = { asr: setup.asr.models[1].job };
  await h.poll();
  await flush();
  assert.match(h.$("prepare-list").textContent, /取得中 · 50% · 232 MB \/ 464 MB · 12 MB\/秒 · model\.bin/);
  assert.equal(buttons(h.$("prepare-list"), /中止/).length, 1);
  assert.equal(h.$("open-prepare").dataset.attention, "busy");
  const healthReads = h.sent("GET", "/api/v1/health").length;
  setup = withModel(setup, "whisper-small", { state: "ready", usable: true, check: { load_ms: 1200, test_ms: 300, device: "cpu", compute_type: "int8", at: "2026-11-04T10:00:00Z" },
    job: { ...setup.asr.models[1].job, state: "done", phase: "load" } });
  setup.asr.active = { engine: "faster-whisper", weights_present: true, description: "faster-whisper faster-whisper-small · CPU int8 · light" };
  setup.jobs = { asr: setup.asr.models[1].job };
  await h.poll();
  await flush();
  assert.match(h.$("session-message").textContent, /whisper-small を読み込んで確認しました/);
  assert.ok(h.sent("GET", "/api/v1/health").length > healthReads);
  assert.match(h.$("prepare-list").textContent, /準備完了/);
  assert.match(h.$("prepare-list").textContent, /読み込み 1\.2 秒 · 試験 0\.3 秒 · CPU int8/);
  assert.equal(h.$("prepare-done").disabled, false);
});

test("choosing summaries adds Ollama with install guidance and the summary models, and speaker labels add their card", async () => {
  let setup = setupStatus({ plan: { summaries: false, speakers: false } });
  setup = withModel(setup, "whisper-small", { state: "ready", usable: true });
  setup.asr.active = { engine: "faster-whisper", weights_present: true, description: "small" };
  const h = harness({ replies: {
    "GET /api/v1/health": healthWith({ setup: { plan: setup.plan, attention: [], busy: false } }),
    "GET /api/v1/setup": () => ({ setup }),
    "PUT /api/v1/setup/plan": (init) => { setup = { ...setup, plan: JSON.parse(init.body) }; return { setup }; },
    "POST /api/v1/setup/llm/pull": () => ({ setup }),
  } });
  await flush();
  assert.ok(h.body.classes.has("mode-idle"), "a captions model is present, so the new-session page shows");
  h.click("open-prepare");
  await flush();
  assert.deepEqual(cardsOf(h), ["asr", "vad", "details"]);
  await h.$("plan-options").children.find((tile) => tile.dataset.plan === "summaries").handlers.get("click")();
  await flush();
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/plan").at(-1).init.body), { summaries: true, translation: false, speakers: false });
  assert.deepEqual(cardsOf(h), ["asr", "vad", "llm-backend", "ollama", "llm", "details"]);
  const ollama = h.$("prepare-list").children[3];
  assert.match(ollama.textContent, /未インストール/);
  assert.match(ollama.textContent, /インストールは Yata からは行いません/);
  assert.equal(ollama.find((node) => node.tagName === "a").href, "https://ollama.com/download");
  assert.match(h.$("prepare-summary").textContent, /要約のモデル/);
  setup = { ...setup, llm: { ...setup.llm, ollama: { ...setup.llm.ollama, state: "running", version: "0.40.0" },
    models: [{ ...setup.llm.models[0], state: "missing" }] } };
  await h.$("prepare-list").children[3].find((node) => node.tagName === "button" && /再確認/.test(node.textContent)).handlers.get("click")();
  await flush();
  const pull = buttons(h.$("prepare-list"), /取得する（4\.9 GB）/)[0];
  await pull.handlers.get("click")();
  await flush();
  assert.deepEqual(JSON.parse(h.sent("POST", "/api/v1/setup/llm/pull")[0].init.body), { model: "qwen3:8b" });
  assert.match(h.$("prepare-list").textContent, /動作に約 7\.0 GB のメモリ/);
  h.$("plan-speakers").handlers.get("change")({ target: { checked: true } });
  await flush();
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/plan").at(-1).init.body), { summaries: true, translation: false, speakers: true });
  const speakers = h.$("prepare-list").children.find((card) => card.dataset.req === "speakers");
  assert.match(speakers.textContent, /uv sync --inexact --extra speakers/);
  assert.equal(buttons(speakers, /準備する（27 MB）/).length, 1);
});

test("failures say what went wrong, a running job can be stopped, and env-fixed choices cannot be changed", async () => {
  let setup = setupStatus({ plan: { summaries: false, speakers: false } });
  setup = withModel(setup, "whisper-small", { partial_bytes: 1000,
    job: { component: "asr", target: "whisper-small", kind: "download", state: "failed", phase: "download", done: 0, total: SMALL,
      error: { kind: "disk", message: "空き容量が足りません（必要 700 MB、空き 200 MB）。" } } });
  setup.jobs = { asr: setup.asr.models[1].job };
  const h = harness({ replies: {
    "GET /api/v1/health": healthWith({ asr: { engine: "faster-whisper", weights_present: false }, setup: { plan: setup.plan, attention: ["asr"], busy: false } }),
    "GET /api/v1/setup": () => ({ setup }),
    "POST /api/v1/setup/jobs/asr/cancel": () => ({ setup }),
  } });
  await flush();
  h.click("open-prepare");
  await flush();
  const small = h.$("prepare-list").children[0].findAll((node) => node.classes?.has("model-row"))[1];
  assert.match(small.textContent, /空き容量：空き容量が足りません/);
  assert.equal(buttons(small, /続きから取得（464 MB）/).length, 1, "a partial download continues where it stopped");
  setup = withModel(setup, "whisper-small", { job: { ...setup.asr.models[1].job, state: "running", error: null, done: 100 } });
  setup.jobs = { asr: setup.asr.models[1].job };
  for (let tick = 0; tick < 5; tick++) { await h.poll(); await flush(); }
  await buttons(h.$("prepare-list"), /中止/)[0].handlers.get("click")();
  assert.equal(h.sent("POST", "/api/v1/setup/jobs/asr/cancel").length, 1);
  setup = { ...setupStatus({ plan: { summaries: false, speakers: false } }), storage: { path: "D:\\fixed", source: "env", free: GB } };
  setup.asr = { ...setup.asr, locked: "YATA_ASR_MODEL_PATH" };
  await h.poll();
  await flush();
  assert.equal(h.$("storage-path").value, "D:\\fixed");
  assert.equal(h.$("storage-path").disabled, true);
  assert.equal(h.$("storage-save").disabled, true);
  assert.match(h.$("storage-note").textContent, /YATA_MODEL_ROOT/);
  assert.match(h.$("prepare-list").textContent, /環境変数 YATA_ASR_MODEL_PATH/);
  assert.ok(h.$("prepare-list").findAll((node) => node.type === "radio").every((radio) => radio.disabled));
});

test("a captions-only use makes captions only the default and asks for no summary model", async () => {
  const h = harness({ replies: { "GET /api/v1/health": healthWith({ setup: { plan: { summaries: false, speakers: false }, attention: [], busy: false } }) } });
  await flush();
  assert.equal(h.$("model-select").value, "");
  assert.equal(h.$("summary-note").hidden, false);
  assert.doesNotMatch(h.$("session-message").textContent, /要約モデル/);
  assert.equal(h.$("open-prepare").dataset.attention, "false");
});

test("until the chosen use has what it needs, the setup page stands in for the new-session page", async () => {
  let attention = ["llm"];
  let setup = setupStatus({ plan: { summaries: true, translation: false, speakers: false },
    recommended: { ...setupStatus().recommended, plan: "summaries", plan_reason: "" } });
  setup = withModel(setup, "whisper-small", { state: "ready", usable: true });
  setup.asr.active = { engine: "faster-whisper", weights_present: true, description: "small" };
  const h = harness({ replies: {
    "GET /api/v1/health": () => healthWith({ setup: { plan: setup.plan, attention, busy: false } })(),
    "GET /api/v1/setup": () => ({ setup }),
    "PUT /api/v1/setup/plan": (init) => { setup = { ...setup, plan: JSON.parse(init.body) }; return { setup }; },
  } });
  await flush();
  await flush();
  assert.ok(h.body.classes.has("mode-prepare"), "summaries are chosen but no summary model is here");
  assert.equal(h.$("plan-advice").hidden, true, "no advice line where there is nothing to advise");
  assert.equal(h.$("prepare-done").disabled, true);
  h.click("new-session");
  await flush();
  assert.ok(h.body.classes.has("mode-prepare"), "the new-session page does not open");
  assert.match(h.$("session-message").textContent, /準備が終わるまで、新しいセッションは開けません/);
  attention = [];
  await h.$("plan-options").children.find((tile) => tile.dataset.plan === "captions").handlers.get("click")();
  await flush();
  assert.equal(h.$("prepare-done").disabled, false);
  await h.$("prepare-done").handlers.get("click")();
  await flush();
  assert.ok(h.body.classes.has("mode-idle"));
  h.click("open-prepare");
  await flush();
  h.click("new-session");
  assert.ok(h.body.classes.has("mode-idle"), "once ready, 新しいセッション opens the new-session page");
});

const tile = (container, backend) => container.find((node) => node.dataset?.backend === backend);
const labelled = (container, label) => container.find((node) => node.attributes?.["aria-label"] === label);
const cardsIn = (node) => node.children.map((card) => card.dataset.req || card.tagName);

test("the first run asks for the summary LLM even with a captions model, and the pick is saved", async () => {
  let setup = setupStatus({ recommended: { ...setupStatus().recommended, plan: "summaries", llm: "qwen3:8b" },
    llm: setupLlm({ backend: null, source: null, chosen: false }) });
  setup = withModel(setup, "whisper-small", { state: "ready", usable: true });
  setup.asr.active = { engine: "faster-whisper", weights_present: true, description: "small" };
  const h = harness({ replies: {
    "GET /api/v1/health": healthWith({ llm: { backend: null, chosen: false, connected: false, models: [], reason: "要約の LLM がまだ選ばれていません。" },
      setup: { plan: null, attention: [], busy: false, llm: { backend: null, chosen: false } } }),
    "GET /api/v1/setup": () => ({ setup }),
    "PUT /api/v1/setup/plan": (init) => { setup = { ...setup, plan: JSON.parse(init.body) }; return { setup }; },
    "PUT /api/v1/setup/llm": (init) => { setup = { ...setup, llm: { ...setup.llm, backend: JSON.parse(init.body).backend, chosen: true, source: "settings" } }; return { setup }; },
    "POST /api/v1/setup/models/llama-cpp/download": () => ({ setup }),
  } });
  await flush();
  await flush();
  assert.ok(h.body.classes.has("mode-prepare"), "nothing is chosen yet, so the setup page comes first");
  assert.deepEqual(cardsOf(h), ["asr", "vad", "llm-backend", "builtin", "details"]);
  const picker = h.$("prepare-list").children[2];
  assert.match(picker.textContent, /まだ選ばれていないため/);
  assert.equal(tile(picker, "builtin").getAttribute("aria-checked"), "true");
  assert.match(tile(picker, "builtin").textContent, /おすすめ/);
  assert.equal(tile(picker, "foundry"), null, "Foundry Local is not offered where it is not installed");
  const builtin = h.$("prepare-list").children[3];
  assert.match(builtin.textContent, /llama\.cpp b11800（Vulkan）/);
  assert.match(builtin.textContent, /動作に約 7\.0 GB のメモリ/);
  assert.match(h.$("prepare-summary").textContent, /要約のモデル/);
  await buttons(builtin, /準備する（35 MB）/)[0].handlers.get("click")();
  await flush();
  assert.equal(h.sent("POST", "/api/v1/setup/models/llama-cpp/download").length, 1);
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/plan")[0].init.body), { summaries: true, translation: false, speakers: false });
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/llm")[0].init.body), { backend: "builtin" });
  await tile(h.$("prepare-list").children[2], "ollama").handlers.get("click")();
  await flush();
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/llm").at(-1).init.body), { backend: "ollama" });
  assert.deepEqual(cardsOf(h), ["asr", "vad", "llm-backend", "ollama", "llm", "details"]);
});

test("the settings hold only the display language; the LLM is changed later on the setup page: an OpenAI-compatible server on this PC, or Foundry Local where it is installed", async () => {
  let setup = setupStatus({ plan: { summaries: true, speakers: false } });
  setup = withModel(setup, "whisper-small", { state: "ready", usable: true });
  setup.asr.active = { engine: "faster-whisper", weights_present: true, description: "small" };
  const h = harness({ replies: {
    "GET /api/v1/health": healthWith({ setup: { plan: setup.plan, attention: [], busy: false, llm: { backend: "ollama", chosen: true } } }),
    "GET /api/v1/setup": () => ({ setup }),
    "PUT /api/v1/setup/llm": (init) => { setup = { ...setup, llm: { ...setup.llm, backend: JSON.parse(init.body).backend, source: "settings" } }; return { setup }; },
    "PUT /api/v1/setup/llm/openai": (init) => {
      const saved = JSON.parse(init.body);
      setup = { ...setup, llm: { ...setup.llm, openai: { ...setup.llm.openai, ...saved, state: "unverified", server_models: ["local-model"] },
        options: setup.llm.options.map((option) => option.id === "openai" ? { ...option, usable: true } : option) } };
      return { setup };
    },
    "POST /api/v1/setup/llm/test": () => ({ setup }),
    "POST /api/v1/setup/llm/foundry/start": () => ({ setup }),
  } });
  await flush();
  assert.ok(h.body.classes.has("mode-idle"), "chosen earlier: no setup page on start");
  h.click("open-settings");
  await flush();
  assert.ok(h.body.classes.has("mode-settings"));
  assert.equal(h.$("active-title").textContent, "設定");
  assert.equal(h.$("active-meta").textContent, "表示言語");
  assert.equal(h.$("settings-llm").children.length, 0);
  h.click("settings-done");
  assert.ok(h.body.classes.has("mode-idle"));
  h.click("open-prepare");
  await flush();
  assert.deepEqual(cardsOf(h), ["asr", "vad", "llm-backend", "ollama", "llm", "details"]);
  await tile(h.$("prepare-list"), "openai").handlers.get("click")();
  await flush();
  assert.deepEqual(cardsOf(h), ["asr", "vad", "llm-backend", "openai", "details"]);
  const form = h.$("prepare-list").children[3];
  assert.match(form.textContent, /この PC（127\.0\.0\.1・localhost・::1）のサーバーだけ/);
  assert.match(form.textContent, /YATA_OPENAI_API_KEY は未設定/);
  assert.match(h.$("prepare-summary").textContent, /要約のモデル/);
  labelled(form, "サーバーの URL").handlers.get("input")({ target: { value: "http://127.0.0.1:1234/v1" } });
  labelled(form, "モデル").handlers.get("input")({ target: { value: "local-model" } });
  await buttons(form, /^保存$/)[0].handlers.get("click")();
  await flush();
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/llm/openai")[0].init.body), { base_url: "http://127.0.0.1:1234/v1", model: "local-model" });
  const saved = h.$("prepare-list").children[3];
  assert.match(saved.textContent, /使えます（未確認）/);
  assert.equal(labelled(saved, "サーバーの URL").value, "http://127.0.0.1:1234/v1");
  assert.doesNotMatch(h.$("prepare-summary").textContent, /要約のモデル/);
  await buttons(saved, /^確認する$/)[0].handlers.get("click")();
  assert.deepEqual(JSON.parse(h.sent("POST", "/api/v1/setup/llm/test")[0].init.body), { model: "local-model", backend: "openai" });
  setup = { ...setup, llm: { ...setup.llm, backend: "foundry",
    options: setup.llm.options.concat([{ id: "foundry", label: "Foundry Local", usable: false }]),
    foundry: { installed: true, service: { state: "stopped", reason: "Foundry Local のサービスが起動していません。" }, job: null,
      models: [{ name: "qwen3-8b", summary: "qwen3:8b と同じ", installed: false, variant: null, recommended: true, state: "unknown", check: null, job: null }] } } };
  await h.poll();
  for (let tick = 0; tick < 5; tick++) { await h.poll(); await flush(); }
  assert.deepEqual(cardsOf(h), ["asr", "vad", "llm-backend", "foundry", "details"]);
  const foundry = h.$("prepare-list").children[3];
  assert.match(foundry.textContent, /サービス停止中/);
  assert.equal(buttons(foundry, /取得する/).length, 0, "no download while the service is stopped");
  await buttons(foundry, /起動する/)[0].handlers.get("click")();
  assert.equal(h.sent("POST", "/api/v1/setup/llm/foundry/start").length, 1);
});

const PLAMO_LANGUAGES = ["ar", "de", "en", "es", "fr", "hi", "id", "it", "ja", "ko", "ru", "th", "vi", "zh"];

function setupTranslator(over = {}) {
  const large = setupModel("plamo-3-translate-2b", "PLaMo 3 Translate 2B（Q4_K_M）", {
    target: "", size: 1_807_192_480, remaining: 1_807_192_480, need: 2.1 * GB, recommended: true, active: true, fits_with_summaries: true,
    license: "PLaMo Community License", license_page: "https://plamo.preferredai.jp/info/plamo-community-license-ja",
    terms: [{ text: "個人の利用と、学術を含む非商用の利用は無償です。", link: "", url: "" },
      { text: "商用の利用は、直近の年間売上が 10 億円以下なら登録すると無償で使えます。", link: "PLaMo 商用利用登録フォーム", url: "https://forms.gle/mTL8tBLrMYXKNZD56" },
      { text: "年間売上が 10 億円を超える場合の商用利用には、PFN との別の契約が必要です。", link: "PLaMo 翻訳モデルのお問い合わせ", url: "https://translate.preferredai.jp/llm/contact/" }],
  });
  // The smaller size, with its license as plain text here so the links the test lists stay those of 2B.
  const small = setupModel("plamo-3-translate-0.8b", "PLaMo 3 Translate 0.8B（Q4_K_M）", {
    target: "", size: 573_116_288, remaining: 573_116_288, need: 0.84 * GB, recommended: false, active: false, fits_with_summaries: true,
    license: "PLaMo Community License",
  });
  return {
    env: null, executable: false, device: null, devices: [], available: false, languages: PLAMO_LANGUAGES, with_summaries: false,
    reason: "要約の LLM は重すぎますが、翻訳専用の PLaMo なら軽く動きます。", server: { running: false },
    runtime: setupModel("llama-cpp", "llama.cpp b11800（Vulkan）", { target: "", size: 36_859_398, remaining: 36_859_398, recommended: false, active: undefined }),
    model: large, models: [large, small], model_chosen: false, engine: "llm", engine_chosen: false, engine_recommended: "llm",
    ...over,
  };
}

const links = (node) => node.findAll((item) => item.tagName === "a").map((item) => [item.textContent, item.href]);

test("captions + translation offers PLaMo in two sizes with its license; with summaries the page asks what translates captions", async () => {
  let setup = setupStatus({ plan: { summaries: false, translation: false, speakers: false }, translator: setupTranslator(),
    recommended: { ...setupStatus().recommended, translator: "plamo-3-translate-2b", translator_reason: "PLaMo なら軽く動きます。" } });
  setup = withModel(setup, "whisper-small", { state: "ready", usable: true });
  setup.asr.active = { engine: "faster-whisper", weights_present: true, description: "small" };
  const h = harness({ replies: {
    "GET /api/v1/health": healthWith({ setup: { plan: setup.plan, attention: [], busy: false } }),
    "GET /api/v1/setup": () => ({ setup }),
    "PUT /api/v1/setup/plan": (init) => { setup = { ...setup, plan: JSON.parse(init.body) }; return { setup }; },
    "POST /api/v1/setup/models/plamo-3-translate-2b/download": () => ({ setup }),
    "PUT /api/v1/setup/translator": (init) => {
      const choice = JSON.parse(init.body);
      const translator = setup.translator;
      setup = { ...setup, translator: { ...translator, ...(choice.engine ? { engine: choice.engine, engine_chosen: true } : {}),
        ...(choice.model ? { model_chosen: true, models: translator.models.map((model) => ({ ...model, active: model.id === choice.model })) } : {}) } };
      return { setup };
    },
  } });
  await flush();
  h.click("open-prepare");
  await flush();
  const tiles = h.$("plan-options").children;
  assert.deepEqual(tiles.map((item) => item.dataset.plan), ["captions", "translation", "summaries"]);
  assert.match(tiles[1].textContent, /字幕＋翻訳翻訳するなら/, "suggested where the summary LLM is too heavy");
  await tiles[1].handlers.get("click")();
  await flush();
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/plan").at(-1).init.body), { summaries: false, translation: true, speakers: false });
  assert.equal(h.$("plan-options").children.find((item) => item.getAttribute("aria-checked") === "true").dataset.plan, "translation");
  assert.deepEqual(cardsOf(h), ["asr", "vad", "translator", "details"]);
  const card = h.$("prepare-list").children[2];
  assert.match(card.textContent, /翻訳特化モデル（PLaMo）準備が必要/);
  assert.doesNotMatch(card.textContent, /実行環境がまだありません/, "the model is not here either: its own button says so");
  assert.match(card.textContent, /要約の LLM を使わずに/);
  assert.match(card.textContent, /2B と 0\.8B から選べます/);
  const sizes = card.findAll((node) => node.tagName === "input" && node.type === "radio");
  assert.deepEqual(sizes.map((node) => node.checked), [true, false], "2B until a size is chosen");
  assert.match(card.textContent, /PLaMo 3 Translate 0\.8B（Q4_K_M）/);
  assert.match(card.textContent, /利用条件：PLaMo Community License です。/);
  assert.match(card.textContent, /非商用の利用は無償/);
  assert.match(card.textContent, /10 億円以下なら登録すると無償/);
  assert.match(card.textContent, /10 億円を超える場合の商用利用には、PFN との別の契約が必要/);
  assert.deepEqual(links(card).filter(([, href]) => !href.startsWith("https://huggingface.co")), [
    ["PLaMo Community License", "https://plamo.preferredai.jp/info/plamo-community-license-ja"],
    ["PLaMo 商用利用登録フォーム", "https://forms.gle/mTL8tBLrMYXKNZD56"],
    ["PLaMo 翻訳モデルのお問い合わせ", "https://translate.preferredai.jp/llm/contact/"],
    ["ライセンスの全文", "https://plamo.preferredai.jp/info/plamo-community-license-ja"],
  ]);
  assert.match(card.textContent, /動作に約 2\.1 GB のメモリ/);
  assert.equal(buttons(card, /準備する（35 MB）/).length, 1, "the llama.cpp runtime it runs on");
  assert.match(h.$("prepare-summary").textContent, /翻訳のモデル/);
  await buttons(card, /準備する（1\.7 GB）/)[0].handlers.get("click")();
  assert.equal(h.sent("POST", "/api/v1/setup/models/plamo-3-translate-2b/download").length, 1);
  await sizes[1].handlers.get("change")();
  await flush();
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/translator").at(-1).init.body), { model: "plamo-3-translate-0.8b" });
  setup = { ...setup, translator: { ...setup.translator, with_summaries: true, engine_recommended: "translator" } };
  await h.$("plan-options").children[2].handlers.get("click")();
  await flush();
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/plan").at(-1).init.body), { summaries: true, translation: false, speakers: false });
  assert.deepEqual(cardsOf(h), ["asr", "vad", "llm-backend", "ollama", "llm", "translation", "translator", "details"]);
  const choice = h.$("prepare-list").children[5];
  assert.match(choice.textContent, /字幕の翻訳/);
  assert.match(choice.textContent, /まだ選ばれていないため/);
  const engines = choice.findAll((node) => Boolean(node.dataset?.engine));
  assert.deepEqual(engines.map((node) => [node.dataset.engine, node.getAttribute("aria-checked")]), [["translator", "false"], ["llm", "true"]]);
  assert.match(engines[0].textContent, /^翻訳特化モデルを使用するおすすめ/);
  assert.match(engines[0].textContent, /要約の LLM に翻訳の負荷をかけません/);
  assert.match(engines[1].textContent, /^要約 LLM と同じモデルを使用する/);
  assert.match(engines[1].textContent, /追加の取得やメモリが要らず/);
  assert.doesNotMatch(h.$("prepare-summary").textContent, /翻訳のモデル/, "the summary LLM translates: nothing more to prepare");
  const optional = h.$("prepare-list").children[6];
  assert.deepEqual([optional.dataset.tone, optional.querySelector(".chip").textContent], ["off", "任意"]);
  assert.match(optional.textContent, /任意です。要約のあるセッションでは要約の LLM が訳します。新しいセッションで要約を「使わない」にしたときは/);
  assert.equal(buttons(optional, /準備する（1\.7 GB）/).length, 1, "it can be fetched from here");
  await engines[0].handlers.get("click")();
  await flush();
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/translator").at(-1).init.body), { engine: "translator" });
  assert.deepEqual(cardsOf(h), ["asr", "vad", "llm-backend", "ollama", "llm", "translation", "translator", "details"]);
  const model = h.$("prepare-list").children[6];
  assert.match(model.textContent, /要約の LLM とは別に、字幕の翻訳だけをこのモデルで行います/);
  assert.deepEqual(model.findAll((node) => node.tagName === "input" && node.type === "radio").map((node) => node.checked), [false, true]);
  assert.match(h.$("prepare-summary").textContent, /翻訳のモデル/);
});

test("the translator card takes a PLaMo GGUF file of the user's own, never fetches it, and can forget it", async () => {
  const custom = "plamo-3-translate-custom";
  let setup = setupStatus({ plan: { summaries: false, translation: true, speakers: false }, translator: setupTranslator() });
  setup = withModel(setup, "whisper-small", { state: "ready", usable: true });
  setup.asr.active = { engine: "faster-whisper", weights_present: true, description: "small" };
  const own = "D:\\models\\plamo-q8.gguf";
  const h = harness({ replies: {
    "GET /api/v1/health": healthWith({ setup: { plan: setup.plan, attention: [], busy: false } }),
    "GET /api/v1/setup": () => ({ setup }),
    "PUT /api/v1/setup/translator/path": (init) => {
      const { path } = JSON.parse(init.body);
      const models = setup.translator.models.filter((item) => item.id !== custom).map((item, index) => ({ ...item, active: !path && index === 0 }));
      if (path) {
        models.push({ ...models[0], id: custom, label: "PLaMo 3 Translate（plamo-q8.gguf）", active: true, location: "custom",
          folder: "D:\\models", page: "", source: "この PC のファイル", state: "unverified", missing: [], remaining: 0 });
      }
      setup = { ...setup, translator: { ...setup.translator, custom_path: path, models, model: models.find((item) => item.active) } };
      return { setup };
    },
  } });
  await flush();
  h.click("open-prepare");
  await flush();
  const card = () => h.$("prepare-list").children[cardsOf(h).indexOf("translator")];
  assert.match(card().textContent, /自分で用意した PLaMo 3 Translate（GGUF ファイル）/);
  const input = card().find((node) => node.tagName === "input" && node.type === "text");
  assert.equal(input.value, "");
  assert.equal(buttons(card(), /指定を外す/).length, 0, "nothing to forget yet");
  input.value = own;
  input.handlers.get("input")({ target: input });
  await buttons(card(), /^保存$/)[0].handlers.get("click")();
  await flush();
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/translator/path").at(-1).init.body), { path: own });
  assert.equal(card().find((node) => node.tagName === "input" && node.type === "text").value, own);
  const row = card().findAll((node) => node.classes?.has("model-row")).at(-1);
  assert.match(row.textContent, /PLaMo 3 Translate（plamo-q8\.gguf）/);
  assert.match(row.textContent, /この PC のファイル/);
  assert.equal(row.findAll((node) => node.tagName === "a" && node.textContent === "この PC のファイル").length, 0, "no page to link to");
  assert.equal(buttons(row, /^確認する$/).length, 1);
  assert.equal(buttons(row, /準備する/).length, 0, "a chosen file is not fetched");
  await buttons(card(), /指定を外す/)[0].handlers.get("click")();
  await flush();
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/translator/path").at(-1).init.body), { path: "" });
  assert.equal(card().findAll((node) => node.classes?.has("model-row")).some((node) => /plamo-q8/.test(node.textContent)), false);
});

test("without summaries a translation can be chosen before PLaMo is here: the page says what is missing and the start waits", async () => {
  let translator = { id: "plamo-3-translate-2b", label: "PLaMo 3 Translate 2B（Q4_K_M）", available: false,
    missing: ["llama.cpp の実行環境", "PLaMo 3 Translate 2B（Q4_K_M）"], languages: PLAMO_LANGUAGES };
  const h = harness({ replies: { "GET /api/v1/health": () => healthWith({ translator,
    setup: { plan: { summaries: true, translation: false, speakers: false }, attention: [], busy: false } })() } });
  await flush();
  const select = h.$("model-select");
  const caption = h.$("caption-language");
  select.value = "";
  select.handlers.get("change")();
  assert.equal(caption.disabled, false, "the translation can be picked without summaries");
  assert.equal(caption.children.find((option) => option.value === "pt").disabled, true, "PLaMo does not translate into Portuguese");
  assert.equal(h.$("launch").disabled, false, "captions only can start");
  caption.value = "en";
  caption.handlers.get("change")();
  const note = h.$("summary-note");
  assert.equal(note.dataset.tone, "warn");
  assert.equal(note.textContent, "字幕の翻訳は翻訳専用のモデル（PLaMo）で行いますが、この PC にはまだ llama.cpp の実行環境、PLaMo 3 Translate 2B（Q4_K_M）がありません。"
    + "「この PC の準備」で準備するか、字幕の翻訳を「—」に戻すと開始できます。");
  assert.equal(h.$("launch").disabled, true, "the server would refuse it after the share picker");
  caption.value = "";
  caption.handlers.get("change")();
  assert.equal(note.dataset.tone, "");
  assert.equal(h.$("launch").disabled, false);
  caption.value = "en";
  select.value = "qwen3:8b";
  select.handlers.get("change")();
  assert.equal(h.$("launch").disabled, false);
  assert.equal(note.hidden, true);
});

test("an older server that does not list what PLaMo lacks gets a note naming both parts, not the model alone", async () => {
  const translator = { id: "plamo-3-translate-2b", label: "PLaMo 3 Translate 2B（Q4_K_M）", available: false, languages: PLAMO_LANGUAGES };
  const h = harness({ replies: { "GET /api/v1/health": () => healthWith({ translator,
    setup: { plan: { summaries: true, translation: false, speakers: false }, attention: [], busy: false } })() } });
  await flush();
  h.$("model-select").value = "";
  h.$("model-select").handlers.get("change")();
  h.$("caption-language").value = "en";
  h.$("caption-language").handlers.get("change")();
  assert.match(h.$("summary-note").textContent, /この PC にはまだ llama\.cpp の実行環境か PLaMo のモデルがありません/);
  assert.doesNotMatch(h.$("summary-note").textContent, /2B/);
});

test("PLaMo's model here without llama.cpp: its card says the runtime is what is missing", async () => {
  const fetched = setupTranslator();
  const models = fetched.models.map((model) => model.id === "plamo-3-translate-2b" ? { ...model, state: "unverified", usable: true, remaining: 0 } : model);
  let setup = setupStatus({ plan: { summaries: true, translation: false, speakers: false }, translator: { ...fetched, model: models[0], models } });
  setup = withModel(setup, "whisper-small", { state: "ready", usable: true });
  setup.asr.active = { engine: "faster-whisper", weights_present: true, description: "small" };
  const h = harness({ replies: {
    "GET /api/v1/health": healthWith({ setup: { plan: setup.plan, attention: [], busy: false } }),
    "GET /api/v1/setup": () => ({ setup }),
  } });
  await flush();
  h.click("open-prepare");
  await flush();
  const card = h.$("prepare-list").children[cardsOf(h).indexOf("translator")];
  assert.match(card.textContent, /PLaMo のモデルはこの PC にありますが、動かすための llama\.cpp の実行環境がまだありません/);
});

test("with PLaMo ready, a session without summaries translates into a language PLaMo knows and says so", async () => {
  const h = harness({ replies: { "GET /api/v1/health": healthWith({
    llm: llmHealth([], { reason: "要約の LLM がありません。" }),
    translator: { id: "plamo-3-translate-2b", label: "PLaMo 3 Translate 2B（Q4_K_M）", available: true, languages: PLAMO_LANGUAGES },
    setup: { plan: { summaries: false, translation: true, speakers: false }, attention: [], busy: false } }) } });
  Object.assign(h.session, { summaries: false, selected_model: "", caption_language: "en", translator: "plamo-3-translate-2b", status: "created" });
  h.session.audio_diagnostics = { ...h.session.audio_diagnostics, summary_state: undefined, translator_state: "busy" };
  await flush();
  const caption = h.$("caption-language");
  assert.equal(h.$("model-select").value, "");
  assert.equal(caption.disabled, false, "translation works without a summary model");
  assert.equal(caption.children.find((option) => option.value === "pt").disabled, true, "PLaMo does not translate into Portuguese");
  assert.equal(caption.children.find((option) => option.value === "en").disabled, false);
  assert.match(h.$("summary-note").textContent, /翻訳専用のモデル（PLaMo）/);
  assert.equal(h.$("slack-delivery").disabled, true);
  caption.value = "en";
  await h.launch();
  const body = JSON.parse(h.sent("POST", "/api/v1/sessions")[0].init.body);
  assert.deepEqual([body.summaries, body.selected_model, body.caption_language], [false, "", "en"]);
  assert.match(h.$("active-meta").textContent, /字幕＋翻訳（PLaMo）$/);
  assert.match(h.$("session-list").textContent, /字幕＋翻訳/);
  assert.ok(h.body.classes.has("summaries-off"));
  assert.equal(h.$("gauge-llm").querySelector("b").textContent, "処理中", "the gauge follows the translator");
});

test("the memo shows the session's notes, keeps what is typed, saves it, and the export downloads through a link", async () => {
  const h = harness({ replies: { "PUT /api/v1/sessions/s1/notes": (init) => ({ notes: JSON.parse(init.body).notes }) } });
  h.session.notes = "前のメモ";
  h.session.clock_started_utc = new Date(Date.now() - 62_000).toISOString();
  await flush();
  assert.equal(h.$("export-session").disabled, true, "nothing to export before a session is open");
  assert.equal(h.$("memo-text").disabled, true);
  await h.launch();
  await flush();
  const box = h.$("memo-text");
  assert.equal(box.disabled, false);
  assert.equal(box.value, "前のメモ");
  box.value = "前のメモ\n[00:01:02] 価格を確認";
  box.handlers.get("input")();
  assert.equal(h.$("memo-state").textContent, "編集中…");
  await h.poll({ session: { ...h.session, notes: "前のメモ" }, events: [] });
  assert.equal(box.value, "前のメモ\n[00:01:02] 価格を確認");
  box.handlers.get("blur")();
  await flush();
  const sent = h.sent("PUT", "/api/v1/sessions/s1/notes");
  assert.deepEqual(JSON.parse(sent.at(-1).init.body), { notes: "前のメモ\n[00:01:02] 価格を確認" });
  assert.equal(h.$("memo-state").textContent, "保存しました");
  box.selectionStart = box.selectionEnd = 0;
  h.click("memo-time");
  assert.match(box.value, /^\[00:01:0[23]\] 前のメモ/, "the session clock 62 s after the run began");
  await h.click("export-session");
  const link = h.doc.clicked.at(-1);
  assert.equal(link.href, "/api/v1/sessions/s1/export.md");
  assert.equal(link.download, "");
  assert.equal(h.sent("PUT", "/api/v1/sessions/s1/notes").length, 2, "a memo still waiting is saved before the export");
});

test("the divider widens the side column, remembers the width, and returns to the default", async () => {
  const h = harness({ storage: { "yata.sideWidth": "520" } });
  await flush();
  const grid = h.$("session-view");
  assert.ok(grid.classes.has("sized"));
  assert.equal(grid.style["--side-width"], "520px");
  // Arrow keys move the divider; without a layout here the column measures as 360 px.
  h.$("column-resizer").handlers.get("keydown")({ key: "ArrowLeft", shiftKey: false });
  assert.equal(grid.style["--side-width"], "392px");
  assert.equal(h.stored.get("yata.sideWidth"), "392");
  h.$("column-resizer").handlers.get("keydown")({ key: "ArrowRight", shiftKey: true });
  assert.equal(grid.style["--side-width"], "300px", "never narrower than 300 px");
  h.$("column-resizer").handlers.get("dblclick")();
  assert.equal(grid.classes.has("sized"), false);
  assert.equal(grid.style["--side-width"], undefined);
  assert.equal(h.stored.has("yata.sideWidth"), false);
});

function i18nIn(languages, locale) {
  const doc = makeDocument();
  doc.documentElement = new Element(doc, "html");
  doc.documentElement.dataset.locale = locale || "";
  const sandbox = { document: doc, navigator: languages ? { languages, language: languages[0] } : undefined };
  sandbox.globalThis = sandbox;
  runInNewContext(read("../web/assets/i18n.js"), sandbox);
  return { i18n: sandbox.YataI18n, root: doc.documentElement };
}

test("the language is the saved choice, else the first supported one in the browser's list, else Japanese", () => {
  const cases = [
    [["en-US", "en"], "", "en"], [["fr-FR", "en-GB", "ja"], "", "en"], [["fr-FR", "de"], "", "ja"], [["ja-JP", "en-US"], "", "ja"],
    [["en-US"], "ja", "ja"], [["ja"], "en", "en"], [["en-US"], "xx", "en"], [null, "", "ja"],
  ];
  for (const [languages, locale, expected] of cases) {
    const { i18n, root } = i18nIn(languages, locale);
    assert.equal(i18n.locale, expected, JSON.stringify([languages, locale]));
    assert.equal(root.getAttribute("lang"), expected);
  }
  const { i18n } = i18nIn(["en"], "");
  assert.equal(i18n.choice, "", "no saved choice: the browser decides");
  assert.equal(i18n.t("sessions.topics", { count: 1 }), "1 topic");
  assert.equal(i18n.t("sessions.topics", { count: 3 }), "3 topics");
  assert.equal(i18n.t("no.such.key"), "no.such.key");
  assert.equal(i18nIn(["ja"], "").i18n.t("sessions.topics", { count: 3 }), "話題 3");
});

test("static markup is translated from its data-i18n keys", () => {
  const { i18n } = i18nIn(["en-US"], "");
  const text = { dataset: { i18n: "nav.newSession" }, textContent: "新しいセッション" };
  const attrs = { dataset: { i18nAttr: "aria-label:nav.list; data-tip:bar.share" }, values: {}, setAttribute(name, value) { this.values[name] = value; } };
  i18n.apply({ querySelectorAll: (selector) => (selector === "[data-i18n]" ? [text] : selector === "[data-i18n-attr]" ? [attrs] : []) });
  assert.equal(text.textContent, "New session");
  assert.deepEqual(attrs.values, { "aria-label": "Session list", "data-tip": "Share a tab" });
});

test("both catalogs have the same keys and placeholders, every key the page uses exists, and index.html matches ja", () => {
  const { i18n } = i18nIn(["ja"], "");
  const { ja, en } = i18n.CATALOG;
  assert.deepEqual(Object.keys(en).sort(), Object.keys(ja).sort());
  const names = (value) => [...new Set([...(typeof value === "object" ? value.one + value.other : value).matchAll(/\{(\w+)\}/g)].map((m) => m[1]))].sort();
  for (const key of Object.keys(ja)) assert.deepEqual(names(en[key]), names(ja[key]), key);
  const areas = new Set(Object.keys(ja).map((key) => key.split(".")[0]));
  for (const file of ["app.js", "prepare.js", "sessions.js", "topics.js", "hud.js", "captions.js", "capture.js", "gaps.js", "audio-diagnostics.js"]) {
    for (const [, key] of read("../web/assets/" + file).matchAll(/"([a-z]+\.[A-Za-z0-9_.]+)"/g)) {
      if (areas.has(key.split(".")[0]) && !key.endsWith(".")) assert.ok(key in ja, file + ": " + key);
    }
  }
  // Keys built from a name.
  ["ready", "unverified", "incomplete", "missing", "conflict", "unknown", "unset", "unreachable"].forEach((name) => assert.ok("prep.state." + name in ja));
  ["download", "verify", "load", "pull", "test", "unpack", "start"].forEach((name) => assert.ok("prep.phase." + name in ja));
  ["network", "disk", "corrupt", "conflict", "permission", "runtime", "ollama", "llm", "cancelled"].forEach((name) => assert.ok("prep.error." + name in ja));
  ["captions", "translation", "summaries"].forEach((name) => assert.ok("plan." + name in ja && "plan." + name + ".note" in ja));
  ["pull", "test", "start", "default"].forEach((name) => assert.ok("job.done." + name in ja));
  ["recognized", "empty", "error", "none"].forEach((name) => assert.ok("diag.result." + name in ja));
  const html = read("../web/index.html");
  let checked = 0;
  for (const [, key, text] of html.matchAll(/data-i18n="([^"]+)"[^>]*>([^<]*)</g)) {
    assert.equal(text, ja[key], "index.html text for " + key);
    checked++;
  }
  for (const [tag] of html.matchAll(/<[a-z][^>]*data-i18n-attr="[^"]*"[^>]*>/g)) {
    const pairs = tag.match(/data-i18n-attr="([^"]*)"/)[1].split(";").map((pair) => pair.split(":").map((part) => part.trim()));
    for (const [name, key] of pairs) {
      assert.ok(key in ja, key);
      assert.equal(tag.match(new RegExp("\\s" + name + '="([^"]*)"'))?.[1], ja[key], "index.html " + name + " for " + key);
      checked++;
    }
  }
  assert.ok(checked > 80, "the markup's text is tagged (" + checked + ")");
  for (const file of ["app.js", "prepare.js", "sessions.js", "topics.js", "hud.js", "capture.js", "gaps.js", "audio-diagnostics.js"]) {
    const code = read("../web/assets/" + file).split("\n").filter((line) => !/^\s*\/\//.test(line) && !/LANGUAGE_NAMES = /.test(line)).join("\n");
    assert.doesNotMatch(code, /"[^"\n]*[\u3040-\u30ff\u4e00-\u9fff][^"\n]*"/, file);
  }
});

test("an English browser gets the English interface: new session, session bar, sidebar and messages", async () => {
  const today = new Date().toISOString();
  const saved = { id: "s9", source_kind: "share", status: "ended", display_title: "Product launch", started_at_utc: today, updated_at_utc: today,
    duration_ms: 60_000, topic_count: 1, summaries: true };
  const h = harness({ languages: ["en-US", "en"], sessions: [saved] });
  await flush();
  assert.equal(h.sandbox.YataI18n.locale, "en");
  assert.equal(h.$("launch-title").textContent, "Share tab audio and start");
  assert.match(h.$("source-options").children[0].textContent, /^Tab audioShares a Chrome tab/);
  assert.equal(h.$("model-select").children.at(-1).textContent, "None (no summaries)");
  assert.equal(h.$("model-select").children[0].textContent, "qwen3.6:35b-a3b (Ollama)");
  assert.equal(h.$("sys-spk").dataset.tip, "Speaker labels: on");
  assert.match(h.$("session-list").textContent, /^TodayProduct launch\d\d:\d\d · 00:01:00 · 1 topic/);
  const remove = h.$("session-list").find((node) => node.classes.has("session-delete"));
  assert.equal(remove.getAttribute("aria-label"), "Delete (Product launch)");
  Object.assign(h.session, { caption_language: null, status: "created" });
  await h.launch();
  assert.match(h.$("active-title").textContent, /^Tab audio 11\/04 \d\d:\d\d$/);
  assert.match(h.$("active-meta").textContent, /^Started 11\/04 \d\d:\d\d · Tab audio$/);
  assert.equal(h.$("output-language").value, "en");
});

test("an English browser gets the English setup page", async () => {
  const setup = setupStatus({ recommended: { ...setupStatus().recommended, asr_reason: "(server text)", plan_reason: "(server plan)" } });
  const h = harness({ languages: ["en-GB"], replies: {
    "GET /api/v1/health": healthWith({ asr: { engine: "faster-whisper", weights_present: false, reason: "No folder is set." }, setup: { plan: null, attention: ["asr"], busy: false } }),
    "GET /api/v1/setup": () => ({ setup }),
  } });
  await flush();
  await flush();
  assert.equal(h.$("active-title").textContent, "Set up this PC");
  assert.equal(h.$("active-meta").textContent, "Check and get models");
  assert.match(h.$("prepare-machine").textContent, /Test CPU \(4 threads\)/);
  assert.match(h.$("prepare-machine").textContent, /None \(runs on the CPU\)/);
  const tiles = h.$("plan-options").children.map((tile) => tile.textContent);
  assert.match(tiles[0], /^Captions onlyRecommendedRecords captions/);
  const asr = h.$("prepare-list").children[0];
  assert.match(asr.textContent, /Captions \(speech recognition\)/);
  assert.match(asr.textContent, /\(server text\) Choose the model to use/);
  assert.match(asr.textContent, /whisper-smallRecommendedFor CPUsNot downloaded/);
  assert.equal(buttons(asr, /^Get \(464 MB\)$/).length, 1);
  assert.equal(buttons(asr, /^Get \(1\.5 GB\)$/).length, 1);
  assert.match(h.$("prepare-list").textContent, /Speech detection/);
  assert.match(h.$("prepare-summary").textContent, /^Still needed: the caption model$/);
  assert.match(h.$("storage-note").textContent, /^The default folder \(Yata's data folder\)\. Choose a folder/);
  assert.equal(h.$("open-prepare").dataset.tip, "Set up this PC: something is missing");
});

test("the display language is chosen in the settings: saved on the server, then the page reloads in it", async () => {
  let failing = false;
  const h = harness({ languages: ["en-US"], replies: {
    "PUT /api/v1/setup/locale": (init) => (failing ? { status: 500, detail: "disk full" } : { locale: { choice: JSON.parse(init.body).locale || null } }),
  } });
  await flush();
  h.click("open-settings");
  assert.ok(h.body.classes.has("mode-settings"));
  assert.equal(h.$("active-meta").textContent, "Display language");
  const select = h.$("ui-language");
  assert.deepEqual(select.children.map((option) => [option.value, option.textContent]),
    [["auto", "Automatic (follow the browser: English)"], ["ja", "日本語"], ["en", "English"]]);
  assert.equal(select.value, "auto", "no saved choice");
  await select.handlers.get("change")({ target: select });
  assert.equal(h.sent("PUT", "/api/v1/setup/locale").length, 0);
  select.value = "ja";
  await select.handlers.get("change")({ target: select });
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/locale")[0].init.body), { locale: "ja" });
  assert.equal(h.reloads(), 1);
  failing = true;
  select.disabled = false;
  select.value = "en";
  await select.handlers.get("change")({ target: select });
  assert.equal(h.reloads(), 1);
  assert.equal(select.value, "auto");
  assert.equal(select.disabled, false);
  assert.match(h.$("session-message").textContent, /Cannot save the display language: disk full/);
});

test("with English saved in the settings, a Japanese browser still gets English, and auto goes back to the browser", async () => {
  const h = harness({ languages: ["ja-JP", "ja"], locale: "en" });
  await flush();
  assert.equal(h.sandbox.YataI18n.locale, "en");
  h.click("open-settings");
  const select = h.$("ui-language");
  assert.equal(select.value, "en");
  assert.equal(select.children[0].textContent, "Automatic (follow the browser: 日本語)");
  select.value = "auto";
  await select.handlers.get("change")({ target: select });
  assert.deepEqual(JSON.parse(h.sent("PUT", "/api/v1/setup/locale")[0].init.body), { locale: "" });
});

test("Qwen3-ASR's manual steps keep the installed extras, and macOS is told it is not available there", async () => {
  for (const [os, expectCommand] of [["Windows", true], ["Darwin", false]]) {
    const setup = setupStatus({ plan: { summaries: false, translation: false, speakers: false } });
    setup.hardware = { ...setup.hardware, os };
    const h = harness({ replies: { "GET /api/v1/setup": () => ({ setup }) } });
    await flush();
    h.click("open-prepare");
    await flush();
    const extras = h.$("prepare-list").children.at(-1);
    assert.equal(/uv sync --inexact --extra qwen/.test(extras.textContent), expectCommand, os);
    if (!expectCommand) assert.match(extras.textContent, /macOS では使えません/);
    else assert.ok(extras.textContent.includes("<保存先>\\Qwen3-ASR-1.7B-hf"), "a Windows path on Windows");
  }
});
