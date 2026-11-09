(() => {
  "use strict";
  const t = (key, params) => YataI18n.t(key, params);
  const KB = 1024;
  const MB = KB * 1024;
  const GB = MB * 1024;

  function bytes(value) {
    if (value == null || !Number.isFinite(value)) return t("common.unknown");
    if (value >= GB) return (value / GB).toFixed(1) + " GB";
    if (value >= MB) return Math.round(value / MB) + " MB";
    return Math.max(1, Math.round(value / KB)) + " KB";
  }

  const TONES = { ready: "ok", unverified: "warn", incomplete: "warn", missing: "off", conflict: "bad", unknown: "off", unset: "warn", unreachable: "bad" };
  const STATE = Object.fromEntries(Object.entries(TONES).map(([name, tone]) => [name, [t("prep.state." + name), tone]]));
  const PHASE = Object.fromEntries(["download", "verify", "load", "pull", "test", "unpack", "start"].map((name) => [name, t("prep.phase." + name)]));
  const ERROR = Object.fromEntries(["network", "disk", "corrupt", "conflict", "permission", "runtime", "ollama", "llm", "cancelled"]
    .map((name) => [name, t("prep.error." + name)]));
  const BACKEND = {
    builtin: ["i-chip", t("prep.backend.builtin")],
    ollama: ["i-brain", t("prep.backend.ollama")],
    openai: ["i-link", t("prep.backend.openai")],
    foundry: ["i-terminal", t("prep.backend.foundry")],
  };
  const labelled = (name) => h("b", { text: t("prep.label", { name }) });
  const withSize = (label, size) => t("prep.withSize", { label, size: bytes(size) });
  const PROPS = new Set(["href", "target", "rel", "type", "name", "checked", "disabled", "hidden", "value", "title"]);

  function h(tag, props, ...children) {
    const node = document.createElement(tag);
    Object.entries(props || {}).forEach(([key, value]) => {
      if (value == null || value === false) return;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "on") Object.entries(value).forEach(([type, handler]) => node.addEventListener(type, handler));
      else if (key === "data") Object.entries(value).forEach(([name, item]) => { node.dataset[name] = item; });
      else if (PROPS.has(key)) node[key] = value;
      else node.setAttribute(key, value);
    });
    children.flat().forEach((child) => { if (child != null && child !== false && child !== "") node.append(child); });
    return node;
  }

  function icon(name) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", "#" + name);
    svg.append(use);
    return svg;
  }

  const button = (label, handler, primary) => h("button", { type: "button", class: "action" + (primary ? " primary" : ""), on: { click: handler } }, label);
  const chip = (text, tone) => h("span", { class: "chip", data: { tone: tone || "" }, text });

  function stamp(iso) {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    const pad = (value) => String(value).padStart(2, "0");
    return pad(date.getMonth() + 1) + "/" + pad(date.getDate()) + " " + pad(date.getHours()) + ":" + pad(date.getMinutes());
  }

  const running = (job) => Boolean(job && job.state === "running");

  function stateOf(item) {
    if (running(item.job)) return [PHASE[item.job.phase] || t("prep.preparing"), "busy"];
    if (item.job && item.job.state === "failed") return [t("prep.failed"), "bad"];
    return STATE[item.state] || ["", "off"];
  }

  function jobView(job) {
    if (!job) return null;
    if (job.state === "running") {
      const counted = job.total > 0 && ["download", "verify", "pull"].includes(job.phase);
      const percent = counted ? Math.min(100, Math.floor((job.done / job.total) * 100)) : null;
      const bar = h("i");
      if (percent != null && bar.style) bar.style.width = percent + "%";
      const detail = [PHASE[job.phase] || t("prep.preparing"), percent != null ? percent + "%" : "",
        counted ? bytes(job.done) + " / " + bytes(job.total) : "", job.rate ? t("prep.perSecond", { size: bytes(job.rate) }) : "",
        job.phase === "pull" ? job.file : job.phase === "download" ? job.file : ""].filter(Boolean).join(" · ");
      return h("div", { class: "req-job" }, h("div", { class: "req-progress" + (percent == null ? " indeterminate" : "") }, bar),
        h("small", { class: "req-job-text", text: detail }));
    }
    if (job.state === "failed" || job.state === "cancelled") {
      const kind = job.error?.kind || "";
      return h("p", { class: "req-error", data: { kind } }, labelled(ERROR[kind] || t("prep.failed")), job.error?.message || "");
    }
    if (job.state === "done" && job.note) return h("p", { class: "req-note", text: job.note });
    return null;
  }

  function checkLine(check) {
    if (!check) return null;
    const parts = [];
    if (check.load_ms != null) parts.push(t("prep.check.load", { seconds: (check.load_ms / 1000).toFixed(1) }));
    if (check.test_ms != null) parts.push(t("prep.check.test", { seconds: (check.test_ms / 1000).toFixed(1) }));
    if (check.device) parts.push(String(check.device).toUpperCase() + " " + (check.compute_type || ""));
    if (check.sample) parts.push(t("prep.check.sample", { sample: check.sample }));
    if (check.at) parts.push(t("prep.check.at", { when: stamp(check.at) }));
    return h("small", { class: "req-check" }, icon("i-check"), parts.join(" · ") + (check.differs ? t("prep.check.differs") : ""));
  }

  // The conditions come from the server in Japanese only for now (legal wording is not translated yet).
  function termsBlock(model) {
    if (!model.terms?.length) return null;
    const link = (url, text) => h("a", { href: url, target: "_blank", rel: "noreferrer", text });
    return h("div", { class: "req-terms" },
      h("p", {}, labelled(t("prep.terms.label")), t("prep.terms.lead", { license: model.license })),
      h("ul", {}, model.terms.map((term) => h("li", {}, term.text, term.url ? [" ", link(term.url, term.link || term.url)] : null))),
      model.license_page ? link(model.license_page, t("prep.terms.full")) : null);
  }

  function modelRow(model, handlers, radio) {
    const [label, tone] = stateOf(model);
    const job = model.job;
    const title = h("div", { class: "model-title" }, h("strong", { text: model.label }),
      model.recommended ? chip(t("prep.recommended"), "accent") : null,
      model.target === "gpu" ? chip(t("prep.forGpu")) : model.target === "cpu" ? chip(t("prep.forCpu")) : null,
      chip(label, tone));
    const where = model.location === "hf-cache" ? t("prep.hfCache", { folder: model.folder }) : model.folder;
    const sizeText = (model.state === "incomplete" ? t("prep.remaining", { size: bytes(model.size), remaining: bytes(model.remaining) })
      : bytes(model.size)) + (model.need ? t("prep.needMemory", { size: bytes(model.need) }) : "");
    const facts = h("dl", { class: "req-facts" },
      h("dt", { text: t("prep.size") }), h("dd", { text: sizeText }),
      h("dt", { text: t("prep.folder") }), h("dd", { class: "path", text: where }),
      h("dt", { text: t("prep.source") }), h("dd", {}, model.page ? h("a", { href: model.page, target: "_blank", rel: "noreferrer", text: model.source }) : model.source, " · ",
        model.license_page ? h("a", { href: model.license_page, target: "_blank", rel: "noreferrer", text: model.license }) : model.license));
    const notes = [];
    if (model.state === "incomplete" && model.missing?.length && !running(job)) {
      notes.push(h("p", { class: "req-note", text: t("prep.missingFiles", { files: model.missing.join(t("prep.fileSeparator")) }) }));
    }
    if (model.state === "conflict") {
      notes.push(h("p", { class: "req-error", data: { kind: "conflict" } }, labelled(ERROR.conflict),
        t("prep.conflict", { files: (model.mismatched || []).join(t("prep.fileSeparator")) })));
    }
    const actions = h("div", { class: "req-actions" });
    if (running(job)) {
      actions.append(button(t("prep.cancel"), () => handlers.cancel(model.component)));
    } else if ((model.state === "missing" || model.state === "incomplete") && model.location !== "hf-cache" && model.location !== "custom") {
      const resume = model.partial_bytes > 0 || (job && job.kind === "download" && job.state !== "done");
      const remaining = Math.max(0, (model.state === "incomplete" ? model.remaining : model.size) - (model.partial_bytes || 0));
      actions.append(button(withSize(t(resume ? "prep.resume" : "prep.get"), remaining), () => handlers.download(model.id), true));
    } else if (model.state === "unverified") {
      actions.append(button(t(job && job.state === "failed" ? "prep.checkAgain" : "prep.check"), () => handlers.verify(model.id), true));
    } else if (model.state === "ready") {
      actions.append(button(t("prep.checkAgain"), () => handlers.verify(model.id)));
    }
    const head = radio
      ? h("label", { class: "model-pick" }, h("input", { type: "radio", name: radio.name, checked: radio.checked, disabled: radio.disabled,
          on: { change: radio.onPick } }), title)
      : title;
    return h("div", { class: "model-row", data: { state: model.state, active: radio && radio.checked ? "true" : "false" } },
      head, h("p", { class: "model-summary", text: model.summary }), facts, termsBlock(model), notes, jobView(job),
      model.state === "ready" ? checkLine(model.check) : null, actions.children.length ? actions : null);
  }

  function card(key, iconName, title, [label, tone], ...body) {
    return h("article", { class: "req", data: { req: key, tone } },
      h("header", { class: "req-head" }, icon(iconName), h("h3", { text: title }), chip(label, tone)), body);
  }

  function asrCard(status, handlers) {
    const asr = status.asr;
    const active = asr.models.find((model) => model.active) || null;
    const engine = asr.active || {};
    const header = engine.weights_present && active && active.state === "ready" ? STATE.ready
      : engine.weights_present ? [t("prep.usableUnverified"), "warn"] : [t("prep.noModel"), "bad"];
    const lead = asr.locked
      ? t("prep.asr.locked", { env: asr.locked, engine: engine.description || engine.engine || "" })
      : t("prep.asr.lead", { reason: status.recommended.asr_reason });
    const rows = asr.models.map((model) => modelRow({ ...model, component: "asr" }, handlers, {
      name: "asr-model", checked: Boolean(model.active), disabled: Boolean(asr.locked) || running(model.job),
      onPick: () => handlers.chooseAsr(model.id),
    }));
    return card("asr", "i-wave", t("prep.asr.title"), running(active?.job) ? stateOf(active) : header,
      h("p", { class: "req-lead", text: lead }),
      engine.description ? h("small", { class: "req-engine", text: t("prep.asr.current", { description: engine.description }) }) : null,
      h("div", { class: "model-list" }, rows));
  }

  function vadCard(status) {
    const vad = status.vad;
    return card("vad", "i-chip", t("prep.vad.title"), vad.state === "ready" ? STATE.ready : [t("prep.notFound"), "bad"],
      h("p", { class: "req-lead", text: t("prep.vad.lead", { label: vad.label, detail: vad.detail }) }));
  }

  function ollamaCard(status, handlers) {
    const ollama = status.llm.ollama;
    const tone = { running: [t("prep.ollama.running"), "ok"], stopped: [t("prep.ollama.stopped"), "warn"], absent: [t("prep.ollama.absent"), "bad"] }[ollama.state]
      || STATE.unknown;
    const text = ollama.state === "running" ? t("prep.ollama.runningText", { version: ollama.version || "" }).replace("  ", " ")
      : t(ollama.state === "stopped" ? "prep.ollama.stoppedText" : "prep.ollama.absentText");
    return card("ollama", "i-brain", t("prep.ollama.title"), tone, h("p", { class: "req-lead", text }),
      h("div", { class: "req-actions" },
        ollama.state !== "running" ? h("a", { class: "action primary", href: ollama.install_url, target: "_blank", rel: "noreferrer",
          text: t(ollama.state === "absent" ? "prep.ollama.openSource" : "prep.ollama.source") }) : null,
        button(t("prep.recheck"), handlers.refresh)));
  }

  function llmCard(status, handlers) {
    const llm = status.llm;
    const up = llm.ollama.state === "running";
    const job = status.jobs?.llm;
    const ready = llm.models.filter((model) => model.installed);
    const header = running(job) ? [PHASE[job.phase] || t("prep.preparing"), "busy"] : !up ? [t("prep.ollama.needed"), "off"]
      : ready.some((model) => model.state === "ready") ? STATE.ready : ready.length ? [t("prep.usableUnverified"), "warn"] : [t("prep.noModel"), "bad"];
    const rows = llm.models.map((model) => {
      const [label, tone] = stateOf(model);
      const actions = h("div", { class: "req-actions" });
      if (running(model.job)) actions.append(button(t("prep.cancel"), () => handlers.cancel("llm")));
      else if (up && !model.installed) actions.append(button(withSize(t(model.job ? "prep.resume" : "prep.fetch"), model.size), () => handlers.pull(model.name), model.recommended));
      else if (up && model.state === "unverified") actions.append(button(t("prep.check"), () => handlers.testLlm(model.name), true));
      else if (up && model.state === "ready") actions.append(button(t("prep.checkAgain"), () => handlers.testLlm(model.name)));
      return h("div", { class: "model-row", data: { state: model.state } },
        h("div", { class: "model-title" }, h("strong", { text: model.name }), model.recommended ? chip(t("prep.recommended"), "accent") : null, chip(label, tone)),
        h("p", { class: "model-summary", text: model.summary }),
        h("dl", { class: "req-facts" }, h("dt", { text: t("prep.size") }),
          h("dd", { text: t("prep.llm.size", { download: bytes(model.size), memory: bytes(model.need) }) })),
        jobView(model.job), model.state === "ready" ? checkLine(model.check) : null, actions.children.length ? actions : null);
    });
    return card("llm", "i-sigma", t("prep.llm.title"), header,
      h("p", { class: "req-lead", text: t("prep.llm.lead", { reason: status.recommended.llm_reason || "" }).trim() }),
      h("small", { class: "req-engine", text: t("prep.llm.store", { path: llm.ollama.store, free: bytes(llm.ollama.free) }) }),
      h("div", { class: "model-list" }, rows));
  }

  function backendOf(status, draft) {
    return draft || status.llm.backend || status.llm.recommended;
  }

  function backendCard(status, shown, handlers) {
    const llm = status.llm;
    const usable = llm.options.find((option) => option.id === shown)?.usable;
    const lead = llm.locked ? t("prep.backend.locked", { env: llm.locked })
      : (llm.chosen ? "" : t("prep.backend.unchosen")) + t("prep.backend.lead");
    const tiles = llm.options.map((option) => {
      const checked = option.id === shown;
      return h("button", { type: "button", class: "source-option", role: "radio", "aria-checked": String(checked), data: { backend: option.id },
        disabled: Boolean(llm.locked) && !checked, on: { click: () => { if (!checked) handlers.chooseLlm(option.id); } } },
      icon(BACKEND[option.id][0]),
      h("strong", {}, option.label, option.id === llm.recommended ? chip(t("prep.recommended"), "accent") : null, option.usable ? chip(t("prep.usable"), "ok") : null),
      h("small", { text: BACKEND[option.id][1] }));
    });
    return card("llm-backend", "i-brain", t("settings.llm"), usable ? [t("prep.usable"), "ok"] : [t("prep.needsSetup"), "warn"],
      h("p", { class: "req-lead", text: lead }),
      h("div", { class: "source-options backend-options", role: "radiogroup", "aria-label": t("settings.llm") }, tiles));
  }

  function deviceText(builtin) {
    if (builtin.device === "none") return t("prep.cpuOnly");
    const found = (builtin.devices || []).find((item) => item.id === builtin.device);
    return found ? t("prep.device", { id: found.id, name: found.name }) : builtin.device;
  }

  function builtinCard(status, handlers) {
    const builtin = status.llm.builtin;
    const busy = [builtin.runtime?.job, ...builtin.models.map((model) => model.job)].find(running);
    const header = busy ? [PHASE[busy.phase] || t("prep.preparing"), "busy"]
      : builtin.executable && builtin.models.some((model) => model.state === "ready") ? STATE.ready
        : builtin.executable && builtin.models.some((model) => model.usable) ? [t("prep.usableUnverified"), "warn"] : [t("prep.needsSetup"), "warn"];
    const rows = (builtin.runtime ? [modelRow({ ...builtin.runtime, component: "llm-runtime" }, handlers)] : [])
      .concat(builtin.models.map((model) => modelRow({ ...model, component: "llm" }, handlers)));
    return card("builtin", "i-chip", t("prep.builtin.title"), header,
      h("p", { class: "req-lead", text: t("prep.builtin.lead") }),
      builtin.env ? h("p", { class: "req-lead", text: t(builtin.executable ? "prep.llamaEnv" : "prep.llamaEnvMissing", { env: builtin.env }) }) : null,
      // macOS and Linux: the page fetches no llama.cpp there, so the server says how to install it.
      builtin.install ? h("p", { class: "req-lead", text: builtin.install }) : null,
      builtin.device ? h("small", { class: "req-engine", text: t("prep.deviceLabel", { device: deviceText(builtin) }) }) : null,
      h("div", { class: "model-list" }, rows));
  }

  // What the user typed into the OpenAI-compatible form, kept across redraws until it is saved.
  const openaiDraft = {};

  function openaiCard(status, handlers) {
    const openai = status.llm.openai;
    const job = openai.job;
    const tone = { ready: STATE.ready, unverified: [t("prep.usableUnverified"), "warn"], missing: [t("prep.noModel"), "bad"] }[openai.state]
      || STATE[openai.state] || STATE.unknown;
    const locked = (openai.locked || []).length > 0;
    const field = (key, label, placeholder, extra) => {
      const input = h("input", { type: "text", value: openaiDraft[key] ?? openai[key] ?? "", placeholder, spellcheck: "false", autocomplete: "off",
        disabled: locked, "aria-label": label, on: { input: (event) => { openaiDraft[key] = event.target.value; } }, ...extra });
      return h("label", { class: "llm-field" }, h("span", { text: label }), input);
    };
    const save = () => {
      const values = { base_url: openaiDraft.base_url ?? openai.base_url ?? "", model: openaiDraft.model ?? openai.model ?? "" };
      delete openaiDraft.base_url;
      delete openaiDraft.model;
      return handlers.saveOpenai(values);
    };
    const actions = h("div", { class: "req-actions" });
    if (running(job)) actions.append(button(t("prep.cancel"), () => handlers.cancel("llm")));
    else {
      if (!locked) actions.append(button(t("prep.save"), save, openai.state === "unset" || openai.state === "unreachable"));
      if (openai.state === "unverified" || openai.state === "ready") {
        actions.append(button(t(openai.state === "ready" ? "prep.checkAgain" : "prep.check"), () => handlers.testLlm(openai.model, "openai"), openai.state === "unverified"));
      }
      actions.append(button(t("prep.recheck"), handlers.refresh));
    }
    return card("openai", "i-link", t("prep.openai.title"), running(job) ? [PHASE[job.phase] || t("prep.preparing"), "busy"] : tone,
      h("p", { class: "req-lead", text: t("prep.openai.lead") }),
      locked ? h("p", { class: "req-lead", text: t("prep.openai.locked", { env: openai.locked.join(t("common.listSeparator")) }) }) : null,
      h("div", { class: "llm-form" },
        field("base_url", t("prep.openai.url"), "http://127.0.0.1:1234/v1"),
        field("model", t("prep.openai.model"), t("prep.openai.modelPlaceholder"), { list: "openai-models" }),
        h("datalist", { id: "openai-models" }, (openai.server_models || []).map((name) => h("option", { value: name })))),
      h("small", { class: "req-engine", text: t(openai.key ? "prep.openai.keySet" : "prep.openai.keyUnset", { env: openai.key_env }) }),
      (openai.server_models || []).length ? h("small", { class: "req-engine", text: t("prep.openai.models", { list: openai.server_models.slice(0, 4).join(t("prep.fileSeparator")) })
        + (openai.server_models.length > 4 ? t("prep.openai.more", { count: openai.server_models.length - 4 }) : "") }) : null,
      openai.reason ? h("p", { class: "req-error", data: { kind: "llm" } }, labelled(t("prep.openai.connection")), openai.reason) : null,
      jobView(job), openai.state === "ready" ? checkLine(openai.check) : null, actions);
  }

  function foundryCard(status, handlers) {
    const foundry = status.llm.foundry;
    const up = foundry.service?.state === "running";
    const busy = [foundry.job, ...foundry.models.map((model) => model.job)].find(running);
    const header = busy ? [PHASE[busy.phase] || t("prep.preparing"), "busy"] : !up ? [t("prep.foundry.stopped"), "warn"]
      : foundry.models.some((model) => model.state === "ready") ? STATE.ready
        : foundry.models.some((model) => model.installed) ? [t("prep.usableUnverified"), "warn"] : [t("prep.noModel"), "bad"];
    const rows = foundry.models.map((model) => {
      const [label, tone] = stateOf(model);
      const actions = h("div", { class: "req-actions" });
      if (running(model.job)) actions.append(button(t("prep.cancel"), () => handlers.cancel("llm")));
      else if (up && !model.installed) actions.append(button(t(model.job ? "prep.fetchAgain" : "prep.fetch"), () => handlers.foundryDownload(model.name), model.recommended));
      else if (up && model.state === "unverified") actions.append(button(t("prep.check"), () => handlers.testLlm(model.name, "foundry"), true));
      else if (up && model.state === "ready") actions.append(button(t("prep.checkAgain"), () => handlers.testLlm(model.name, "foundry")));
      return h("div", { class: "model-row", data: { state: model.state } },
        h("div", { class: "model-title" }, h("strong", { text: model.name }), model.recommended ? chip(t("prep.recommended"), "accent") : null, chip(label, tone)),
        h("p", { class: "model-summary", text: model.summary }),
        model.variant ? h("dl", { class: "req-facts" }, h("dt", { text: t("prep.foundry.variant") }), h("dd", { class: "path", text: model.variant })) : null,
        jobView(model.job), model.state === "ready" ? checkLine(model.check) : null, actions.children.length ? actions : null);
    });
    const lead = up ? t("prep.foundry.running")
      : (foundry.service?.reason || t("prep.foundry.notRunning")) + t("prep.foundry.startHint");
    return card("foundry", "i-terminal", "Foundry Local", header, h("p", { class: "req-lead", text: lead }), jobView(foundry.job),
      h("div", { class: "req-actions" }, !up && !running(foundry.job) ? button(t("prep.foundry.start"), handlers.foundryStart, true) : null,
        button(t("prep.recheck"), handlers.refresh)),
      h("div", { class: "model-list" }, rows));
  }

  function llmCards(status, handlers, draft) {
    const shown = backendOf(status, draft);
    const cards = [backendCard(status, shown, handlers)];
    if (shown === "builtin") cards.push(builtinCard(status, handlers));
    else if (shown === "ollama") cards.push(ollamaCard(status, handlers), llmCard(status, handlers));
    else if (shown === "openai") cards.push(openaiCard(status, handlers));
    else if (shown === "foundry") cards.push(foundryCard(status, handlers));
    return cards;
  }

  function engineOf(status, draft) {
    return draft || status.translator?.engine || "llm";
  }

  function translationCard(status, shown, handlers) {
    const translator = status.translator;
    const lead = (translator.engine_chosen ? "" : t("prep.translation.unchosen")) + t("prep.translation.lead");
    const tiles = [["translator", "i-translate"], ["llm", "i-sigma"]].map(([engine, iconName]) => {
      const checked = engine === shown;
      return h("button", { type: "button", class: "source-option", role: "radio", "aria-checked": String(checked), data: { engine },
        on: { click: () => { if (!checked) handlers.chooseTranslation({ engine }); } } },
      icon(iconName),
      h("strong", {}, t("prep.translation." + engine), engine === translator.engine_recommended ? chip(t("prep.recommended"), "accent") : null,
        engine === "translator" && translator.available ? chip(t("prep.usable"), "ok") : null),
      h("small", { text: t("prep.translation." + engine + ".note") }));
    });
    const usable = shown === "llm" || translator.available;
    return card("translation", "i-translate", t("prep.translation.title"), usable ? [t("prep.usable"), "ok"] : [t("prep.needsSetup"), "warn"],
      h("p", { class: "req-lead", text: lead }),
      h("div", { class: "source-options backend-options", role: "radiogroup", "aria-label": t("prep.translation.title") }, tiles));
  }

  // The GGUF path typed into the translator card, kept across redraws until it is saved.
  const translatorDraft = {};

  function customTranslatorForm(translator, handlers) {
    const saved = translator.custom_path || "";
    const input = h("input", { type: "text", value: translatorDraft.path ?? saved, placeholder: t("prep.translator.customPlaceholder"),
      spellcheck: "false", autocomplete: "off", "aria-label": t("prep.translator.customPath"),
      on: { input: (event) => { translatorDraft.path = event.target.value; } } });
    const save = (path) => {
      delete translatorDraft.path;
      return handlers.saveTranslatorPath(path);
    };
    const actions = h("div", { class: "req-actions" }, button(t("prep.save"), () => save(translatorDraft.path ?? saved), !saved),
      saved ? button(t("prep.translator.customClear"), () => save("")) : null);
    return h("div", { class: "custom-model" },
      h("label", { class: "llm-field" }, h("span", { text: t("prep.translator.customPath") }), input),
      h("small", { class: "req-engine", text: t("prep.translator.customNote") }), actions);
  }

  function translatorCard(status, handlers, role, runtimeShown) {
    const translator = status.translator;
    const models = translator.models || [translator.model];
    const active = models.find((model) => model.active) || translator.model;
    const busy = [translator.runtime?.job, ...models.map((model) => model.job)].find(running);
    const header = busy ? [PHASE[busy.phase] || t("prep.preparing"), "busy"]
      : translator.available && active.state === "ready" ? STATE.ready
        : translator.available ? [t("prep.usableUnverified"), "warn"] : role === "optional" ? [t("prep.optional"), "off"] : [t("prep.needsSetup"), "warn"];
    const lead = t({ plan: "prep.translator.lead", summaries: "prep.translator.summariesLead", optional: "prep.translator.optionalLead" }[role])
      + t("prep.translator.sizes");
    const rows = (translator.runtime && !runtimeShown ? [modelRow({ ...translator.runtime, component: "llm-runtime" }, handlers)] : [])
      .concat(models.map((model) => modelRow({ ...model, component: "translator" }, handlers, {
        name: "translator-model", checked: Boolean(model.active), disabled: running(model.job),
        onPick: () => handlers.chooseTranslation({ model: model.id }),
      })));
    const crowded = role === "summaries" && status.hardware?.gpu?.cuda && active.fits_with_summaries === false;
    // The model is here but not the fetched runtime that runs it: the size's own row reads as done, so say so.
    const runtimeMissing = Boolean(translator.runtime) && !translator.executable && ["ready", "unverified"].includes(active.state);
    // The reason says how it runs here without a summary LLM, so it belongs to the use "captions + translation".
    return card("translator", "i-translate", t("prep.translator.title"), header,
      h("p", { class: "req-lead", text: (role === "plan" && translator.reason ? translator.reason + " " : "") + lead }),
      crowded ? h("p", { class: "req-note", text: t("prep.translator.crowded") }) : null,
      runtimeMissing ? h("p", { class: "req-note", text: t("prep.translator.runtimeMissing") }) : null,
      translator.env ? h("p", { class: "req-lead", text: t(translator.executable ? "prep.llamaEnv" : "prep.llamaEnvMissing", { env: translator.env }) }) : null,
      translator.install && !runtimeShown ? h("p", { class: "req-lead", text: translator.install }) : null,
      translator.device ? h("small", { class: "req-engine", text: t("prep.deviceLabel", { device: deviceText(translator) }) }) : null,
      h("div", { class: "model-list" }, rows), customTranslatorForm(translator, handlers));
  }

  function speakersCard(status, handlers) {
    const speakers = status.speakers;
    const header = speakers.enabled ? STATE.ready : running(speakers.model?.job) ? stateOf(speakers.model) : [t("prep.needsSetup"), "warn"];
    const packageLine = speakers.package
      ? h("p", { class: "req-lead" }, labelled("sherpa-onnx"), t("prep.speakers.installed"))
      : h("p", { class: "req-lead" }, labelled("sherpa-onnx"), t("prep.speakers.needBefore"),
        h("code", { text: speakers.command }), t("prep.speakers.needAfter"));
    return card("speakers", "i-user", t("sys.speakers"), header, packageLine,
      speakers.env ? h("p", { class: "req-lead", text: t("prep.speakers.env") }) : null,
      speakers.model ? h("div", { class: "model-list" }, modelRow({ ...speakers.model, component: "speakers" }, handlers)) : null);
  }

  // Qwen3-ASR's packages pin NVIDIA's CUDA build of torch (Windows and Linux only), so macOS gets a note instead.
  function extrasCard(status) {
    const os = status.hardware?.os;
    const body = os === "Darwin"
      ? h("p", { class: "req-lead" }, labelled("Qwen3-ASR"), t("prep.extras.qwenMac"))
      : h("p", { class: "req-lead" }, labelled("Qwen3-ASR"),
        t("prep.extras.qwen1"), h("code", { text: "uv sync --inexact --extra qwen" }), t("prep.extras.qwen2"),
        h("code", { text: "hf download Qwen/Qwen3-ASR-1.7B-hf --local-dir " + t("prep.extras.folder") + (os === "Windows" ? "\\" : "/") + "Qwen3-ASR-1.7B-hf" }),
        t("prep.extras.qwen3"));
    return h("details", { class: "req req-extra" }, h("summary", {}, icon("i-chevron"), h("span", { text: t("prep.extras.title") })), body);
  }

  function readiness(status, plan, drafts) {
    const captions = Boolean(status.asr.active?.weights_present);
    const backend = backendOf(status, drafts?.llm);
    const summaries = Boolean(status.llm.options.find((option) => option.id === backend)?.usable);
    const speakers = Boolean(status.speakers.enabled);
    const translation = Boolean(status.translator?.available);
    const translates = plan.summaries ? Boolean(status.translator) && engineOf(status, drafts?.translation) === "translator" : plan.translation;
    const missing = [];
    if (!captions) missing.push(t("prep.missing.captions"));
    if (plan.summaries && !summaries) missing.push(t("prep.missing.summaries"));
    if (translates && !translation) missing.push(t("prep.missing.translation"));
    if (plan.speakers && !speakers) missing.push(t("sys.speakers"));
    return { captions, summaries, translation, speakers, missing };
  }

  function planOf(status) {
    const plan = status.plan || { summaries: status.recommended.plan === "summaries", speakers: false };
    return { summaries: Boolean(plan.summaries), translation: !plan.summaries && Boolean(plan.translation), speakers: Boolean(plan.speakers) };
  }

  const planKey = (plan) => (plan.summaries ? "summaries" : plan.translation ? "translation" : "captions");

  function machine(status) {
    const hw = status.hardware;
    const gpu = hw.gpu;
    const items = [
      ["CPU", t("prep.machine.cpu", { name: hw.cpu?.name || "", count: hw.cpu?.logical || "?" })],
      [t("prep.machine.memory"), t("prep.machine.free", { total: bytes(hw.memory?.total), free: bytes(hw.memory?.available) })],
      ["GPU", gpu ? gpu.name + (gpu.vram ? " · VRAM " + bytes(gpu.vram) : "") + (gpu.cuda ? "" : t("prep.machine.noCuda")) : t("prep.machine.noGpu")],
      [t("prep.machine.storageFree"), bytes(status.storage?.free)],
    ];
    return items.map(([name, value]) => h("div", { class: "machine-item" }, h("span", { text: name }), h("strong", { text: value })));
  }

  function planTiles(status, plan, onPick) {
    const recommended = status.recommended.plan;
    const translatorSuits = Boolean(status.recommended.translator) && recommended !== "summaries";
    return [
      ["captions", "i-wave"], ["translation", "i-translate"], ["summaries", "i-sigma"],
    ].filter(([key]) => key !== "translation" || status.translator).map(([key, iconName]) => {
      const [label, note] = [t("plan." + key), t("plan." + key + ".note")];
      const checked = key === planKey(plan);
      const tile = h("button", { type: "button", class: "source-option", role: "radio", "aria-checked": String(checked), data: { plan: key },
        on: { click: () => onPick({ summaries: key === "summaries", translation: key === "translation" }) } }, icon(iconName),
      h("strong", {}, label, key === recommended ? chip(t("prep.recommended"), "accent") : key === "translation" && translatorSuits ? chip(t("plan.forTranslation"), "accent") : null),
      h("small", { text: note }));
      return tile;
    });
  }

  function requirements(status, plan, handlers, drafts) {
    const cards = [asrCard(status, handlers), vadCard(status)];
    if (plan.summaries) cards.push(...llmCards(status, handlers, drafts?.llm));
    const translator = status.translator;
    if (translator && !plan.summaries && plan.translation) cards.push(translatorCard(status, handlers, "plan", false));
    // While the summary LLM translates, the translator stays on offer: a session started without summaries can still
    // translate with it.
    if (translator && plan.summaries) {
      const engine = engineOf(status, drafts?.translation);
      cards.push(translationCard(status, engine, handlers),
        translatorCard(status, handlers, engine === "translator" ? "summaries" : "optional", backendOf(status, drafts?.llm) === "builtin"));
    }
    if (plan.speakers) cards.push(speakersCard(status, handlers));
    cards.push(extrasCard(status));
    return cards;
  }

  globalThis.YataPrepare = { backendOf, bytes, engineOf, h, llmCards, machine, planKey, planOf, planTiles, readiness, requirements, STATE, PHASE, ERROR };
})();
