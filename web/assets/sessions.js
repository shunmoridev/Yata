(() => {
  "use strict";
  const t = (key, params) => globalThis.YataI18n.t(key, params);
  // The soft theme labels a row's state in words; the HUD keeps its codes.
  const code = (word) => (word && globalThis.YataTheme?.get?.() !== "hud" ? t("code." + word) : word);
  const SOURCE_LABELS = { share: "sessions.source.share" };
  const STATES = {
    created: ["live", "LIVE"], starting_audio: ["live", "LIVE"], capturing_audio: ["live", "LIVE"], reconnecting: ["live", "RELINK"],
    draining: ["live", "FINAL"], ended: ["ended", ""], cancelled: ["cancelled", t("sessions.state.cancelled")], error: ["error", "ERROR"],
  };
  const WEEKDAYS = t("sessions.weekdays").split(",");
  const pad2 = (value) => String(value).padStart(2, "0");
  const finished = (status) => ["ended", "cancelled", "error"].includes(status);

  function parse(iso) {
    const date = new Date(iso);
    return Number.isFinite(date.getTime()) ? date : null;
  }

  function startedLabel(iso) {
    const date = parse(iso);
    if (!date) return "";
    return pad2(date.getMonth() + 1) + "/" + pad2(date.getDate()) + " " + pad2(date.getHours()) + ":" + pad2(date.getMinutes());
  }

  function timeLabel(iso) {
    const date = parse(iso);
    return date ? pad2(date.getHours()) + ":" + pad2(date.getMinutes()) : "";
  }

  function dayLabel(iso, now) {
    const date = parse(iso);
    if (!date) return "";
    const day = (value) => new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
    const diff = Math.round((day(now) - day(date)) / 86_400_000);
    if (diff === 0) return t("sessions.today");
    if (diff === 1) return t("sessions.yesterday");
    return t("sessions.day", { date: pad2(date.getMonth() + 1) + "/" + pad2(date.getDate()), weekday: WEEKDAYS[date.getDay()] });
  }

  function sourceLabel(kind) {
    return t(SOURCE_LABELS[kind] || "nav.sessions");
  }

  function placeholder(session) {
    return (sourceLabel(session.source_kind) + " " + startedLabel(session.started_at_utc)).trim();
  }

  function title(session) {
    return session.display_title || placeholder(session);
  }

  function el(doc, tag, className, text) {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function icon(doc, name) {
    const svg = doc.createElementNS("http://www.w3.org/2000/svg", "svg");
    const use = doc.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", "#" + name);
    svg.append(use);
    return svg;
  }

  function renderList(container, sessions, options) {
    const doc = container.ownerDocument;
    const opts = options || {};
    const now = opts.now || new Date();
    const query = String(opts.query || "").trim().toLowerCase();
    container.replaceChildren();
    const shown = query ? sessions.filter((session) => title(session).toLowerCase().includes(query)) : sessions;
    if (!shown.length) {
      container.append(el(doc, "p", "session-empty", t(query ? "sessions.noMatch" : "sessions.empty")));
      return;
    }
    let group = null;
    shown.forEach((session) => {
      const day = dayLabel(session.updated_at_utc || session.started_at_utc, now);
      if (day !== group) {
        group = day;
        container.append(el(doc, "h3", "session-group", day));
      }
      const [tone, known] = STATES[session.status] || ["ended", null];
    const label = known === null ? String(session.status || "").toUpperCase() : known === t("sessions.state.cancelled") ? known : code(known);
      const current = session.id === opts.currentId;
      const row = el(doc, "div", "session-row" + (current ? " active" : ""));
      row.dataset.state = tone;
      row.dataset.sessionId = session.id;

      const open = el(doc, "button", "session-open");
      open.disabled = Boolean(opts.locked) && !current;
      open.setAttribute?.("aria-current", String(current));
      if (open.disabled) open.dataset.tip = t("sessions.locked");
      const meta = el(doc, "small", "session-meta");
      if (label) meta.append(el(doc, "b", "session-state", label));
      meta.append(el(doc, "span", null, [
        timeLabel(session.started_at_utc),
        finished(session.status) && opts.formatTime ? opts.formatTime(session.duration_ms) : "",
        session.summaries === false ? t(session.translator ? "meta.captionsTranslation" : "meta.captionsOnly")
          : t("sessions.topics", { count: session.topic_count || 0 }),
      ].filter(Boolean).join(" · ")));
      open.append(el(doc, "i", "session-dot"), el(doc, "strong", "session-title", title(session)), meta);
      open.addEventListener("click", () => opts.onOpen && opts.onOpen(session.id));

      const remove = el(doc, "button", "icon-button session-delete");
      const armed = opts.armedId === session.id;
      const deletable = finished(session.status) && !(opts.locked && current);
      remove.disabled = !deletable;
      remove.dataset.armed = String(armed);
      const tip = t(armed ? "sessions.confirmDelete" : "sessions.delete");
      remove.dataset.tip = tip;
      remove.setAttribute?.("aria-label", t("sessions.deleteLabel", { action: tip, title: title(session) }));
      remove.append(icon(doc, armed ? "i-alert" : "i-trash"));
      remove.addEventListener("click", () => opts.onDelete && opts.onDelete(session.id));

      row.append(open, remove);
      container.append(row);
    });
  }

  globalThis.YataSessions = { renderList, placeholder, title, startedLabel, dayLabel, sourceLabel };
})();
