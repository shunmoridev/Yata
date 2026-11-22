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

  const DRAG_TYPE = "application/x-yata-session";

  // The sidebar: one folder per project with its sessions, then the unassigned sessions grouped by day.
  // Without projects it is the day-grouped list alone.
  function renderList(container, sessions, options) {
    const doc = container.ownerDocument;
    const opts = options || {};
    const now = opts.now || new Date();
    const query = String(opts.query || "").trim().toLowerCase();
    const projects = opts.projects || [];
    const known = new Set(projects.map((project) => project.id));
    const collapsed = opts.collapsed || new Set();
    container.replaceChildren();
    const shown = query ? sessions.filter((session) => title(session).toLowerCase().includes(query)) : sessions;
    const inProject = (id) => shown.filter((session) => session.project_id === id);
    const unassigned = shown.filter((session) => !known.has(session.project_id));
    const visibleProjects = query ? projects.filter((project) => inProject(project.id).length) : projects;
    if (!shown.length && !visibleProjects.length) {
      container.append(el(doc, "p", "session-empty", t(query ? "sessions.noMatch" : "sessions.empty")));
      return;
    }
    let focus = null;
    visibleProjects.forEach((project) => {
      const rows = inProject(project.id);
      // A search opens every folder it found something in.
      const open = Boolean(query) || !collapsed.has(project.id);
      const folder = el(doc, "section", "project" + (open ? " open" : ""));
      folder.dataset.projectId = project.id;
      const head = projectHead(doc, project, rows.length, open, opts);
      if (head.input) focus = head.input;
      folder.append(head.node);
      if (open) {
        const list = el(doc, "div", "project-sessions");
        if (!rows.length) list.append(el(doc, "p", "project-empty", t("projects.empty")));
        rows.forEach((session) => list.append(...sessionRow(doc, session, opts, startedLabel(session.started_at_utc))));
        folder.append(list);
      }
      container.append(folder);
    });
    if (unassigned.length && projects.length) {
      const head = el(doc, "h3", "session-group unassigned", t("projects.unassigned"));
      dropTarget(head, null, opts);
      container.append(head);
    }
    let group = null;
    unassigned.forEach((session) => {
      const day = dayLabel(session.updated_at_utc || session.started_at_utc, now);
      if (day !== group) {
        group = day;
        const heading = el(doc, "h3", "session-group", day);
        if (projects.length) dropTarget(heading, null, opts);
        container.append(heading);
      }
      container.append(...sessionRow(doc, session, opts, timeLabel(session.started_at_utc)));
    });
    focus?.focus?.();
    focus?.select?.();
  }

  // Dropping a dragged session on a project's head (or on an unassigned heading) files it there.
  function dropTarget(node, projectId, opts) {
    const dragged = (event) => Array.from(event.dataTransfer?.types || []).includes(DRAG_TYPE);
    node.addEventListener("dragover", (event) => {
      if (!dragged(event)) return;
      event.preventDefault();
      node.classList.add("drop");
    });
    node.addEventListener("dragleave", () => node.classList.remove("drop"));
    node.addEventListener("drop", (event) => {
      node.classList.remove("drop");
      const id = event.dataTransfer?.getData(DRAG_TYPE);
      if (!id) return;
      event.preventDefault();
      if (opts.onMove) opts.onMove(id, projectId);
    });
  }

  function projectHead(doc, project, count, open, opts) {
    const head = el(doc, "div", "project-head");
    dropTarget(head, project.id, opts);
    if (opts.editingId === project.id) {
      const input = el(doc, "input", "project-name-input");
      input.value = project.name;
      input.maxLength = 80;
      input.setAttribute?.("aria-label", t("projects.nameLabel"));
      let done = false;
      const finish = (value) => {
        if (done) return;
        done = true;
        if (opts.onRename) opts.onRename(project.id, value);
      };
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") { event.preventDefault?.(); finish(input.value); }
        else if (event.key === "Escape") { event.preventDefault?.(); finish(null); }
      });
      input.addEventListener("blur", () => finish(input.value));
      head.append(icon(doc, "i-folder"), input);
      return { node: head, input };
    }
    const toggle = el(doc, "button", "project-toggle");
    toggle.setAttribute?.("aria-expanded", String(open));
    toggle.append(icon(doc, "i-chevron"), icon(doc, "i-folder"), el(doc, "strong", "project-name", project.name), el(doc, "b", "count", String(count)));
    toggle.addEventListener("click", () => opts.onToggle && opts.onToggle(project.id));
    toggle.addEventListener("dblclick", () => opts.onStartRename && opts.onStartRename(project.id));

    const rename = el(doc, "button", "icon-button project-action project-rename");
    rename.dataset.tip = t("projects.rename");
    rename.setAttribute?.("aria-label", t("projects.renameLabel", { name: project.name }));
    rename.append(icon(doc, "i-pencil"));
    rename.addEventListener("click", () => opts.onStartRename && opts.onStartRename(project.id));

    const remove = el(doc, "button", "icon-button project-action project-delete");
    const armed = opts.armedId === "project:" + project.id;
    remove.dataset.armed = String(armed);
    const tip = t(armed ? "projects.confirmDelete" : "projects.delete");
    remove.dataset.tip = tip;
    remove.setAttribute?.("aria-label", t("sessions.deleteLabel", { action: tip, title: project.name }));
    remove.append(icon(doc, armed ? "i-alert" : "i-trash"));
    remove.addEventListener("click", () => opts.onDeleteProject && opts.onDeleteProject(project.id));
    head.append(toggle, rename, remove);
    return { node: head, input: null };
  }

  function sessionRow(doc, session, opts, when) {
    const [tone, known] = STATES[session.status] || ["ended", null];
    const label = known === null ? String(session.status || "").toUpperCase() : known === t("sessions.state.cancelled") ? known : code(known);
    const current = session.id === opts.currentId;
    const projects = opts.projects || [];
    const row = el(doc, "div", "session-row" + (current ? " active" : ""));
    row.dataset.state = tone;
    row.dataset.sessionId = session.id;
    if (projects.length) {
      row.draggable = true;
      row.addEventListener("dragstart", (event) => {
        event.dataTransfer?.setData(DRAG_TYPE, session.id);
        if (event.dataTransfer) event.dataTransfer.effectAllowed = "move";
      });
    }

    const open = el(doc, "button", "session-open");
    open.disabled = Boolean(opts.locked) && !current;
    open.setAttribute?.("aria-current", String(current));
    if (open.disabled) open.dataset.tip = t("sessions.locked");
    const meta = el(doc, "small", "session-meta");
    if (label) meta.append(el(doc, "b", "session-state", label));
    meta.append(el(doc, "span", null, [
      when,
      finished(session.status) && opts.formatTime ? opts.formatTime(session.duration_ms) : "",
      session.summaries === false ? t(session.translator ? "meta.captionsTranslation" : "meta.captionsOnly")
        : t("sessions.topics", { count: session.topic_count || 0 }),
    ].filter(Boolean).join(" · ")));
    open.append(el(doc, "i", "session-dot"), el(doc, "strong", "session-title", title(session)), meta);
    open.addEventListener("click", () => opts.onOpen && opts.onOpen(session.id));
    row.append(open);

    const menuOpen = opts.menuId === session.id;
    if (projects.length) {
      const move = el(doc, "button", "icon-button session-move");
      move.dataset.tip = t("projects.move");
      move.dataset.open = String(menuOpen);
      move.setAttribute?.("aria-label", t("projects.moveLabel", { title: title(session) }));
      move.setAttribute?.("aria-haspopup", "menu");
      move.setAttribute?.("aria-expanded", String(menuOpen));
      move.append(icon(doc, "i-folder"));
      move.addEventListener("click", () => opts.onMenu && opts.onMenu(menuOpen ? null : session.id));
      row.append(move);
    }

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
    row.append(remove);
    if (!menuOpen) return [row];

    const filed = new Set(projects.map((project) => project.id));
    const menu = el(doc, "div", "move-menu");
    menu.setAttribute?.("role", "menu");
    menu.setAttribute?.("aria-label", t("projects.moveLabel", { title: title(session) }));
    projects.map((project) => [project.id, project.name]).concat([[null, t("projects.unassigned")]]).forEach(([id, name]) => {
      const item = el(doc, "button", "move-item");
      const here = id === null ? !filed.has(session.project_id) : session.project_id === id;
      item.setAttribute?.("role", "menuitemradio");
      item.setAttribute?.("aria-checked", String(here));
      item.append(icon(doc, here ? "i-check" : "i-folder"), el(doc, "span", null, name));
      item.addEventListener("click", () => opts.onMove && opts.onMove(session.id, id));
      menu.append(item);
    });
    return [row, menu];
  }

  globalThis.YataSessions = { renderList, placeholder, title, startedLabel, dayLabel, sourceLabel };
})();
