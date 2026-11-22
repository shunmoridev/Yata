(() => {
  "use strict";
  const LANGUAGES = [
    ["ja", "日本語"], ["en", "English"], ["zh", "中文"], ["ko", "한국어"], ["fr", "Français"], ["es", "Español"],
    ["de", "Deutsch"], ["it", "Italiano"], ["pt", "Português"], ["ru", "Русский"], ["hi", "हिन्दी"], ["id", "Bahasa Indonesia"],
    ["th", "ไทย"], ["vi", "Tiếng Việt"], ["ar", "العربية"],
  ];
  const languageLabel = (code) => (LANGUAGES.find(([value]) => value === code) || [code, code || globalThis.YataI18n.t("common.unknown")])[1];
  const speakerTone = (speaker) => {
    const number = Number(String(speaker || "").replace(/\D/g, ""));
    return number ? ((number - 1) % 6) + 1 : 0;
  };

  class CaptionBoard {
    // The board keeps every final line of the session; a view that shows fewer (the small window) slices lines().
    constructor(maxFinals = Infinity) {
      this.maxFinals = maxFinals;
      this.finals = [];
      this.keys = new Set();
      this.partial = null;
      this.translations = new Map();
      // The final lines change only when a final or a translation arrives; partial updates reuse them.
      this.cache = null;
    }

    addFinal(caption) {
      if (!caption || !caption.text) return;
      const key = caption.segment_id || caption.utterance_id || caption.id;
      if (this.keys.has(key)) return;
      const item = {
        key, utterance: caption.utterance_id || null, text: caption.text, language: caption.language || null,
        speaker: caption.speaker || null, start: caption.capture_start_ms ?? null,
      };
      this.finals.push(item);
      this.keys.add(key);
      // Lines nearly always arrive in order; sort only when one did not.
      const previous = this.finals[this.finals.length - 2];
      if (previous && (previous.start ?? 0) > (item.start ?? 0)) this.finals.sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
      if (this.finals.length > this.maxFinals) {
        this.finals.splice(0, this.finals.length - this.maxFinals).forEach((dropped) => this.keys.delete(dropped.key));
      }
      if (this.partial && this.partial.utterance_id === caption.utterance_id) this.partial = null;
      this.cache = null;
    }

    setPartial(caption) {
      this.partial = caption && caption.text
        ? { utterance_id: caption.utterance_id, text: caption.text, language: caption.language || null, speaker: caption.speaker || null }
        : null;
    }

    clear(utteranceId) {
      if (this.partial && (!utteranceId || this.partial.utterance_id === utteranceId)) this.partial = null;
    }

    addTranslation(segmentId, text) {
      if (!segmentId || !text || this.translations.get(segmentId) === text) return;
      this.translations.set(segmentId, text);
      this.cache = null;
    }

    // A new array each call (callers may keep it); its final line objects are shared until the finals change.
    lines() {
      this.cache ||= this.finals.map((item) => ({
        kind: "final", key: item.key, utterance: item.utterance, text: item.text, language: item.language,
        speaker: item.speaker, translation: this.translations.get(item.key) || null,
      }));
      const lines = this.cache.slice();
      if (this.partial) {
        lines.push({
          kind: "partial", key: "partial:" + this.partial.utterance_id, utterance: this.partial.utterance_id,
          text: this.partial.text, language: this.partial.language, speaker: this.partial.speaker, translation: null,
        });
      }
      return lines;
    }
  }

  const MAX_ANIMATED_CHARS = 48;
  // Only the newest rows are on screen while the view follows, so only they are measured for the slide.
  const SLIDE_ROWS = 8;
  const FOLLOW_SLACK_PX = 32;
  // Keep only a bounded slice of the session in the DOM. Both directions can page across the full history.
  const WINDOW_ROWS = 300;
  const OLDER_ROWS = 200;
  const MAX_WINDOW_ROWS = 700;
  const OLDER_TRIGGER_PX = 120;

  class CaptionView {
    constructor(container, options) {
      this.container = container;
      this.doc = container.ownerDocument;
      this.nodes = new Map();
      this.placeholder = null;
      this.windowed = Boolean(options?.windowed);
      this.firstKey = null;
      this.lastKey = null;
      this.start = 0;
      this.end = 0;
      this.older = false;
      this.newer = false;
      this.forceFollow = false;
      this.shifting = false;
      this.last = null;
      if (this.windowed) container.addEventListener?.("scroll", () => this.maybeShift());
    }

    // Swap a page of rows in either direction while keeping the same visible anchor in place.
    // Merely appending older rows would eventually put the entire transcript back in the DOM.
    maybeShift() {
      const box = this.container;
      if (this.shifting || !this.last) return;
      const older = box.scrollTop <= OLDER_TRIGGER_PX && this.start > 0;
      const newer = !older && this.end < this.last.lines.length
        && box.scrollHeight - box.scrollTop - box.clientHeight <= OLDER_TRIGGER_PX;
      if (!older && !newer) return;
      const anchor = this.nodes.get(older ? this.firstKey : this.lastKey)?.row;
      const before = anchor?.getBoundingClientRect?.().top;
      const height = box.scrollHeight;
      this.shifting = true;
      try {
        this.older = older;
        this.newer = newer;
        this.update(this.last.lines, this.last.showTranslation);
        if (before != null && anchor?.getBoundingClientRect) {
          box.scrollTop += anchor.getBoundingClientRect().top - before;
        } else if (older) {
          // DOM-free test harnesses do not expose per-row geometry.
          box.scrollTop += box.scrollHeight - height;
        }
      } finally {
        this.shifting = false;
      }
    }

    windowRange(lines, follow, previousLength) {
      if (!this.windowed) return [0, lines.length];
      if (follow || this.firstKey === null) return [Math.max(0, lines.length - WINDOW_ROWS), lines.length];
      const at = lines[this.start]?.key === this.firstKey ? this.start : lines.findIndex((line) => line.key === this.firstKey);
      let start = at >= 0 ? at : Math.max(0, this.start);
      let end = this.end === previousLength ? lines.length : Math.min(lines.length, this.end);
      if (this.older) start = Math.max(0, start - OLDER_ROWS);
      if (this.newer) end = Math.min(lines.length, end + OLDER_ROWS);
      if (end - start > MAX_WINDOW_ROWS) {
        // When paging backward, evict the newest rows; when paging forward, evict the oldest.
        if (this.older) end = start + MAX_WINDOW_ROWS;
        else start = end - MAX_WINDOW_ROWS;
      }
      return [start, end];
    }

    jumpToLatest() {
      if (!this.last) return;
      this.forceFollow = true;
      this.update(this.last.lines, this.last.showTranslation);
    }

    // A view that scrolls follows the newest line unless the reader has scrolled up to read back.
    following() {
      const box = this.container;
      if (!box.scrollHeight || !box.clientHeight) return true;
      return box.scrollHeight - box.scrollTop - box.clientHeight <= FOLLOW_SLACK_PX;
    }

    follow() {
      const box = this.container;
      if (box.scrollHeight) box.scrollTop = box.scrollHeight;
    }

    update(lines, showTranslation) {
      const doc = this.doc;
      const previousLength = this.last?.lines.length ?? lines.length;
      const follow = this.forceFollow || (!this.windowed || !this.last || this.end === previousLength) && this.following();
      this.forceFollow = false;
      this.last = { lines, showTranslation };
      if (!lines.length) {
        this.clear();
        if (!this.placeholder) {
          this.placeholder = doc.createElement("div");
          this.placeholder.className = "caption-idle";
          for (let i = 0; i < 3; i++) this.placeholder.append(doc.createElement("i"));
          this.container.append(this.placeholder);
        }
        return;
      }
      if (this.placeholder) {
        this.placeholder.remove?.();
        this.placeholder = null;
      }
      const before = this.positions(lines);
      const [start, end] = this.windowRange(lines, follow, previousLength);
      this.start = start;
      this.end = end;
      this.firstKey = lines[start].key;
      this.lastKey = lines[end - 1].key;
      if (this.windowed) this.container.dataset.historyBehind = String(end < lines.length);
      const wanted = new Set();
      const order = [];
      lines.slice(start, end).forEach((line, offset) => {
        const index = start + offset;
        const key = line.key;
        let node = this.nodes.get(key);
        const partialKey = line.utterance ? "partial:" + line.utterance : null;
        if (!node && line.kind === "final" && partialKey && this.nodes.has(partialKey)) {
          node = this.nodes.get(partialKey);
          this.nodes.delete(partialKey);
          this.nodes.set(key, node);
        }
        if (!node) {
          node = this.create(line);
          this.nodes.set(key, node);
          this.container.append(node.row);
        }
        wanted.add(key);
        order.push(node.row);
        this.fill(node, line, showTranslation, index === lines.length - 1);
      });
      for (const [key, node] of [...this.nodes]) {
        if (wanted.has(key)) continue;
        this.nodes.delete(key);
        // Rows leaving the stage's window go at once; a short view (the small window) lets its old line rise out.
        if (this.windowed) node.row.remove?.();
        else this.leave(node.row);
      }
      // Moving a node restarts its animations, so reorder only when the order really changed.
      const ordered = new Set(order);
      const current = [...this.container.children].filter((row) => !row.classList?.contains("leaving") && ordered.has(row));
      if (current.some((row, index) => row !== order[index])) order.forEach((row) => this.container.append(row));
      if (follow) this.follow();
      this.slide(before);
    }

    clear() {
      for (const node of this.nodes.values()) node.row.remove?.();
      this.nodes.clear();
      this.firstKey = null;
      this.lastKey = null;
      this.start = 0;
      this.end = 0;
      if (this.windowed) this.container.dataset.historyBehind = "false";
      this.container.replaceChildren(...(this.placeholder ? [this.placeholder] : []));
    }

    create(line) {
      const doc = this.doc;
      const row = doc.createElement("p");
      row.className = "caption-line entering";
      const speaker = doc.createElement("span");
      speaker.className = "caption-speaker";
      const text = doc.createElement("span");
      text.className = "caption-original";
      const translation = doc.createElement("span");
      translation.className = "caption-translation";
      row.append(speaker, text, translation);
      return { row, speaker, text, translation, value: "", kind: null, translated: "", latest: null, speakerName: null, animated: false };
    }

    fill(node, line, showTranslation, latest) {
      const translated = showTranslation && line.translation ? line.translation : "";
      // With the whole history on the board most rows are unchanged on each update; leave them alone.
      if (node.kind === line.kind && node.latest === latest && node.value === line.text && node.speakerName === (line.speaker || "")
        && node.translated === translated) return;
      const tone = speakerTone(line.speaker);
      node.row.className = "caption-line " + line.kind + (latest ? " latest" : " past") + (tone ? " tone-" + tone : "")
        + (node.kind === "partial" && line.kind === "final" ? " locking" : "") + (node.value ? "" : " entering");
      if (node.speakerName !== (line.speaker || "")) {
        node.speaker.textContent = line.speaker || "";
        node.speaker.hidden = !line.speaker;
        node.speakerName = line.speaker || "";
      }
      if (line.text !== node.value) this.type(node, line.text, latest);
      else if (!latest && node.animated) this.settle(node);
      node.kind = line.kind;
      node.latest = latest;
      if (translated !== node.translated) {
        node.translation.textContent = translated;
        node.translation.className = "caption-translation" + (translated ? " shown" : "");
        node.translated = translated;
      }
    }

    type(node, next, animate = true) {
      const doc = this.doc;
      const previous = node.value;
      if (!animate) {
        node.text.replaceChildren();
        node.text.textContent = next;
        node.value = next;
        node.animated = false;
        return;
      }
      let common = 0;
      while (common < previous.length && common < next.length && previous[common] === next[common]) common++;
      node.text.replaceChildren();
      const stable = doc.createElement("span");
      stable.textContent = next.slice(0, common);
      node.text.append(stable);
      const tail = next.slice(common);
      const animated = tail.slice(0, MAX_ANIMATED_CHARS);
      [...animated].forEach((char, index) => {
        const span = doc.createElement("span");
        span.className = "ch";
        span.textContent = char;
        if (span.style) span.style.animationDelay = Math.min(index * 18, 600) + "ms";
        node.text.append(span);
      });
      if (tail.length > animated.length) {
        const rest = doc.createElement("span");
        rest.textContent = tail.slice(animated.length);
        node.text.append(rest);
      }
      node.value = next;
      node.animated = true;
    }

    // A row that is no longer the newest drops its per-character spans: the history stays one text node a row.
    settle(node) {
      node.text.replaceChildren();
      node.text.textContent = node.value;
      node.animated = false;
    }

    positions(lines) {
      const out = new Map();
      for (const line of lines.slice(-SLIDE_ROWS)) {
        const node = this.nodes.get(line.key) || (line.utterance && this.nodes.get("partial:" + line.utterance));
        if (node?.row.getBoundingClientRect) out.set(node.row, node.row.getBoundingClientRect().top);
      }
      return out;
    }

    // FLIP: rows that moved animate from their old position to the new one.
    slide(before) {
      for (const row of before.keys()) {
        if (!row.getBoundingClientRect || !row.animate) continue;
        const delta = before.get(row) - row.getBoundingClientRect().top;
        if (Math.abs(delta) > 1) row.animate([{ transform: `translateY(${delta}px)` }, { transform: "translateY(0)" }], { duration: 380, easing: "cubic-bezier(.2,.8,.2,1)" });
      }
    }

    leave(row) {
      if (!row.animate) {
        row.remove?.();
        return;
      }
      row.classList?.add("leaving");
      const done = () => row.remove();
      const animation = row.animate([{ opacity: 1, transform: "translateY(0)" }, { opacity: 0, transform: "translateY(-18px)" }], { duration: 320, easing: "ease-in" });
      animation.onfinish = done;
    }
  }

  const views = new WeakMap();
  // options.windowed: the stage, which scrolls through the whole history (see WINDOW_ROWS).
  function render(container, lines, showTranslation, options) {
    let view = views.get(container);
    if (!view) {
      view = new CaptionView(container, options);
      views.set(container, view);
    }
    view.update(lines, showTranslation);
  }

  function jumpToLatest(container) {
    views.get(container)?.jumpToLatest();
  }

  globalThis.YataCaptions = { CaptionBoard, CaptionView, LANGUAGES, languageLabel, render, jumpToLatest, speakerTone, WINDOW_ROWS, OLDER_ROWS, MAX_WINDOW_ROWS };
})();
