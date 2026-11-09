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
    constructor(maxFinals = 2) {
      this.maxFinals = maxFinals;
      this.finals = [];
      this.partial = null;
      this.translations = new Map();
    }

    addFinal(caption) {
      if (!caption || !caption.text) return;
      const key = caption.segment_id || caption.utterance_id || caption.id;
      if (this.finals.some((item) => item.key === key)) return;
      this.finals.push({
        key, utterance: caption.utterance_id || null, text: caption.text, language: caption.language || null,
        speaker: caption.speaker || null, start: caption.capture_start_ms ?? null,
      });
      this.finals.sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
      this.finals = this.finals.slice(-this.maxFinals);
      if (this.partial && this.partial.utterance_id === caption.utterance_id) this.partial = null;
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
      if (segmentId && text) this.translations.set(segmentId, text);
    }

    lines() {
      const lines = this.finals.map((item) => ({
        kind: "final", key: item.key, utterance: item.utterance, text: item.text, language: item.language,
        speaker: item.speaker, translation: this.translations.get(item.key) || null,
      }));
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

  class CaptionView {
    constructor(container) {
      this.container = container;
      this.doc = container.ownerDocument;
      this.nodes = new Map();
      this.placeholder = null;
    }

    update(lines, showTranslation) {
      const doc = this.doc;
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
      const before = this.positions();
      const wanted = new Set();
      const order = [];
      lines.forEach((line, index) => {
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
        this.leave(node.row);
      }
      // Moving a node restarts its animations, so reorder only when the order really changed.
      const current = [...this.container.children].filter((row) => !row.classList?.contains("leaving") && order.includes(row));
      if (current.some((row, index) => row !== order[index])) order.forEach((row) => this.container.append(row));
      this.slide(before);
    }

    clear() {
      for (const node of this.nodes.values()) node.row.remove?.();
      this.nodes.clear();
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
      return { row, speaker, text, translation, value: "", kind: null, translated: "" };
    }

    fill(node, line, showTranslation, latest) {
      const tone = speakerTone(line.speaker);
      node.row.className = "caption-line " + line.kind + (latest ? " latest" : " past") + (tone ? " tone-" + tone : "")
        + (node.kind === "partial" && line.kind === "final" ? " locking" : "") + (node.value ? "" : " entering");
      node.speaker.textContent = line.speaker || "";
      node.speaker.hidden = !line.speaker;
      if (line.text !== node.value) this.type(node, line.text);
      node.kind = line.kind;
      const translated = showTranslation && line.translation ? line.translation : "";
      if (translated !== node.translated) {
        node.translation.textContent = translated;
        node.translation.className = "caption-translation" + (translated ? " shown" : "");
        node.translated = translated;
      }
    }

    type(node, next) {
      const doc = this.doc;
      const previous = node.value;
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
    }

    positions() {
      const out = new Map();
      for (const node of this.nodes.values()) {
        if (node.row.getBoundingClientRect) out.set(node.row, node.row.getBoundingClientRect().top);
      }
      return out;
    }

    // FLIP: rows that moved animate from their old position to the new one.
    slide(before) {
      for (const node of this.nodes.values()) {
        const row = node.row;
        if (!before.has(row) || !row.getBoundingClientRect || !row.animate) continue;
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
  function render(container, lines, showTranslation) {
    let view = views.get(container);
    if (!view) {
      view = new CaptionView(container);
      views.set(container, view);
    }
    view.update(lines, showTranslation);
  }

  globalThis.YataCaptions = { CaptionBoard, CaptionView, LANGUAGES, languageLabel, render, speakerTone };
})();
