(() => {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const t = (key, params) => globalThis.YataI18n.t(key, params);

  function svg(doc, tag, attrs) {
    const node = doc.createElementNS ? doc.createElementNS(NS, tag) : doc.createElement(tag);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute?.(key, String(value)));
    return node;
  }

  const clock = (ms) => {
    const s = Math.max(0, Math.floor((ms || 0) / 1000));
    const h = Math.floor(s / 3600);
    const m = String(Math.floor(s / 60) % 60).padStart(2, "0");
    const sec = String(s % 60).padStart(2, "0");
    return (h ? String(h).padStart(2, "0") + ":" : "") + m + ":" + sec;
  };

  function gauge(el, value, text, tone) {
    if (!el) return;
    if (el.dataset) el.dataset.tone = tone || "ok";
    const readout = el.querySelector ? el.querySelector("b") : null;
    if (readout) readout.textContent = text;
    const bar = el.querySelector ? el.querySelector("i") : null;
    const clamped = Math.max(0, Math.min(1, Number(value) || 0));
    if (bar?.style) bar.style.transform = `scaleX(${clamped.toFixed(3)})`;
  }

  function radar(container, counts, active) {
    const doc = container.ownerDocument;
    let view = container.querySelector ? container.querySelector("svg.radar-svg") : null;
    if (!view) {
      container.replaceChildren?.();
      view = svg(doc, "svg", { viewBox: "-100 -100 200 200", class: "radar-svg", role: "img", "aria-label": t("hud.radar") });
      [30, 60, 90].forEach((r) => view.append(svg(doc, "circle", { r, class: "ring" })));
      view.append(svg(doc, "path", { d: "M-90 0H90M0 -90V90", class: "cross" }));
      const sweep = svg(doc, "g", { class: "sweep" });
      sweep.append(svg(doc, "path", { d: "M0 0 L0 -90 A90 90 0 0 1 63.6 -63.6 Z", class: "sweep-wedge" }));
      view.append(sweep, svg(doc, "g", { class: "blips" }));
      const legend = doc.createElement("div");
      legend.className = "radar-legend";
      container.append(view, legend);
    }
    const blips = view.querySelector(".blips");
    const legend = container.querySelector(".radar-legend");
    const entries = [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0], undefined, { numeric: true }));
    const total = entries.reduce((sum, [, count]) => sum + count, 0) || 1;
    blips.replaceChildren?.();
    legend.replaceChildren();
    container.classList?.toggle("empty", !entries.length);
    entries.forEach(([label, count], index) => {
      const share = count / total;
      const bearing = (-90 + index * (360 / Math.max(entries.length, 1)) + 25) * Math.PI / 180;
      const distance = 24 + share * 58;
      const x = Math.cos(bearing) * distance;
      const y = Math.sin(bearing) * distance;
      const tone = globalThis.YataCaptions ? YataCaptions.speakerTone(label) : index + 1;
      const group = svg(doc, "g", { class: "blip tone-" + tone + (label === active ? " active" : ""), transform: `translate(${x.toFixed(1)} ${y.toFixed(1)})` });
      group.append(svg(doc, "circle", { r: 14, class: "blip-pulse" }), svg(doc, "circle", { r: (4 + share * 7).toFixed(1), class: "blip-dot" }));
      const name = svg(doc, "text", { x: 10, y: -8, class: "blip-label" });
      name.textContent = label;
      group.append(name);
      blips.append(group);
      const row = doc.createElement("div");
      row.className = "radar-row tone-" + tone + (label === active ? " active" : "");
      const tag = doc.createElement("b");
      tag.textContent = label;
      const bar = doc.createElement("i");
      const fill = doc.createElement("span");
      if (fill.style) fill.style.width = Math.round(share * 100) + "%";
      bar.append(fill);
      const value = doc.createElement("em");
      value.textContent = Math.round(share * 100) + "%";
      row.append(tag, bar, value);
      legend.append(row);
    });
  }

  function sessionMap(container, topics, gaps, nowMs, onPick) {
    const doc = container.ownerDocument;
    const ends = topics.map((topic) => topic.end_ms || 0).concat(gaps.map((gap) => gap.captureEndMs || 0));
    const total = Math.max(60_000, nowMs || 0, ...ends) * 1.02;
    const pct = (ms) => Math.max(0, Math.min(100, (ms / total) * 100)) + "%";
    container.replaceChildren();
    const track = doc.createElement("div");
    track.className = "map-track";
    topics.forEach((topic) => {
      const start = topic.start_ms || 0;
      const end = topic.status === "open" ? Math.max(nowMs || 0, topic.end_ms || start) : (topic.end_ms || start);
      const segment = doc.createElement("button");
      segment.className = "map-segment hue-" + (((topic.index - 1) % 6) + 1) + (topic.status === "open" ? " open" : "")
        + (topic.summary_state === "ready" ? " ready" : "");
      segment.dataset.tip = (topic.title || t("hud.topic", { index: topic.index })) + " · " + clock(start) + "–" + clock(end);
      segment.setAttribute?.("aria-label", (t("hud.topic", { index: topic.index }) + " " + (topic.title || "")).trim());
      if (segment.style) {
        segment.style.left = pct(start);
        segment.style.width = `max(6px, ${pct(end - start)})`;
      }
      const number = doc.createElement("span");
      number.textContent = String(topic.index).padStart(2, "0");
      segment.append(number);
      if (onPick) segment.addEventListener("click", () => onPick(topic.id));
      track.append(segment);
    });
    gaps.forEach((gap) => {
      if (!Number.isFinite(gap.captureStartMs)) return;
      const tick = doc.createElement("i");
      tick.className = "map-gap";
      tick.dataset.tip = gap.reason;
      if (tick.style) tick.style.left = pct(gap.captureStartMs);
      track.append(tick);
    });
    const head = doc.createElement("i");
    head.className = "map-head";
    if (head.style) head.style.left = pct(nowMs || 0);
    track.append(head);
    const ruler = doc.createElement("div");
    ruler.className = "map-ruler";
    for (let i = 0; i <= 4; i++) {
      const mark = doc.createElement("span");
      mark.textContent = clock((total / 1.02) * i / 4);
      ruler.append(mark);
    }
    container.append(track, ruler);
  }

  globalThis.YataHud = { gauge, radar, sessionMap, clock };
})();
