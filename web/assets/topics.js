(() => {
  "use strict";
  const t = (key, params) => globalThis.YataI18n.t(key, params);
  const REVIEW = t("topics.review");
  const SUMMARY_STATE = {
    pending: { icon: "i-refresh", tip: t("topics.state.pending"), tone: "busy" },
    ready: { icon: "i-check", tip: t("topics.state.ready"), tone: "ok" },
    error: { icon: "i-alert", tip: t("topics.state.error"), tone: "bad" },
    none: { icon: "i-refresh", tip: t("topics.state.none"), tone: "busy" },
  };
  const DELIVERY = { pending: t("topics.delivery.pending"), sent: t("topics.delivery.sent"), failed: t("topics.delivery.failed") };

  function el(doc, tag, className, text) {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function icon(doc, id, className) {
    const holder = el(doc, "span", "icon " + (className || ""));
    if (doc.createElementNS) {
      const svg = doc.createElementNS("http://www.w3.org/2000/svg", "svg");
      const use = doc.createElementNS("http://www.w3.org/2000/svg", "use");
      use.setAttribute("href", "#" + id);
      svg.append(use);
      holder.append(svg);
    }
    return holder;
  }

  function tip(node, text) {
    if (node.dataset) node.dataset.tip = text;
    node.setAttribute?.("aria-label", text);
    return node;
  }

  function quotes(doc, ids, evidence, formatTime) {
    const list = el(doc, "ul", "quotes");
    (ids || []).forEach((id) => {
      const item = evidence.get(id);
      if (!item) return;
      const li = el(doc, "li", item.speaker ? "tone-" + (globalThis.YataCaptions ? YataCaptions.speakerTone(item.speaker) : 1) : null);
      const start = item.media_start_ms != null ? item.media_start_ms : item.capture_start_ms;
      li.append(el(doc, "time", null, formatTime(start)));
      if (item.speaker) li.append(el(doc, "b", null, item.speaker));
      li.append(el(doc, "q", null, item.text));
      list.append(li);
    });
    return list;
  }

  function point(doc, item, evidence, formatTime) {
    const li = el(doc, "li", "point" + (item.needs_review ? " review" : ""));
    const body = el(doc, "div", "point-body");
    if (item.needs_review) {
      body.append(tip(icon(doc, "i-alert", "hazard"), REVIEW));
    } else {
      body.append(el(doc, "span", "point-text", item.text));
    }
    const details = el(doc, "details", "evidence");
    const summary = el(doc, "summary", null);
    summary.append(icon(doc, "i-quote"), el(doc, "span", null, String(item.evidence_ids.length)));
    tip(summary, t("topics.evidence", { count: item.evidence_ids.length }));
    details.append(summary, quotes(doc, item.evidence_ids, evidence, formatTime));
    if (item.needs_review) details.open = true;
    body.append(details);
    li.append(body);
    return li;
  }

  function summaryBlock(doc, summary, evidence, formatTime) {
    const block = el(doc, "div", "topic-body");
    if (summary.overview) block.append(el(doc, "p", "overview", summary.overview));
    if (summary.overview_needs_review) block.append(tip(icon(doc, "i-alert", "hazard"), REVIEW));
    if (summary.key_facts && summary.key_facts.length) {
      const facts = el(doc, "div", "facts");
      summary.key_facts.forEach((fact) => {
        const tile = el(doc, "div", "fact");
        tile.append(el(doc, "span", "fact-label", fact.label), el(doc, "strong", "fact-value", fact.value));
        tip(tile, fact.label + ": " + fact.value);
        facts.append(tile);
      });
      block.append(facts);
    }
    if (summary.points && summary.points.length) {
      const list = el(doc, "ul", "points");
      summary.points.forEach((item) => list.append(point(doc, item, evidence, formatTime)));
      block.append(list);
    }
    return block;
  }

  function header(doc, topic, formatTime, redeliver) {
    const head = el(doc, "header", "topic-head");
    head.append(el(doc, "span", "topic-no", "T" + String(topic.index).padStart(2, "0")));
    head.append(el(doc, "time", "topic-time", formatTime(topic.start_ms) + " – " + formatTime(topic.end_ms)));
    if (topic.status === "open") {
      head.append(tip(el(doc, "span", "topic-live"), t("topics.live")));
    } else {
      const state = SUMMARY_STATE[topic.summary_state] || SUMMARY_STATE.none;
      head.append(tip(icon(doc, state.icon, "state " + state.tone), state.tip));
    }
    if (topic.close_reason === "max_length") head.append(tip(icon(doc, "i-chevron", "split"), t("topics.split")));
    if (redeliver && DELIVERY[topic.delivery_state]) {
      const failed = topic.delivery_state === "failed";
      const delivery = el(doc, failed ? "button" : "span", "delivery " + topic.delivery_state);
      delivery.append(icon(doc, "i-send"));
      tip(delivery, failed && topic.delivery_reason ? t("topics.delivery.retry", { state: DELIVERY.failed, reason: topic.delivery_reason })
        : DELIVERY[topic.delivery_state]);
      if (failed) delivery.addEventListener("click", () => redeliver(topic.id));
      head.append(delivery);
    }
    return head;
  }

  function renderTopics(container, topics, evidence, formatTime, redeliver) {
    const doc = container.ownerDocument;
    container.replaceChildren();
    // Cards are rebuilt on every update; only a topic the list has not shown yet animates in.
    const seen = container.yataSeen || (container.yataSeen = new Set());
    if (!topics.length) {
      seen.clear();
      const ghost = el(doc, "div", "topic ghost");
      for (let i = 0; i < 3; i++) ghost.append(el(doc, "i"));
      container.append(ghost);
      return;
    }
    const open = topics.filter((topic) => topic.status === "open");
    const closed = topics.filter((topic) => topic.status !== "open").reverse();
    [...open, ...closed].forEach((topic) => {
      const live = topic.status === "open";
      const card = el(doc, "article", "topic hue-" + (((topic.index - 1) % 6) + 1) + (live ? " open" : " closed") + (seen.has(topic.id) ? "" : " entering"));
      seen.add(topic.id);
      if (card.dataset) card.dataset.topicId = topic.id;
      card.append(header(doc, topic, formatTime, redeliver));
      card.append(el(doc, "h3", "topic-title" + (topic.title ? "" : " pending-title"), topic.title || ""));
      const summary = live ? topic.progress : topic.summary;
      if (summary) {
        const block = summaryBlock(doc, summary, evidence, formatTime);
        if (live) block.classList?.add("draft");
        card.append(block);
      }
      const pips = el(doc, "div", "pips");
      tip(pips, t("topics.lines", { count: topic.segment_ids.length }));
      const shown = Math.min(topic.segment_ids.length, 24);
      for (let i = 0; i < shown; i++) pips.append(el(doc, "i"));
      if (topic.segment_ids.length > shown) pips.append(el(doc, "em", null, "+" + (topic.segment_ids.length - shown)));
      card.append(pips);
      container.append(card);
    });
  }

  function renderDigest(container, digest, topics) {
    const doc = container.ownerDocument;
    container.replaceChildren();
    container.hidden = !digest;
    if (!digest) return;
    const head = el(doc, "header", "digest-head");
    // The HUD theme shows the code, the soft theme the word in the page's language (style.css .en / .ja).
    head.append(icon(doc, "i-flag"), el(doc, "span", "en", "DEBRIEF"), el(doc, "span", "ja", t("topics.digest")));
    tip(head, t("topics.digest"));
    container.append(head);
    if (digest.overview) container.append(el(doc, "p", "overview", digest.overview));
    if (digest.overview_needs_review) container.append(tip(icon(doc, "i-alert", "hazard"), REVIEW));
    const list = el(doc, "ul", "highlights");
    (digest.highlights || []).forEach((item) => {
      const li = el(doc, "li", item.needs_review ? "review" : null);
      item.topic_ids.map((id) => topics.get(id)).filter(Boolean).forEach((topic) => {
        li.append(tip(el(doc, "span", "topic-ref hue-" + (((topic.index - 1) % 6) + 1), "T" + String(topic.index).padStart(2, "0")), topic.title || ""));
      });
      if (item.needs_review) li.append(tip(icon(doc, "i-alert", "hazard"), REVIEW));
      else li.append(el(doc, "span", "point-text", item.text));
      list.append(li);
    });
    container.append(list);
  }

  globalThis.YataTopics = { renderTopics, renderDigest, REVIEW };
})();
