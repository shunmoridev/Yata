(() => {
  "use strict";
  class GapLedger {
    constructor() {
      this.byReason = new Map();
      this.lastCursor = 0;
    }

    add(event) {
      if (!event || !Number.isInteger(event.cursor) || event.cursor <= this.lastCursor) return false;
      this.lastCursor = event.cursor;
      const data = event.data || {};
      const reason = String(data.reason || globalThis.YataI18n.t("gaps.unknown")).trim();
      const entry = this.byReason.get(reason) || { reason, count: 0 };
      entry.count++;
      entry.lastCursor = event.cursor;
      entry.lastSeen = data.observed_at_utc || new Date().toISOString();
      entry.captureStartMs = data.capture_start_ms;
      entry.captureEndMs = data.capture_end_ms;
      this.byReason.set(reason, entry);
      return true;
    }

    entries() {
      return [...this.byReason.values()].sort((a, b) => b.lastCursor - a.lastCursor);
    }
  }
  globalThis.YataGapLedger = GapLedger;
})();
