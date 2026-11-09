(() => {
  "use strict";
  const THEMES = ["soft", "hud"];
  const DEFAULT = THEMES[0];
  const KEY = "yata.theme";
  const root = globalThis.document?.documentElement || { dataset: {} };
  const switchable = String(root.dataset?.features || "").split(/\s+/).includes("theme_switch");
  const listeners = [];

  function apply(name) {
    const theme = switchable && THEMES.includes(name) ? name : DEFAULT;
    root.dataset.theme = theme;
    return theme;
  }

  function stored() {
    if (!switchable) return null;
    try { return globalThis.localStorage?.getItem(KEY) || null; } catch { return null; }
  }

  function set(name) {
    if (!switchable) return apply(DEFAULT);
    const theme = apply(name);
    try { globalThis.localStorage?.setItem(KEY, theme); } catch { /* private window or blocked storage */ }
    listeners.forEach((listener) => listener(theme));
    return theme;
  }

  apply(stored());
  globalThis.YataTheme = {
    THEMES,
    DEFAULT,
    switchable,
    get: () => root.dataset.theme || DEFAULT,
    set,
    onChange: (listener) => listeners.push(listener),
  };
})();
