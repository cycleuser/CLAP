/* CLAP theme — light / dark / follow-system, applied before the stylesheet
   so the first paint is already correct (no flash).
   The choice lives in localStorage; `data-theme` on <html> drives app.css. */
(function (global) {
  "use strict";
  const KEY = "clap-theme";
  const mq = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  let mode = "auto";
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark" || saved === "auto") mode = saved;
  } catch (e) { /* private mode */ }
  try {
    const fromUrl = new URLSearchParams(window.location.search || "").get("theme");
    if (fromUrl === "light" || fromUrl === "dark" || fromUrl === "auto") mode = fromUrl;
  } catch (e) { /* old browser */ }

  const listeners = [];

  function resolved() {
    if (mode === "light" || mode === "dark") return mode;
    return mq && mq.matches ? "dark" : "light";
  }

  function apply() {
    const theme = resolved();
    document.documentElement.setAttribute("data-theme", theme);
    document.documentElement.style.colorScheme = theme;
    listeners.forEach((fn) => {
      try { fn(theme, mode); } catch (e) { /* listener error */ }
    });
  }

  function setMode(next) {
    if (next !== "light" && next !== "dark" && next !== "auto") return;
    mode = next;
    try { localStorage.setItem(KEY, next); } catch (e) { /* private mode */ }
    apply();
  }

  if (mq) {
    const onScheme = () => { if (mode === "auto") apply(); };
    if (mq.addEventListener) mq.addEventListener("change", onScheme);
    else if (mq.addListener) mq.addListener(onScheme);
  }

  global.CLAPtheme = {
    get mode() { return mode; },
    get theme() { return resolved(); },
    setMode,
    apply,
    onChange: (fn) => listeners.push(fn),
    options: ["auto", "light", "dark"]
  };

  apply();
})(window);
