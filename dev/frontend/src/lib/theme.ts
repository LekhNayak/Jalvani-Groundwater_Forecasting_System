/** Light / dark / follow the device. The choice is remembered per browser; index.html applies it before first paint. */
export type ThemePref = "light" | "dark" | "system";
const KEY = "jalvaani.theme";

export function readTheme(): ThemePref {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "light" || v === "dark") return v;
  } catch { /* storage blocked */ }
  return "system";
}

export function applyTheme(p: ThemePref) {
  const root = document.documentElement;
  if (p === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", p);
  try {
    if (p === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, p);
  } catch { /* ignore */ }
}

export const systemDark = () => typeof window !== "undefined" && !!window.matchMedia?.("(prefers-color-scheme: dark)").matches;

/** Resolved colour of a CSS custom property (for Leaflet, which needs real colours, not var()). */
export const cssVar = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
