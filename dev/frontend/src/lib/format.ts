import { bcp47, type Lang } from "./langs";


/** "4 Jun 2026" in the chosen language (Latin digits everywhere, for consistency with the charts). */
export function fmtDate(iso: string | null | undefined, lang: Lang): string {
  if (!iso) return "–";
  const d = new Date(`${iso}T00:00:00`);
  return new Intl.DateTimeFormat(`${bcp47(lang)}-u-nu-latn`, { day: "numeric", month: "short", year: "numeric" }).format(d);
}

/** Numbers rounded for people: metres to 1 decimal, percentages to whole numbers. */
export function fmtNum(x: number | null | undefined, digits = 1, lang: Lang = "en"): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  return new Intl.NumberFormat(`${bcp47(lang)}-u-nu-latn`, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(x);
}

/** Level (negative = below ground) -> depth below ground, positive metres. */
export const depth = (level: number | null | undefined) => (level === null || level === undefined ? null : -level);

/** Direction word for a change in level (positive = water rose). Small changes count as steady. */
export function changeDir(change: number | null | undefined, steady = 0.1): "rise" | "fall" | "steady" | null {
  if (change === null || change === undefined || !Number.isFinite(change)) return null;
  if (Math.abs(change) < steady) return "steady";
  return change > 0 ? "rise" : "fall";
}

export function slugify(name: string) {
  return name.toLowerCase().replace(/ /g, "_");
}
