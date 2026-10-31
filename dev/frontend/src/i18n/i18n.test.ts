import { describe, expect, it } from "vitest";
import en from "./en.json";
import districts from "./districts.json";
import { LANG_CODES } from "../lib/langs";

type Tree = { [k: string]: string | Tree };
const FILES = import.meta.glob<Tree>("./*.json", { eager: true, import: "default" });

function flatten(t: Tree, prefix = ""): Record<string, string> {
  return Object.entries(t).reduce((acc, [k, v]) => {
    const key = prefix ? `${prefix}.${k}` : k;
    return typeof v === "string" ? { ...acc, [key]: v } : { ...acc, ...flatten(v, key) };
  }, {} as Record<string, string>);
}
const vars = (s: string) => (s.match(/\{\{\s*\w+\s*\}\}/g) ?? []).map((x) => x.replace(/\s/g, "")).sort();

describe("translations", () => {
  const E = flatten(en as Tree);
  for (const code of LANG_CODES.filter((c) => c !== "en")) {
    const tree = FILES[`./${code}.json`];
    it(`${code} has a translation file`, () => expect(tree).toBeTruthy());
    if (!tree) continue;
    const L = flatten(tree);
    it(`${code} has every English key`, () => {
      expect(Object.keys(E).filter((k) => !(k in L))).toEqual([]);
    });
    it(`${code} has no extra keys`, () => {
      expect(Object.keys(L).filter((k) => !(k in E))).toEqual([]);
    });
    it(`${code} keeps every {{placeholder}}`, () => {
      expect(Object.keys(E).filter((k) => JSON.stringify(vars(E[k])) !== JSON.stringify(vars(L[k] ?? "")))).toEqual([]);
    });
    it(`${code} is translated (not left in English)`, () => {
      const same = Object.keys(E).filter((k) => E[k].length > 25 && L[k] === E[k]);
      expect(same).toEqual([]);
    });
  }
  it("names all 36 districts in every language", () => {
    const d = Object.entries(districts).filter(([k]) => !k.startsWith("_")) as [string, Record<string, string>][];
    expect(d.length).toBe(36);
    const missing = d.flatMap(([k, v]) => LANG_CODES.filter((c) => c !== "en" && !v[c]).map((c) => `${k}:${c}`));
    expect(missing).toEqual([]);
  });
  it("covers every advice and flag key the backend can send", () => {
    const advice = ["falling", "rising", "uncertain", "deeper_than_usual", "rain_deficit", "rain_surplus", "extraction_high",
                    "reliability_limited", "no_groundwater_data", "contact_local", "forecast_outdated"];
    const flags = ["no_monitoring", "insufficient_data", "single_well", "extraction_critical", "extraction_semi_critical",
                   "forecast_decline", "deeper_than_usual", "rain_deficit", "reliability_limited"];
    expect(advice.filter((k) => !(`advice.${k}` in E))).toEqual([]);
    expect(flags.filter((k) => !(`gov.flag.${k}` in E))).toEqual([]);
  });
});
