import { describe, expect, it } from "vitest";
import en from "./en.json";
import hi from "./hi.json";
import mr from "./mr.json";
import districts from "./districts.json";

type Tree = { [k: string]: string | Tree };

function flatten(t: Tree, prefix = ""): Record<string, string> {
  return Object.entries(t).reduce((acc, [k, v]) => {
    const key = prefix ? `${prefix}.${k}` : k;
    return typeof v === "string" ? { ...acc, [key]: v } : { ...acc, ...flatten(v, key) };
  }, {} as Record<string, string>);
}
const vars = (s: string) => (s.match(/\{\{\s*\w+\s*\}\}/g) ?? []).map((x) => x.replace(/\s/g, "")).sort();

describe("translations", () => {
  const E = flatten(en as Tree);
  for (const [name, tree] of [["mr", mr], ["hi", hi]] as const) {
    const L = flatten(tree as Tree);
    it(`${name} has every English key`, () => {
      expect(Object.keys(E).filter((k) => !(k in L))).toEqual([]);
    });
    it(`${name} has no extra keys`, () => {
      expect(Object.keys(L).filter((k) => !(k in E))).toEqual([]);
    });
    it(`${name} keeps every {{placeholder}}`, () => {
      expect(Object.keys(E).filter((k) => JSON.stringify(vars(E[k])) !== JSON.stringify(vars(L[k] ?? "")))).toEqual([]);
    });
  }
  it("names all 36 districts in Marathi and Hindi", () => {
    const d = Object.entries(districts).filter(([k]) => !k.startsWith("_")) as [string, { mr: string; hi: string }][];
    expect(d.length).toBe(36);
    expect(d.filter(([, v]) => !v.mr || !v.hi)).toEqual([]);
  });
  it("covers every advice and flag key the backend can send", () => {
    const advice = ["falling", "rising", "uncertain", "deeper_than_usual", "rain_deficit", "rain_surplus", "extraction_high",
                    "reliability_limited", "no_groundwater_data", "contact_local"];
    const flags = ["no_monitoring", "insufficient_data", "single_well", "extraction_critical", "extraction_semi_critical",
                   "forecast_decline", "deeper_than_usual", "rain_deficit", "reliability_limited"];
    expect(advice.filter((k) => !(`advice.${k}` in E))).toEqual([]);
    expect(flags.filter((k) => !(`gov.flag.${k}` in E))).toEqual([]);
  });
});
