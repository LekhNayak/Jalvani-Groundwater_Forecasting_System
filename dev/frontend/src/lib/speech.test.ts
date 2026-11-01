import { describe, expect, it } from "vitest";
import { splitForSpeech } from "./speech";

describe("splitForSpeech", () => {
  it("keeps short text whole", () => {
    expect(splitForSpeech("7.7 metres below ground")).toEqual(["7.7 metres below ground"]);
  });
  it("never splits a decimal number", () => {
    const parts = splitForSpeech("Water is 7.1 m below ground. It rose 0.4 m in 30 days.");
    expect(parts).toEqual(["Water is 7.1 m below ground.", "It rose 0.4 m in 30 days."]);
  });
  it("splits a long sentence at commas, with a short first piece", () => {
    const t = "In 30 days: about 6.8 m below ground, most likely 6.2 to 7.3 m, could be 6.0 to 8.0 m, could go either way. In 7 days: about 7.1 m.";
    const parts = splitForSpeech(t);
    expect(parts.length).toBeGreaterThan(2);
    expect(parts[0].length).toBeLessThanOrEqual(55);
    expect(parts.join(" ")).toBe(t);                        // nothing lost, nothing added
  });
  it("handles Devanagari and Gujarati sentence ends", () => {
    expect(splitForSpeech("पाणी वर आले। पाणी खाली गेले।").length).toBe(2);
    expect(splitForSpeech("પાણી ઉપર આવ્યું. પાણી નીચે ગયું.").length).toBe(2);
  });
  it("never returns an empty list", () => {
    expect(splitForSpeech("")).toEqual([""]);
  });
});
