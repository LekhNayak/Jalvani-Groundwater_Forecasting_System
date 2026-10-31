import { describe, expect, it } from "vitest";
import { changeDir, depth, fmtDate, fmtNum, slugify } from "./format";

describe("format", () => {
  it("turns a level (negative = below ground) into a positive depth", () => {
    expect(depth(-5.28)).toBeCloseTo(5.28);
    expect(depth(null)).toBeNull();
  });
  it("calls small changes steady and keeps the sign meaning (positive = water rose)", () => {
    expect(changeDir(0.05)).toBe("steady");
    expect(changeDir(0.4)).toBe("rise");
    expect(changeDir(-0.4)).toBe("fall");
    expect(changeDir(null)).toBeNull();
  });
  it("formats numbers with Latin digits in every language", () => {
    expect(fmtNum(5.284, 1, "en")).toBe("5.3");
    expect(fmtNum(5.284, 1, "mr")).toMatch(/^5[.,]3$/);
    expect(fmtNum(null)).toBe("–");
  });
  it("formats dates per language", () => {
    expect(fmtDate("2026-06-04", "en")).toMatch(/2026/);
    expect(fmtDate(null, "hi")).toBe("–");
  });
  it("matches the backend slugs", () => {
    expect(slugify("Mumbai Suburban")).toBe("mumbai_suburban");
  });
});
