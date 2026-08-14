import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { WordleBridge } from "../src/bridge.js";

describe("Python JSONL bridge", () => {
  let bridge: WordleBridge;
  beforeAll(() => {
    bridge = new WordleBridge();
  });
  afterAll(() => bridge.close());

  it("matches duplicate-letter feedback", async () => {
    const result = await bridge.feedback("cacao", "anana");
    expect(result.feedback).toBe("10100");
  });

  it("filters candidates and returns a deterministic best guess", async () => {
    const result = await bridge.candidates([{ guess: "audio", feedback: "00000" }]);
    expect(result.candidates.every((word) => !/[audio]/.test(word))).toBe(true);
    const best = await bridge.bestGuess([{ guess: "audio", feedback: "00000" }]);
    expect(best.guess).toMatch(/^[a-zñ]{5}$/u);
  });
});
