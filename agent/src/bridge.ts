import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { createInterface, Interface } from "node:readline";

export type HistoryRow = { guess: string; feedback: string | number };

type BridgeResponse<T> = {
  id: number;
  ok: boolean;
  result?: T;
  error?: string;
};

export class WordleBridge {
  private readonly process: ChildProcessWithoutNullStreams;
  private readonly lines: Interface;
  private nextId = 1;
  private readonly pending = new Map<
    number,
    { resolve: (value: unknown) => void; reject: (reason: Error) => void }
  >();

  constructor() {
    this.process = spawn("uv", ["run", "python", "-m", "wordle_slm.bridge"], {
      cwd: process.cwd(),
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.lines = createInterface({ input: this.process.stdout });
    this.lines.on("line", (line) => {
      const response = JSON.parse(line) as BridgeResponse<unknown>;
      const waiter = this.pending.get(response.id);
      if (!waiter) return;
      this.pending.delete(response.id);
      if (response.ok) waiter.resolve(response.result);
      else waiter.reject(new Error(response.error ?? "unknown Wordle bridge error"));
    });
    this.process.stderr.on("data", (chunk) => process.stderr.write(chunk));
    this.process.on("exit", (code) => {
      for (const waiter of this.pending.values()) {
        waiter.reject(new Error(`Wordle bridge exited with code ${code}`));
      }
      this.pending.clear();
    });
  }

  request<T>(op: string, payload: Record<string, unknown>): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: (value) => resolve(value as T),
        reject,
      });
      this.process.stdin.write(`${JSON.stringify({ id, op, ...payload })}\n`);
    });
  }

  feedback(target: string, guess: string) {
    return this.request<{ code: number; feedback: string }>("feedback", { target, guess });
  }

  validateWord(word: string) {
    return this.request<{ valid: boolean; word: string | null }>("validate_word", { word });
  }

  candidates(history: HistoryRow[]) {
    return this.request<{ count: number; candidates: string[] }>("get_candidates", { history });
  }

  bestGuess(history: HistoryRow[]) {
    return this.request<{ guess: string }>("best_guess", { history });
  }

  close() {
    this.lines.close();
    this.process.stdin.end();
    this.process.kill();
  }
}
