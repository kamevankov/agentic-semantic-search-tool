import { spawn } from "node:child_process";
import { access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { SearchMode, SemanticSearchHit } from "./contracts.js";
import { SemanticSearchError } from "./contracts.js";

const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_MAX_STDOUT_BYTES = 8 * 1024 * 1024;
const DEFAULT_MAX_STDERR_BYTES = 1024 * 1024;

export function bundledScriptPath(): string {
  return fileURLToPath(new URL("../python/agentic_semantic_search.py", import.meta.url));
}

export async function resolvePythonBin(configured?: string, env: NodeJS.ProcessEnv = process.env): Promise<string> {
  const selected = configured?.trim() || env.AGENTIC_SEMANTIC_SEARCH_PYTHON?.trim();
  if (selected) return selected;
  const homebrewPython = "/opt/homebrew/bin/python3";
  try {
    await access(homebrewPython);
    return homebrewPython;
  } catch {
    return "python3";
  }
}

interface ProcessInput {
  query: string;
  roots: string[];
  limit: number;
  mode: SearchMode;
  noCache: boolean;
  rgTrace: boolean;
}

interface ProcessOptions {
  cwd: string;
  pythonBin: string;
  scriptPath: string;
  timeoutMs?: number;
  env?: NodeJS.ProcessEnv;
  cacheDirectory?: string;
  maxStdoutBytes?: number;
  maxStderrBytes?: number;
  signal?: AbortSignal;
  onStderr?: (text: string) => void;
}

export interface ProcessResult {
  exitCode: 0 | 2;
  hits: SemanticSearchHit[];
  stderr: string;
  durationMs: number;
}

function parseHit(value: unknown): SemanticSearchHit | undefined {
  if (!value || typeof value !== "object") return undefined;
  const raw = value as Record<string, unknown>;
  if (typeof raw.file !== "string" || !raw.file.trim()) return undefined;
  if (!Array.isArray(raw.lines) || raw.lines.length < 2) return undefined;
  const start = Number(raw.lines[0]);
  const end = Number(raw.lines[1]);
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start) return undefined;
  const score = Number(raw.score);
  if (!Number.isFinite(score)) return undefined;
  const symbol = typeof raw.symbol === "string" ? raw.symbol : null;
  const why = typeof raw.why === "string" ? raw.why : JSON.stringify(raw.why ?? "");
  return { file: raw.file, lines: [start, end], symbol, score, why };
}

function parseHits(stdout: string): SemanticSearchHit[] {
  if (!stdout.trim()) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(stdout);
  } catch (error) {
    throw new SemanticSearchError("INVALID_OUTPUT", `Search engine returned invalid JSON: ${(error as Error).message}`, {
      stdoutPreview: stdout.slice(0, 2000),
    });
  }
  if (!Array.isArray(parsed)) {
    throw new SemanticSearchError("INVALID_OUTPUT", "Search engine JSON must be an array", {
      stdoutPreview: stdout.slice(0, 2000),
    });
  }
  const hits = parsed.map(parseHit);
  if (hits.some((hit) => hit === undefined)) {
    throw new SemanticSearchError("INVALID_OUTPUT", "Search engine returned a malformed hit", {
      stdoutPreview: stdout.slice(0, 2000),
    });
  }
  return hits as SemanticSearchHit[];
}

export async function runSearchProcess(input: ProcessInput, options: ProcessOptions): Promise<ProcessResult> {
  if (options.signal?.aborted) throw new SemanticSearchError("ABORTED", "Search was aborted before launch");
  try {
    await access(options.scriptPath);
  } catch {
    throw new SemanticSearchError("SCRIPT_NOT_FOUND", `Semantic search engine not found at ${options.scriptPath}`, {
      scriptPath: options.scriptPath,
    });
  }

  const args = [
    options.scriptPath,
    "--query", input.query,
    "--limit", String(input.limit),
    "--mode", input.mode,
  ];
  for (const root of input.roots) args.push("--root", root);
  if (input.noCache) args.push("--no-cache");
  if (input.rgTrace) args.push("--rg-trace");

  const env: NodeJS.ProcessEnv = { ...process.env, ...options.env };
  if (options.cacheDirectory) env.AGENTIC_SEMANTIC_SEARCH_CACHE_DIR = options.cacheDirectory;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxStdoutBytes = options.maxStdoutBytes ?? DEFAULT_MAX_STDOUT_BYTES;
  const maxStderrBytes = options.maxStderrBytes ?? DEFAULT_MAX_STDERR_BYTES;
  const startedAt = Date.now();

  return await new Promise<ProcessResult>((resolve, reject) => {
    const child = spawn(options.pythonBin, args, {
      cwd: path.resolve(options.cwd),
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let termination: "timeout" | "abort" | "output" | undefined;
    let settled = false;

    const finishReject = (error: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };
    const kill = () => {
      try { child.kill("SIGKILL"); } catch { /* best effort */ }
    };
    const timer = setTimeout(() => {
      termination = "timeout";
      kill();
    }, timeoutMs);
    const onAbort = () => {
      termination = "abort";
      kill();
    };
    const cleanup = () => {
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", onAbort);
    };
    options.signal?.addEventListener("abort", onAbort, { once: true });

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
      if (Buffer.byteLength(stdout, "utf8") > maxStdoutBytes && !termination) {
        termination = "output";
        kill();
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stderr += text;
      try { options.onStderr?.(text); } catch { /* telemetry is isolated */ }
      if (Buffer.byteLength(stderr, "utf8") > maxStderrBytes && !termination) {
        termination = "output";
        kill();
      }
    });
    child.on("error", (error: Error) => {
      finishReject(new SemanticSearchError("SPAWN_FAILED", `Failed to start ${options.pythonBin}: ${error.message}`, {
        pythonBin: options.pythonBin,
      }));
    });
    child.on("close", (code: number | null) => {
      if (settled) return;
      cleanup();
      if (termination === "abort") {
        finishReject(new SemanticSearchError("ABORTED", "Search was aborted"));
        return;
      }
      if (termination === "timeout") {
        finishReject(new SemanticSearchError("TIMED_OUT", `Search timed out after ${timeoutMs}ms`, { timeoutMs }));
        return;
      }
      if (termination === "output") {
        finishReject(new SemanticSearchError("OUTPUT_LIMIT", "Search engine exceeded its bounded output limit", {
          maxStdoutBytes,
          maxStderrBytes,
        }));
        return;
      }
      if (code !== 0 && code !== 2) {
        finishReject(new SemanticSearchError("PROCESS_FAILED", `Semantic search failed with exit code ${code ?? "unknown"}`, {
          exitCode: code,
          stderrTail: stderr.slice(-1000),
        }));
        return;
      }
      try {
        const hits = parseHits(stdout);
        settled = true;
        resolve({
          exitCode: code,
          hits,
          stderr,
          durationMs: Date.now() - startedAt,
        });
      } catch (error) {
        finishReject(error as Error);
      }
    });
  });
}
