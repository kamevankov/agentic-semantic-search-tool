import path from "node:path";
import type {
  GenericSemanticSearchTool,
  SearchMode,
  SearchReadHit,
  SemanticSearchEvent,
  SemanticSearchExecutionContext,
  SemanticSearchHit,
  SemanticSearchInput,
  SemanticSearchResponse,
  SemanticSearchRuntime,
  SemanticSearchRuntimeOptions,
} from "./contracts.js";
import { SEARCH_MODES, SemanticSearchError } from "./contracts.js";
import { bundledScriptPath, resolvePythonBin, runSearchProcess } from "./process.js";
import { semanticSearchSchema } from "./schemas.js";

interface NormalizedInput {
  query: string;
  roots: string[];
  limit: number;
  mode: SearchMode;
  noCache: boolean;
  rgTrace: boolean;
}

function normalizeInput(raw: SemanticSearchInput | Record<string, unknown>, cwd: string): NormalizedInput {
  const query = typeof raw.query === "string" ? raw.query.trim() : "";
  if (!query) throw new SemanticSearchError("INVALID_INPUT", "query is required and must be a non-empty string");
  const rawRoots = Array.isArray(raw.roots) ? raw.roots : [];
  const roots = rawRoots
    .filter((root): root is string => typeof root === "string" && root.trim().length > 0)
    .map((root) => path.resolve(cwd, root.trim()));
  if (roots.length === 0) roots.push(path.resolve(cwd));
  const uniqueRoots = [...new Set(roots)];
  const limit = typeof raw.limit === "number" && Number.isInteger(raw.limit)
    ? Math.min(200, Math.max(1, raw.limit))
    : 20;
  const mode = typeof raw.mode === "string" && (SEARCH_MODES as readonly string[]).includes(raw.mode)
    ? raw.mode as SearchMode
    : "hybrid";
  return {
    query,
    roots: uniqueRoots,
    limit,
    mode,
    noCache: raw.noCache === true,
    rgTrace: raw.rgTrace === true,
  };
}

export function extractSearchReadHits(hits: readonly SemanticSearchHit[], root = process.cwd()): SearchReadHit[] {
  const out: SearchReadHit[] = [];
  const seen = new Set<string>();
  for (const hit of hits) {
    const [start, end] = hit.lines;
    if (!hit.file || start < 1 || end < start) continue;
    const filePath = path.isAbsolute(hit.file) ? path.resolve(hit.file) : path.resolve(root, hit.file);
    const key = `${filePath}:${start}:${end}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ filePath, lineSpan: { start, end } });
  }
  return out;
}

function errorCode(error: unknown): SemanticSearchError["code"] {
  return error instanceof SemanticSearchError ? error.code : "PROCESS_FAILED";
}

export function createSemanticSearchRuntime(options: SemanticSearchRuntimeOptions = {}): SemanticSearchRuntime {
  const cwd = path.resolve(options.cwd ?? process.cwd());
  if (options.gpuLayers !== undefined && (!Number.isInteger(options.gpuLayers) || options.gpuLayers < -1)) {
    throw new SemanticSearchError("INVALID_INPUT", "gpuLayers must be an integer of -1 or greater");
  }
  const now = options.now ?? Date.now;
  const emit = (event: SemanticSearchEvent, context?: SemanticSearchExecutionContext) => {
    try { options.onEvent?.(event); } catch { /* telemetry is isolated */ }
    if (context?.onEvent !== options.onEvent) {
      try { context?.onEvent?.(event); } catch { /* telemetry is isolated */ }
    }
  };

  const search = async (
    raw: SemanticSearchInput | Record<string, unknown>,
    context: SemanticSearchExecutionContext = {},
  ): Promise<SemanticSearchResponse> => {
    let normalized: NormalizedInput;
    try {
      normalized = normalizeInput(raw, cwd);
      if (options.cacheDirectory) {
        const cacheDirectory = path.normalize(options.cacheDirectory.trim());
        if (
          !cacheDirectory ||
          path.isAbsolute(cacheDirectory) ||
          cacheDirectory === ".." ||
          cacheDirectory.startsWith(`..${path.sep}`)
        ) {
          throw new SemanticSearchError(
            "INVALID_INPUT",
            "cacheDirectory must be a non-empty path relative to each search root",
          );
        }
      }
    } catch (error) {
      emit({ type: "search.failed", code: errorCode(error), message: (error as Error).message, at: now() }, context);
      throw error;
    }
    emit({
      type: "search.started",
      query: normalized.query,
      roots: normalized.roots,
      mode: normalized.mode,
      at: now(),
    }, context);
    try {
      const pythonBin = await resolvePythonBin(options.pythonBin, { ...process.env, ...options.env });
      const scriptPath = options.scriptPath ?? bundledScriptPath();
      const result = await runSearchProcess(normalized, {
        cwd,
        pythonBin,
        scriptPath,
        timeoutMs: options.timeoutMs,
        env: options.env,
        cacheDirectory: options.cacheDirectory,
        embeddingModelPath: options.embeddingModelPath,
        modelCacheDirectory: options.modelCacheDirectory,
        offline: options.offline,
        autoDownload: options.autoDownload,
        gpuLayers: options.gpuLayers,
        maxStdoutBytes: options.maxStdoutBytes,
        maxStderrBytes: options.maxStderrBytes,
        signal: context.signal,
        onStderr: (text) => emit({ type: "search.stderr", text, at: now() }, context),
      });
      const response: SemanticSearchResponse = {
        hits: result.hits,
        stderr: result.stderr,
        details: {
          status: "completed",
          ok: true,
          exitCode: result.exitCode,
          query: normalized.query,
          roots: normalized.roots,
          limit: normalized.limit,
          mode: normalized.mode,
          noCache: normalized.noCache,
          rgTrace: normalized.rgTrace,
          pythonBin,
          scriptPath,
          durationMs: result.durationMs,
          stderrTail: result.stderr.slice(-1000),
          hits: result.hits,
        },
      };
      emit({ type: "search.hits", hits: result.hits, at: now() }, context);
      const readHits = extractSearchReadHits(result.hits, normalized.roots[0]);
      const callbacks = [options.onHits, context.onHits].filter(
        (callback, index, all): callback is NonNullable<typeof callback> => Boolean(callback) && all.indexOf(callback) === index,
      );
      for (const callback of callbacks) {
        try {
          await callback(readHits, response);
        } catch (error) {
          const message = (error as Error)?.message ?? String(error);
          emit({ type: "search.hit_callback_failed", message, at: now() }, context);
          if (options.strictHitCallbackErrors) {
            throw new SemanticSearchError("HIT_CALLBACK_FAILED", `Search hit callback failed: ${message}`);
          }
        }
      }
      emit({
        type: "search.completed",
        hitCount: result.hits.length,
        exitCode: result.exitCode,
        durationMs: result.durationMs,
        at: now(),
      }, context);
      return response;
    } catch (error) {
      emit({ type: "search.failed", code: errorCode(error), message: (error as Error).message, at: now() }, context);
      throw error;
    }
  };

  const tool: GenericSemanticSearchTool = {
    name: "codebase_semantic_search",
    label: "Codebase Semantic Search",
    description: "Hybrid lexical, structural, and optional semantic codebase search. Returns ranked {file, lines:[start,end], symbol, score, why} hits across one or more roots.",
    parameters: semanticSearchSchema,
    execute: async (_toolCallId, input, signal) => {
      const response = await search(input, { signal });
      return {
        content: [{ type: "text", text: JSON.stringify(response.details, null, 2) }],
        details: response.details,
      };
    },
  };

  return { tool, search };
}

export function createSemanticSearchTool(options: SemanticSearchRuntimeOptions = {}): GenericSemanticSearchTool {
  return createSemanticSearchRuntime(options).tool;
}
