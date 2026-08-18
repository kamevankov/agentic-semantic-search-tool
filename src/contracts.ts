export const SEARCH_MODES = ["hybrid", "lexical", "symbols", "semantic"] as const;
export type SearchMode = (typeof SEARCH_MODES)[number];

export type SearchErrorCode =
  | "INVALID_INPUT"
  | "SCRIPT_NOT_FOUND"
  | "SPAWN_FAILED"
  | "PROCESS_FAILED"
  | "INVALID_OUTPUT"
  | "OUTPUT_LIMIT"
  | "TIMED_OUT"
  | "ABORTED"
  | "HIT_CALLBACK_FAILED";

export class SemanticSearchError extends Error {
  constructor(
    public readonly code: SearchErrorCode,
    message: string,
    public readonly details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "SemanticSearchError";
  }
}

export interface SemanticSearchInput {
  query: string;
  roots?: string[];
  limit?: number;
  mode?: SearchMode;
  noCache?: boolean;
  rgTrace?: boolean;
}

export interface SemanticSearchHit {
  file: string;
  lines: [number, number];
  symbol: string | null;
  score: number;
  why: string;
}

export interface SearchReadHit {
  filePath: string;
  lineSpan: { start: number; end: number };
}

export interface SemanticSearchDetails {
  status: "completed";
  ok: true;
  exitCode: 0 | 2;
  query: string;
  roots: string[];
  limit: number;
  mode: SearchMode;
  noCache: boolean;
  rgTrace: boolean;
  pythonBin: string;
  scriptPath: string;
  durationMs: number;
  stderrTail: string;
  hits: SemanticSearchHit[];
}

export interface SemanticSearchResponse {
  hits: SemanticSearchHit[];
  stderr: string;
  details: SemanticSearchDetails;
}

export interface TextContent { type: "text"; text: string }
export interface SemanticSearchToolResult {
  content: TextContent[];
  details: SemanticSearchDetails;
}

export type SemanticSearchEvent =
  | { type: "search.started"; query: string; roots: string[]; mode: SearchMode; at: number }
  | { type: "search.stderr"; text: string; at: number }
  | { type: "search.hits"; hits: SemanticSearchHit[]; at: number }
  | { type: "search.hit_callback_failed"; message: string; at: number }
  | { type: "search.completed"; hitCount: number; exitCode: 0 | 2; durationMs: number; at: number }
  | { type: "search.failed"; code: SearchErrorCode; message: string; at: number };

export interface SemanticSearchExecutionContext {
  signal?: AbortSignal;
  invocationId?: string;
  onEvent?: (event: SemanticSearchEvent) => void;
  onHits?: (hits: SearchReadHit[], response: SemanticSearchResponse) => void | Promise<void>;
}

export interface SemanticSearchRuntimeOptions {
  cwd?: string;
  pythonBin?: string;
  scriptPath?: string;
  timeoutMs?: number;
  env?: NodeJS.ProcessEnv;
  cacheDirectory?: string;
  maxStdoutBytes?: number;
  maxStderrBytes?: number;
  strictHitCallbackErrors?: boolean;
  onEvent?: (event: SemanticSearchEvent) => void;
  onHits?: (hits: SearchReadHit[], response: SemanticSearchResponse) => void | Promise<void>;
  now?: () => number;
}

export interface GenericSemanticSearchTool {
  name: "codebase_semantic_search";
  label: string;
  description: string;
  parameters: Record<string, unknown>;
  execute(
    toolCallId: string,
    input: SemanticSearchInput | Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<SemanticSearchToolResult>;
}

export interface SemanticSearchRuntime {
  readonly tool: GenericSemanticSearchTool;
  search(
    input: SemanticSearchInput | Record<string, unknown>,
    context?: SemanticSearchExecutionContext,
  ): Promise<SemanticSearchResponse>;
}
