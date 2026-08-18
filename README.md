# agentic-semantic-search-tool

A standalone, harness-neutral codebase search tool for Node agent runtimes. It combines ripgrep recall, BM25 ranking, structural symbol chunks, and optional semantic reranking while returning bounded, source-readable spans.

This is the complete semantic-search stack extracted from myclaw/OpenClaw. It contains no OpenClaw plugin registration, session dependency, SDK dependency, or runtime cache from the source installation.

## Install

```bash
npm install agentic-semantic-search-tool
```

Requirements:

- Node.js 22 or newer
- Python 3.10 or newer
- [`ripgrep`](https://github.com/BurntSushi/ripgrep) (`rg`) on `PATH`

The core engine otherwise uses only Python's standard library. Optional structural grammars can be installed with:

```bash
python3 -m pip install -r node_modules/agentic-semantic-search-tool/python/requirements-treesitter.txt
```

Without them, the engine falls back to its bundled regex symbol detection. For local cross-encoder reranking:

```bash
python3 -m pip install -r node_modules/agentic-semantic-search-tool/python/requirements-semantic.txt
```

## Minimal Node usage

```js
import { createSemanticSearchRuntime } from "agentic-semantic-search-tool";

const search = createSemanticSearchRuntime({ cwd: process.cwd() });
const result = await search.search({
  query: "registerModeSystemTools",
  roots: ["src", "packages"],
  mode: "hybrid",
  limit: 10,
});

console.log(result.hits);
```

Each hit has a stable shape:

```ts
{
  file: string;
  lines: [start: number, end: number]; // 1-indexed, inclusive
  symbol: string | null;
  score: number;
  why: string;
}
```

Ranges are capped at 180 lines. Very large enclosing symbols are narrowed around the densest matching lines while the full symbol still participates in ranking.

## Generic agent tool

`createSemanticSearchTool` returns an SDK-independent function-tool object with a plain JSON Schema:

```js
import { createSemanticSearchTool } from "agentic-semantic-search-tool";

const tool = createSemanticSearchTool({ cwd: process.cwd() });
registerWithYourHarness({
  name: tool.name,
  description: tool.description,
  inputSchema: tool.parameters,
  execute: (args, context) => tool.execute(context.callId, args, context.signal),
});
```

The compatible tool name is `codebase_semantic_search`. Its inputs are `query`, `roots`, `limit`, `mode`, `noCache`, and `rgTrace`.

## Search modes

- `hybrid` (default): BM25 plus hit-density, symbol-name, path, and exact-identifier bonuses.
- `lexical`: pure BM25 over matched chunks.
- `symbols`: hybrid ranking, excluding hits that cannot be assigned to a symbol.
- `semantic`: hybrid candidate generation, then a cross-encoder rerank over the top 50. If unavailable, it tries Ollama cosine similarity; if that is unavailable, it returns the weighted hybrid order. The `why` field identifies the actual path used.

Recall runs exact, case-sensitive fixed-string searches for identifier-like terms before broad case-insensitive token recall. CamelCase and snake_case identifiers are split into component terms so queries can cross naming styles.

## Structural languages

Tree-sitter support is loaded lazily for Python, JavaScript, JSX, TypeScript, TSX, Go, Rust, Java, C, and C headers. It detects functions, methods, nested definitions, classes, interfaces, type aliases, enums, function-bound variables, Rust impl blocks, and common C definitions. C++, headers without a grammar, and installations without optional grammars use the regex fallback.

Parsed syntax trees use a bounded 256-entry, process-local cache. Search roots skip common generated/vendor locations, lock files, minified JavaScript, and files over 1.5 MB.

## Read-state integration

Search results often need to count as reads before an edit tool accepts changes. Use `onHits` to connect the packages without coupling them:

```js
import { createSemanticSearchRuntime } from "agentic-semantic-search-tool";
import { createFileToolsRuntime } from "agentic-file-tools";

const files = createFileToolsRuntime({ root: process.cwd() });
const search = createSemanticSearchRuntime({
  cwd: process.cwd(),
  onHits: (hits) => files.recordSearchHits(
    hits.map(({ filePath, lineSpan }) => ({ path: filePath, lineSpan })),
  ),
});
```

Callbacks are awaited before the search resolves, so the next tool call sees the registered spans. Callback errors emit `search.hit_callback_failed` and are non-fatal by default; set `strictHitCallbackErrors: true` to fail the search instead.

## CLI

```bash
npx agentic-semantic-search \
  --query "read before edit state" \
  --root ./src \
  --mode semantic \
  --limit 10 \
  --pretty
```

Repeat `--root` for multiple repositories. `--rg-trace` writes the raw recall hits to stderr. Exit codes match the engine: `0` results, `2` no results, `3` invalid input or missing roots, and `4` missing ripgrep.

## Configuration

Runtime options include:

- `cwd`, `pythonBin`, and `scriptPath`
- `timeoutMs` (default 60 seconds) and `AbortSignal` per invocation
- `cacheDirectory` (relative to every root)
- bounded `maxStdoutBytes` and `maxStderrBytes`
- `env`, `onEvent`, `onHits`, and `strictHitCallbackErrors`

Environment variables:

- `AGENTIC_SEMANTIC_SEARCH_PYTHON`: Python executable
- `AGENTIC_SEMANTIC_SEARCH_CACHE_DIR`: cache path relative to each search root (default `.agentic-semantic-search`)
- `AGENTIC_SEMANTIC_SEARCH_EMBED_MODEL`: Ollama embedding model (default `nomic-embed-text`)
- `AGENTIC_SEMANTIC_SEARCH_CROSS_ENCODER`: sentence-transformers model (default `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `OLLAMA_URL`: Ollama endpoint (default `http://127.0.0.1:11434`)

The legacy `CODEBASE_SEARCH_EMBED_MODEL` and `CODEBASE_SEARCH_CROSS_ENCODER` names remain accepted by the Python engine for migration compatibility.

## Cache behavior

Each root gets `.agentic-semantic-search/index-v1.json` by default. Entries record the root's git commit and invalidate individual files using size, nanosecond mtime, and SHA-1 content fingerprints. Symbol spans and successful Ollama chunk embeddings are cached. Use `noCache: true` or `--no-cache` to bypass all persistent cache reads and writes.

## Privacy and execution boundary

Lexical, BM25, symbol, and cache work is local. `semantic` mode may:

- send query/chunk text to the configured Ollama endpoint; and
- load or download the configured sentence-transformers model through that Python library.

Do not point `OLLAMA_URL` at a remote service unless sending source excerpts there is acceptable. The Node runtime uses argument-vector process spawning with no shell, enforces wall-clock and output limits, and terminates the engine on abort or timeout.

## What is intentionally excluded

The package excludes the OpenClaw extension manifest, plugin API adapter, per-session wiring, configuration schema, and existing `.openclaw` cache data. Their reusable behaviors are represented by the generic tool schema, structured details, events, and hit callback.
