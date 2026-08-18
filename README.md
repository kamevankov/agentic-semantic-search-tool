# agentic-semantic-search-tool

A standalone, harness-neutral code search tool for Node agent runtimes. It combines ripgrep recall, BM25 ranking, structural symbol chunks, and fully local semantic reranking while returning bounded, source-readable spans.

## Install

```bash
npm install agentic-semantic-search-tool
```

Requirements:

- Node.js 22 or newer
- Python 3.10 or newer
- [`ripgrep`](https://github.com/BurntSushi/ripgrep) (`rg`) on `PATH`

The default lexical and hybrid modes use only Python's standard library. Install optional tree-sitter grammars with:

```bash
python3 -m pip install -r node_modules/agentic-semantic-search-tool/python/requirements-treesitter.txt
```

Install local EmbeddingGemma support for `semantic` mode with:

```bash
python3 -m pip install -r node_modules/agentic-semantic-search-tool/python/requirements-semantic.txt
```

The first semantic search downloads `embeddinggemma-300M-Q8_0.gguf` from the [`ggml-org/embeddinggemma-300M-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) repository on Hugging Face. Inference runs locally through `llama-cpp-python`; no query or source text is sent to an embedding service.

An optional cross-encoder can refine the embedding-ranked candidates:

```bash
python3 -m pip install -r node_modules/agentic-semantic-search-tool/python/requirements-cross-encoder.txt
```

## Node API

```js
import { createSemanticSearchRuntime } from "agentic-semantic-search-tool";

const search = createSemanticSearchRuntime({ cwd: process.cwd() });
const result = await search.search({
  query: "registerModeSystemTools",
  roots: ["src", "packages"],
  mode: "semantic",
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

`createSemanticSearchTool` returns an SDK-independent function-tool object with plain JSON Schema:

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

The tool name is `codebase_semantic_search`. Inputs are `query`, `roots`, `limit`, `mode`, `noCache`, and `rgTrace`.

## Search modes

- `hybrid` (default): BM25 plus hit-density, symbol-name, path, and exact-identifier bonuses.
- `lexical`: pure BM25 over matched chunks.
- `symbols`: hybrid ranking, excluding hits that cannot be assigned to a symbol.
- `semantic`: hybrid candidate generation over the top 50, local EmbeddingGemma cosine ranking, then an optional cross-encoder. Missing optional components degrade cleanly to the remaining stages and finally to hybrid order. The `why` field records the path used.

Exact, case-sensitive fixed-string recall for identifier-like terms runs before broad case-insensitive token recall. CamelCase and snake_case identifiers are split into components so queries can cross naming styles.

EmbeddingGemma receives QMD-compatible prefixes:

```text
task: search result | query: <query>
title: <symbol-or-filename> | text: <source chunk>
```

## Structural languages

Tree-sitter support is loaded lazily for Python, JavaScript, JSX, TypeScript, TSX, Go, Rust, Java, C, and C headers. It detects functions, methods, nested definitions, classes, interfaces, type aliases, enums, function-bound variables, Rust impl blocks, and common C definitions. C++, headers without a grammar, and installations without optional grammars use the bundled regex fallback.

Parsed trees use a bounded 256-entry process-local cache. Searches skip common generated/vendor locations, lock files, minified JavaScript, and files over 1.5 MB.

## Read-state integration

Use `onHits` to connect search results to a file-tool runtime without coupling either package:

```js
const search = createSemanticSearchRuntime({
  cwd: process.cwd(),
  onHits: (hits) => files.recordSearchHits(
    hits.map(({ filePath, lineSpan }) => ({ path: filePath, lineSpan })),
  ),
});
```

Callbacks are awaited before search resolves. Callback errors emit `search.hit_callback_failed` and are non-fatal by default; set `strictHitCallbackErrors: true` to fail instead.

## CLI

```bash
npx agentic-semantic-search \
  --query "read before edit state" \
  --root ./src \
  --mode semantic \
  --limit 10 \
  --pretty
```

Repeat `--root` for multiple repositories. `--rg-trace` writes raw recall hits to stderr. Exit codes are `0` for results, `2` for no results, `3` for invalid input or missing roots, and `4` when ripgrep is unavailable.

Model controls are also available through `--embedding-model`, `--model-cache`, `--offline`, `--no-model-download`, and `--gpu-layers`.

## Runtime and model configuration

Runtime options include:

- `cwd`, `pythonBin`, and `scriptPath`
- `timeoutMs` (60 seconds by default) and a per-invocation `AbortSignal`
- `cacheDirectory` (relative to every search root)
- `embeddingModelPath`, `modelCacheDirectory`, `offline`, `autoDownload`, and `gpuLayers`
- bounded `maxStdoutBytes` and `maxStderrBytes`
- `env`, `onEvent`, `onHits`, and `strictHitCallbackErrors`

Environment variables:

- `AGENTIC_SEMANTIC_SEARCH_PYTHON`: Python executable
- `AGENTIC_SEMANTIC_SEARCH_CACHE_DIR`: per-root cache path (default `.agentic-semantic-search`)
- `AGENTIC_SEMANTIC_SEARCH_EMBED_MODEL_PATH`: existing local GGUF file
- `AGENTIC_SEMANTIC_SEARCH_MODEL_CACHE`: model directory (default `~/.cache/agentic-semantic-search/models`)
- `AGENTIC_SEMANTIC_SEARCH_AUTO_DOWNLOAD`: set to `0` to disable downloads
- `AGENTIC_SEMANTIC_SEARCH_OFFLINE`: set to `1` to prohibit downloads
- `AGENTIC_SEMANTIC_SEARCH_GPU_LAYERS`: llama.cpp GPU layer count (`-1` for all)
- `AGENTIC_SEMANTIC_SEARCH_DOWNLOAD_TIMEOUT`: model-download timeout in seconds
- `AGENTIC_SEMANTIC_SEARCH_EMBED_REPO`, `AGENTIC_SEMANTIC_SEARCH_EMBED_FILE`, and `AGENTIC_SEMANTIC_SEARCH_EMBED_URL`: advanced artifact overrides
- `AGENTIC_SEMANTIC_SEARCH_EMBED_SHA256`: expected model digest; defaults to the pinned artifact's published LFS SHA-256
- `AGENTIC_SEMANTIC_SEARCH_CROSS_ENCODER`: optional sentence-transformers model

If the initial model download takes longer than the runtime's default timeout, set a larger `timeoutMs` for that invocation or download the GGUF separately and provide `embeddingModelPath`. Offline mode never performs a network request.

## Cache and privacy

Each search root gets `.agentic-semantic-search/index-v2.json` by default. File entries invalidate on size, nanosecond mtime, and SHA-1 content fingerprint. Chunk embeddings are additionally keyed by model identity, so changing the GGUF cannot reuse incompatible vectors. Set `noCache: true` or `--no-cache` to bypass persistent reads and writes.

Search, parsing, ranking, and inference are local. The only built-in network operation is downloading a missing GGUF over HTTPS from its configured model URL. The Node runtime spawns Python without a shell, bounds stdout/stderr, enforces a wall-clock timeout, and terminates the process on abort or timeout.

## Development

```bash
npm test
npm pack --dry-run
```

The test suite covers the Node API and CLI, process cancellation and bounds, Python ranking modes, model resolution and atomic downloads, embedding normalization and caching, optional-backend fallbacks, and package-content scrubbing.
