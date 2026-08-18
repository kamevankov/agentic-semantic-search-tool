# Feature inventory

This inventory maps the extracted source stack to its standalone surface.

| Capability | Standalone surface |
| --- | --- |
| Exact identifier recall before broad search | Bundled Python engine |
| ripgrep token expansion, excludes, hit caps, 1.5 MB file cap | Bundled Python engine |
| CamelCase/snake_case tokenization and BM25 | Bundled Python engine |
| Symbol/path/density/exact-identifier bonuses | `hybrid` and `symbols` modes |
| Tree-sitter with regex fallback | `_treesitter_chunks.py` plus optional requirements |
| Python, JS/JSX, TS/TSX, Go, Rust, Java, C/H definitions | Bundled detectors |
| Lexical, symbols, hybrid, semantic modes | Node API, JSON Schema, and CLI |
| Cross-encoder → Ollama → hybrid fallback chain | `semantic` mode with explanatory `why` strings |
| Persistent file-span and embedding cache | Per-root standalone cache; configurable/bypassable |
| Multiple roots and longest-root cache ownership | Node API and repeatable CLI `--root` |
| Bounded output spans | 180-line maximum with dense-hit narrowing |
| Raw ripgrep trace | `rgTrace`, `search.stderr`, and CLI stderr |
| No-result/misuse/missing-rg exit semantics | Process wrapper and CLI |
| Python preference and overrides | Homebrew preference, option, and environment variable |
| 60-second default timeout | Configurable runtime timeout |
| Cancellation | Per-call `AbortSignal` |
| Bounded stdout/stderr | Configurable byte ceilings |
| Structured tool result | Plain SDK-neutral result and JSON Schema |
| Scoped read registration | Awaited `onHits` callback with normalized line spans |
| OpenClaw extension registration | Excluded; harness adapter is intentionally generic |
| Existing runtime indexes | Excluded; caches are generated locally by consumers |
