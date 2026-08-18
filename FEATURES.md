# Feature inventory

| Capability | Standalone surface |
| --- | --- |
| Exact identifier recall before broad search | Bundled Python engine |
| ripgrep expansion, excludes, hit caps, and file-size cap | Bundled Python engine |
| CamelCase/snake_case tokenization and BM25 | All ranking modes |
| Symbol, path, density, and exact-identifier bonuses | `hybrid` and `symbols` modes |
| Tree-sitter with regex fallback | `_treesitter_chunks.py` and optional requirements |
| Python, JS/JSX, TS/TSX, Go, Rust, Java, and C/H definitions | Bundled detectors |
| Lexical, symbols, hybrid, and semantic modes | Node API, JSON Schema, and CLI |
| EmbeddingGemma → optional cross-encoder → hybrid fallback | `semantic` mode with explanatory `why` strings |
| Exact QMD-compatible query/document prefixes | Local embedding backend |
| HTTPS model download with SHA-256 verification and atomic installation | Configurable model cache |
| Explicit local model, offline mode, and download disable switch | Runtime options, CLI, and environment |
| Model-aware file-span and embedding cache | Per-root standalone cache |
| Multiple roots and longest-root cache ownership | Node API and repeatable CLI `--root` |
| Bounded output spans | 180-line maximum with dense-hit narrowing |
| Raw ripgrep trace | `rgTrace`, events, and CLI stderr |
| No-result/misuse/missing-rg exit semantics | Process wrapper and CLI |
| Cancellation, timeout, and bounded output | Runtime controls |
| Structured tool result and JSON Schema | SDK-neutral adapter surface |
| Scoped read registration | Awaited `onHits` callback with normalized spans |
| Enforced line/branch coverage and opt-in real-model tests | Development test commands |
