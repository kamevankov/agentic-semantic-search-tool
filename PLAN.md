# Standalone package plan

1. Preserve lexical recall, BM25, structural chunking, multiple roots, cache behavior, bounded output, and result explanations.
2. Expose the engine through a typed Node runtime, generic JSON Schema tool, CLI, events, cancellation, and an awaited search-hit callback.
3. Run the exact `embeddinggemma-300M-Q8_0.gguf` artifact locally through `llama-cpp-python`, using QMD-compatible query and document formatting.
4. Resolve models from an explicit path or configurable cache, download missing artifacts over HTTPS with atomic installation, and support strict offline/no-download operation.
5. Key cached vectors by file fingerprint, span, and model identity; preserve the optional cross-encoder and deterministic hybrid fallback.
6. Test Python ranking, model resolution/download, normalization, cache invalidation, fallbacks, Node runtime configuration, CLI behavior, cancellation, output bounds, and package-content scrubbing.
7. Verify the TypeScript build, all test suites, npm tarball contents, and a clean consumer install before publishing.
