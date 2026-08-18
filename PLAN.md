# Extraction plan

1. Inventory the OpenClaw wrapper, Python search engine, tree-sitter layer, persistent cache, optional rerankers, read-state integration, and tests.
2. Preserve all reusable search and ranking behavior inside bundled package assets.
3. Replace OpenClaw-specific registration and session wiring with a Node runtime, JSON Schema tool, CLI, events, cancellation, and hit callback.
4. Move cache defaults to standalone naming while retaining migration aliases for model configuration.
5. Port the complete Python suite and add Node API, tool, callback, cancellation, error, cache, and CLI tests.
6. Verify TypeScript build, both suites, tarball contents, and clean consumer installation.
7. Publish only this folder to a dedicated public GitHub repository and npm as version 0.1.0.
