import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, readdir, realpath, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  SemanticSearchError,
  createSemanticSearchRuntime,
  createSemanticSearchTool,
  extractSearchReadHits,
  semanticSearchSchema,
} from "../dist/index.js";

async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), "agentic-semantic-search-"));
  await mkdir(path.join(root, "src"));
  const target = path.join(root, "src", "mode-system-tools.ts");
  await writeFile(target, [
    "export function registerModeSystemTools(api: unknown) {",
    "  return api;",
    "}",
  ].join("\n"));
  await writeFile(path.join(root, "src", "noise.ts"), Array.from(
    { length: 80 },
    (_, index) => `export type ModeSystemNoise${index} = { mode: string };`,
  ).join("\n"));
  return { root, target };
}

test("exports a plain JSON Schema and compatible tool identity", () => {
  assert.equal(semanticSearchSchema.type, "object");
  assert.deepEqual(semanticSearchSchema.required, ["query"]);
  const tool = createSemanticSearchTool();
  assert.equal(tool.name, "codebase_semantic_search");
  assert.equal(tool.parameters, semanticSearchSchema);
});

test("hybrid engine finds an exact symbol and awaits read-hit registration", async () => {
  const { root, target } = await fixture();
  let callbackFinished = false;
  let recorded = [];
  const events = [];
  const runtime = createSemanticSearchRuntime({
    cwd: root,
    onEvent: (event) => events.push(event.type),
    onHits: async (hits) => {
      await readFile(hits[0].filePath);
      recorded = hits;
      callbackFinished = true;
    },
  });
  const response = await runtime.search({
    query: "registerModeSystemTools",
    roots: ["src"],
    limit: 3,
    mode: "symbols",
    noCache: true,
    rgTrace: true,
  });
  assert.equal(callbackFinished, true);
  const canonicalTarget = await realpath(target);
  assert.equal(path.resolve(response.hits[0].file), canonicalTarget);
  assert.equal(response.hits[0].symbol, "registerModeSystemTools");
  assert.match(response.hits[0].why, /exact_identifier=\+/);
  assert.match(response.stderr, /mode-system-tools\.ts:1:/);
  assert.deepEqual(recorded[0], { filePath: canonicalTarget, lineSpan: { start: 1, end: 3 } });
  assert.deepEqual(events.filter((event) => event !== "search.stderr"), [
    "search.started", "search.hits", "search.completed",
  ]);
});

test("tool wrapper returns the generic structured result shape", async () => {
  const { root } = await fixture();
  const tool = createSemanticSearchTool({ cwd: root });
  const result = await tool.execute("call-1", {
    query: "registerModeSystemTools",
    roots: ["src"],
    noCache: true,
  });
  assert.equal(result.details.status, "completed");
  assert.ok(result.details.hits.length > 0);
  assert.deepEqual(JSON.parse(result.content[0].text), result.details);
});

test("no results is a successful invocation with exit code 2", async () => {
  const { root } = await fixture();
  const response = await createSemanticSearchRuntime({ cwd: root }).search({
    query: "identifierThatDefinitelyDoesNotExist987654321",
    noCache: true,
  });
  assert.equal(response.details.exitCode, 2);
  assert.deepEqual(response.hits, []);
});

test("persistent cache defaults to a standalone directory", async () => {
  const { root } = await fixture();
  await createSemanticSearchRuntime({ cwd: root }).search({ query: "registerModeSystemTools" });
  const info = await stat(path.join(root, ".agentic-semantic-search", "index-v2.json"));
  assert.equal(info.isFile(), true);
});

test("structured hit extraction resolves relative paths and deduplicates spans", () => {
  const root = path.resolve("/tmp/project");
  const hit = { file: "src/a.ts", lines: [2, 4], symbol: null, score: 1, why: "test" };
  assert.deepEqual(extractSearchReadHits([hit, hit], root), [{
    filePath: path.join(root, "src/a.ts"),
    lineSpan: { start: 2, end: 4 },
  }]);
});

test("AbortSignal terminates the Python engine", async () => {
  const scriptPath = path.resolve("test/fixtures/slow_engine.py");
  const runtime = createSemanticSearchRuntime({ scriptPath, timeoutMs: 20_000 });
  const controller = new AbortController();
  const pending = runtime.search({ query: "needle", noCache: true }, { signal: controller.signal });
  setTimeout(() => controller.abort(), 30);
  await assert.rejects(pending, (error) => error instanceof SemanticSearchError && error.code === "ABORTED");
});

test("invalid engine JSON fails with a stable error code", async () => {
  const scriptPath = path.resolve("test/fixtures/invalid_engine.py");
  const runtime = createSemanticSearchRuntime({ scriptPath });
  await assert.rejects(
    runtime.search({ query: "needle", noCache: true }),
    (error) => error instanceof SemanticSearchError && error.code === "INVALID_OUTPUT",
  );
});

test("runtime rejects cache directories that escape search roots", async () => {
  const runtime = createSemanticSearchRuntime({ cacheDirectory: "../shared-cache" });
  await assert.rejects(
    runtime.search({ query: "needle" }),
    (error) => error instanceof SemanticSearchError && error.code === "INVALID_INPUT",
  );
});

test("runtime forwards local embedding controls to the Python process", async () => {
  const { root } = await fixture();
  const scriptPath = path.resolve("test/fixtures/env_engine.py");
  const runtime = createSemanticSearchRuntime({
    cwd: root,
    scriptPath,
    embeddingModelPath: "models/local.gguf",
    modelCacheDirectory: "model-cache",
    offline: true,
    autoDownload: false,
    gpuLayers: -1,
  });
  const response = await runtime.search({ query: "needle", mode: "semantic" });
  const received = JSON.parse(response.stderr.trim());
  assert.equal(received.AGENTIC_SEMANTIC_SEARCH_EMBED_MODEL_PATH, path.join(root, "models/local.gguf"));
  assert.equal(received.AGENTIC_SEMANTIC_SEARCH_MODEL_CACHE, path.join(root, "model-cache"));
  assert.equal(received.AGENTIC_SEMANTIC_SEARCH_OFFLINE, "1");
  assert.equal(received.AGENTIC_SEMANTIC_SEARCH_AUTO_DOWNLOAD, "0");
  assert.equal(received.AGENTIC_SEMANTIC_SEARCH_GPU_LAYERS, "-1");
});

test("runtime rejects invalid GPU layer configuration", () => {
  assert.throws(
    () => createSemanticSearchRuntime({ gpuLayers: -2 }),
    (error) => error instanceof SemanticSearchError && error.code === "INVALID_INPUT",
  );
});

test("public package sources contain no source-project or retired-backend references", async () => {
  const ignored = new Set([".git", "node_modules", "__pycache__"]);
  const extensions = new Set([".md", ".json", ".ts", ".js", ".mjs", ".py", ".txt"]);
  const forbidden = [
    ["ol", "lama"].join(""),
    ["open", "claw"].join(""),
    ["my", "claw"].join(""),
  ];
  const forbiddenPattern = new RegExp(forbidden.join("|"), "i");
  const violations = [];
  async function scan(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      if (ignored.has(entry.name) || entry.name.endsWith(".tgz")) continue;
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) await scan(absolute);
      else if (extensions.has(path.extname(entry.name))) {
        const contents = await readFile(absolute, "utf8");
        if (forbiddenPattern.test(contents)) violations.push(path.relative(process.cwd(), absolute));
      }
    }
  }
  await scan(process.cwd());
  assert.deepEqual(violations, []);
});
