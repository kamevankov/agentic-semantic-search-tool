import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

test("CLI searches the bundled engine and preserves no-result exit code", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "agentic-semantic-cli-"));
  await writeFile(path.join(root, "demo.py"), "def find_me():\n    return 7\n");
  const found = spawnSync(process.execPath, [
    "dist/cli.js", "--query", "find_me", "--root", root, "--mode", "symbols", "--no-cache",
  ], { encoding: "utf8" });
  assert.equal(found.status, 0, found.stderr);
  const hits = JSON.parse(found.stdout);
  assert.equal(hits[0].symbol, "find_me");

  const missing = spawnSync(process.execPath, [
    "dist/cli.js", "--query", "missing_identifier_987654321", "--root", root, "--no-cache",
  ], { encoding: "utf8" });
  assert.equal(missing.status, 2, missing.stderr);
  assert.deepEqual(JSON.parse(missing.stdout), []);
});
