import { spawnSync } from "node:child_process";

const python = process.env.AGENTIC_SEMANTIC_SEARCH_PYTHON || "python3";
const commands = [
  ["-m", "coverage", "erase"],
  ["-m", "coverage", "run", "-m", "unittest", "discover", "-s", "test/python", "-p", "test_*.py"],
  ["-m", "coverage", "report"],
];

for (const args of commands) {
  const result = spawnSync(python, args, { stdio: "inherit" });
  if (result.error) {
    console.error(`failed to start ${python}: ${result.error.message}`);
    process.exit(result.status ?? 1);
  }
  if (result.status !== 0) process.exit(result.status ?? 1);
}
