import { spawn } from "node:child_process";

const python = process.env.AGENTIC_SEMANTIC_SEARCH_PYTHON || "python3";
const child = spawn(python, [
  "-m", "unittest", "discover", "-s", "test/python",
  "-p", "test_model_integration.py", "-v",
], {
  env: { ...process.env, AGENTIC_SEMANTIC_SEARCH_RUN_INTEGRATION: "1" },
  stdio: "inherit",
});

child.on("error", (error) => {
  console.error(`failed to start ${python}: ${error.message}`);
  process.exitCode = 1;
});
child.on("close", (code) => {
  process.exitCode = code ?? 1;
});
