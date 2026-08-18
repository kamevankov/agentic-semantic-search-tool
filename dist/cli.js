#!/usr/bin/env node
import process from "node:process";
import { SEARCH_MODES, SemanticSearchError } from "./contracts.js";
import { createSemanticSearchRuntime } from "./runtime.js";
const usage = `agentic-semantic-search 0.1.0

Hybrid lexical, structural, and optional semantic codebase search.

Usage:
  agentic-semantic-search --query TEXT [options]

Options:
  -q, --query TEXT             Search query (required)
  -r, --root PATH              Search root; repeat for multiple roots (default: cwd)
  -n, --limit N                Maximum results (default: 20, max: 200)
  -m, --mode MODE              hybrid | lexical | symbols | semantic
      --no-cache               Bypass persistent caches
      --rg-trace               Write ripgrep-style recall hits to stderr
      --pretty                 Pretty-print result JSON
      --python PATH             Python interpreter
      --script PATH             Alternate engine script
      --cache-directory PATH    Relative cache directory within each root
      --timeout-ms N            Wall-clock timeout (default: 60000)
  -h, --help                   Show this help
  -v, --version                Show the version

Exit codes: 0 results, 2 no results, 3 invalid input, 4 ripgrep unavailable.
`;
function takeValue(args, index, flag) {
    const value = args[index + 1];
    if (!value || value.startsWith("-"))
        throw new SemanticSearchError("INVALID_INPUT", `${flag} requires a value`);
    return value;
}
async function main() {
    const args = process.argv.slice(2);
    if (args.includes("--help") || args.includes("-h")) {
        process.stdout.write(usage);
        return 0;
    }
    if (args.includes("--version") || args.includes("-v")) {
        process.stdout.write("0.1.0\n");
        return 0;
    }
    const input = { query: "" };
    const roots = [];
    let pretty = false;
    let pythonBin;
    let scriptPath;
    let cacheDirectory;
    let timeoutMs;
    for (let index = 0; index < args.length; index += 1) {
        const flag = args[index];
        if (flag === "--query" || flag === "-q")
            input.query = takeValue(args, index++, flag);
        else if (flag === "--root" || flag === "-r")
            roots.push(takeValue(args, index++, flag));
        else if (flag === "--limit" || flag === "-n")
            input.limit = Number(takeValue(args, index++, flag));
        else if (flag === "--mode" || flag === "-m") {
            const mode = takeValue(args, index++, flag);
            if (!SEARCH_MODES.includes(mode))
                throw new SemanticSearchError("INVALID_INPUT", `Unknown mode: ${mode}`);
            input.mode = mode;
        }
        else if (flag === "--no-cache")
            input.noCache = true;
        else if (flag === "--rg-trace")
            input.rgTrace = true;
        else if (flag === "--pretty")
            pretty = true;
        else if (flag === "--python")
            pythonBin = takeValue(args, index++, flag);
        else if (flag === "--script")
            scriptPath = takeValue(args, index++, flag);
        else if (flag === "--cache-directory")
            cacheDirectory = takeValue(args, index++, flag);
        else if (flag === "--timeout-ms")
            timeoutMs = Number(takeValue(args, index++, flag));
        else
            throw new SemanticSearchError("INVALID_INPUT", `Unknown argument: ${flag}`);
    }
    if (roots.length)
        input.roots = roots;
    if (input.limit !== undefined && (!Number.isInteger(input.limit) || input.limit < 1 || input.limit > 200)) {
        throw new SemanticSearchError("INVALID_INPUT", "--limit must be an integer from 1 to 200");
    }
    if (timeoutMs !== undefined && (!Number.isFinite(timeoutMs) || timeoutMs < 1)) {
        throw new SemanticSearchError("INVALID_INPUT", "--timeout-ms must be a positive number");
    }
    const controller = new AbortController();
    const onInterrupt = () => controller.abort();
    process.once("SIGINT", onInterrupt);
    try {
        const runtime = createSemanticSearchRuntime({ pythonBin, scriptPath, cacheDirectory, timeoutMs });
        const response = await runtime.search(input, { signal: controller.signal });
        if (response.stderr)
            process.stderr.write(response.stderr);
        process.stdout.write(`${JSON.stringify(response.hits, null, pretty ? 2 : undefined)}\n`);
        return response.details.exitCode;
    }
    finally {
        process.removeListener("SIGINT", onInterrupt);
    }
}
main().then((code) => { process.exitCode = code; }, (error) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`error: ${message}\n`);
    if (error instanceof SemanticSearchError && error.code === "PROCESS_FAILED") {
        const exitCode = error.details.exitCode;
        process.exitCode = typeof exitCode === "number" ? exitCode : 1;
    }
    else {
        process.exitCode = error instanceof SemanticSearchError && error.code === "INVALID_INPUT" ? 3 : 1;
    }
});
//# sourceMappingURL=cli.js.map