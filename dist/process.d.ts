import type { SearchMode, SemanticSearchHit } from "./contracts.js";
export declare function bundledScriptPath(): string;
export declare function resolvePythonBin(configured?: string, env?: NodeJS.ProcessEnv): Promise<string>;
interface ProcessInput {
    query: string;
    roots: string[];
    limit: number;
    mode: SearchMode;
    noCache: boolean;
    rgTrace: boolean;
}
interface ProcessOptions {
    cwd: string;
    pythonBin: string;
    scriptPath: string;
    timeoutMs?: number;
    env?: NodeJS.ProcessEnv;
    cacheDirectory?: string;
    embeddingModelPath?: string;
    modelCacheDirectory?: string;
    offline?: boolean;
    autoDownload?: boolean;
    gpuLayers?: number;
    maxStdoutBytes?: number;
    maxStderrBytes?: number;
    signal?: AbortSignal;
    onStderr?: (text: string) => void;
}
export interface ProcessResult {
    exitCode: 0 | 2;
    hits: SemanticSearchHit[];
    stderr: string;
    durationMs: number;
}
export declare function runSearchProcess(input: ProcessInput, options: ProcessOptions): Promise<ProcessResult>;
export {};
//# sourceMappingURL=process.d.ts.map