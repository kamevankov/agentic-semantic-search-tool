import type { GenericSemanticSearchTool, SearchReadHit, SemanticSearchHit, SemanticSearchRuntime, SemanticSearchRuntimeOptions } from "./contracts.js";
export declare function extractSearchReadHits(hits: readonly SemanticSearchHit[], root?: string): SearchReadHit[];
export declare function createSemanticSearchRuntime(options?: SemanticSearchRuntimeOptions): SemanticSearchRuntime;
export declare function createSemanticSearchTool(options?: SemanticSearchRuntimeOptions): GenericSemanticSearchTool;
//# sourceMappingURL=runtime.d.ts.map