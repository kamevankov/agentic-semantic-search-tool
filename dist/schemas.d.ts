export declare const semanticSearchSchema: {
    readonly type: "object";
    readonly additionalProperties: false;
    readonly required: readonly ["query"];
    readonly properties: {
        readonly query: {
            readonly type: "string";
            readonly minLength: 1;
            readonly description: "Search query, as free text or an exact identifier.";
        };
        readonly roots: {
            readonly type: "array";
            readonly items: {
                readonly type: "string";
                readonly minLength: 1;
            };
            readonly description: "Repository roots to search. Defaults to the runtime working directory.";
        };
        readonly limit: {
            readonly type: "integer";
            readonly minimum: 1;
            readonly maximum: 200;
            readonly description: "Maximum number of ranked hits to return. Defaults to 20.";
        };
        readonly mode: {
            readonly type: "string";
            readonly enum: readonly ["hybrid", "lexical", "symbols", "semantic"];
            readonly description: "Ranking mode. Semantic mode degrades to hybrid when optional rerankers are unavailable.";
        };
        readonly noCache: {
            readonly type: "boolean";
            readonly description: "Bypass the persistent symbol-span and embedding cache.";
        };
        readonly rgTrace: {
            readonly type: "boolean";
            readonly description: "Emit ripgrep-style recall hits through search.stderr events and response stderr.";
        };
    };
};
//# sourceMappingURL=schemas.d.ts.map