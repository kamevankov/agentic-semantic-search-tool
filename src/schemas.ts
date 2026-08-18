export const semanticSearchSchema = {
  type: "object",
  additionalProperties: false,
  required: ["query"],
  properties: {
    query: {
      type: "string",
      minLength: 1,
      description: "Search query, as free text or an exact identifier.",
    },
    roots: {
      type: "array",
      items: { type: "string", minLength: 1 },
      description: "Repository roots to search. Defaults to the runtime working directory.",
    },
    limit: {
      type: "integer",
      minimum: 1,
      maximum: 200,
      description: "Maximum number of ranked hits to return. Defaults to 20.",
    },
    mode: {
      type: "string",
      enum: ["hybrid", "lexical", "symbols", "semantic"],
      description: "Ranking mode. Semantic mode degrades to hybrid when optional rerankers are unavailable.",
    },
    noCache: {
      type: "boolean",
      description: "Bypass the persistent symbol-span and embedding cache.",
    },
    rgTrace: {
      type: "boolean",
      description: "Emit ripgrep-style recall hits through search.stderr events and response stderr.",
    },
  },
} as const;
