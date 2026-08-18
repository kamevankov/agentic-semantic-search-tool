export const SEARCH_MODES = ["hybrid", "lexical", "symbols", "semantic"];
export class SemanticSearchError extends Error {
    code;
    details;
    constructor(code, message, details = {}) {
        super(message);
        this.code = code;
        this.details = details;
        this.name = "SemanticSearchError";
    }
}
//# sourceMappingURL=contracts.js.map