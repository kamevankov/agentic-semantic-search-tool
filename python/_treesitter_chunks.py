"""Tree-sitter symbol detection for agentic_semantic_search.

Provides `detect_symbol_treesitter(path, line_no, ctx_lines)` returning
(name, start_line, end_line) or None when grammar is unavailable / no
enclosing definition is found. Result is 1-indexed inclusive.

Designed as a drop-in upgrade over the regex SYMBOL_PATTERNS in the main
module, with lazy loading and a quiet fallback (returns None) so callers
can fall back to the regex implementation.

Supported languages by extension:
  py                          -> python
  js, mjs, cjs, jsx           -> javascript
  ts                          -> typescript
  tsx                         -> tsx
  go                          -> go
  rs                          -> rust
  java                        -> java
  c, h                        -> c
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Lazy parser cache                                                            #
# --------------------------------------------------------------------------- #

_PARSERS: dict[str, object] = {}
_DISABLED: bool = False  # set True if py-tree-sitter itself is missing
_LOAD_FAILED: set[str] = set()  # langs whose grammar import failed

# Parsed-tree cache keyed by file fingerprint. Mirrors the chunk-level
# fingerprint cache in agentic_semantic_search (size + mtime_ns) so a single
# query that hits the same file multiple times (e.g. several detect_symbol()
# calls per result) only parses each file once.
#
# Key: (str(path), lang, st_mtime_ns, st_size, ctx_len)
# Value: (tree, source_bytes) -- source kept alive for tree-sitter node API.
import os as _os

_TREE_CACHE: "dict[tuple[str, str, int, int, int], tuple[object, bytes]]" = {}
_TREE_CACHE_MAX = 256  # FIFO cap; query-scoped reuse, bounded memory


def _tree_cache_key(file_path: Path, lang: str, ctx_len: int):
    try:
        st = _os.stat(file_path)
    except OSError:
        return None
    return (str(file_path), lang, st.st_mtime_ns, st.st_size, ctx_len)


def _tree_cache_get(key):
    if key is None:
        return None
    return _TREE_CACHE.get(key)


def _tree_cache_put(key, tree, source: bytes) -> None:
    if key is None:
        return
    if len(_TREE_CACHE) >= _TREE_CACHE_MAX:
        try:
            _TREE_CACHE.pop(next(iter(_TREE_CACHE)))
        except StopIteration:
            pass
    _TREE_CACHE[key] = (tree, source)


def _get_parser(lang: str):
    """Return a cached tree_sitter.Parser for `lang`, or None on failure."""
    global _DISABLED
    if _DISABLED:
        return None
    if lang in _PARSERS:
        return _PARSERS[lang]
    if lang in _LOAD_FAILED:
        return None
    try:
        from tree_sitter import Language, Parser  # type: ignore
    except Exception:
        _DISABLED = True
        return None

    try:
        if lang == "python":
            import tree_sitter_python as m  # type: ignore
            language = Language(m.language())
        elif lang == "javascript":
            import tree_sitter_javascript as m  # type: ignore
            language = Language(m.language())
        elif lang == "typescript":
            import tree_sitter_typescript as m  # type: ignore
            language = Language(m.language_typescript())
        elif lang == "tsx":
            import tree_sitter_typescript as m  # type: ignore
            language = Language(m.language_tsx())
        elif lang == "go":
            import tree_sitter_go as m  # type: ignore
            language = Language(m.language())
        elif lang == "rust":
            import tree_sitter_rust as m  # type: ignore
            language = Language(m.language())
        elif lang == "java":
            import tree_sitter_java as m  # type: ignore
            language = Language(m.language())
        elif lang == "c":
            import tree_sitter_c as m  # type: ignore
            language = Language(m.language())
        else:
            return None
        parser = Parser(language)
        _PARSERS[lang] = parser
        return parser
    except Exception:
        _LOAD_FAILED.add(lang)
        return None


# --------------------------------------------------------------------------- #
# Per-language definition node descriptors                                     #
# --------------------------------------------------------------------------- #

# Each entry is a set of node types that count as "named definitions". The
# name is extracted by `_extract_name` below. Nodes not in this set are
# skipped when walking up from the target line.
_DEF_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "method_definition",
        "lexical_declaration",   # const/let/var ... = (arrow|function)
        "variable_declaration",
        "function_expression",
    },
    "typescript": {
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "method_definition",
        "lexical_declaration",
        "variable_declaration",
        "function_expression",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "abstract_class_declaration",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
        "type_item",
    },
    "java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "annotation_type_declaration",
        "method_declaration",
        "constructor_declaration",
    },
    "c": {
        "function_definition",
        "struct_specifier",
        "union_specifier",
        "enum_specifier",
        "type_definition",
    },
}
_DEF_TYPES["tsx"] = _DEF_TYPES["typescript"]


_EXT_TO_LANG: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "tsx",
    "go": "go",
    "rs": "rust",
    "java": "java",
    "c": "c",
    "h": "c",
}


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _name_from_field(node, field: str) -> Optional[str]:
    n = node.child_by_field_name(field)
    if n is None:
        return None
    return _node_text(n, _SOURCE_HOLDER["src"])


# Trick: use a small mutable holder so helpers can read the current source
# without threading it through every call. Only used during a single
# parse; we set it before walking.
_SOURCE_HOLDER: dict[str, bytes] = {"src": b""}


def _extract_name(node, lang: str) -> Optional[str]:
    src = _SOURCE_HOLDER["src"]
    t = node.type

    # Most defs expose a "name" field.
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(name_node, src)

    # JS/TS arrow / function expression bound to a const/let/var.
    if t in {"lexical_declaration", "variable_declaration"}:
        # Find first variable_declarator with an arrow_function or function_expression value.
        for child in node.named_children:
            if child.type == "variable_declarator":
                value = child.child_by_field_name("value")
                if value is None:
                    continue
                if value.type in {"arrow_function", "function", "function_expression"}:
                    nm = child.child_by_field_name("name")
                    if nm is not None:
                        return _node_text(nm, src)
        return None

    # Rust impl blocks: name from "type" field.
    if t == "impl_item":
        ty = node.child_by_field_name("type")
        if ty is not None:
            return _node_text(ty, src)

    # C function_definition: name lives inside declarator -> function_declarator -> identifier.
    if t == "function_definition":
        d = node.child_by_field_name("declarator")
        # Strip pointer_declarator wrappers (e.g. int *foo(...)).
        while d is not None and d.type == "pointer_declarator":
            d = d.child_by_field_name("declarator")
        if d is not None and d.type == "function_declarator":
            inner = d.child_by_field_name("declarator")
            while inner is not None and inner.type in {"parenthesized_declarator", "pointer_declarator"}:
                inner = inner.child_by_field_name("declarator") or (
                    inner.named_children[0] if inner.named_children else None
                )
            if inner is not None:
                return _node_text(inner, src)

    # C typedef: name is the declarator (last identifier-ish child).
    if t == "type_definition":
        d = node.child_by_field_name("declarator")
        while d is not None and d.type in {"pointer_declarator", "array_declarator", "function_declarator"}:
            d = d.child_by_field_name("declarator")
        if d is not None:
            return _node_text(d, src)

    return None


def _is_definition(node, lang: str) -> bool:
    types = _DEF_TYPES.get(lang)
    if not types or node.type not in types:
        return False
    # Filter lexical/variable_declaration that are not function-bound; we
    # don't want every `const x = 5` polluting symbol detection.
    if node.type in {"lexical_declaration", "variable_declaration"}:
        for child in node.named_children:
            if child.type == "variable_declarator":
                value = child.child_by_field_name("value")
                if value is not None and value.type in {
                    "arrow_function", "function", "function_expression"
                }:
                    return True
        return False
    return True


def _walk_innermost(root, target_row: int, lang: str):
    """Return the innermost definition node containing target_row (0-indexed)."""
    best = None
    stack = [root]
    while stack:
        node = stack.pop()
        if node.start_point[0] > target_row or node.end_point[0] < target_row:
            continue
        if _is_definition(node, lang):
            if best is None or (
                node.start_point[0] >= best.start_point[0]
                and node.end_point[0] <= best.end_point[0]
            ):
                best = node
        for child in node.named_children:
            stack.append(child)
    return best


def detect_symbol_treesitter(
    file_path: Path, line_no: int, ctx_lines: list[str]
) -> Optional[tuple[str, int, int]]:
    """Return (symbol_name, start_line, end_line) using tree-sitter.

    Returns None when:
      - tree_sitter or the language grammar is not installed
      - the file extension isn't in _EXT_TO_LANG
      - parsing yields no enclosing definition for `line_no`

    Lines are 1-indexed inclusive on both ends.
    """
    ext = file_path.suffix.lstrip(".").lower()
    lang = _EXT_TO_LANG.get(ext)
    if lang is None:
        return None
    parser = _get_parser(lang)
    if parser is None:
        return None
    cache_key = _tree_cache_key(file_path, lang, len(ctx_lines))
    cached = _tree_cache_get(cache_key)
    if cached is not None:
        tree, source = cached
        _SOURCE_HOLDER["src"] = source
    else:
        try:
            source = ("\n".join(ctx_lines)).encode("utf-8", "replace")
            _SOURCE_HOLDER["src"] = source
            tree = parser.parse(source)
        except Exception:
            return None
        _tree_cache_put(cache_key, tree, source)

    target_row = max(0, line_no - 1)
    node = _walk_innermost(tree.root_node, target_row, lang)
    if node is None:
        return None
    name = _extract_name(node, lang)
    if name is None:
        return None
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    return (name, start, max(start, end))


def is_available() -> bool:
    """True if py-tree-sitter import succeeds and at least one grammar loads."""
    if _DISABLED:
        return False
    for lang in ("python", "javascript", "typescript", "go", "rust", "java", "c"):
        if _get_parser(lang) is not None:
            return True
    return False
