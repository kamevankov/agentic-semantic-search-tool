#!/usr/bin/env python3
"""
agentic_semantic_search — hybrid lexical + structural codebase search.

Capabilities (shipped):
- Lexical recall via ripgrep (fixed-string + regex token expansion).
- BM25 reranking over matched chunks built around hits (~symbol or window).
- Structural symbol detection: tree-sitter for py/js/jsx/ts/tsx/go/rust when
  grammars are installed, regex SYMBOL_PATTERNS fallback otherwise.
- Persistent incremental index per repo root. It records the git commit and
  invalidates each file by path + size + mtime-ns + sha1 under
  `<repo-root>/.agentic-semantic-search/` by default.
- Embedding via Ollama `nomic-embed-text` for `--mode semantic` and the
  optional cosine rerank in `hybrid`; degrades cleanly to weighted hybrid
  when Ollama is unreachable (`why` records `semantic=unavailable`).
- Optional cross-encoder rerank over the top candidates in `--mode semantic`;
  falls back to embedding cosine when CE is unavailable (`why` records
  `ce_rerank=unavailable`).
- Output: JSON list of {file, lines, symbol, score, why}.
- Emits ripgrep-style `path:line:content` lines on stderr (--rg-trace) so
  harnesses can register returned spans as scoped reads. The bundled Node
  runtime also exposes structured search-hit callbacks for this purpose.

Usage:
  agentic_semantic_search.py --query "..." [--root PATH ...] [--limit N]
                             [--mode hybrid|lexical|symbols|semantic] [--rg-trace]

Exit codes: 0 ok, 2 no-results, 3 misuse, 4 ripgrep missing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
# Symbol detection (tree-sitter primary, regex SYMBOL_PATTERNS as fallback).  #
# --------------------------------------------------------------------------- #

# (extension -> list of (regex, group-index-of-name))
SYMBOL_PATTERNS: dict[str, list[tuple[re.Pattern[str], int]]] = {
    "py": [
        (re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)"), 1),
        (re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)"), 1),
    ],
    "ts": [
        (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"), 1),
        (re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"), 1),
        (re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)"), 1),
        (re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)"), 1),
        (re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*[:=]"), 1),
    ],
    "go": [
        (re.compile(r"^\s*func\s+(?:\([^)]*\)\s+)?([A-Za-z_][A-Za-z0-9_]*)"), 1),
        (re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:struct|interface)"), 1),
    ],
    "rs": [
        (re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)"), 1),
        (re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)"), 1),
        (re.compile(r"^\s*impl(?:<[^>]*>)?\s+([A-Za-z_][A-Za-z0-9_:<>]*)"), 1),
    ],
    "java": [
        (re.compile(r"^\s*(?:public|private|protected|static|final|\s)*\s+(?:class|interface)\s+([A-Za-z_][A-Za-z0-9_]*)"), 1),
        (re.compile(r"^\s*(?:public|private|protected|static|final|\s)*\s+[\w<>\[\],\s]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("), 1),
    ],
    "c": [
        (re.compile(r"^\s*[\w*\s]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{?\s*$"), 1),
    ],
}
SYMBOL_PATTERNS["js"] = SYMBOL_PATTERNS["ts"]
SYMBOL_PATTERNS["jsx"] = SYMBOL_PATTERNS["ts"]
SYMBOL_PATTERNS["tsx"] = SYMBOL_PATTERNS["ts"]
SYMBOL_PATTERNS["mjs"] = SYMBOL_PATTERNS["ts"]
SYMBOL_PATTERNS["cjs"] = SYMBOL_PATTERNS["ts"]
SYMBOL_PATTERNS["h"] = SYMBOL_PATTERNS["c"]
SYMBOL_PATTERNS["cpp"] = SYMBOL_PATTERNS["c"]
SYMBOL_PATTERNS["hpp"] = SYMBOL_PATTERNS["c"]


def detect_symbol(file_path: Path, line_no: int, ctx_lines: list[str]) -> tuple[str | None, int, int]:
    """Return (symbol_name, span_start, span_end) for the symbol enclosing
    the given line. Falls back to a fixed-window span when no symbol is found.

    Strategy: tree-sitter when its grammar is available for the file
    extension (py/js/jsx/ts/tsx/go/rust/java/c/h), regex SYMBOL_PATTERNS
    as fallback for languages without a wired grammar (e.g. cpp/hpp).

    `ctx_lines` is the full file content as a list (1-indexed via line_no - 1).
    """
    # --- tree-sitter primary path ----------------------------------------- #
    try:
        from . import _treesitter_chunks as _tsc  # type: ignore
    except Exception:
        try:
            import _treesitter_chunks as _tsc  # type: ignore
        except Exception:
            _tsc = None  # type: ignore
    if _tsc is not None:
        try:
            ts_hit = _tsc.detect_symbol_treesitter(file_path, line_no, ctx_lines)
        except Exception:
            ts_hit = None
        if ts_hit is not None:
            return ts_hit
    # --- regex fallback --------------------------------------------------- #
    ext = file_path.suffix.lstrip(".").lower()
    patterns = SYMBOL_PATTERNS.get(ext, [])
    if not patterns:
        start = max(1, line_no - 10)
        end = min(len(ctx_lines), line_no + 10)
        return (None, start, end)
    # Walk upward to find the nearest enclosing symbol header.
    target_idx = max(0, line_no - 1)
    name: str | None = None
    sym_line: int | None = None
    sym_indent: int | None = None
    for i in range(target_idx, -1, -1):
        line = ctx_lines[i] if i < len(ctx_lines) else ""
        for pat, group in patterns:
            m = pat.match(line)
            if m:
                indent = len(line) - len(line.lstrip(" \t"))
                if sym_indent is None or indent <= sym_indent:
                    name = m.group(group)
                    sym_line = i + 1
                    sym_indent = indent
                    break
        if name is not None and sym_indent == 0:
            break
    if sym_line is None:
        start = max(1, line_no - 10)
        end = min(len(ctx_lines), line_no + 10)
        return (None, start, end)
    # Find a reasonable end line: next line at indent <= sym_indent (non-blank), or +60 lines.
    end = min(len(ctx_lines), sym_line + 60)
    for j in range(sym_line, min(len(ctx_lines), sym_line + 400)):
        line = ctx_lines[j] if j < len(ctx_lines) else ""
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if indent <= (sym_indent or 0) and j + 1 != sym_line:
            end = j  # exclusive of the next sibling header line
            break
    return (name, sym_line, max(sym_line, end))


# --------------------------------------------------------------------------- #
# Tokenization + BM25                                                          #
# --------------------------------------------------------------------------- #

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(text: str) -> list[str]:
    raw = TOKEN_RE.findall(text)
    out: list[str] = []
    for tok in raw:
        # split CamelCase and snake_case so a query of "readState" matches "read_state".
        parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", tok) or [tok]
        out.append(tok.lower())
        for p in parts:
            if p and p.lower() != tok.lower():
                out.append(p.lower())
    return out


@dataclass
class Chunk:
    file: str
    start: int
    end: int
    symbol: str | None
    text: str
    hit_lines: list[int]


def bm25_scores(chunks: list[Chunk], query_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    n = len(chunks)
    if n == 0:
        return []
    tokenized = [tokenize(c.text) for c in chunks]
    lens = [len(t) for t in tokenized]
    avgdl = sum(lens) / n if n > 0 else 0.0
    df: Counter[str] = Counter()
    for toks in tokenized:
        for t in set(toks):
            df[t] += 1
    scores = [0.0] * n
    qcount = Counter(query_tokens)
    for term, qf in qcount.items():
        d = df.get(term, 0)
        if d == 0:
            continue
        idf = math.log(1 + (n - d + 0.5) / (d + 0.5))
        for i, toks in enumerate(tokenized):
            tf = toks.count(term)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * (lens[i] / avgdl if avgdl else 1))
            scores[i] += idf * (tf * (k1 + 1)) / denom * (1 + 0.05 * qf)
    return scores


# --------------------------------------------------------------------------- #
# Ripgrep recall                                                               #
# --------------------------------------------------------------------------- #

DEFAULT_GLOB_EXCLUDES = [
    "!**/node_modules/**",
    "!**/.git/**",
    "!**/dist/**",
    "!**/build/**",
    "!**/.openclaw/**",
    "!**/.agentic-semantic-search/**",
    "!**/__pycache__/**",
    "!**/*.min.js",
    "!**/*.lock",
    "!**/package-lock.json",
]

MAX_FILE_BYTES = 1_500_000  # skip very large files


def _parse_rg_lines(stdout: str, max_hits: int) -> list[tuple[Path, int, str]]:
    out: list[tuple[Path, int, str]] = []
    for line in stdout.splitlines():
        m = re.match(r"^(.*?):(\d+):(.*)$", line)
        if not m:
            continue
        try:
            out.append((Path(m.group(1)), int(m.group(2)), m.group(3)))
        except (ValueError, OSError):
            continue
        if len(out) >= max_hits:
            break
    return out


def exact_identifier_terms(query: str) -> list[str]:
    """Return identifier-like query terms that deserve exact recall.

    BM25 token expansion is deliberately broad, but broad terms like
    "mode", "system", and "tools" can drown out a precise symbol such as
    `registerModeSystemTools`. Exact identifier recall runs first so obvious
    definitions and call sites always enter the candidate set.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for tok in TOKEN_RE.findall(query):
        if len(tok) < 4:
            continue
        # Prefer terms that look like symbols rather than prose words.
        looks_symbolic = "_" in tok or "$" in tok or any(c.isupper() for c in tok[1:]) or len(tok) >= 12
        if not looks_symbolic:
            continue
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)
    return terms


def _run_rg(pattern: str, roots: list[Path], *, ignore_case: bool, max_count: int, max_hits: int, fixed_strings: bool = False) -> list[tuple[Path, int, str]]:
    rg = shutil.which("rg")
    if not rg:
        print("error: ripgrep (rg) not found in PATH", file=sys.stderr)
        sys.exit(4)
    args = [
        rg,
        "--no-heading",
        "--line-number",
        "--no-config",
        "--max-count", str(max_count),
        "--max-filesize", "1500000",
        "-e", pattern,
    ]
    if fixed_strings:
        args.insert(4, "--fixed-strings")
    if ignore_case:
        args.insert(4, "--ignore-case")
    for g in DEFAULT_GLOB_EXCLUDES:
        args += ["-g", g]
    args += [str(r) for r in roots]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
    except Exception as e:  # noqa: BLE001
        print(f"error: ripgrep failed: {e}", file=sys.stderr)
        return []
    return _parse_rg_lines(proc.stdout, max_hits)


def run_ripgrep(query: str, roots: list[Path], max_hits: int = 800) -> list[tuple[Path, int, str]]:
    # First, force exact identifier candidates into recall. This prevents
    # broad token expansion from starving exact symbol hits before ranking.
    exact_terms = exact_identifier_terms(query)
    exact_hits: list[tuple[Path, int, str]] = []
    if exact_terms:
        # Use fixed-string rg for exact identifiers. Rust regex does not support
        # lookarounds by default, and asking for PCRE2 here would make recall
        # depend on the user's rg build.
        for term in exact_terms:
            exact_hits.extend(
                _run_rg(term, roots, ignore_case=False, max_count=20, max_hits=min(300, max_hits), fixed_strings=True)
            )

    # Build a permissive query: tokenize, then OR them. Also include the raw query.
    tokens = sorted(set(t for t in tokenize(query) if len(t) >= 2))
    pattern_terms = list({re.escape(t) for t in tokens})
    if query.strip():
        pattern_terms.append(re.escape(query.strip()))
    if not pattern_terms:
        return exact_hits
    broad_hits = _run_rg("|".join(pattern_terms), roots, ignore_case=True, max_count=8, max_hits=max_hits)

    merged: list[tuple[Path, int, str]] = []
    seen: set[tuple[str, int]] = set()
    for hit in [*exact_hits, *broad_hits]:
        key = (str(hit[0]), hit[1])
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
        if len(merged) >= max_hits:
            break
    return merged


# --------------------------------------------------------------------------- #
# Main pipeline                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    file: str
    lines: list[int]      # [start, end]
    symbol: str | None
    score: float
    why: str


OUTPUT_MAX_LINE_SPAN = 180
OUTPUT_HIT_CONTEXT_LINES = 35


def output_range_for_chunk(chunk: Chunk) -> tuple[list[int], str | None]:
    """Return a read-friendly output range for a ranked chunk.

    Ranking can use a whole enclosing symbol, but returning a 600+ line symbol
    defeats the tool's purpose for agents. When a symbol is large, emit the
    densest hit-centered window instead while keeping scoring unchanged.
    """
    original_span = chunk.end - chunk.start + 1
    if original_span <= OUTPUT_MAX_LINE_SPAN:
        return [chunk.start, chunk.end], None
    hits = sorted({line for line in chunk.hit_lines if chunk.start <= line <= chunk.end})
    if not hits:
        narrowed_start = chunk.start
    else:
        width = min(OUTPUT_MAX_LINE_SPAN, original_span)
        best_start = max(chunk.start, min(hits[0] - OUTPUT_HIT_CONTEXT_LINES, chunk.end - width + 1))
        best_count = -1
        best_distance = 10**9
        median_hit = hits[len(hits) // 2]
        for hit in hits:
            candidate_start = max(chunk.start, min(hit - OUTPUT_HIT_CONTEXT_LINES, chunk.end - width + 1))
            candidate_end = candidate_start + width - 1
            count = sum(1 for h in hits if candidate_start <= h <= candidate_end)
            distance = abs((candidate_start + candidate_end) // 2 - median_hit)
            if count > best_count or (count == best_count and distance < best_distance):
                best_start = candidate_start
                best_count = count
                best_distance = distance
        narrowed_start = best_start
    narrowed_end = min(chunk.end, narrowed_start + min(OUTPUT_MAX_LINE_SPAN, original_span) - 1)
    return [narrowed_start, narrowed_end], f"output_range=narrowed_from:{chunk.start}-{chunk.end}"


def result_from_chunk(chunk: Chunk, score: float, why: str) -> Result:
    lines, range_note = output_range_for_chunk(chunk)
    if range_note:
        why = f"{why}, {range_note}"
    return Result(
        file=chunk.file,
        lines=lines,
        symbol=chunk.symbol,
        score=round(score, 4),
        why=why,
    )


def build_chunks_from_hits(
    hits: list[tuple[Path, int, str]],
    mode: str,
    roots: list[Path] | None = None,
    use_cache: bool = True,
) -> list[Chunk]:
    by_file: dict[Path, list[tuple[int, str]]] = {}
    for path, line, content in hits:
        by_file.setdefault(path, []).append((line, content))
    chunks: list[Chunk] = []
    for path, file_hits in by_file.items():
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            file_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except (OSError, UnicodeError):
            continue
        # Group hits into the same enclosing symbol/window.
        seen_spans: dict[tuple[int, int], Chunk] = {}
        for line_no, _content in file_hits:
            if use_cache and roots:
                symbol, start, end = cached_chunk_for(path, line_no, file_lines, roots)
            else:
                symbol, start, end = detect_symbol(path, line_no, file_lines)
            # In "symbols" mode, skip hits without a real symbol.
            if mode == "symbols" and symbol is None:
                continue
            key = (start, end)
            if key in seen_spans:
                seen_spans[key].hit_lines.append(line_no)
                continue
            text = "\n".join(file_lines[start - 1:end])
            seen_spans[key] = Chunk(
                file=str(path),
                start=start,
                end=end,
                symbol=symbol,
                text=text,
                hit_lines=[line_no],
            )
        chunks.extend(seen_spans.values())
    return chunks


def hybrid_score(chunk: Chunk, bm25: float, hit_count: int, query_tokens: list[str], exact_terms: list[str] | None = None) -> tuple[float, str]:
    # Combine BM25 with bonuses: hits density, symbol-name match, file-path match.
    sym_bonus = 0.0
    if chunk.symbol:
        sym_l = chunk.symbol.lower()
        for t in set(query_tokens):
            if t and t in sym_l:
                sym_bonus += 1.0
    path_bonus = 0.0
    f_l = chunk.file.lower()
    for t in set(query_tokens):
        if t and len(t) >= 3 and t in f_l:
            path_bonus += 0.4
    exact_bonus = 0.0
    if exact_terms:
        text_l = chunk.text.lower()
        sym_l = (chunk.symbol or "").lower()
        for term in exact_terms:
            term_l = term.lower()
            if sym_l == term_l:
                exact_bonus += 8.0
            elif term_l in sym_l:
                exact_bonus += 4.0
            elif re.search(rf"(?<![A-Za-z0-9_$]){re.escape(term_l)}(?![A-Za-z0-9_$])", text_l):
                exact_bonus += 2.0
    density = math.log1p(hit_count)
    score = bm25 + 0.8 * density + 1.2 * sym_bonus + path_bonus + exact_bonus
    parts = [f"bm25={bm25:.2f}", f"hits={hit_count}"]
    if sym_bonus:
        parts.append(f"symbol_match=+{sym_bonus:.1f}")
    if path_bonus:
        parts.append(f"path_match=+{path_bonus:.1f}")
    if exact_bonus:
        parts.append(f"exact_identifier=+{exact_bonus:.1f}")
    return score, ", ".join(parts)


def search(
    query: str,
    roots: list[Path],
    limit: int,
    mode: str,
    rg_trace: bool,
    use_cache: bool = True,
) -> list[Result]:
    raw_hits = run_ripgrep(query, roots)
    if rg_trace:
        for path, line, content in raw_hits:
            sys.stderr.write(f"{path}:{line}:{content}\n")
    if not raw_hits:
        return []
    # In semantic mode, run the hybrid pipeline first then rerank by embedding.
    chunk_mode = "hybrid" if mode == "semantic" else mode
    chunks = build_chunks_from_hits(raw_hits, chunk_mode, roots=roots, use_cache=use_cache)
    if not chunks:
        return []
    qtok = tokenize(query)
    exact_terms = exact_identifier_terms(query)
    if mode == "lexical":
        # Pure BM25, no bonuses.
        scores = bm25_scores(chunks, qtok)
        ranked = list(zip(chunks, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        results = [
            result_from_chunk(c, s, f"bm25={s:.2f}, hits={len(c.hit_lines)}")
            for c, s in ranked[:limit]
        ]
        return results
    bm25 = bm25_scores(chunks, qtok)
    scored: list[tuple[Chunk, float, str]] = []
    for c, b in zip(chunks, bm25):
        s, why = hybrid_score(c, b, len(c.hit_lines), qtok, exact_terms)
        scored.append((c, s, why))
    scored.sort(key=lambda x: x[1], reverse=True)
    if mode == "semantic":
        top = scored[:50]
        # Stage 1: cross-encoder rerank (preferred when sentence-transformers is available).
        ce_scores = _cross_encoder_rerank(query, [c for c, _, _ in top])
        if ce_scores is not None:
            reranked_ce: list[tuple[Chunk, float, str, float]] = []
            for (c, s, why), ce in zip(top, ce_scores):
                reranked_ce.append((c, float(ce), why, float(ce)))
            reranked_ce.sort(key=lambda x: x[1], reverse=True)
            results: list[Result] = []
            for c, combined, why, ce in reranked_ce[:limit]:
                results.append(result_from_chunk(
                    c,
                    combined,
                    f"{why}, ce_rerank={ce:.3f}; mode=semantic (BM25+hybrid then cross-encoder {CROSS_ENCODER_MODEL} rerank)",
                ))
            return results
        # Stage 2 fallback: Ollama nomic-embed-text cosine rerank.
        qvec = _ollama_embed(query)
        if qvec is None:
            results = []
            for c, s, why in top[:limit]:
                results.append(result_from_chunk(
                    c,
                    s,
                    why + ", ce_rerank=unavailable, semantic=unavailable; mode=semantic (cross-encoder + Ollama offline; weighted hybrid fallback)",
                ))
            return results
        reranked: list[tuple[Chunk, float, str, float]] = []
        for c, s, why in top:
            vec = _embed_for_chunk(c, roots) if use_cache else _ollama_embed(c.text[:8000])
            sim = _cosine(qvec, vec) if vec else 0.0
            # Keep exact-symbol/lexical signal strong in the fallback path.
            # Embedding-only rerank is too fuzzy for code identifiers.
            combined = 0.35 * sim + 0.65 * (s / (max(scored[0][1], 1e-6)))
            reranked.append((c, combined, why, sim))
        reranked.sort(key=lambda x: x[1], reverse=True)
        results = []
        for c, combined, why, sim in reranked[:limit]:
            results.append(result_from_chunk(
                c,
                combined,
                f"{why}, ce_rerank=unavailable, semantic={sim:.2f}; mode=semantic (BM25+hybrid then Ollama nomic-embed-text cosine rerank; cross-encoder unavailable)",
            ))
        return results
    results = []
    for c, s, why in scored[:limit]:
        suffix = "; mode=hybrid (lexical+BM25+structural; enable semantic/rerank with --mode semantic)"
        if mode == "symbols":
            suffix = "; mode=symbols (regex symbol detection)"
        results.append(result_from_chunk(c, s, why + suffix))
    return results


def main() -> int:
    p = argparse.ArgumentParser(description="Hybrid lexical+structural codebase search")
    p.add_argument("--query", "-q", required=True, help="Search query")
    p.add_argument("--root", "-r", action="append", default=[], help="Search root (repeatable). Default: cwd.")
    p.add_argument("--limit", "-n", type=int, default=20)
    p.add_argument("--mode", "-m", choices=["hybrid", "lexical", "symbols", "semantic"], default="hybrid")
    p.add_argument("--no-cache", action="store_true", help="Bypass the persistent chunk/embedding cache.")
    p.add_argument("--rg-trace", action="store_true",
                   help="Emit ripgrep-style 'path:line:content' on stderr for scoped-read recording.")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = p.parse_args()

    roots = [Path(r).resolve() for r in args.root] or [Path.cwd().resolve()]
    bad = [r for r in roots if not r.exists()]
    if bad:
        print(f"error: missing roots: {[str(b) for b in bad]}", file=sys.stderr)
        return 3

    results = search(args.query, roots, max(1, args.limit), args.mode, args.rg_trace, use_cache=not args.no_cache)
    payload = [asdict(r) for r in results]
    if args.pretty:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload))
    return 0 if results else 2


# --------------------------------------------------------------------------- #
# Persistent incremental index + optional Ollama semantic rerank.
# Appended in v2 (task #161). Keep additive; don't break the v1 surface.
# --------------------------------------------------------------------------- #
import hashlib
import json as _json
import os as _os
import time as _time
import urllib.request as _urlreq
import urllib.error as _urlerr
import subprocess as _subp
from pathlib import Path as _Path

CACHE_DIRNAME = _os.environ.get(
    "AGENTIC_SEMANTIC_SEARCH_CACHE_DIR",
    ".agentic-semantic-search",
)
CACHE_VERSION = 1


def _git_commit_for(root: _Path) -> str:
    try:
        out = _subp.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        sha = (out.stdout or "").strip()
        if sha and len(sha) >= 7:
            return sha
    except Exception:  # noqa: BLE001
        pass
    return "nogit"


def _cache_root_for(root: _Path) -> _Path:
    base = root / CACHE_DIRNAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def _file_fingerprint(path: _Path) -> tuple[int, int, str] | None:
    """Return (size, mtime_ns, sha1) for a file, or None if unreadable."""
    try:
        st = path.stat()
    except OSError:
        return None
    if st.st_size > MAX_FILE_BYTES:
        return None
    h = hashlib.sha1()
    try:
        with path.open("rb") as fh:
            for buf in iter(lambda: fh.read(65536), b""):
                h.update(buf)
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns, h.hexdigest())


def _index_path_for(root: _Path) -> _Path:
    return _cache_root_for(root) / f"index-v{CACHE_VERSION}.json"


def _load_index(root: _Path) -> dict:
    p = _index_path_for(root)
    if not p.exists():
        return {"version": CACHE_VERSION, "commit": _git_commit_for(root), "files": {}}
    try:
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"version": CACHE_VERSION, "commit": _git_commit_for(root), "files": {}}


def _save_index(root: _Path, idx: dict) -> None:
    p = _index_path_for(root)
    try:
        p.write_text(_json.dumps(idx), encoding="utf-8")
    except OSError:
        pass


def _index_root_for(file_path: _Path, roots: list[_Path]) -> _Path | None:
    """Pick the longest matching search root for a file."""
    fp = file_path.resolve()
    best: _Path | None = None
    best_len = -1
    for r in roots:
        rr = r.resolve()
        try:
            fp.relative_to(rr)
        except ValueError:
            continue
        l = len(str(rr))
        if l > best_len:
            best = rr
            best_len = l
    return best


def cached_chunk_for(
    file_path: _Path,
    line_no: int,
    file_lines: list[str],
    roots: list[_Path],
) -> tuple[str | None, int, int]:
    """Like detect_symbol() but consults an mtime+size+sha1 cache before recomputing.
    Falls back to detect_symbol() on miss and stores the result."""
    root = _index_root_for(file_path, roots)
    if root is None:
        return detect_symbol(file_path, line_no, file_lines)
    fp = _file_fingerprint(file_path)
    if fp is None:
        return detect_symbol(file_path, line_no, file_lines)
    idx = _load_index(root)
    files = idx.setdefault("files", {})
    rel = str(file_path.resolve().relative_to(root))
    entry = files.get(rel)
    key = [fp[0], fp[1], fp[2]]
    if not entry or entry.get("key") != key:
        entry = {"key": key, "spans": {}}
        files[rel] = entry
    spans = entry.setdefault("spans", {})
    sline = str(line_no)
    if sline in spans:
        s = spans[sline]
        return (s.get("symbol"), int(s["start"]), int(s["end"]))
    symbol, start, end = detect_symbol(file_path, line_no, file_lines)
    spans[sline] = {"symbol": symbol, "start": start, "end": end}
    _save_index(root, idx)
    return (symbol, start, end)


# ---- Ollama embedding (semantic mode) ------------------------------------- #

OLLAMA_URL = _os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = _os.environ.get(
    "AGENTIC_SEMANTIC_SEARCH_EMBED_MODEL",
    _os.environ.get("CODEBASE_SEARCH_EMBED_MODEL", "nomic-embed-text"),
)


def _ollama_embed(text: str, timeout: float = 6.0) -> list[float] | None:
    body = _json.dumps({"model": OLLAMA_EMBED_MODEL, "prompt": text}).encode("utf-8")
    req = _urlreq.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
        emb = payload.get("embedding")
        if isinstance(emb, list) and emb:
            return [float(x) for x in emb]
    except (_urlerr.URLError, _urlerr.HTTPError, OSError, ValueError):
        return None
    return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _embed_for_chunk(
    chunk: "Chunk",
    roots: list[_Path],
) -> list[float] | None:
    """Embed a chunk, caching per file fingerprint + (start, end)."""
    p = _Path(chunk.file)
    root = _index_root_for(p, roots)
    if root is None:
        return _ollama_embed(chunk.text[:8000])
    fp = _file_fingerprint(p)
    if fp is None:
        return _ollama_embed(chunk.text[:8000])
    idx = _load_index(root)
    files = idx.setdefault("files", {})
    rel = str(p.resolve().relative_to(root))
    entry = files.get(rel)
    key = [fp[0], fp[1], fp[2]]
    if not entry or entry.get("key") != key:
        entry = {"key": key, "spans": {}, "embeds": {}}
        files[rel] = entry
    embeds = entry.setdefault("embeds", {})
    span_key = f"{chunk.start}-{chunk.end}"
    if span_key in embeds:
        return embeds[span_key]
    vec = _ollama_embed(chunk.text[:8000])
    if vec is not None:
        embeds[span_key] = vec
        _save_index(root, idx)
    return vec



# ---- Cross-encoder rerank (sentence-transformers) ------------------------- #

CROSS_ENCODER_MODEL = _os.environ.get(
    "AGENTIC_SEMANTIC_SEARCH_CROSS_ENCODER",
    _os.environ.get(
        "CODEBASE_SEARCH_CROSS_ENCODER",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ),
)
_CROSS_ENCODER_CACHE: dict[str, object] = {}
_CROSS_ENCODER_DISABLED = False


def _load_cross_encoder():
    """Lazily load a sentence-transformers CrossEncoder. Returns None if unavailable."""
    global _CROSS_ENCODER_DISABLED
    if _CROSS_ENCODER_DISABLED:
        return None
    cached = _CROSS_ENCODER_CACHE.get(CROSS_ENCODER_MODEL)
    if cached is not None:
        return cached
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as exc:
        sys.stderr.write(
            f"cross-encoder: sentence-transformers unavailable ({exc!r}); "
            "falling back to Ollama cosine rerank\n"
        )
        _CROSS_ENCODER_DISABLED = True
        return None
    try:
        model = CrossEncoder(CROSS_ENCODER_MODEL)
    except Exception as exc:
        sys.stderr.write(
            f"cross-encoder: failed to load {CROSS_ENCODER_MODEL!r} ({exc!r}); "
            "falling back to Ollama cosine rerank\n"
        )
        _CROSS_ENCODER_DISABLED = True
        return None
    _CROSS_ENCODER_CACHE[CROSS_ENCODER_MODEL] = model
    return model


def _cross_encoder_rerank(
    query: str,
    chunks: list["Chunk"],
    char_cap: int = 2000,
) -> list[float] | None:
    """Score (query, chunk) pairs with a cross-encoder. Returns None if unavailable."""
    model = _load_cross_encoder()
    if model is None or not chunks:
        return None
    pairs = [(query, c.text[:char_cap]) for c in chunks]
    try:
        scores = model.predict(pairs)
    except Exception as exc:
        sys.stderr.write(f"cross-encoder: predict failed ({exc!r}); falling back\n")
        return None
    try:
        return [float(s) for s in scores]
    except Exception:
        return None



if __name__ == "__main__":
    sys.exit(main())
