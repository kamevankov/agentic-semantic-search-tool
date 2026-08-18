"""Smoke + unit tests for the bundled agentic semantic search engine.

Run: python3 -m unittest discover -s test/python -p 'test_*.py' -v
"""
from __future__ import annotations
import importlib.util
import io
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).parent
PYTHON_DIR = HERE.parent.parent / "python"
SCRIPT = PYTHON_DIR / "agentic_semantic_search.py"
_sys_path_value = str(PYTHON_DIR)
import sys as _sys
if _sys_path_value not in _sys.path:
    _sys.path.insert(0, _sys_path_value)

spec = importlib.util.spec_from_file_location("agentic_semantic_search", SCRIPT)
css = importlib.util.module_from_spec(spec)
_sys.modules["agentic_semantic_search"] = css
spec.loader.exec_module(css)  # type: ignore[union-attr]


class Tokenize(unittest.TestCase):
    def test_camelcase_split(self):
        toks = css.tokenize("readState getUser")
        self.assertIn("readstate", toks)
        self.assertIn("read", toks)
        self.assertIn("state", toks)
        self.assertIn("getuser", toks)
        self.assertIn("user", toks)

    def test_snake_case(self):
        toks = css.tokenize("read_state")
        self.assertIn("read_state", toks)
        self.assertIn("read", toks)
        self.assertIn("state", toks)


class SymbolDetect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "a.py").write_text(
            "import os\n\n"
            "def alpha(x):\n"
            "    return x + 1\n"
            "\n"
            "class Beta:\n"
            "    def method(self):\n"
            "        return 42\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_python_function(self):
        lines = (self.root / "a.py").read_text().splitlines()
        sym, start, end = css.detect_symbol(self.root / "a.py", 4, lines)
        self.assertEqual(sym, "alpha")
        self.assertEqual(start, 3)
        self.assertGreaterEqual(end, 4)

    def test_python_class_method(self):
        lines = (self.root / "a.py").read_text().splitlines()
        sym, start, end = css.detect_symbol(self.root / "a.py", 7, lines)
        # Walk-up grabs the outermost (class) when nested method is hit.
        self.assertIn(sym, {"Beta", "method"})


class BM25(unittest.TestCase):
    def test_bm25_ordering(self):
        chunks = [
            css.Chunk(file="a", start=1, end=10, symbol=None,
                      text="alpha alpha beta", hit_lines=[1]),
            css.Chunk(file="b", start=1, end=10, symbol=None,
                      text="gamma delta", hit_lines=[1]),
        ]
        scores = css.bm25_scores(chunks, css.tokenize("alpha"))
        self.assertGreater(scores[0], scores[1])


class CacheRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "x.py").write_text(
            "def needle():\n    return 1\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_cached_chunk_persists_and_reuses(self):
        f = self.root / "x.py"
        lines = f.read_text().splitlines()
        sym, start, end = css.cached_chunk_for(f, 1, lines, [self.root])
        self.assertEqual(sym, "needle")
        # Verify cache file exists.
        idx = css._index_path_for(self.root)
        self.assertTrue(idx.exists())
        data = json.loads(idx.read_text())
        self.assertIn("x.py", data["files"])
        self.assertEqual(data["files"]["x.py"]["spans"]["1"]["symbol"], "needle")
        # Second call hits the cache (we monkey-patch detect_symbol to ensure no recompute).
        called = {"n": 0}
        orig = css.detect_symbol
        def stub(*a, **k):
            called["n"] += 1
            return orig(*a, **k)
        css.detect_symbol = stub
        try:
            css.cached_chunk_for(f, 1, lines, [self.root])
        finally:
            css.detect_symbol = orig
        self.assertEqual(called["n"], 0, "cache must not call detect_symbol on hit")

    def test_cache_invalidates_on_change(self):
        f = self.root / "x.py"
        lines = f.read_text().splitlines()
        css.cached_chunk_for(f, 1, lines, [self.root])
        # Mutate the file: cache key (size+mtime+sha1) changes, entry resets.
        f.write_text("def renamed():\n    return 2\n")
        new_lines = f.read_text().splitlines()
        sym, _, _ = css.cached_chunk_for(f, 1, new_lines, [self.root])
        self.assertEqual(sym, "renamed")


class CLISmoke(unittest.TestCase):
    def test_runs_and_returns_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "demo.py").write_text("def find_me():\n    return 7\n")
            proc = subprocess.run(
                [_sys.executable, str(SCRIPT),
                 "--query", "find_me", "--root", str(d), "--limit", "3"],
                capture_output=True, text=True, check=False, timeout=15,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(proc.stdout)
            self.assertGreaterEqual(len(data), 1)
            self.assertEqual(data[0]["symbol"], "find_me")
            self.assertEqual(data[0]["lines"], [1, 2])

    def test_exact_symbol_recall_beats_generic_token_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            src = d / "src" / "config"
            src.mkdir(parents=True)
            noisy = d / "src" / "channels" / "session.types.ts"
            noisy.parent.mkdir(parents=True)
            noisy.write_text("\n".join(
                f"export type ModeSystemGeneric{i} = {{ mode: string; tools: string[] }};"
                for i in range(80)
            ))
            target = src / "mode-system-tools.ts"
            target.write_text(
                "export function registerModeSystemTools(api: unknown) {\n"
                "  return api;\n"
                "}\n"
            )
            proc = subprocess.run(
                [_sys.executable, str(SCRIPT),
                 "--query", "registerModeSystemTools", "--root", str(d / "src"),
                 "--limit", "3", "--mode", "symbols", "--no-cache"],
                capture_output=True, text=True, check=False, timeout=15,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(proc.stdout)
            self.assertGreaterEqual(len(data), 1)
            self.assertEqual(Path(data[0]["file"]).resolve(), target.resolve())
            self.assertEqual(data[0]["symbol"], "registerModeSystemTools")
            self.assertIn("exact_identifier=+", data[0]["why"])

    def test_large_symbol_output_range_is_narrowed_around_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            src = d / "src"
            src.mkdir()
            target = src / "large.ts"
            filler = "\n".join(f"  const filler{i} = {i};" for i in range(260))
            target.write_text(
                "export function hugeSessionInit() {\n"
                f"{filler}\n"
                "  const modelOverride = 'gpt-5.5';\n"
                "  const tierLockedLevel = 'high';\n"
                "  return { modelOverride, tierLockedLevel };\n"
                "}\n"
            )
            proc = subprocess.run(
                [_sys.executable, str(SCRIPT),
                 "--query", "modelOverride tierLockedLevel", "--root", str(src),
                 "--limit", "1", "--mode", "hybrid", "--no-cache"],
                capture_output=True, text=True, check=False, timeout=15,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(proc.stdout)
            self.assertGreaterEqual(len(data), 1)
            start, end = data[0]["lines"]
            self.assertLessEqual(end - start + 1, css.OUTPUT_MAX_LINE_SPAN)
            self.assertLessEqual(start, 262)
            self.assertGreaterEqual(end, 263)
            self.assertIn("output_range=narrowed_from:", data[0]["why"])


class TreeSitterDetect(unittest.TestCase):
    """Tree-sitter chunking: nested classes and arrow-function const exports."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _ts_available(self) -> bool:
        try:
            import _treesitter_chunks  # type: ignore
        except Exception:
            # _treesitter_chunks lives next to css; import via spec.
            spec = importlib.util.spec_from_file_location(
                "_treesitter_chunks", PYTHON_DIR / "_treesitter_chunks.py"
            )
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
            except Exception:
                return False
            _sys.modules["_treesitter_chunks"] = mod
        import _treesitter_chunks as _tsc  # type: ignore
        return _tsc.is_available()

    def test_nested_python_class(self):
        if not self._ts_available():
            self.skipTest("tree-sitter not installed")
        f = self.root / "nested.py"
        f.write_text(
            "class Outer:\n"
            "    class Inner:\n"
            "        def method(self):\n"
            "            return 'deep'\n"
        )
        lines = f.read_text().splitlines()
        sym, start, end = css.detect_symbol(f, 4, lines)
        # Innermost enclosing definition for line 4 is `method`.
        self.assertEqual(sym, "method")
        self.assertEqual(start, 3)
        self.assertGreaterEqual(end, 4)

    def test_nested_typescript_class(self):
        if not self._ts_available():
            self.skipTest("tree-sitter not installed")
        f = self.root / "nested.ts"
        f.write_text(
            "export class Outer {\n"
            "  inner = class Inner {\n"
            "    method(): number {\n"
            "      return 42;\n"
            "    }\n"
            "  };\n"
            "}\n"
        )
        lines = f.read_text().splitlines()
        sym, start, end = css.detect_symbol(f, 4, lines)
        # Innermost enclosing definition for line 4 is the method.
        self.assertEqual(sym, "method")

    def test_arrow_function_const_export_ts(self):
        if not self._ts_available():
            self.skipTest("tree-sitter not installed")
        f = self.root / "arrow.ts"
        f.write_text(
            "export const greet = (name: string): string => {\n"
            "  return `hi ${name}`;\n"
            "};\n"
        )
        lines = f.read_text().splitlines()
        sym, start, end = css.detect_symbol(f, 2, lines)
        self.assertEqual(sym, "greet")
        self.assertEqual(start, 1)

    def test_arrow_function_const_export_js(self):
        if not self._ts_available():
            self.skipTest("tree-sitter not installed")
        f = self.root / "arrow.js"
        f.write_text(
            "export const add = (a, b) => {\n"
            "  return a + b;\n"
            "};\n"
        )
        lines = f.read_text().splitlines()
        sym, _, _ = css.detect_symbol(f, 2, lines)
        self.assertEqual(sym, "add")

    def test_treesitter_for_java(self):
        try:
            import tree_sitter_java  # type: ignore  # noqa: F401
        except Exception:
            self.skipTest("tree-sitter-java not installed")
        f = self.root / "Foo.java"
        f.write_text(
            "public class Foo {\n"
            "  public int bar() {\n"
            "    return 1;\n"
            "  }\n"
            "}\n"
        )
        lines = f.read_text().splitlines()
        sym, _, _ = css.detect_symbol(f, 3, lines)
        self.assertEqual(sym, "bar")
        sym_top, _, _ = css.detect_symbol(f, 1, lines)
        self.assertEqual(sym_top, "Foo")

    def test_treesitter_for_c(self):
        try:
            import tree_sitter_c  # type: ignore  # noqa: F401
        except Exception:
            self.skipTest("tree-sitter-c not installed")
        f = self.root / "foo.c"
        f.write_text(
            "int add(int a, int b) {\n"
            "  return a + b;\n"
            "}\n"
            "struct Point { int x; int y; };\n"
        )
        lines = f.read_text().splitlines()
        sym, _, _ = css.detect_symbol(f, 2, lines)
        self.assertEqual(sym, "add")
        sym2, _, _ = css.detect_symbol(f, 4, lines)
        self.assertEqual(sym2, "Point")

if __name__ == "__main__":
    unittest.main()


class CrossEncoderRerank(unittest.TestCase):
    """Cover CE-available and CE-unavailable paths in semantic mode (top-50 rerank)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Two files so ripgrep produces multiple chunks for hybrid+rerank.
        (self.root / "a.py").write_text(
            "def needle_alpha():\n    # needle alpha alpha\n    return 1\n"
        )
        (self.root / "b.py").write_text(
            "def needle_beta():\n    # needle beta\n    return 2\n"
        )
        # Snapshot module-level toggles so tests don't bleed.
        self._saved_disabled = css._CROSS_ENCODER_DISABLED
        self._saved_cache = dict(css._CROSS_ENCODER_CACHE)
        self._saved_loader = css._load_cross_encoder
        self._saved_embed = css._local_embed_many
        css._local_embed_many = lambda *a, **k: None

    def tearDown(self):
        css._CROSS_ENCODER_DISABLED = self._saved_disabled
        css._CROSS_ENCODER_CACHE.clear()
        css._CROSS_ENCODER_CACHE.update(self._saved_cache)
        css._load_cross_encoder = self._saved_loader
        css._local_embed_many = self._saved_embed
        self.tmp.cleanup()

    def test_ce_available_path_emits_ce_rerank(self):
        class FakeCE:
            def predict(self, pairs):
                # Reverse rank: first chunk gets lowest, last gets highest, so order changes.
                return [float(i) * 0.5 for i in range(len(pairs))]
        css._CROSS_ENCODER_DISABLED = False
        css._load_cross_encoder = lambda: FakeCE()
        results = css.search(
            query="needle", roots=[self.root], limit=5,
            mode="semantic", rg_trace=False, use_cache=False,
        )
        self.assertGreaterEqual(len(results), 1)
        joined = " | ".join(r.why for r in results)
        self.assertIn("ce_rerank=", joined)
        self.assertIn("cross-encoder", joined)
        self.assertNotIn("ce_rerank=unavailable", joined)

    def test_ce_unavailable_falls_back_with_explanation(self):
        # Force both optional stages off so we hit the weighted-hybrid fallback.
        css._CROSS_ENCODER_DISABLED = True
        css._load_cross_encoder = lambda: None
        css._local_embed_many = lambda *a, **k: None
        results = css.search(
            query="needle", roots=[self.root], limit=5,
            mode="semantic", rg_trace=False, use_cache=False,
        )
        self.assertGreaterEqual(len(results), 1)
        joined = " | ".join(r.why for r in results)
        self.assertIn("ce_rerank=unavailable", joined)
        # Original hybrid signal is preserved in why.
        self.assertIn("bm25=", joined)


if __name__ == "__main__":
    unittest.main()


class LoadCrossEncoder(unittest.TestCase):
    """Direct unit tests for _load_cross_encoder() using sys.modules monkeypatching.

    Covers:
    - ImportError path (sentence_transformers missing) -> returns None, disables further attempts.
    - CrossEncoder constructor failure -> returns None, disables further attempts.
    - Success path -> returns stub model and caches it so subsequent calls reuse the same instance.
    """

    def setUp(self):
        self._saved_disabled = css._CROSS_ENCODER_DISABLED
        self._saved_cache = dict(css._CROSS_ENCODER_CACHE)
        self._saved_st = _sys.modules.get("sentence_transformers")
        css._CROSS_ENCODER_DISABLED = False
        css._CROSS_ENCODER_CACHE.clear()

    def tearDown(self):
        css._CROSS_ENCODER_DISABLED = self._saved_disabled
        css._CROSS_ENCODER_CACHE.clear()
        css._CROSS_ENCODER_CACHE.update(self._saved_cache)
        if self._saved_st is None:
            _sys.modules.pop("sentence_transformers", None)
        else:
            _sys.modules["sentence_transformers"] = self._saved_st

    def test_import_error_returns_none_and_disables(self):
        # Setting the sys.modules entry to None makes `from sentence_transformers import ...` raise ImportError.
        _sys.modules["sentence_transformers"] = None  # type: ignore[assignment]
        self.assertIsNone(css._load_cross_encoder())
        self.assertTrue(css._CROSS_ENCODER_DISABLED)
        # Subsequent call short-circuits via the disabled flag (no second import attempt needed).
        self.assertIsNone(css._load_cross_encoder())

    def test_constructor_failure_returns_none_and_disables(self):
        import types as _types
        fake = _types.ModuleType("sentence_transformers")
        class BoomCE:
            def __init__(self, *a, **k):
                raise RuntimeError("model download blocked")
        fake.CrossEncoder = BoomCE  # type: ignore[attr-defined]
        _sys.modules["sentence_transformers"] = fake
        self.assertIsNone(css._load_cross_encoder())
        self.assertTrue(css._CROSS_ENCODER_DISABLED)
        self.assertNotIn(css.CROSS_ENCODER_MODEL, css._CROSS_ENCODER_CACHE)

    def test_success_returns_stub_model_and_caches(self):
        import types as _types
        fake = _types.ModuleType("sentence_transformers")
        instances: list[object] = []
        class StubCE:
            def __init__(self, name):
                self.name = name
                instances.append(self)
            def predict(self, pairs):
                return [0.0 for _ in pairs]
        fake.CrossEncoder = StubCE  # type: ignore[attr-defined]
        _sys.modules["sentence_transformers"] = fake
        model = css._load_cross_encoder()
        self.assertIsNotNone(model)
        self.assertIsInstance(model, StubCE)
        self.assertFalse(css._CROSS_ENCODER_DISABLED)
        # Cached: subsequent call returns same instance, no second construction.
        again = css._load_cross_encoder()
        self.assertIs(again, model)
        self.assertEqual(len(instances), 1)


class SemanticRerankShape(unittest.TestCase):
    """Lock the JSON shape of semantic-mode results and the why-string stage marker.

    Ensures CI catches regressions in the public output contract when the cross-encoder
    fallback chain is exercised, without ever downloading a model.
    """

    EXPECTED_KEYS = {"file", "lines", "symbol", "score", "why"}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "a.py").write_text(
            "def needle_alpha():\n    # needle alpha alpha\n    return 1\n"
        )
        (self.root / "b.py").write_text(
            "def needle_beta():\n    # needle beta\n    return 2\n"
        )
        self._saved_disabled = css._CROSS_ENCODER_DISABLED
        self._saved_cache = dict(css._CROSS_ENCODER_CACHE)
        self._saved_loader = css._load_cross_encoder
        self._saved_embed = css._local_embed_many
        css._local_embed_many = lambda *a, **k: None

    def tearDown(self):
        css._CROSS_ENCODER_DISABLED = self._saved_disabled
        css._CROSS_ENCODER_CACHE.clear()
        css._CROSS_ENCODER_CACHE.update(self._saved_cache)
        css._load_cross_encoder = self._saved_loader
        css._local_embed_many = self._saved_embed
        self.tmp.cleanup()

    def _shape_ok(self, results):
        self.assertGreaterEqual(len(results), 1)
        for r in results:
            d = r.__dict__ if hasattr(r, "__dict__") else r
            self.assertTrue(self.EXPECTED_KEYS.issubset(set(d.keys())),
                            f"missing keys: {self.EXPECTED_KEYS - set(d.keys())}")
            # Score must be numeric, why must be a non-empty string mentioning the mode.
            self.assertIsInstance(d["score"], (int, float))
            self.assertIsInstance(d["why"], str)
            self.assertIn("mode=semantic", d["why"])

    def test_shape_preserved_when_ce_available(self):
        class FakeCE:
            def predict(self, pairs):
                return [float(i) * 0.5 for i in range(len(pairs))]
        css._CROSS_ENCODER_DISABLED = False
        css._load_cross_encoder = lambda: FakeCE()
        results = css.search(
            query="needle", roots=[self.root], limit=5,
            mode="semantic", rg_trace=False, use_cache=False,
        )
        self._shape_ok(results)
        why = " | ".join(r.why for r in results)
        # Why string reflects the chosen stage (cross-encoder rerank).
        self.assertIn("ce_rerank=", why)
        self.assertNotIn("ce_rerank=unavailable", why)
        self.assertIn("cross-encoder", why)

    def test_shape_preserved_on_full_fallback(self):
        css._CROSS_ENCODER_DISABLED = True
        css._load_cross_encoder = lambda: None
        css._local_embed_many = lambda *a, **k: None
        results = css.search(
            query="needle", roots=[self.root], limit=5,
            mode="semantic", rg_trace=False, use_cache=False,
        )
        self._shape_ok(results)
        why = " | ".join(r.why for r in results)
        # Why string flips to the unavailable-fallback marker.
        self.assertIn("ce_rerank=unavailable", why)
        self.assertIn("semantic=unavailable", why)

    def test_shape_preserved_when_ce_predict_raises(self):
        """Stub CE whose predict() raises must trigger the cosine fallback path,
        not crash, and the result shape must still be valid."""
        class BoomCE:
            def predict(self, pairs):
                raise RuntimeError("predict exploded")
        css._CROSS_ENCODER_DISABLED = False
        css._load_cross_encoder = lambda: BoomCE()
        # The local backend returns deterministic vectors for query and documents.
        css._local_embed_many = lambda texts: [[1.0, 0.0] for _ in texts]
        results = css.search(
            query="needle", roots=[self.root], limit=5,
            mode="semantic", rg_trace=False, use_cache=False,
        )
        self._shape_ok(results)
        why = " | ".join(r.why for r in results)
        # CE was attempted but predict raised, so local cosine ranking is retained.
        self.assertIn("ce_rerank=unavailable", why)
        self.assertIn("local EmbeddingGemma cosine rerank", why)
        # Per-result cosine signal recorded as semantic=<float>, not "unavailable".
        self.assertNotIn("semantic=unavailable", why)

    def test_shape_preserved_on_cosine_fallback_when_ce_none(self):
        """CE unavailable + local embeddings must use the weighted cosine path
        and emit a numeric semantic= score per result."""
        css._CROSS_ENCODER_DISABLED = True
        css._load_cross_encoder = lambda: None
        css._local_embed_many = lambda texts: [[0.5, 0.5] for _ in texts]
        results = css.search(
            query="needle", roots=[self.root], limit=5,
            mode="semantic", rg_trace=False, use_cache=False,
        )
        self._shape_ok(results)
        why = " | ".join(r.why for r in results)
        self.assertIn("ce_rerank=unavailable", why)
        self.assertIn("local EmbeddingGemma cosine rerank", why)
        self.assertNotIn("semantic=unavailable", why)


class LocalEmbeddingBackend(unittest.TestCase):
    """Exercise model resolution, safe downloads, loader compatibility, and cache keys."""

    CONFIG_NAMES = (
        "EMBEDDING_MODEL_REPO",
        "EMBEDDING_MODEL_FILE",
        "EMBEDDING_MODEL_URL",
        "EMBEDDING_MODEL_SHA256",
        "EMBEDDING_MODEL_PATH",
        "EMBEDDING_MODEL_CACHE",
        "EMBEDDING_AUTO_DOWNLOAD",
        "EMBEDDING_OFFLINE",
        "EMBEDDING_DOWNLOAD_TIMEOUT",
        "EMBEDDING_GPU_LAYERS",
    )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.saved = {name: getattr(css, name) for name in self.CONFIG_NAMES}
        self.saved_llama = _sys.modules.get("llama_cpp")
        self.saved_download = css._download_model_file
        self._reset_backend()

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(css, name, value)
        css._download_model_file = self.saved_download
        if self.saved_llama is None:
            _sys.modules.pop("llama_cpp", None)
        else:
            _sys.modules["llama_cpp"] = self.saved_llama
        self._reset_backend()
        self.tmp.cleanup()

    def _reset_backend(self):
        css._LOCAL_EMBEDDER = None
        css._LOCAL_EMBEDDER_DISABLED = False
        css._LOCAL_EMBEDDER_ERROR = None
        css._LOCAL_MODEL_PATH = None

    def test_qmd_compatible_prompt_format(self):
        chunk = css.Chunk(
            file="src/example.py", start=1, end=2, symbol="answer",
            text="def answer():\n    return 42", hit_lines=[1],
        )
        self.assertEqual(
            css.format_query_for_embedding("find the answer"),
            "task: search result | query: find the answer",
        )
        self.assertEqual(
            css.format_document_for_embedding(chunk),
            "title: answer | text: def answer():\n    return 42",
        )

    def test_download_is_atomic_and_rejects_incomplete_payload(self):
        class Response:
            def __init__(self, payload, declared):
                self.stream = io.BytesIO(payload)
                self.headers = {"Content-Length": str(declared)}
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, size):
                return self.stream.read(size)

        original_urlopen = css._urlreq.urlopen
        try:
            css._urlreq.urlopen = lambda *a, **k: Response(b"model-data", 10)
            target = self.root / "models" / "model.gguf"
            self.assertEqual(css._download_model_file("https://example.test/model", target), target)
            self.assertEqual(target.read_bytes(), b"model-data")
            self.assertEqual(list(target.parent.glob("*.part")), [])

            css._urlreq.urlopen = lambda *a, **k: Response(b"short", 50)
            bad_target = self.root / "models" / "bad.gguf"
            with self.assertRaises(OSError):
                css._download_model_file("https://example.test/bad", bad_target)
            self.assertFalse(bad_target.exists())
            self.assertEqual(list(bad_target.parent.glob("*.part")), [])
            with self.assertRaises(ValueError):
                css._download_model_file("http://example.test/model", bad_target)
            css._urlreq.urlopen = lambda *a, **k: Response(b"model-data", 10)
            with self.assertRaises(OSError):
                css._download_model_file(
                    "https://example.test/model",
                    bad_target,
                    "0" * 64,
                )
            self.assertFalse(bad_target.exists())
        finally:
            css._urlreq.urlopen = original_urlopen

    def test_offline_missing_model_never_downloads(self):
        css.EMBEDDING_MODEL_REPO = "example/custom"
        css.EMBEDDING_MODEL_FILE = "missing.gguf"
        css.EMBEDDING_MODEL_PATH = ""
        css.EMBEDDING_MODEL_CACHE = str(self.root / "empty-cache")
        css.EMBEDDING_OFFLINE = True
        called = {"count": 0}
        def fail_download(*args, **kwargs):
            called["count"] += 1
            raise AssertionError("offline resolution attempted a download")
        css._download_model_file = fail_download
        self.assertIsNone(css._resolve_embedding_model_path())
        self.assertEqual(called["count"], 0)

    def test_missing_llama_dependency_does_not_download_model(self):
        _sys.modules["llama_cpp"] = None  # type: ignore[assignment]
        css.EMBEDDING_MODEL_PATH = ""
        css.EMBEDDING_MODEL_CACHE = str(self.root / "empty-cache")
        called = {"count": 0}
        def count_download(*args, **kwargs):
            called["count"] += 1
            return self.root / "never.gguf"
        css._download_model_file = count_download
        self.assertIsNone(css._load_local_embedder())
        self.assertEqual(called["count"], 0)

    def test_loader_supports_embedding_flag_and_normalizes_vectors(self):
        model_path = self.root / "model.gguf"
        model_path.write_bytes(b"gguf")
        css.EMBEDDING_MODEL_PATH = str(model_path)
        instances = []
        fake = types.ModuleType("llama_cpp")

        class FakeLlama:
            def __init__(self, model_path, embedding=False, **kwargs):
                self.model_path = model_path
                self.embedding = embedding
                instances.append(self)
            def embed(self, texts, normalize=True, truncate=True):
                self.call = (list(texts), normalize, truncate)
                return [[3.0, 4.0] for _ in texts]

        fake.Llama = FakeLlama  # type: ignore[attr-defined]
        _sys.modules["llama_cpp"] = fake
        vectors = css._local_embed_many(["one", "two"])
        self.assertEqual(vectors, [[0.6, 0.8], [0.6, 0.8]])
        self.assertTrue(instances[0].embedding)
        self.assertEqual(instances[0].call, (["one", "two"], True, True))

    def test_loader_supports_plural_flag_and_openai_response(self):
        model_path = self.root / "model.gguf"
        model_path.write_bytes(b"gguf")
        css.EMBEDDING_MODEL_PATH = str(model_path)
        instances = []
        fake = types.ModuleType("llama_cpp")

        class FakeLlama:
            def __init__(self, model_path, embeddings=False, **kwargs):
                self.embeddings = embeddings
                instances.append(self)
            def create_embedding(self, texts):
                return {"data": [
                    {"index": index, "embedding": [0.0, float(index + 1)]}
                    for index, _ in enumerate(texts)
                ]}

        fake.Llama = FakeLlama  # type: ignore[attr-defined]
        _sys.modules["llama_cpp"] = fake
        vectors = css._local_embed_many(["one", "two"])
        self.assertEqual(vectors, [[0.0, 1.0], [0.0, 1.0]])
        self.assertTrue(instances[0].embeddings)

    def test_chunk_embeddings_are_batched_cached_and_model_aware(self):
        source = self.root / "source.py"
        source.write_text("def answer():\n    return 42\n")
        model_path = self.root / "model.gguf"
        model_path.write_bytes(b"model-a")
        css._LOCAL_MODEL_PATH = model_path
        chunk = css.Chunk(
            file=str(source), start=1, end=2, symbol="answer",
            text=source.read_text(), hit_lines=[1],
        )
        original_embed = css._local_embed_many
        calls = []
        css._local_embed_many = lambda texts: calls.append(list(texts)) or [[1.0, 0.0] for _ in texts]
        try:
            first = css._embeddings_for_chunks([chunk], [self.root], use_cache=True)
            second = css._embeddings_for_chunks([chunk], [self.root], use_cache=True)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            identity_a = css._embedding_model_identity()
            model_path.write_bytes(b"model-b-with-different-size")
            identity_b = css._embedding_model_identity()
            self.assertNotEqual(identity_a, identity_b)
            css._embeddings_for_chunks([chunk], [self.root], use_cache=True)
            self.assertEqual(len(calls), 2)
        finally:
            css._local_embed_many = original_embed
