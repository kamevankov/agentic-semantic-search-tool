"""Opt-in network/model integration tests.

Run through `npm run test:integration` after installing the semantic and
cross-encoder requirements. The ordinary unit suite imports this file but
skips the heavyweight checks.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
PYTHON_DIR = HERE.parent.parent / "python"
SCRIPT = PYTHON_DIR / "agentic_semantic_search.py"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

spec = importlib.util.spec_from_file_location("agentic_semantic_search_integration", SCRIPT)
engine = importlib.util.module_from_spec(spec)
sys.modules["agentic_semantic_search_integration"] = engine
spec.loader.exec_module(engine)  # type: ignore[union-attr]

RUN_INTEGRATION = os.environ.get("AGENTIC_SEMANTIC_SEARCH_RUN_INTEGRATION") == "1"
POINTER_TEXT = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63\n"
    "size 333590944\n"
)
POINTER_URL = (
    "https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/raw/main/"
    "embeddinggemma-300M-Q8_0.gguf"
)


@unittest.skipUnless(RUN_INTEGRATION, "set AGENTIC_SEMANTIC_SEARCH_RUN_INTEGRATION=1")
class RealModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_directory = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.model_directory.name) / engine.DEFAULT_EMBEDDING_MODEL_FILE
        engine._download_model_file(
            engine.EMBEDDING_MODEL_URL,
            cls.model_path,
            engine.DEFAULT_EMBEDDING_MODEL_SHA256,
        )

    @classmethod
    def tearDownClass(cls):
        cls.model_directory.cleanup()

    def test_full_hugging_face_gguf_download_is_verified(self):
        self.assertTrue(engine._model_file_is_valid(
            self.model_path,
            expected_sha256=engine.DEFAULT_EMBEDDING_MODEL_SHA256,
            expected_size=engine.DEFAULT_EMBEDDING_MODEL_SIZE,
        ))

    def test_hugging_face_transport_and_pinned_lfs_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.pointer"
            digest = hashlib.sha256(POINTER_TEXT.encode("utf-8")).hexdigest()
            engine._download_model_file(POINTER_URL, target, digest)
            self.assertEqual(target.read_text(encoding="utf-8"), POINTER_TEXT)

    def test_real_embeddinggemma_gguf_inference(self):
        model_path = self.model_path
        self.assertTrue(engine._model_file_is_valid(
            model_path,
            expected_sha256=engine.DEFAULT_EMBEDDING_MODEL_SHA256,
            expected_size=engine.DEFAULT_EMBEDDING_MODEL_SIZE,
        ))
        engine.EMBEDDING_MODEL_PATH = str(model_path)
        engine.EMBEDDING_MODEL_SHA256 = engine.DEFAULT_EMBEDDING_MODEL_SHA256
        engine.EMBEDDING_OFFLINE = True
        engine._LOCAL_EMBEDDER = None
        engine._LOCAL_EMBEDDER_DISABLED = False
        engine._LOCAL_EMBEDDER_ERROR = None
        engine._LOCAL_MODEL_PATH = None
        vectors = engine._local_embed_many([
            engine.format_query_for_embedding("find authentication middleware"),
            "title: authenticate | text: def authenticate(request): return request.user",
        ])
        self.assertIsNotNone(vectors)
        self.assertEqual(len(vectors), 2)
        self.assertEqual({len(vector) for vector in vectors}, {768})
        for vector in vectors:
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0, places=5)
        self.assertGreater(engine._cosine(vectors[0], vectors[1]), -1.0)

    def test_real_cross_encoder_scores_pairs(self):
        engine._CROSS_ENCODER_DISABLED = False
        engine._CROSS_ENCODER_CACHE.clear()
        model = engine._load_cross_encoder()
        self.assertIsNotNone(model, "sentence-transformers cross-encoder failed to load")
        scores = engine._cross_encoder_rerank(
            "find authentication middleware",
            [
                engine.Chunk("auth.py", 1, 2, "authenticate", "authenticate the incoming user", [1]),
                engine.Chunk("colors.py", 1, 2, "palette", "select a background color", [1]),
            ],
        )
        self.assertIsNotNone(scores)
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(math.isfinite(score) for score in scores))
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
