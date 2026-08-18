#!/usr/bin/env python3
import json
import os
import sys

names = [
    "AGENTIC_SEMANTIC_SEARCH_EMBED_MODEL_PATH",
    "AGENTIC_SEMANTIC_SEARCH_MODEL_CACHE",
    "AGENTIC_SEMANTIC_SEARCH_OFFLINE",
    "AGENTIC_SEMANTIC_SEARCH_AUTO_DOWNLOAD",
    "AGENTIC_SEMANTIC_SEARCH_GPU_LAYERS",
]
print(json.dumps({name: os.environ.get(name) for name in names}), file=sys.stderr)
print("[]")
