"""Pytest configuration: stub heavy optional dependencies (onnxruntime, rembg)
so that server-level tests can run without the full ML stack installed."""

import sys
from unittest.mock import MagicMock

# Stub onnxruntime and rembg before any test module imports them so that
# tests/test_server.py can import src.server without the ML stack.
for _mod in ("onnxruntime", "rembg"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
