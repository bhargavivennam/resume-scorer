"""Test-session setup.

Do NOT import torch here. torch and LightGBM each bundle an OpenMP runtime, and
loading both into one process segfaults LightGBM's `predict` on macOS. Semantic
matching is opt-in (RESUME_SCORER_SEMANTIC=1) precisely so torch stays out of
the default test run; importing it here would defeat that.

The semantic tests enable the encoder themselves and are the only place torch
loads — run them on their own if you have both installed:

    pytest tests --ignore=tests/test_semantic.py     # default, no torch
    pytest tests/test_semantic.py                     # encoder tests
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
