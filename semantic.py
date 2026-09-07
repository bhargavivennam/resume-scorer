"""
Semantic similarity via pretrained sentence embeddings.

Why embeddings and not LSA
--------------------------
The obvious cheap alternative was LSA (TruncatedSVD over the existing TF-IDF).
It was rejected for a concrete reason, not a stylistic one: **LSA learns its
semantic space from the corpus you fit it on**, and ours is 1,500 synthetic
resumes from `data.py`. It would have learned the generator's phrasing rather
than English meaning — re-introducing exactly the kind of artifact the project
spent three rounds of shortcut-fixing to remove.

A pretrained sentence encoder needs no corpus. `all-MiniLM-L6-v2` is trained on
over a billion sentence pairs, is ~90MB, and runs on CPU in milliseconds.

What this fixes
---------------
Lexical matching only connects a requirement to a resume when the literal
tokens overlap. These pairs mean the same thing and share almost no tokens:

    "experience training machine learning models"
        vs "trained LightGBM models and evaluated with AUC-ROC"
    "comfortable owning services in production"
        vs "handled L3 production incidents and on-call rotations"
    "build models from scratch"
        vs "designed a graded ranking model from first principles"

The ontology handles synonyms for *named skills*. Embeddings handle everything
else — which, on a real posting, is most of it.

Optional by design
------------------
`sentence-transformers` pulls in torch, which is a heavy dependency for a text
scorer. Every entry point degrades to lexical matching when it is absent, so the
project stays installable from `requirements.txt` alone. `is_available()` says
which mode you are in.
"""

from __future__ import annotations

import functools
import os
import threading
from typing import List, Optional, Sequence

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# OFF by default. Two measured reasons, not a stylistic preference:
#
#   1. On a real resume/posting pair it moved the match score by ONE point
#      (63 -> 64). The ontology and implication graph were already catching
#      what it was supposed to add.
#   2. torch and LightGBM each bundle an OpenMP runtime, and loading both in
#      one process segfaults on macOS. That is a hard failure in the default
#      `pytest tests` run, traded for a rounding error in accuracy.
#
# Enable with RESUME_SCORER_SEMANTIC=1 when you want it — the code, and the
# tests that prove it works, are all still here.
ENV_FLAG = "RESUME_SCORER_SEMANTIC"

# Loading the model is slow (~1-2s) and thread-unsafe under concurrent first
# use, which a web server will produce on its first two requests.
_lock = threading.Lock()
_load_failed = False


@functools.lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def _encoder_installed() -> bool:
    """Whether the library is present, independent of the opt-in flag."""
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def get_model():
    """Return the encoder, or None if unavailable/disabled. Never raises."""
    global _load_failed
    if _load_failed or not enabled():
        return None
    with _lock:
        try:
            return _model()
        except Exception:
            # Missing dependency, no local cache and no network, corrupt
            # download — all mean the same thing to callers: use lexical.
            _load_failed = True
            return None


def is_available() -> bool:
    return get_model() is not None


def encode(texts: Sequence[str]) -> Optional[np.ndarray]:
    """L2-normalised embeddings, or None when the encoder is unavailable."""
    model = get_model()
    if model is None or not texts:
        return None
    try:
        return model.encode(list(texts), normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
    except Exception:
        return None


def similarity(a: str, b: str) -> Optional[float]:
    """Cosine similarity of two texts in [0, 1], or None."""
    vecs = encode([a, b])
    if vecs is None:
        return None
    return float(np.clip(vecs[0] @ vecs[1], 0.0, 1.0))


def best_match(query: str, candidates: Sequence[str]) -> Optional[tuple[int, float]]:
    """Index and score of the candidate closest in meaning to `query`."""
    if not candidates:
        return None
    vecs = encode([query, *candidates])
    if vecs is None:
        return None
    scores = vecs[1:] @ vecs[0]
    idx = int(np.argmax(scores))
    return idx, float(np.clip(scores[idx], 0.0, 1.0))


def rescale(raw: float, floor: float = 0.25, ceiling: float = 0.75) -> float:
    """Map a raw cosine onto a usable 0-1 range.

    Sentence encoders put almost all unrelated English text in the 0.0-0.3 band
    and near-paraphrases around 0.7-0.85; a raw cosine of 0.5 is already a
    strong match. Reporting the raw number as a percentage would make every
    score look mediocre, so the useful band is stretched across the full range.
    """
    if ceiling <= floor:
        return raw
    return float(np.clip((raw - floor) / (ceiling - floor), 0.0, 1.0))
