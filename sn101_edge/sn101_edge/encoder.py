"""Shared, cached embedding model.

The validator scores with all-MiniLM-L6-v2. We load the same checkpoint so our
local diversity and validity estimates are not approximations -- they are the
identical computation the validator will run.

Every candidate tag gets embedded many times during the triple search, so the
cache is what makes exhaustive selection affordable inside a 10s budget.
"""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

from . import config


class CachedEncoder:
    """Embeds text to L2-normalised vectors, memoising by exact string.

    Exposes `.encode(...)` with the sentence-transformers signature so it can be
    handed straight to the validator's own scorer classes as `model=`.
    """

    def __init__(self, model=None, dim: int | None = None):
        self._model = model
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._dim = dim

    # -- sentence-transformers compatible surface ---------------------------
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False, **_kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        missing = []
        with self._lock:
            for t in texts:
                if t not in self._cache:
                    missing.append(t)
        missing = list(dict.fromkeys(missing))

        if missing:
            vectors = self._encode_uncached(missing)
            with self._lock:
                for text, vec in zip(missing, vectors):
                    self._cache[text] = vec

        with self._lock:
            out = np.stack([self._cache[t] for t in texts])
        return out

    def _encode_uncached(self, texts: Sequence[str]) -> np.ndarray:
        model = self.model
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors

    # -- lazy model loading --------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(config.EMBED_MODEL)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode(["probe"]).shape[1])
        return self._dim

    def similarity_matrix(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.encode(list(texts))
        return vecs @ vecs.T

    def max_pairwise_similarity(self, texts: Sequence[str]) -> float:
        if len(texts) < 2:
            return 0.0
        sim = self.similarity_matrix(texts)
        np.fill_diagonal(sim, -1.0)
        return float(np.max(sim))

    def warm(self) -> None:
        """Force model load + a trivial encode so the first real task does not
        pay the ~2-4s import and checkpoint cost inside a validator deadline.

        Also preloads sklearn, whose first import alone costs ~1.8s and would
        otherwise land inside the first task's budget.
        """
        try:
            import sklearn.cluster  # noqa: F401
        except ImportError:
            pass
        self.encode(["warmup"])


_ENCODER: CachedEncoder | None = None
_ENCODER_LOCK = threading.Lock()


def get_encoder() -> CachedEncoder:
    global _ENCODER
    if _ENCODER is None:
        with _ENCODER_LOCK:
            if _ENCODER is None:
                _ENCODER = CachedEncoder()
    return _ENCODER


def set_encoder(encoder: CachedEncoder) -> None:
    """Injection point for offline tests."""
    global _ENCODER
    with _ENCODER_LOCK:
        _ENCODER = encoder


def warm_in_background() -> None:
    thread = threading.Thread(target=lambda: get_encoder().warm(), daemon=True)
    thread.start()


class HashEncoder:
    """Deterministic stand-in used by the offline self-test.

    Produces stable pseudo-random unit vectors with light lexical coupling, so
    selection logic can be exercised without downloading a checkpoint. NEVER
    use this in production -- diversity estimates would be meaningless.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False, **_kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in text.lower().split():
                rng = np.random.default_rng(abs(hash(token)) % (2**32))
                vec += rng.normal(size=self.dim).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            out.append(vec / norm if norm > 0 else vec)
        return np.stack(out) if out else np.zeros((0, self.dim), dtype=np.float32)
