from __future__ import annotations

from typing import Any

import numpy as np

from .preprocessing import ScoringContext, build_scoring_context


class DiversityScorer:
    """Independent diversity scorer for intra-miner tag redundancy."""

    DIVERSE_SIMILARITY_THRESHOLD = 0.55
    DUPLICATE_SIMILARITY_THRESHOLD = 0.85

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_tags_per_miner: int = 3,
        model: Any | None = None,
        diverse_similarity_threshold: float = DIVERSE_SIMILARITY_THRESHOLD,
        duplicate_similarity_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.n_tags_per_miner = n_tags_per_miner
        self.diverse_similarity_threshold = diverse_similarity_threshold
        self.duplicate_similarity_threshold = duplicate_similarity_threshold
        self._model = model or self._load_model(model_name)

    def score(self, post: str, responses: list[list[str]]) -> dict[str, Any]:
        context = build_scoring_context(
            post=post,
            responses=responses,
            n_tags_per_miner=self.n_tags_per_miner,
        )
        return self.score_from_context(context)

    def score_from_context(self, context: ScoringContext) -> dict[str, Any]:
        if context.miner_count == 0:
            return {
                "diversity_scores": [],
                "diversity_details": [],
                "normalized_responses": [],
            }

        if not context.flat_tags:
            return {
                "diversity_scores": [[] for _ in range(context.miner_count)],
                "diversity_details": [[] for _ in range(context.miner_count)],
                "normalized_responses": context.normalized_responses,
            }

        tag_embeddings = self._embed_texts(context.flat_tags)
        diversity_scores, diversity_details = self._compute_scores(
            normalized_responses=context.normalized_responses,
            tag_embeddings=tag_embeddings,
        )

        return {
            "diversity_scores": diversity_scores,
            "diversity_details": diversity_details,
            "normalized_responses": context.normalized_responses,
        }

    def _load_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install it with "
                "'pip install sentence-transformers'."
            ) from exc
        return SentenceTransformer(model_name)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors

    def _compute_scores(
        self,
        normalized_responses: list[list[str]],
        tag_embeddings: np.ndarray,
    ) -> tuple[list[list[float]], list[list[dict[str, Any]]]]:
        scores: list[list[float]] = []
        details: list[list[dict[str, Any]]] = []
        cursor = 0

        for tags in normalized_responses:
            count = len(tags)
            miner_embeddings = tag_embeddings[cursor: cursor + count]
            miner_scores: list[float] = []
            miner_details: list[dict[str, Any]] = []

            for idx, _tag in enumerate(tags):
                if count <= 1:
                    miner_scores.append(1.0)
                    miner_details.append(
                        {
                            "nearest_tag": None,
                            "max_similarity": 0.0,
                        }
                    )
                    continue

                similarities = np.dot(miner_embeddings, miner_embeddings[idx])
                similarities[idx] = -1.0
                nearest_idx = int(np.argmax(similarities))
                max_similarity = float(
                    np.clip(similarities[nearest_idx], 0.0, 1.0)
                )
                diversity = self._novelty_from_similarity(max_similarity)

                miner_scores.append(diversity)
                miner_details.append(
                    {
                        "nearest_tag": tags[nearest_idx],
                        "max_similarity": max_similarity,
                    }
                )

            scores.append(miner_scores)
            details.append(miner_details)
            cursor += count

        return scores, details

    def _novelty_from_similarity(self, similarity: float) -> float:
        low = self.diverse_similarity_threshold
        high = self.duplicate_similarity_threshold
        if similarity <= low:
            return 1.0
        if similarity >= high:
            return 0.0
        return float(1.0 - ((similarity - low) / (high - low)))
