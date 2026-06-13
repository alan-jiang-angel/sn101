from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .preprocessing import ScoringContext, build_scoring_context


@dataclass
class ConsensusDebug:
    cluster_id: int
    support: float
    tags: list[str]
    miner_indices: list[int]


class ConsensusScorer:
    """Independent consensus scorer with embedding + clustering."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_tags_per_miner: int = 3,
        proximity_rank_decay: float = 1.0,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.n_clusters = n_tags_per_miner * 2
        self.n_tags_per_miner = n_tags_per_miner
        self.proximity_rank_decay = proximity_rank_decay
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
                "consensus_scores": [],
                "clusters": [],
                "normalized_responses": [],
            }

        if not context.flat_tags:
            return {
                "consensus_scores": [[] for _ in range(context.miner_count)],
                "clusters": [],
                "normalized_responses": context.normalized_responses,
            }

        tag_embeddings = self._embed_texts(context.flat_tags)
        (
            flat_consensus_scores,
            labels,
            cluster_supports,
            cluster_debug,
        ) = self._compute_consensus_scores(
            tag_embeddings=tag_embeddings,
            flat_tags=context.flat_tags,
            flat_miner_idx=context.flat_miner_idx,
            miner_count=context.miner_count,
        )

        consensus_scores = self._unflatten_scores(
            normalized_responses=context.normalized_responses,
            flat_scores=flat_consensus_scores,
        )
        clusters = self._build_cluster_records(
            labels=labels,
            supports=cluster_supports,
            cluster_debug=cluster_debug,
        )
        return {
            "consensus_scores": consensus_scores,
            "clusters": clusters,
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

    def _tag_proximity_by_rank(
        self,
        indices: list[int],
        flat_tags: list[str],
        tag_embeddings: np.ndarray,
    ) -> dict[str, float]:
        """Map each tag string to exp(-decay * (rank-1)) by centroid distance."""
        if not indices:
            return {}

        cluster_vectors = tag_embeddings[indices]
        centroid = cluster_vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        tag_vectors: dict[str, list[np.ndarray]] = {}
        for idx in indices:
            tag_vectors.setdefault(flat_tags[idx], []).append(tag_embeddings[idx])

        distances: dict[str, float] = {}
        for tag, vectors in tag_vectors.items():
            tag_vector = np.mean(vectors, axis=0)
            tag_norm = np.linalg.norm(tag_vector)
            if tag_norm > 0:
                tag_vector = tag_vector / tag_norm
            distances[tag] = float(1.0 - np.dot(tag_vector, centroid))

        decay = self.proximity_rank_decay
        out: dict[str, float] = {}
        for tag, distance in distances.items():
            rank = 1 + sum(
                1
                for other_distance in distances.values()
                if other_distance < distance
                and not np.isclose(other_distance, distance)
            )
            out[tag] = float(np.exp(-decay * (rank - 1)))
        return out

    def _compute_consensus_scores(
        self,
        tag_embeddings: np.ndarray,
        flat_tags: list[str],
        flat_miner_idx: list[int],
        miner_count: int,
    ) -> tuple[
        list[float],
        np.ndarray,
        dict[int, float],
        list[ConsensusDebug],
    ]:
        n_tags = tag_embeddings.shape[0]
        if n_tags == 1:
            labels = np.array([0], dtype=int)
        else:
            n_clusters = min(self.n_clusters, len(set(flat_tags)))
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="cosine",
                    linkage="average",
                )
            except TypeError:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity="cosine",
                    linkage="average",
                )
            labels = clustering.fit_predict(tag_embeddings)

        cluster_to_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            cluster_to_indices.setdefault(int(label), []).append(idx)

        cluster_supports: dict[int, float] = {}
        cluster_tag_proximity: dict[int, dict[str, float]] = {}
        cluster_debug: list[ConsensusDebug] = []

        for cluster_id, indices in cluster_to_indices.items():
            miner_set = {flat_miner_idx[i] for i in indices}
            support = len(miner_set) / max(miner_count, 1)
            cluster_supports[cluster_id] = support
            cluster_tag_proximity[cluster_id] = self._tag_proximity_by_rank(
                indices, flat_tags, tag_embeddings
            )

            cluster_debug.append(
                ConsensusDebug(
                    cluster_id=cluster_id,
                    support=support,
                    tags=[flat_tags[i] for i in indices],
                    miner_indices=sorted(miner_set),
                )
            )

        consensus_scores: list[float] = []
        for idx, label in enumerate(labels):
            cluster_id = int(label)
            support = cluster_supports[cluster_id]
            tag = flat_tags[idx]
            centroid_proximity = cluster_tag_proximity[cluster_id].get(tag, 1.0)
            consensus = support * centroid_proximity
            consensus_scores.append(float(np.clip(consensus, 0.0, 1.0)))

        return consensus_scores, labels, cluster_supports, cluster_debug

    def _unflatten_scores(
        self,
        normalized_responses: list[list[str]],
        flat_scores: list[float],
    ) -> list[list[float]]:
        scores: list[list[float]] = []
        cursor = 0
        for tags in normalized_responses:
            count = len(tags)
            scores.append(flat_scores[cursor : cursor + count])
            cursor += count
        return scores

    def _build_cluster_records(
        self,
        labels: np.ndarray,
        supports: dict[int, float],
        cluster_debug: list[ConsensusDebug],
    ) -> list[dict[str, Any]]:
        _ = labels
        records: list[dict[str, Any]] = []
        debug_map = {item.cluster_id: item for item in cluster_debug}
        for cluster_id in sorted(supports):
            info = debug_map[cluster_id]
            records.append(
                {
                    "cluster_id": cluster_id,
                    "support": supports[cluster_id],
                    "tags": info.tags,
                    "miner_indices": info.miner_indices,
                }
            )
        return records
