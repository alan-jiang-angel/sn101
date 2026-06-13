from __future__ import annotations

from typing import Any

from .consensus import ConsensusScorer
from .diversity import DiversityScorer
from .preprocessing import (
    aggregate_miner_score,
    build_scoring_context,
    unflatten_scores,
)
from .validity import ValidityScorer


class TagScorer:
    """Phase 1 scorer composed from independent tag scorers."""

    N_TAGS_PER_MINER = 3
    CONSENSUS_WEIGHT = 0.60
    VALIDITY_DIVERSITY_WEIGHT = 0.40

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_tags_per_miner: int = N_TAGS_PER_MINER,
        proximity_rank_decay: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.n_tags_per_miner = n_tags_per_miner
        self.proximity_rank_decay = proximity_rank_decay
        self._model = self._load_model(model_name)
        self._consensus_scorer = ConsensusScorer(
            model_name=model_name,
            n_tags_per_miner=n_tags_per_miner,
            proximity_rank_decay=proximity_rank_decay,
            model=self._model,
        )
        self._validity_scorer = ValidityScorer(
            model_name=model_name,
            n_tags_per_miner=n_tags_per_miner,
            model=self._model,
        )
        self._diversity_scorer = DiversityScorer(
            model_name=model_name,
            n_tags_per_miner=n_tags_per_miner,
            model=self._model,
        )

    def score(self, post: str, responses: list[list[str]]) -> dict[str, Any]:
        context = build_scoring_context(
            post=post,
            responses=responses,
            n_tags_per_miner=self.n_tags_per_miner,
        )
        consensus_result = self._consensus_scorer.score_from_context(context)
        validity_result = self._validity_scorer.score_from_context(context)
        diversity_result = self._diversity_scorer.score_from_context(context)

        normalized_responses = consensus_result["normalized_responses"]
        consensus_scores = consensus_result["consensus_scores"]
        validity_scores = validity_result["validity_scores"]
        diversity_scores = diversity_result["diversity_scores"]

        if not normalized_responses:
            return {
                "tag_scores": [],
                "miner_scores": [],
                "consensus_scores": [],
                "validity_scores": [],
                "diversity_scores": [],
                "clusters": [],
                "normalized_responses": [],
            }

        flat_consensus = self._flatten_scores(consensus_scores)
        flat_validity = self._flatten_scores(validity_scores)
        flat_diversity = self._flatten_scores(diversity_scores)
        flat_tag_scores = [
            (self.CONSENSUS_WEIGHT * c + self.VALIDITY_DIVERSITY_WEIGHT * v * d)
            for c, v, d in zip(flat_consensus, flat_validity, flat_diversity)
        ]

        tag_scores, _, _ = unflatten_scores(
            normalized_responses=normalized_responses,
            flat_tag_scores=flat_tag_scores,
            flat_consensus_scores=flat_consensus,
            flat_validity_scores=flat_validity,
        )
        miner_scores = [
            aggregate_miner_score(scores=scores, top_k=self.n_tags_per_miner)
            for scores in tag_scores
        ]
        return {
            "tag_scores": tag_scores,
            "miner_scores": miner_scores,
            "consensus_scores": consensus_scores,
            "validity_scores": validity_scores,
            "diversity_scores": diversity_scores,
            "clusters": consensus_result["clusters"],
            "normalized_responses": normalized_responses,
            "spans": validity_result["spans"],
            "validity_details": validity_result["validity_details"],
            "diversity_details": diversity_result["diversity_details"],
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

    def _flatten_scores(self, scores: list[list[float]]) -> list[float]:
        return [score for miner_scores in scores for score in miner_scores]
