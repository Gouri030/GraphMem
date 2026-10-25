"""Learned candidate ranking backed by LightGBM LambdaMART.

The model re-ranks the candidate scopes already retrieved by GraphMem; it does
not replace candidate generation or access gold patches at inference time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SIGNAL_NAMES = ("lexical", "graph", "dataflow", "history", "artifact", "semantic")
ENTITY_TYPES = ("file", "module", "class", "function", "method", "statement", "other")
FEATURE_NAMES = tuple(
    [f"signal_{name}" for name in SIGNAL_NAMES]
    + ["baseline_score", "scope_lines", "path_depth", "name_token_overlap", "path_token_overlap"]
    + [f"type_{name}" for name in ENTITY_TYPES]
)


def _tokens(value: str) -> set[str]:
    import re
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", value.lower()))


def scope_features(hit: Any, issue: str) -> list[float]:
    """Create stable numeric features from a Localizer ScopeHit."""
    entity = hit.entity
    signals = hit.signals or {}
    path = entity.path or ""
    name = entity.qualified_name or entity.name or ""
    query_tokens = _tokens(issue)
    name_tokens = _tokens(name)
    path_tokens = _tokens(path)
    overlap_name = len(query_tokens & name_tokens) / max(1, len(query_tokens))
    overlap_path = len(query_tokens & path_tokens) / max(1, len(query_tokens))
    start = entity.start_line or 0
    end = entity.end_line or start
    typ = getattr(entity.type, "value", str(entity.type)).lower()
    type_values = [1.0 if typ == item else 0.0 for item in ENTITY_TYPES[:-1]]
    type_values.append(1.0 if typ not in ENTITY_TYPES[:-1] else 0.0)
    return (
        [float(signals.get(name, 0.0) or 0.0) for name in SIGNAL_NAMES]
        + [float(hit.score), float(max(0, end - start + 1)), float(path.count("/")), overlap_name, overlap_path]
        + type_values
    )


class LearnedRanker:
    """Wrapper around a trained LightGBM model and its feature schema."""

    def __init__(self, model: Any):
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRanker":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("Install LightGBM with `pip install lightgbm` to use a learned ranker.") from exc
        path = Path(path)
        metadata_path = path.with_suffix(path.suffix + ".json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("feature_names") != list(FEATURE_NAMES):
                raise ValueError("Ranker feature schema does not match this GraphMem version.")
        return cls(lgb.Booster(model_file=str(path)))

    def predict(self, hits: list[Any], issue: str) -> list[float]:
        if not hits:
            return []
        features = [scope_features(hit, issue) for hit in hits]
        return [float(value) for value in self.model.predict(features)]

    def rerank(
        self,
        hits: list[Any],
        issue: str,
        statement_scope_mix: float = 0.55,
    ) -> list[Any]:
        """Rerank scopes and propagate the learned scope ranking to statements.

        LambdaMART outputs are query-relative ranking scores, not calibrated
        probabilities. We preserve raw scores for scope/file ordering, while
        min-max normalizing them only for the existing statement/scope blend.
        """
        if not hits:
            return []

        old_scores = [float(hit.score) for hit in hits]
        scores = self.predict(hits, issue)

        low, high = min(scores), max(scores)
        if high > low:
            normalized = [(score - low) / (high - low) for score in scores]
        else:
            normalized = [0.5 for _ in scores]

        mix = min(1.0, max(0.0, float(statement_scope_mix)))
        for hit, old_scope_score, score, normalized_score in zip(
            hits, old_scores, scores, normalized
        ):
            # StatementHit.score was originally (1-mix)*local + mix*old_scope.
            # Recover the local component, then recompute using learned scope
            # relevance so statement-level metrics respond to learned reranking.
            if mix < 1.0:
                for statement in getattr(hit, "statements", []):
                    local_score = (float(statement.score) - mix * old_scope_score) / (1.0 - mix)
                    statement.score = (1.0 - mix) * local_score + mix * normalized_score
            else:
                for statement in getattr(hit, "statements", []):
                    statement.score = normalized_score
            hit.score = float(score)

        return sorted(hits, key=lambda hit: hit.score, reverse=True)
