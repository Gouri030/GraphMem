from types import SimpleNamespace

from graphmem.ranking.learned import FEATURE_NAMES, LearnedRanker, scope_features


def test_scope_features_are_stable_numeric_vector():
    entity = SimpleNamespace(
        path="src/parser.py",
        name="parse",
        qualified_name="Parser.parse",
        start_line=10,
        end_line=24,
        type=SimpleNamespace(value="method"),
    )
    hit = SimpleNamespace(
        entity=entity,
        signals={"lexical": 0.8, "graph": 0.4, "dataflow": 0.2},
        score=0.65,
    )
    features = scope_features(hit, "Parser parse fails")
    assert len(features) == len(FEATURE_NAMES)
    assert all(isinstance(value, float) for value in features)
    assert features[0] == 0.8
    assert features[1] == 0.4
    assert features[6] == 0.65
    assert features[7] == 15.0
    assert features[-3] == 1.0  # method indicator


def test_missing_signals_default_to_zero():
    entity = SimpleNamespace(path="a.py", name="f", qualified_name="f", start_line=1,
                             end_line=1, type=SimpleNamespace(value="function"))
    hit = SimpleNamespace(entity=entity, signals={}, score=0.0)
    features = scope_features(hit, "unknown issue")
    assert len(features) == len(FEATURE_NAMES)
    assert features[:6] == [0.0] * 6


def test_rerank_propagates_learned_scores_to_statements_and_handles_negative_scores():
    from graphmem.models.entities import CodeEntity, EntityType
    from graphmem.retrieval.localizer import LocalizationResult, ScopeHit, StatementHit

    class FakeModel:
        def predict(self, features):
            # All-negative scores expose file aggregation bugs that clamp to zero.
            return [-2.0, -1.0]

    statement_a = StatementHit(entity=None, score=0.8, evidence=[], start_line=10, end_line=10)
    statement_b = StatementHit(entity=None, score=0.3, evidence=[], start_line=20, end_line=20)
    hit_a = ScopeHit(
        entity=CodeEntity("a", EntityType.FUNCTION, "a", path="a.py", start_line=1, end_line=15),
        score=0.7, signals={}, evidence=[], statements=[statement_a],
    )
    hit_b = ScopeHit(
        entity=CodeEntity("b", EntityType.FUNCTION, "b", path="b.py", start_line=16, end_line=25),
        score=0.2, signals={}, evidence=[], statements=[statement_b],
    )

    reranked = LearnedRanker(FakeModel()).rerank([hit_a, hit_b], "issue", statement_scope_mix=0.5)
    assert [hit.entity.path for hit in reranked] == ["b.py", "a.py"]
    assert statement_a.score != 0.8
    assert statement_b.score != 0.3

    result = LocalizationResult("issue", "learned", reranked, {})
    assert result.files == [("b.py", -1.0), ("a.py", -2.0)]
