from types import SimpleNamespace

from graphmem.agent.context_builder import build_context
from graphmem.models.entities import CodeEntity, EntityType
from graphmem.retrieval.evidence import Evidence
from graphmem.retrieval.localizer import ScopeHit


def test_context_stays_within_budget_and_is_readable(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("\n".join(f"line_{i} = {i}" for i in range(1, 80)))
    hits = []
    for i, score in [(30, 0.9), (50, 0.7), (70, 0.2)]:
        entity = CodeEntity(str(i), EntityType.FUNCTION, f"func_{i}", path="sample.py",
                            qualified_name=f"sample.func_{i}", start_line=i-2, end_line=i+2)
        hits.append(ScopeHit(entity, score, {}, [Evidence("lexical", "issue term matched", 0.8)]))
    result = SimpleNamespace(scopes=hits)

    output = build_context(result, "why is the value wrong?", tmp_path, token_budget=180)

    assert output["tokens_estimate"] == (len(output["text"]) + 3) // 4
    assert output["tokens_estimate"] <= 180
    assert isinstance(output["items"], list)
    assert "Why:" in output["text"] or not output["items"]
    assert output["items"] == sorted(output["items"], key=lambda item: item["score"], reverse=True)


def test_tiny_budget_never_exceeds_limit(tmp_path):
    result = SimpleNamespace(scopes=[])
    output = build_context(result, "an issue", tmp_path, token_budget=1)
    assert output["tokens_estimate"] <= 1


def test_code_span_is_around_candidate(tmp_path):
    (tmp_path / "a.py").write_text("\n".join(f"x{i}" for i in range(1, 15)))
    entity = CodeEntity("f", EntityType.FUNCTION, "f", path="a.py", start_line=7, end_line=7)
    result = SimpleNamespace(scopes=[ScopeHit(entity, 1.0, {}, [])])
    output = build_context(result, "issue", tmp_path, token_budget=300)
    assert "4: x4" in output["items"][0]["code_span"]
    assert "10: x10" in output["items"][0]["code_span"]
