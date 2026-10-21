from graphmem.graph.builder import GraphBuilder
from graphmem.parsing.python_parser import PythonParser
from graphmem.ranking.config import ablation_configs
from graphmem.retrieval.localizer import Localizer


def make_localizer(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def hash_password(raw):\n"
        "    salt = 'x'\n"
        "    return raw + salt\n"
        "\n"
        "def login(user, raw):\n"
        "    hashed = hash_password(raw)\n"
        "    return hashed == user.password\n",
        encoding="utf-8",
    )
    (tmp_path / "billing.py").write_text(
        "def charge(amount):\n"
        "    total = amount * 2\n"
        "    return total\n",
        encoding="utf-8",
    )
    (tmp_path / "calc.py").write_text(
        "def compute(x):\n"            # 1
        "    base = x * 2\n"           # 2
        "    tax = base * 0.2\n"       # 3
        "    total = base + tax\n"     # 4
        "    log = 'done'\n"           # 5
        "    return total\n",          # 6
        encoding="utf-8",
    )
    parsed = PythonParser().parse_repository(str(tmp_path))
    return Localizer(GraphBuilder().build(parsed))


def test_exact_symbol_ranks_first(tmp_path):
    result = make_localizer(tmp_path).localize("login is broken")
    assert result.scopes[0].entity.qualified_name == "login"


def test_graph_signal_adds_callee_only_when_enabled(tmp_path):
    loc = make_localizer(tmp_path)
    cfgs = ablation_configs()

    bm25 = {h.entity.qualified_name for h in loc.localize("login is broken", config=cfgs["bm25"]).scopes}
    assert "hash_password" not in bm25

    graph = {h.entity.qualified_name: h for h in loc.localize("login is broken", config=cfgs["graph"]).scopes}
    assert "hash_password" in graph
    assert any(e.type == "graph" for e in graph["hash_password"].evidence)


def test_dataflow_extends_from_matched_statement(tmp_path):
    loc = make_localizer(tmp_path)
    result = loc.localize("compute returns wrong tax", config=ablation_configs()["graph+dataflow"])
    hit = result.scopes[0]
    assert hit.entity.qualified_name == "compute"

    assert hit.statements[0].start_line == 3  # the 'tax' statement
    df_lines = {
        st.start_line for st in hit.statements
        if any(e.type == "dataflow" for e in st.evidence)
    }
    assert {2, 4}.issubset(df_lines)
    assert 5 not in df_lines  # unrelated statement stays out


def test_ablation_removes_dataflow_evidence(tmp_path):
    loc = make_localizer(tmp_path)
    result = loc.localize("compute returns wrong tax", config=ablation_configs()["graph"])
    for hit in result.scopes:
        for st in hit.statements:
            assert all(e.type != "dataflow" for e in st.evidence)


def test_output_shape(tmp_path):
    data = make_localizer(tmp_path).localize("compute returns wrong tax").to_dict()
    assert {"issue", "files", "scopes", "candidates", "timings_ms"} <= data.keys()
    cand = data["candidates"][0]
    assert {"file", "scope", "start_line", "end_line", "score", "evidence"} <= cand.keys()
    assert all({"type", "source", "weight"} <= e.keys() for e in cand["evidence"])

def test_single_word_symbols_are_extracted_and_ranked(tmp_path):
    from graphmem.indexing.code_index import extract_symbols

    assert "parse" not in extract_symbols("parse is broken")
    assert {"parse", "Parser"} <= extract_symbols("`parse` `Parser`")

    (tmp_path / "parser.py").write_text(
        "class Parser:\n"
        "    def parse(self, value):\n"
        "        return filter(value)\n",
        encoding="utf-8",
    )
    loc = make_localizer(tmp_path)
    # Existing helper repo is extended with a one-word symbol.
    result = loc.localize("`parse` is broken")
    assert any(h.entity.name == "parse" for h in result.scopes)


def test_common_prose_word_does_not_get_exact_symbol_bonus(tmp_path):
    from graphmem.indexing.code_index import extract_paths, extract_symbols

    issue = "The merge operation produces incorrect output in `merge`. See src/forms/widgets.py."
    symbols = extract_symbols(issue)
    assert "merge" in symbols  # Only the backticked occurrence is explicit.
    assert "merge" not in extract_symbols("The merge operation produces incorrect output")
    assert "merge" in extract_symbols("The 'merge' function fails")
    assert "Media._css" in extract_symbols("Media._css() raises an exception")
    assert extract_paths(issue) == {"src/forms/widgets.py"}

    (tmp_path / "ops.py").write_text(
        "def merge(left, right):\n"
        "    return left + right\n",
        encoding="utf-8",
    )
    loc = make_localizer(tmp_path)

    ordinary = loc.localize("The merge operation produces incorrect output")
    merge_hit = next(hit for hit in ordinary.scopes if hit.entity.name == "merge")
    assert all("verified explicit symbol" not in item.source for item in merge_hit.evidence)

    explicit = loc.localize("The `merge` function produces incorrect output")
    merge_hit = next(hit for hit in explicit.scopes if hit.entity.name == "merge")
    assert any("verified explicit symbol: merge" in item.source for item in merge_hit.evidence)


def test_traceback_extracts_and_uses_file_function_and_line(tmp_path):
    from graphmem.retrieval.issue_signals import IssueSignals

    issue = '''The parser crashes.
Traceback (most recent call last):
  File "auth.py", line 5, in login
    hashed = hash_password(raw)
TypeError: bad value
'''
    signals = IssueSignals.parse(issue)
    assert signals.frames[0].path == "auth.py"
    assert signals.frames[0].line == 5
    assert signals.frames[0].function == "login"
    assert signals.exceptions[0][0] == "TypeError"

    loc = make_localizer(tmp_path)
    result = loc.localize(issue)
    assert result.scopes[0].entity.name == "login"
    assert any("traceback" in e.source for e in result.scopes[0].evidence)
