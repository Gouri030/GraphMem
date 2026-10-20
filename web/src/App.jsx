
import { useEffect, useMemo, useState } from "react";
import {
  getRepos,
  getConfigs,
  localizeIssue,
  getCode,
} from "./api";
import NeighborhoodGraph from "./NeighborhoodGraph";
import "./App.css";

const EVIDENCE_TYPES = [
  "lexical",
  "graph",
  "dataflow",
  "history",
  "artifact",
];

function evidenceLabel(type) {
  return type || "unknown";
}

function scorePercent(score) {
  return `${Math.round((Number(score) || 0) * 100)}%`;
}

function App() {
  const [repos, setRepos] = useState([]);
  const [configs, setConfigs] = useState([]);
  const [repoId, setRepoId] = useState("sample");
  const [config, setConfig] = useState("full");
  const [issue, setIssue] = useState("");
  const [topK, setTopK] = useState(10);
  const [result, setResult] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [code, setCode] = useState(null);
  const [loading, setLoading] = useState(false);
  const [codeLoading, setCodeLoading] = useState(false);
  const [error, setError] = useState("");
  const [codeError, setCodeError] = useState("");
  const [startupError, setStartupError] = useState("");

  useEffect(() => {
    let active = true;

    Promise.all([getRepos(), getConfigs()])
      .then(([repoData, configData]) => {
        if (!active) return;
        setRepos(repoData);
        setConfigs(configData);

        if (repoData.length) {
          setRepoId((current) =>
            repoData.some((repo) => repo.repo_id === current)
              ? current
              : repoData[0].repo_id
          );
        }

        if (configData.length) {
          setConfig((current) =>
            configData.some((item) => item.name === current)
              ? current
              : configData[configData.length - 1].name
          );
        }
      })
      .catch((err) => {
        if (active) setStartupError(err.message);
      });

    return () => {
      active = false;
    };
  }, []);

  const candidates = useMemo(
    () => result?.candidates ?? [],
    [result]
  );

  const selected = candidates[selectedIndex] ?? null;

  useEffect(() => {
    if (!selected || !repoId) {
      setCode(null);
      setCodeError("");
      return;
    }

    let active = true;
    setCode(null);
    setCodeError("");
    setCodeLoading(true);

    getCode({
      repo_id: repoId,
      path: selected.file,
      start_line: Math.max(1, selected.start_line - 5),
      end_line: selected.end_line + 5,
    })
      .then((data) => {
        if (active) setCode(data);
      })
      .catch((err) => {
        if (active) setCodeError(err.message);
      })
      .finally(() => {
        if (active) setCodeLoading(false);
      });

    return () => {
      active = false;
    };
  }, [selected, repoId]);

  async function handleLocalize(event) {
    event.preventDefault();
    if (!issue.trim()) {
      setError("Enter an issue description before localizing.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setCode(null);
    setSelectedIndex(0);

    try {
      const data = await localizeIssue({
        repo_id: repoId,
        issue: issue.trim(),
        config,
        top_k: Number(topK),
      });

      setResult(data.result ?? data);
    } catch (err) {
      setError(err.message || "Localization failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">G</div>
          <div>
            <h1>GraphMem</h1>
            <p>Repository issue localization</p>
          </div>
        </div>
        <div className="connection">
          <span className="connection-dot" />
          Live API
        </div>
      </header>

      <section className="workspace">
        <aside className="input-panel panel">
          <div className="panel-heading">
            <span className="eyebrow">01 / INPUT</span>
            <h2>Localize an issue</h2>
            <p>Describe the bug or unexpected behavior.</p>
          </div>

          <form onSubmit={handleLocalize}>
            <label htmlFor="repo">Repository</label>
            <select
              id="repo"
              value={repoId}
              onChange={(event) => setRepoId(event.target.value)}
              disabled={!repos.length}
            >
              {repos.map((repo) => (
                <option key={repo.repo_id} value={repo.repo_id}>
                  {repo.name || repo.repo_id}
                </option>
              ))}
            </select>

            <div className="repo-meta">
              {repos.find((repo) => repo.repo_id === repoId)?.entities ?? 0}
              {" "}entities indexed
            </div>

            <label htmlFor="config">Ranking configuration</label>
            <select
              id="config"
              value={config}
              onChange={(event) => setConfig(event.target.value)}
              disabled={!configs.length}
            >
              {configs.map((item) => (
                <option key={item.name} value={item.name}>
                  {item.name}
                </option>
              ))}
            </select>

            <label htmlFor="issue">Issue description</label>
            <textarea
              id="issue"
              value={issue}
              onChange={(event) => setIssue(event.target.value)}
              placeholder="e.g. Booking a seat sometimes fails after the flight has been cancelled..."
              rows={9}
            />

            <div className="form-bottom">
              <label htmlFor="topK">Candidates</label>
              <select
                id="topK"
                value={topK}
                onChange={(event) => setTopK(event.target.value)}
              >
                {[5, 10, 15, 20, 30].map((value) => (
                  <option key={value} value={value}>
                    Top {value}
                  </option>
                ))}
              </select>
            </div>

            <button
              className="localize-button"
              type="submit"
              disabled={loading || !repos.length || !configs.length}
            >
              {loading ? "Localizing..." : "Localize issue"}
              {!loading && <span aria-hidden="true">↗</span>}
            </button>
          </form>

          {startupError && (
            <div className="error-message" role="alert">
              Could not load API settings: {startupError}
            </div>
          )}
        </aside>

        <section className="results-panel panel">
          <div className="results-heading">
            <div>
              <span className="eyebrow">02 / RANKING</span>
              <h2>Ranked candidates</h2>
            </div>
            <span className="count-badge">{candidates.length} results</span>
          </div>

          {loading && (
            <div className="empty-state">
              <div className="spinner" />
              <h3>Analyzing repository</h3>
              <p>Searching code and ranking candidate locations…</p>
            </div>
          )}

          {!loading && error && (
            <div className="error-message" role="alert">
              <strong>Localization failed</strong>
              <p>{error}</p>
            </div>
          )}

          {!loading && !error && !result && (
            <div className="empty-state">
              <div className="empty-icon">⌕</div>
              <h3>Ready to investigate</h3>
              <p>Submit an issue to see the most relevant code locations.</p>
            </div>
          )}

          {!loading && result && candidates.length === 0 && (
            <div className="empty-state">
              <h3>No candidates returned</h3>
              <p>Try a more specific issue description or another configuration.</p>
            </div>
          )}

          {!loading && candidates.length > 0 && (
            <div className="candidate-list">
              {candidates.map((candidate, index) => {
                const evidence = candidate.evidence ?? [];
                const types = [
                  ...new Set(evidence.map((item) => item.type)),
                ];

                return (
                  <button
                    type="button"
                    key={`${candidate.file}:${candidate.start_line}:${index}`}
                    className={`candidate-card ${
                      index === selectedIndex ? "selected" : ""
                    }`}
                    onClick={() => setSelectedIndex(index)}
                  >
                    <div className="candidate-topline">
                      <span className="candidate-file">
                        {candidate.file}:{candidate.start_line}
                        {candidate.end_line !== candidate.start_line &&
                          `–${candidate.end_line}`}
                      </span>
                      <span className="score-number">
                        {scorePercent(candidate.score)}
                      </span>
                    </div>

                    <div className="candidate-scope">
                      {candidate.scope || "Unnamed scope"}
                    </div>

                    <div className="candidate-meta">
                      <span
                        className={`level-badge ${candidate.level || "scope"}`}
                      >
                        {candidate.level || "scope"}
                      </span>
                      <span className="evidence-count">
                        {evidence.length} evidence items
                      </span>
                    </div>

                    <div className="score-track">
                      <div
                        className="score-fill"
                        style={{
                          width: scorePercent(candidate.score),
                        }}
                      />
                    </div>

                    <div className="chips">
                      {types.map((type) => (
                        <span
                          className={`evidence-chip chip-${type}`}
                          key={type}
                        >
                          {evidenceLabel(type)}
                        </span>
                      ))}
                      {!types.length && (
                        <span className="muted">No evidence supplied</span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </section>

        <aside className="detail-panel panel">
          <div className="panel-heading">
            <span className="eyebrow">03 / INSPECTION</span>
            <h2>Candidate details</h2>
            <p>Evidence and source context for the selected location.</p>
          </div>

          {!selected ? (
            <div className="detail-empty">
              Select a ranked candidate to inspect its evidence and code.
            </div>
          ) : (
            <>
              <div className="selected-location">
                <span className={`level-badge ${selected.level}`}>
                  {selected.level}
                </span>
                <h3>{selected.scope || selected.file}</h3>
                <p>
                  {selected.file}:{selected.start_line}–{selected.end_line}
                </p>
                <div className="detail-score">
                  <span>Relevance score</span>
                  <strong>{scorePercent(selected.score)}</strong>
                </div>
              </div>

              <div className="section-title">
                <h3>Evidence</h3>
                <span>{(selected.evidence ?? []).length} signals</span>
              </div>

              <div className="evidence-list">
                {(selected.evidence ?? []).length === 0 && (
                  <p className="muted">No evidence details were returned.</p>
                )}

                {(selected.evidence ?? []).map((item, index) => (
                  <article className="evidence-item" key={`${item.type}-${index}`}>
                    <div className="evidence-item-top">
                      <span className={`evidence-chip chip-${item.type}`}>
                        {evidenceLabel(item.type)}
                      </span>
                      <strong>{Number(item.weight ?? 0).toFixed(2)}</strong>
                    </div>
                    <p>{item.source || item.detail?.description || "No source description provided."}</p>
                    {item.detail && Object.keys(item.detail).length > 0 && (
                      <details>
                        <summary>More details</summary>
                        <pre>{JSON.stringify(item.detail, null, 2)}</pre>
                      </details>
                    )}
                  </article>
                ))}
              </div>

              <div className="section-title code-title">
                <h3>Source preview</h3>
                <span>{selected.file}</span>
              </div>

              {codeLoading && (
                <p className="muted">Loading source code…</p>
              )}
              {codeError && (
                <div className="error-message" role="alert">
                  Could not load code: {codeError}
                </div>
              )}
              {!codeLoading && code && (
                <div className="code-window">
                  <div className="code-toolbar">
                    <span className="code-dot" />
                    <span className="code-dot" />
                    <span className="code-dot" />
                    <span className="code-filename">{code.path || selected.file}</span>
                  </div>
                  <div className="code-lines">
                    {(code.lines ?? []).map((line) => (
                      <div
                        className={`code-line ${
                          line.line >= selected.start_line &&
                          line.line <= selected.end_line
                            ? "highlighted"
                            : ""
                        }`}
                        key={line.line}
                      >
                        <span className="line-number">{line.line}</span>
                        <code>{line.code}</code>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <NeighborhoodGraph
                repoId={repoId}
                candidate={selected}
              />
            </>
          )}
        </aside>
      </section>

      <footer className="timing-footer">
        <span className="footer-label">EXECUTION PROFILE</span>
        {["total", "lexical", "graph", "statements"].map((key) => (
          <div className="timing-item" key={key}>
            <span>{key}</span>
            <strong>
              {result?.timings_ms?.[key] != null
                ? `${Number(result.timings_ms[key]).toFixed(1)} ms`
                : "—"}
            </strong>
          </div>
        ))}
        <span className="footer-status">
          {result ? "Analysis complete" : "Awaiting issue"}
        </span>
      </footer>
    </main>
  );
}

export default App;
