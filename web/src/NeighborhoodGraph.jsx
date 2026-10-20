
import { useEffect, useRef, useState } from "react";
import { Network } from "vis-network";
import { DataSet } from "vis-data";
import { getNeighborhood } from "./api";
import "vis-network/styles/vis-network.css";

const NODE_COLORS = {
  repository: "#FFD166",
  directory: "#06D6A0",
  file: "#118AB2",
  class: "#9B5DE5",
  function: "#F15BB5",
  method: "#F15BB5",
  statement: "#EF476F",
};

const NODE_SHAPES = {
  repository: "diamond",
  directory: "box",
  file: "box",
  class: "ellipse",
  function: "dot",
  method: "dot",
  statement: "square",
};

const EDGE_COLORS = {
  contains: "#8ECAE6",
  imports: "#219EBC",
  calls: "#FB8500",
  inherits: "#9B5DE5",
  references: "#FFB703",
  defines: "#90BE6D",
  uses: "#F8961E",
  dataflow_def_use: "#E63946",
  mentions: "#A8DADC",
  changes: "#457B9D",
  fixes: "#2A9D8F",
};

function displayType(type = "unknown") {
  return type.replaceAll("_", " ");
}

export default function NeighborhoodGraph({ repoId, candidate }) {
  const containerRef = useRef(null);
  const networkRef = useRef(null);

  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!candidate || !repoId) {
      setGraph(null);
      setLoading(false);
      setError("");
      return;
    }

    let active = true;

    setLoading(true);
    setError("");
    setGraph(null);

    getNeighborhood({
      repo_id: repoId,
      path: candidate.file,
      name: candidate.scope,
      hops: 2,
      limit: 60,
    })
      .then((data) => {
        if (active) setGraph(data);
      })
      .catch((err) => {
        if (active) {
          setError(err.message || "Could not load neighbourhood.");
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [repoId, candidate?.file, candidate?.scope]);

  useEffect(() => {
    if (!containerRef.current || !graph) return;

    const nodes = new DataSet(
      (graph.nodes ?? []).map((node) => {
        const type = String(node.type ?? "unknown").toLowerCase();
        const color = NODE_COLORS[type] ?? "#ADB5BD";
        const isSeed = node.id === graph.seed;
        const label = node.name || node.qualified_name || node.id;

        return {
          id: node.id,
          label,
          shape: NODE_SHAPES[type] ?? "dot",
          color: {
            background: color,
            border: isSeed ? "#FFFFFF" : "#263244",
            highlight: { background: color, border: "#FFFFFF" },
            hover: { background: color, border: "#FFFFFF" },
          },
          borderWidth: isSeed ? 4 : 1,
          size: isSeed ? 26 : 18,
          font: {
            color: "#E6EDF7",
            size: isSeed ? 14 : 12,
            face: "monospace",
          },
          title: [
            label,
            `Type: ${type}`,
            `File: ${node.path ?? "unknown"}`,
            node.start_line != null ? `Line: ${node.start_line}` : "",
            isSeed ? "Selected candidate" : "",
          ]
            .filter(Boolean)
            .join("\n"),
        };
      })
    );

    const edges = new DataSet(
      (graph.edges ?? []).map((edge, index) => ({
        id: `${edge.source}-${edge.target}-${edge.type}-${index}`,
        from: edge.source,
        to: edge.target,
        label: displayType(edge.type),
        color: {
          color: EDGE_COLORS[edge.type] ?? "#777777",
          highlight: "#FFFFFF",
          hover: "#FFFFFF",
        },
        font: {
          color: "#C5D0E0",
          size: 10,
          align: "middle",
          strokeWidth: 3,
          strokeColor: "#10151F",
        },
        arrows: { to: { enabled: true, scaleFactor: 0.65 } },
        width: 1.5,
      }))
    );

    const network = new Network(
      containerRef.current,
      { nodes, edges },
      {
        autoResize: true,
        height: "360px",
        nodes: { margin: 10 },
        edges: { smooth: { type: "dynamic" } },
        interaction: {
          hover: true,
          navigationButtons: true,
          keyboard: true,
        },
        physics: {
          stabilization: { iterations: 150 },
          barnesHut: {
            gravitationalConstant: -2500,
            springLength: 115,
            springConstant: 0.04,
          },
        },
      }
    );

    networkRef.current = network;
    network.fit();

    return () => {
      network.destroy();
      networkRef.current = null;
    };
  }, [graph]);

  return (
    <section className="neighborhood-section">
      <div className="section-title">
        <h3>Graph neighbourhood</h3>
        <span>
          {graph ? `${graph.nodes?.length ?? 0} nodes` : "Depth 2"}
        </span>
      </div>

      <p className="neighborhood-description">
        Connections around the selected candidate. Select a different
        candidate to explore its neighbourhood.
      </p>

      {loading && (
        <p className="muted">Loading graph from the live API…</p>
      )}

      {error && (
        <div className="error-message" role="alert">
          Could not load graph: {error}
        </div>
      )}

      {!candidate && !loading && (
        <p className="muted">Select a candidate to view its graph.</p>
      )}

      {!loading && !error && graph && (
        <>
          <div className="neighborhood-canvas" ref={containerRef} />

          <div className="graph-legend">
            <div className="graph-legend-group">
              <h4>Node types</h4>

              <div className="graph-legend-items">
                {Object.entries(NODE_COLORS).map(([type, color]) => (
                  <span className="legend-item" key={type}>
                    <span
                      className="legend-swatch"
                      style={{ backgroundColor: color }}
                    />
                    {displayType(type)}
                  </span>
                ))}

                <span className="legend-item">
                  <span className="legend-swatch selected-swatch" />
                  Selected candidate
                </span>
              </div>
            </div>

            <div className="graph-legend-group">
              <h4>Relationship types</h4>

              <div className="graph-legend-items">
                {Object.entries(EDGE_COLORS).map(([type, color]) => (
                  <span className="legend-item" key={type}>
                    <span
                      className="legend-line"
                      style={{ backgroundColor: color }}
                    />
                    {displayType(type)}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <p className="muted graph-count">
            {graph.nodes?.length ?? 0} nodes ·{" "}
            {graph.edges?.length ?? 0} edges
            {graph.nodes?.length >= 60 ? " · Node limit reached" : ""}
          </p>
        </>
      )}
    </section>
  );
}
