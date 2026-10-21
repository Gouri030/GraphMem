"""Function/class/file-level view of the statement-level graph."""
from __future__ import annotations

from collections import defaultdict

from graphmem.graph.store import GraphStore
from graphmem.indexing.code_index import SCOPE_TYPES
from graphmem.models.entities import CodeEntity, EntityType
from graphmem.models.relations import RelationType

RELS = {
    RelationType.CONTAINS,
    RelationType.CALLS,
    RelationType.REFERENCES,
    RelationType.INHERITS,
    RelationType.IMPORTS,
}


class ScopeGraph:
    """CALLS/REFERENCES edges start at statements; this lifts them to the
    enclosing function/method/file so scopes can be expanded directly.
    Each edge is stored in both directions, tagged 'out' (callee side) or
    'in' (caller side), so the evidence can say which way it was reached."""

    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph
        self.adj: dict[str, set[tuple[str, RelationType, str]]] = defaultdict(set)
        self._build()

    def scope_of(self, entity: CodeEntity) -> CodeEntity | None:
        if entity.type == EntityType.STATEMENT:
            return self.graph.get_entity(entity.parent_id) if entity.parent_id else None
        return entity if entity.type in SCOPE_TYPES else None

    def _build(self) -> None:
        sources: list[CodeEntity] = []
        for t in (EntityType.STATEMENT, EntityType.CLASS, EntityType.FUNCTION,
                  EntityType.METHOD, EntityType.FILE):
            sources.extend(self.graph.get_entities_by_type(t))

        for entity in sources:
            src = self.scope_of(entity)
            if src is None:
                continue
            for target, rel, _meta in self.graph.out_edges(entity.id):
                if rel not in RELS:
                    continue
                dst = self.scope_of(target)
                if dst is None or dst.id == src.id:
                    continue
                self.adj[src.id].add((dst.id, rel, "out"))
                self.adj[dst.id].add((src.id, rel, "in"))

    def neighbors(self, scope_id: str, edge_weights: dict, limit: int):
        items = [
            (n, rel, d)
            for n, rel, d in self.adj.get(scope_id, ())
            if edge_weights.get(rel, 0.0) > 0
        ]
        items.sort(key=lambda x: (-edge_weights[x[1]], x[0]))
        # A file or class can legitimately contain more than `limit` scopes.
        # Keep all direct descendants so hierarchy traversal does not silently
        # discard the function that actually owns the fault.
        descendants = [
            item for item in items
            if item[1] == RelationType.CONTAINS and item[2] == "out"
        ]
        other = [item for item in items if item not in descendants]
        return descendants + other[:limit]
