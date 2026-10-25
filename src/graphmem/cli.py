
import argparse

from graphmem.ingestion.pipeline import RepositoryIngestor


def main():
    parser = argparse.ArgumentParser(
        description="GraphMem repository intelligence"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index",
        help="Parse and index a repository",
    )
    index_parser.add_argument("repository", help="Path to the repository")

    localize_parser = subparsers.add_parser(
        "localize",
        help="Rank likely fault locations for a bug report",
    )
    localize_parser.add_argument("repository", help="Path to the repository")
    localize_parser.add_argument("issue", help="Natural-language bug report")
    localize_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of candidate scopes to return",
    )
    localize_parser.add_argument(
        "--ranker",
        help="Path to a trained LightGBM LambdaMART model to re-rank retrieved scopes",
    )
    localize_parser.add_argument(
        "--json",
        action="store_true",
        help="Print localization results as JSON",
    )

    args = parser.parse_args()

    ingestor = RepositoryIngestor()
    result = ingestor.ingest(args.repository)

    if args.command == "index":
        print(f"Repository:  {result.repository_path}")
        print(f"Entities:    {len(result.parsed.entities)}")
        print(f"Relations:   {len(result.parsed.relations)}")
        print(f"Graph nodes: {result.graph.entity_count()}")
        print(f"Graph edges: {result.graph.relation_count()}")

        languages = {
            entity.metadata.get("language")
            for entity in result.parsed.entities
            if entity.metadata.get("language")
        }
        print(f"Languages:   {', '.join(sorted(languages))}")
        return

    if args.ranker:
        from graphmem.ranking.learned import LearnedRanker
        result.localizer.learned_ranker = LearnedRanker.load(args.ranker)

    localization = result.localizer.localize(
        args.issue,
        top_k=args.top_k,
    )

    if args.json:
        import json
        print(json.dumps(localization.to_dict(), indent=2))
        return

    print(f"Repository: {result.repository_path}")
    print(f"Issue:      {localization.issue}")
    print("\nTop candidates:")

    for rank, hit in enumerate(localization.scopes, start=1):
        entity = hit.entity
        location = (
            f"{entity.path or '<unknown>'}:"
            f"{entity.start_line or '?'}"
        )
        name = entity.qualified_name or entity.name
        print(f"{rank:2}. {location}  {name}  score={hit.score:.4f}")


if __name__ == "__main__":
    main()
