"""CLI entry point: index | query [question]."""
import argparse
import sys
from pathlib import Path

# Ensure project root is on path when run as python -m shanjeev_rag
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def cmd_index(reset: bool) -> None:
    from shanjeev_rag.ingest import build_index
    build_index(reset=reset)


def cmd_query(question: str | None, verbose: bool) -> None:
    from shanjeev_rag.query import run_query

    if question:
        print(run_query(question, verbose=verbose))
        return

    # Interactive
    print("ShanjeevRAG (insurance). Type a question and press Enter. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        print(run_query(q, verbose=verbose))
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="ShanjeevRAG - Insurance domain RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    idx = sub.add_parser("index", help="Build or update the vector index from data/raw/")
    idx.add_argument("--reset", action="store_true", help="Clear existing index and rebuild from scratch")
    idx.set_defaults(func=lambda a: cmd_index(a.reset))

    qry = sub.add_parser("query", help="Query the RAG (interactive or single question)")
    qry.add_argument("question", nargs="?", help="Optional: ask this question and exit")
    qry.add_argument("--no-verbose", action="store_true", help="Hide retrieved chunks, show only answer")
    qry.set_defaults(func=lambda a: cmd_query(a.question, verbose=not a.no_verbose))

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
