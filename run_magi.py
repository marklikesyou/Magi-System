from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from magi.app.cli import DEFAULT_STORE, command_chat, command_ingest, ensure_provider_setup


def prompt_list(message: str) -> List[str]:
    try:
        raw = input(message).strip()
    except (EOFError, KeyboardInterrupt):
        return []
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def prompt_text(message: str) -> str:
    try:
        return input(message).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def normalize_paths(paths: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping.")
            continue
        resolved.append(str(path))
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MAGI once with optional ingestion prompts."
    )
    parser.add_argument(
        "--docs", nargs="*", help="Documents to ingest before answering."
    )
    parser.add_argument("--query", help="Question to send to the MAGI system.")
    parser.add_argument("--constraints", help="Constraint string applied to personas.")
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size used for ingestion."
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=64, help="Overlap used for ingestion."
    )
    parser.add_argument(
        "--store", type=Path, default=DEFAULT_STORE, help="Vector store path."
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.store.parent.mkdir(parents=True, exist_ok=True)
    if not ensure_provider_setup():
        return 1

    docs = args.docs or prompt_list(
        "Enter document paths (comma-separated), or leave blank: "
    )
    normalized = normalize_paths(docs) if docs else []
    if normalized:
        ingest_args = argparse.Namespace(
            paths=normalized,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            store=args.store,
        )
        ingest_status = command_ingest(ingest_args)
        if ingest_status:
            return ingest_status

    interactive_query = args.query is None
    query = args.query or prompt_text("Enter your query: ")
    if not query:
        return 0
    constraints = args.constraints if args.constraints is not None else ""
    if interactive_query and args.constraints is None:
        constraints = prompt_text("Enter constraints (optional): ")
    chat_args = argparse.Namespace(
        query=query, constraints=constraints or "", store=args.store
    )
    return command_chat(chat_args)


if __name__ == "__main__":
    raise SystemExit(main())
