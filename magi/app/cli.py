from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import sys
from typing import Dict, Iterable, List, cast

from magi.app.service import ChatSessionResult, DecisionTrace, run_chat_session
from magi.core.config import Settings, get_settings
from magi.core.embeddings import HashingEmbedder, build_embedder
from magi.core.rag import RagRetriever
from magi.core.storage import (
    describe_store_destination,
    initialize_store,
    persist_store,
    save_json_document,
)
from magi.core.vectorstore import VectorEntry, VectorStore
from magi.data_pipeline.chunkers import sliding_window_chunk
from magi.data_pipeline.embed import embed_chunks
from magi.data_pipeline.ingest import ingest_paths
from magi.decision.schema import PersonaOutput
from magi.dspy_programs.personas import USING_STUB

DEFAULT_STORE = Path(__file__).resolve().parents[1] / "storage" / "vector_store.json"


_PERSONA_TAG_RE = re.compile(
    r"^\[(?:APPROVE|REJECT|REVISE)\]\s*\[(?:MELCHIOR|BALTHASAR|CASPER)\]\s*",
    re.IGNORECASE,
)


def _strip_persona_tags(text: str) -> str:
    """Remove leading [STANCE] [NAME] tags from persona text for clean display."""
    return _PERSONA_TAG_RE.sub("", text).strip()


def _truncate_trace_text(text: object, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    trimmed = compact[:limit]
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return trimmed + "..."


def _vector_entries(payload: Iterable[Dict[str, object]]) -> List[VectorEntry]:
    entries = []
    for record in payload:
        metadata = dict(cast(Dict[str, object], record.get("metadata", {})))
        metadata.setdefault("source", str(record["id"]).split("::")[0])
        entries.append(
            VectorEntry(
                document_id=str(record["id"]),
                embedding=list(cast(Iterable[float], record["embedding"])),
                text=str(record["text"]),
                metadata=metadata,
            )
        )
    return entries


def _decision_record_payload(result: ChatSessionResult) -> Dict[str, object]:
    return {
        "decision": result.final_decision.model_dump(mode="json"),
        "fused": result.fused.model_dump(mode="json"),
        "decision_trace": asdict(result.decision_trace),
        "effective_mode": result.effective_mode,
        "model": result.model,
    }


def _decision_record_path(
    args: argparse.Namespace, settings: Settings, result: ChatSessionResult
) -> Path | None:
    explicit = getattr(args, "decision_record_out", None)
    if explicit:
        return explicit
    trace_dir = str(getattr(settings, "decision_trace_dir", "") or "").strip()
    if not trace_dir:
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(trace_dir) / f"{result.decision_trace.query_hash}-{timestamp}.json"


def _persist_decision_record(
    args: argparse.Namespace, settings: Settings, result: ChatSessionResult
) -> Path | None:
    path = _decision_record_path(args, settings, result)
    if path is None:
        return None
    save_json_document(path, _decision_record_payload(result))
    return path


def _existing_content_hashes(store: VectorStore) -> set[str]:
    return {
        str(entry.metadata.get("content_hash", "")).strip()
        for entry in store.entries
        if str(entry.metadata.get("content_hash", "")).strip()
    }


def _filter_new_documents(
    documents: Iterable[Dict[str, object]], existing_hashes: set[str]
) -> tuple[List[Dict[str, object]], int]:
    skipped_existing = 0
    fresh_documents: List[Dict[str, object]] = []
    for document in documents:
        metadata = cast(Dict[str, object], document.get("metadata", {}))
        content_hash = str(metadata.get("content_hash", "")).strip()
        if content_hash and content_hash in existing_hashes:
            skipped_existing += 1
            continue
        if content_hash:
            existing_hashes.add(content_hash)
        fresh_documents.append(document)
    return fresh_documents, skipped_existing


def _chunk_documents(
    documents: Iterable[Dict[str, object]],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, object]]:
    chunks: List[Dict[str, object]] = []
    for document in documents:
        chunks.extend(
            sliding_window_chunk(
                document,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )
        )
    return chunks


def _print_embedder_mode(embedder: object, settings: Settings) -> None:
    if isinstance(embedder, HashingEmbedder):
        print("[verbose] Using hashing embedder (offline mode).")
        return
    print(f"[verbose] Using OpenAI embeddings ({settings.openai_embedding_model}).")


def _print_chat_preamble(
    args: argparse.Namespace,
    settings: Settings,
    embedder: object,
    store: VectorStore,
) -> None:
    _print_embedder_mode(embedder, settings)
    if USING_STUB and not isinstance(embedder, HashingEmbedder):
        print(
            "[verbose] Deterministic reasoning fallback active; embeddings remain provider-backed."
        )
    print(f"[verbose] Store loaded from {args.store} ({len(store.entries)} entries).")


def _print_trace_summary(
    result: ChatSessionResult, decision_record_path: Path | None
) -> None:
    trace = result.decision_trace
    print(f"[verbose] Received responses from {len(result.personas)} persona(s).")
    print(
        "[verbose] Decision trace: "
        f"query_hash={trace.query_hash} "
        f"used_evidence={len(trace.used_evidence_ids)} "
        f"cited_evidence={len(trace.cited_evidence_ids)} "
        f"blocked_evidence={len(trace.blocked_evidence_ids)} "
        f"safety={trace.safety_outcome} "
        f"review_required={trace.requires_human_review} "
        f"citation_hit_rate={trace.citation_hit_rate:.2f} "
        f"answer_support={trace.answer_support_score:.2f}"
    )
    if decision_record_path is not None:
        print(f"[verbose] Decision record saved to {decision_record_path}")


def _print_bullet_section(
    title: str,
    items: Iterable[object],
    *,
    leading_newline: bool = False,
) -> None:
    lines = []
    for item in items:
        text = item.strip() if isinstance(item, str) else str(item).strip()
        if text:
            lines.append(text)
    if not lines:
        return
    if leading_newline:
        print()
    print(f"{title}:")
    for line in lines:
        print(f"  - {line}")


def _print_trace_evidence(trace: DecisionTrace) -> None:
    if trace.cited_evidence:
        print("Cited Evidence:")
        for cited_item in trace.cited_evidence:
            snippet = _truncate_trace_text(cited_item.text)
            print(f"  - {cited_item.citation} {cited_item.source}: {snippet}")
        print()
    if trace.blocked_evidence:
        print("Blocked Evidence:")
        for blocked_item in trace.blocked_evidence:
            reasons = ", ".join(blocked_item.safety_reasons) or "blocked"
            snippet = _truncate_trace_text(blocked_item.text)
            print(f"  - {blocked_item.citation} {blocked_item.source}: {reasons}")
            if snippet:
                print(f"    {snippet}")
        print()


def _print_persona_outputs(persona_outputs: Iterable[PersonaOutput]) -> None:
    print(f"{'-' * 60}")
    print("Persona Perspectives:\n")
    for persona in persona_outputs:
        clean_text = _strip_persona_tags(persona.text)
        print(f"  [{persona.name.title()}] (confidence {persona.confidence:.2f})")
        for line in clean_text.splitlines():
            stripped = line.strip()
            if stripped:
                print(f"    {stripped}")
        print()


def _print_chat_report(result: ChatSessionResult) -> None:
    decision = result.final_decision
    trace = result.decision_trace

    print(f"\n{'=' * 60}")
    print(f"Verdict: {decision.verdict.upper()}")
    print(f"Residual Risk: {decision.residual_risk}")
    if decision.requires_human_review:
        print("Human Review: REQUIRED")
    print(f"{'=' * 60}\n")
    print(f"{decision.justification}\n")
    if decision.requires_human_review and decision.review_reason:
        print(f"Review Reason: {decision.review_reason}\n")
    _print_trace_evidence(trace)
    _print_persona_outputs(decision.persona_outputs)
    _print_bullet_section("Risks", decision.risks)
    _print_bullet_section(
        "Mitigations",
        decision.mitigations,
        leading_newline=bool(decision.risks),
    )
    if any(
        "Unsafe retrieved instructions" in item
        for item in getattr(result.fused, "consensus_points", [])
    ):
        _print_bullet_section(
            "Safety",
            ["Unsafe retrieved instructions were excluded from synthesis."],
            leading_newline=True,
        )
    print()


def _print_error(message: str) -> None:
    print(message, file=sys.stderr)


def command_ingest(args: argparse.Namespace) -> int:
    verbose = getattr(args, "verbose", False)
    try:
        settings = get_settings()
        embedder = build_embedder(settings)
        store = initialize_store(args.store, embedder)

        doc_paths = [Path(p) for p in args.paths]
        documents, skipped_existing = _filter_new_documents(
            ingest_paths(doc_paths),
            _existing_content_hashes(store),
        )
        if not documents:
            print("No new documents to ingest; all content already exists in the store.")
            return 0

        chunks = _chunk_documents(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        if verbose:
            print(
                f"[verbose] Chunked {len(documents)} document(s) into {len(chunks)} chunks "
                f"(size={args.chunk_size}, overlap={args.chunk_overlap})."
            )

        embedded = embed_chunks(chunks, embedder)
        entries = _vector_entries(embedded)
        store.add(entries)
        persist_store(args.store, store)

        print(f"Ingested {len(entries)} chunks from {len(documents)} document(s).")
        if skipped_existing:
            print(
                f"Skipped {skipped_existing} duplicate document(s) already present in the store."
            )
        print(describe_store_destination(args.store, store))
        if verbose:
            _print_embedder_mode(embedder, settings)
        return 0
    except FileNotFoundError as e:
        _print_error(f"Error: File not found - {e}")
        return 1
    except RuntimeError as e:
        _print_error(f"Error: {e}")
        return 1
    except ValueError as e:
        _print_error(f"Error: Invalid value - {e}")
        return 1
    except Exception as e:
        _print_error(f"Error: Unexpected error - {e}")
        return 1


def command_chat(args: argparse.Namespace) -> int:
    verbose = getattr(args, "verbose", False)
    json_output = getattr(args, "json", False)
    try:
        settings = get_settings()
        embedder = build_embedder(settings)
        store = initialize_store(args.store, embedder)
        retriever = RagRetriever(embedder, store)

        if verbose and not json_output:
            _print_chat_preamble(args, settings, embedder, store)

        result = run_chat_session(args.query, args.constraints or "", retriever)
        decision_record_path = _persist_decision_record(args, settings, result)

        if verbose and not json_output:
            _print_trace_summary(result, decision_record_path)

        if json_output:
            payload = _decision_record_payload(result)
            payload["decision_record_path"] = (
                str(decision_record_path) if decision_record_path is not None else ""
            )
            print(json.dumps(payload, ensure_ascii=True, indent=2))
            return 0

        _print_chat_report(result)
        return 0
    except FileNotFoundError as e:
        _print_error(f"Error: File not found - {e}")
        return 1
    except RuntimeError as e:
        _print_error(f"Error: {e}")
        return 1
    except ValueError as e:
        _print_error(f"Error: Invalid value - {e}")
        return 1
    except Exception as e:
        _print_error(f"Error: Unexpected error - {e}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAGI terminal helper")
    parser.set_defaults(handler=None)

    def add_common_options(
        target: argparse.ArgumentParser, *, with_defaults: bool
    ) -> None:
        target.add_argument(
            "--store",
            type=Path,
            default=DEFAULT_STORE if with_defaults else argparse.SUPPRESS,
            help=f"Path to persisted vector store (default: {DEFAULT_STORE})",
        )
        target.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            default=False if with_defaults else argparse.SUPPRESS,
            help="Print additional diagnostic information (embedder, chunks, scores).",
        )

    add_common_options(parser, with_defaults=True)

    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest one or more documents."
    )
    add_common_options(ingest_parser, with_defaults=False)
    ingest_parser.add_argument(
        "paths", nargs="+", help="Paths to documents for ingestion."
    )
    ingest_parser.add_argument(
        "--chunk-size", type=int, default=1500, help="Chunk size in characters."
    )
    ingest_parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="Overlap between chunks."
    )
    ingest_parser.set_defaults(handler=command_ingest)

    chat_parser = subparsers.add_parser(
        "chat", help="Ask a question against ingested documents."
    )
    add_common_options(chat_parser, with_defaults=False)
    chat_parser.add_argument("query", help="User query to send to the MAGI system.")
    chat_parser.add_argument(
        "--constraints", help="Optional constraints for Balthasar."
    )
    chat_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print the chat result as JSON.",
    )
    chat_parser.add_argument(
        "--decision-record-out",
        type=Path,
        help="Optional path to persist the structured decision record as JSON.",
    )
    chat_parser.set_defaults(handler=command_chat)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 0
    args.store.parent.mkdir(parents=True, exist_ok=True)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
