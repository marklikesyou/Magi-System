from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Literal, cast

from magi.app.service import run_chat_session
from magi.core.config import get_settings
from magi.core.embeddings import HashingEmbedder, build_embedder
from magi.core.rag import RagRetriever
from magi.core.storage import (
    describe_store_destination,
    initialize_store,
    persist_store,
    save_json_document,
)
from magi.core.vectorstore import VectorEntry
from magi.data_pipeline.chunkers import sliding_window_chunk
from magi.data_pipeline.embed import embed_chunks
from magi.data_pipeline.ingest import ingest_paths
from magi.dspy_programs.personas import USING_STUB

DEFAULT_STORE = Path(__file__).resolve().parents[1] / "storage" / "vector_store.json"


_PERSONA_TAG_RE = re.compile(
    r"^\[(?:APPROVE|REJECT|REVISE)\]\s*\[(?:MELCHIOR|BALTHASAR|CASPER)\]\s*",
    re.IGNORECASE,
)


def _strip_persona_tags(text: str) -> str:
    """Remove leading [STANCE] [NAME] tags from persona text for clean display."""
    return _PERSONA_TAG_RE.sub("", text).strip()


def _normalize_residual_label(value: object) -> Literal["low", "medium", "high"]:
    if not value:
        return "medium"
    label = str(value).strip().lower()
    if not label:
        return "medium"
    mapping: dict[str, Literal["low", "medium", "high"]] = {
        "low": "low",
        "minimal": "low",
        "minor": "low",
        "medium": "medium",
        "moderate": "medium",
        "balanced": "medium",
        "manageable": "medium",
        "high": "high",
        "elevated": "high",
        "critical": "high",
    }
    for key, normalized in mapping.items():
        if key in label:
            return normalized
    return "medium"


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


def _decision_record_payload(result) -> Dict[str, object]:
    return {
        "decision": result.final_decision.model_dump(mode="json"),
        "fused": result.fused.model_dump(mode="json"),
        "decision_trace": asdict(result.decision_trace),
        "effective_mode": result.effective_mode,
        "model": result.model,
    }


def _decision_record_path(
    args: argparse.Namespace, settings, result
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
    args: argparse.Namespace, settings, result
) -> Path | None:
    path = _decision_record_path(args, settings, result)
    if path is None:
        return None
    save_json_document(path, _decision_record_payload(result))
    return path


def command_ingest(args: argparse.Namespace) -> int:
    verbose = getattr(args, "verbose", False)
    try:
        settings = get_settings()
        embedder = build_embedder(settings)
        store = initialize_store(args.store, embedder)

        doc_paths = [Path(p) for p in args.paths]
        documents = ingest_paths(doc_paths)
        existing_hashes = {
            str(entry.metadata.get("content_hash", "")).strip()
            for entry in store.entries
            if str(entry.metadata.get("content_hash", "")).strip()
        }
        skipped_existing = 0
        fresh_documents = []
        for document in documents:
            metadata = cast(Dict[str, object], document.get("metadata", {}))
            content_hash = str(metadata.get("content_hash", "")).strip()
            if content_hash and content_hash in existing_hashes:
                skipped_existing += 1
                continue
            if content_hash:
                existing_hashes.add(content_hash)
            fresh_documents.append(document)
        documents = fresh_documents
        if not documents:
            print("No new documents to ingest; all content already exists in the store.")
            return 0

        chunks = []
        for doc in documents:
            chunks.extend(
                sliding_window_chunk(
                    doc,
                    chunk_size=args.chunk_size,
                    overlap=args.chunk_overlap,
                )
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
            if isinstance(embedder, HashingEmbedder):
                print("[verbose] Using hashing embedder (offline mode).")
            else:
                print(
                    f"[verbose] Using OpenAI embeddings ({settings.openai_embedding_model})."
                )
        return 0
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: Invalid value - {e}")
        return 1
    except Exception as e:
        print(f"Error: Unexpected error - {e}")
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
            if isinstance(embedder, HashingEmbedder):
                print("[verbose] Using hashing embedder (offline mode).")
            else:
                print(
                    f"[verbose] Using OpenAI embeddings ({settings.openai_embedding_model})."
                )
            if USING_STUB and not isinstance(embedder, HashingEmbedder):
                print(
                    "[verbose] Deterministic reasoning fallback active; embeddings remain provider-backed."
                )
            print(
                f"[verbose] Store loaded from {args.store} ({len(store.entries)} entries)."
            )

        result = run_chat_session(args.query, args.constraints or "", retriever)
        decision = result.final_decision
        fused = result.fused
        personas = result.personas
        decision_record_path = _persist_decision_record(args, settings, result)

        if verbose and not json_output:
            print(f"[verbose] Received responses from {len(personas)} persona(s).")
            trace = result.decision_trace
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

        if json_output:
            payload = _decision_record_payload(result)
            payload["decision_record_path"] = (
                str(decision_record_path) if decision_record_path is not None else ""
            )
            print(json.dumps(payload, ensure_ascii=True, indent=2))
            return 0

        print(f"\n{'=' * 60}")
        print(f"Verdict: {decision.verdict.upper()}")
        print(f"Residual Risk: {decision.residual_risk}")
        if decision.requires_human_review:
            print("Human Review: REQUIRED")
        print(f"{'=' * 60}\n")
        print(f"{decision.justification}\n")
        if decision.requires_human_review and decision.review_reason:
            print(f"Review Reason: {decision.review_reason}\n")
        trace = result.decision_trace
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
        print(f"{'-' * 60}")
        print("Persona Perspectives:\n")
        for persona in decision.persona_outputs:
            clean_text = _strip_persona_tags(persona.text)
            print(f"  [{persona.name.title()}] (confidence {persona.confidence:.2f})")

            for line in clean_text.splitlines():
                stripped = line.strip()
                if stripped:
                    print(f"    {stripped}")
            print()
        if decision.risks:
            print(f"{'-' * 60}")
            print("Risks:")
            for risk in decision.risks:
                risk_text = risk.strip() if isinstance(risk, str) else str(risk)
                if risk_text:
                    print(f"  - {risk_text}")
        if decision.mitigations:
            print("\nMitigations:")
            for mitigation in decision.mitigations:
                mit_text = (
                    mitigation.strip()
                    if isinstance(mitigation, str)
                    else str(mitigation)
                )
                if mit_text:
                    print(f"  - {mit_text}")
        blocked_count = len(
            [
                item
                for item in getattr(fused, "consensus_points", [])
                if "Unsafe retrieved instructions" in item
            ]
        )
        if blocked_count:
            print("\nSafety:")
            print("  - Unsafe retrieved instructions were excluded from synthesis.")
        print()
        return 0
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: Invalid value - {e}")
        return 1
    except Exception as e:
        print(f"Error: Unexpected error - {e}")
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
