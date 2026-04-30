from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import getpass
import importlib.util
import json
import os
import re
import shlex
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List, Sequence, cast

from magi.app.artifacts import (
    artifact_dir,
    decision_payload,
    diff_run_artifacts,
    load_run_artifact,
    persist_run_artifact,
    render_artifact_diff,
    render_run_artifact,
    resolve_artifact_path,
)
from magi.app.presentation import format_chat_report, response_format_guidance
from magi.app.service import ChatSessionResult, run_chat_session
from magi.core.config import Settings, get_settings, user_env_file
from magi.core.embeddings import build_embedder
from magi.core.profiles import Profile, list_profiles, load_profile
from magi.core.rag import RagRetriever
from magi.core.storage import (
    describe_store_destination,
    initialize_store,
    load_store_bundle,
    persist_store,
    save_json_document,
    store_metadata,
)
from magi.core.vectorstore import VectorEntry, VectorStore
from magi.data_pipeline.chunkers import sliding_window_chunk
from magi.data_pipeline.embed import embed_chunks
from magi.data_pipeline.ingest import ingest_paths
from magi.eval.dataset import export_feature_log, load_dataset
from magi.eval.metrics import accuracy, classification_report, confidence_interval
from magi.eval.reporting import (
    compare_reports,
    failing_regressions,
    load_report,
    render_report_comparison,
)
from magi.eval.run_bench import evaluate_dataset
from magi.eval.scenario_harness import (
    load_scenario_dataset,
    render_scenario_report,
    run_scenario_suite,
    write_scenario_report,
)

DEFAULT_STORE = Path(__file__).resolve().parents[1] / "storage" / "vector_store.json"
MAGI_ASCII_LOGO = r"""
 __  __    _    ____ ___
|  \/  |  / \  / ___|_ _|
| |\/| | / _ \| |  _ | |
| |  | |/ ___ \ |_| || |
|_|  |_/_/   \_\____|___|

multi-agent governance interface
""".strip()
_SETUP_PROVIDERS = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}
_PROVIDER_REQUIRED_COMMANDS = {"ask", "ingest", "chat", "batch", "compare", "replay"}
_ENV_KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
_ROUTE_CHOICES = ("summarize", "extract", "fact_check", "recommend", "decision")


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _offline_mode_allowed() -> bool:
    return _truthy(os.getenv("MAGI_ALLOW_OFFLINE", ""))


def _provider_setup_issue(settings: Settings) -> str:
    openai_key = str(getattr(settings, "openai_api_key", "") or "").strip()
    google_key = str(getattr(settings, "google_api_key", "") or "").strip()

    if openai_key:
        if not _module_available("openai"):
            return (
                "OPENAI_API_KEY is set, but the openai package is not installed. "
                "Reinstall with `magi-system[openai]` or rerun the curl installer."
            )
        return ""
    if google_key:
        if not _module_available("google.genai"):
            return (
                "GOOGLE_API_KEY is set, but the google-genai package is not installed. "
                "Reinstall with `magi-system[google]` or rerun the curl installer."
            )
        return ""
    return "No AI provider API key is configured."


def _requires_provider_setup(args: argparse.Namespace) -> bool:
    command = str(getattr(args, "command", "") or "")
    if command in _PROVIDER_REQUIRED_COMMANDS:
        return True
    if command == "docs":
        return str(getattr(args, "docs_command", "") or "") == "add"
    if command == "eval":
        return (
            str(getattr(args, "eval_command", "") or "") == "run"
            and str(getattr(args, "mode", "") or "") == "live"
        )
    return False


def ensure_provider_setup() -> bool:
    if _offline_mode_allowed():
        return True

    issue = _provider_setup_issue(get_settings())
    if not issue:
        return True

    _print_error(f"Error: {issue}")
    _print_error("Run `magi setup` to save an OpenAI or Google API key before using MAGI.")
    _print_error(f"Config file: {user_env_file()}")
    return False


def _quote_env_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _write_user_env_value(key: str, value: str) -> Path:
    path = user_env_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    else:
        lines = [
            "# MAGI user configuration.\n",
            "# Managed by `magi setup`.\n",
            "\n",
        ]

    replacement = f"{key}={_quote_env_value(value)}\n"
    wrote_key = False
    for index, line in enumerate(lines):
        match = _ENV_KEY_RE.match(line)
        if match and match.group(1) == key:
            lines[index] = replacement
            wrote_key = True
            break
    if not wrote_key:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = f"{lines[-1]}\n"
        lines.append(replacement)

    path.write_text("".join(lines), encoding="utf-8")
    return path


def _remove_user_env_values(keys: set[str]) -> Path:
    path = user_env_file()
    if not path.exists():
        return path
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept_lines: list[str] = []
    for line in lines:
        match = _ENV_KEY_RE.match(line)
        if match and match.group(1) in keys:
            continue
        kept_lines.append(line)
    path.write_text("".join(kept_lines), encoding="utf-8")
    return path


def _masked_key_status(value: object) -> str:
    key = str(value or "").strip()
    if not key:
        return "unset"
    return "set"


def _prompt_provider() -> str:
    if not sys.stdin.isatty():
        return ""
    raw = input("AI provider [openai/google] (openai): ").strip().lower()
    provider = raw or "openai"
    if provider in {"o", "openai"}:
        return "openai"
    if provider in {"g", "google", "gemini"}:
        return "google"
    return provider


def command_setup(args: argparse.Namespace) -> int:
    try:
        config_path = user_env_file()
        if getattr(args, "reset", False):
            provider = str(getattr(args, "provider", "") or "").strip().lower()
            if provider and provider not in _SETUP_PROVIDERS:
                _print_error("Error: provider must be `openai` or `google`.")
                return 1
            keys = (
                {_SETUP_PROVIDERS[provider]}
                if provider
                else set(_SETUP_PROVIDERS.values())
            )
            path = _remove_user_env_values(keys)
            get_settings.cache_clear()
            print(f"Removed {', '.join(sorted(keys))} from {path}")
            print("Run `magi setup` to configure a new provider key.")
            return 0

        settings = get_settings()
        issue = _provider_setup_issue(settings)

        if getattr(args, "status", False):
            print(f"Config file: {config_path}")
            print(
                "OpenAI API key: "
                f"{_masked_key_status(getattr(settings, 'openai_api_key', ''))}"
            )
            print(
                "Google API key: "
                f"{_masked_key_status(getattr(settings, 'google_api_key', ''))}"
            )
            print(f"Ready: {'yes' if not issue else 'no'}")
            if issue:
                print(f"Issue: {issue}")
                print("Next: run `magi setup`")
            return 0

        if getattr(args, "check", False):
            if issue:
                _print_error(f"Error: {issue}")
                _print_error(f"Config file: {config_path}")
                return 1
            print("MAGI provider setup is ready.")
            return 0

        provider = str(getattr(args, "provider", "") or "").strip().lower()
        if not provider:
            provider = _prompt_provider()
        if provider not in _SETUP_PROVIDERS:
            _print_error("Error: provider must be `openai` or `google`.")
            return 1

        env_name = _SETUP_PROVIDERS[provider]
        api_key = str(getattr(args, "api_key", "") or "").strip()
        if not api_key:
            if not sys.stdin.isatty():
                _print_error(
                    f"Error: pass --provider {provider} --api-key <key>, or run `magi setup` interactively."
                )
                return 1
            api_key = getpass.getpass(f"{env_name}: ").strip()
        if not api_key:
            _print_error("Error: API key cannot be empty.")
            return 1

        path = _write_user_env_value(env_name, api_key)
        get_settings.cache_clear()
        print(f"Saved {env_name} to {path}")

        package_issue = _provider_setup_issue(get_settings())
        if package_issue:
            _print_error(f"Warning: {package_issue}")
            return 1
        print("MAGI provider setup is ready.")
        print("Next: run `magi status` or add docs with `magi ingest path/to/file.pdf`.")
        return 0
    except Exception as e:
        _print_error(f"Error: {e}")
        return 1


def _configured_vector_db_url(settings: Settings) -> str:
    return str(getattr(settings, "vector_db_url", "") or "").strip()


def _status_entry_count(settings: Settings, store_path: Path) -> str:
    if _configured_vector_db_url(settings):
        return "unknown (postgresql backend)"
    try:
        metadata, entries = load_store_bundle(store_path)
    except Exception as exc:
        return f"unavailable ({exc})"
    metadata_count = metadata.get("entry_count") if metadata else None
    if metadata_count is not None:
        return str(metadata_count)
    return str(len(entries))


def _status_store_backend(settings: Settings) -> str:
    if _configured_vector_db_url(settings):
        return "postgresql + pgvector"
    return "local json + numpy exact search"


def command_status(args: argparse.Namespace) -> int:
    try:
        settings = get_settings()
        issue = _provider_setup_issue(settings)
        store_path = Path(getattr(args, "store", DEFAULT_STORE))
        entry_count = _status_entry_count(settings, store_path)
        profile_dir = str(getattr(settings, "profile_dir", "") or "").strip()
        trace_dir = str(getattr(settings, "decision_trace_dir", "") or "").strip()

        print("MAGI status")
        print(f"Provider ready: {'yes' if not issue else 'no'}")
        if issue:
            print(f"Provider issue: {issue}")
        print(f"Config file: {user_env_file()}")
        print(
            "OpenAI API key: "
            f"{_masked_key_status(getattr(settings, 'openai_api_key', ''))}"
        )
        print(
            "Google API key: "
            f"{_masked_key_status(getattr(settings, 'google_api_key', ''))}"
        )
        print(f"Store backend: {_status_store_backend(settings)}")
        print(f"Store path: {store_path}")
        print(f"Store entries: {entry_count}")
        print(f"Artifact dir: {artifact_dir(settings)}")
        print(f"Decision trace dir: {trace_dir or 'disabled'}")
        print(f"Profile dir: {profile_dir or 'built-in profiles only'}")

        if issue:
            print("Next: run `magi setup`")
        elif entry_count in {"0", "unavailable"} or entry_count.startswith(
            "unavailable"
        ):
            print("Next: add docs with `magi ingest path/to/file.pdf`")
        else:
            print('Next: ask a question with `magi ask "what should i know?"`')
        return 0
    except Exception as e:
        _print_error(f"Error: {e}")
        return 1


def _shell_status_summary(args: argparse.Namespace) -> str:
    try:
        settings = get_settings()
        provider = "ready" if not _provider_setup_issue(settings) else "needs setup"
        store_path = Path(getattr(args, "store", DEFAULT_STORE))
        entries = _status_entry_count(settings, store_path)
        return f"provider: {provider}; store entries: {entries}; type `help` for commands"
    except Exception:
        return "type `help` for commands"


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
    save_json_document(path, decision_payload(result))
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


def _print_chat_report(
    result: ChatSessionResult,
    artifact_path: Path,
    profile: Profile | None,
) -> None:
    print(format_chat_report(result, artifact_path, profile))


def _print_error(message: str) -> None:
    print(message, file=sys.stderr)


def _load_profile_from_args(
    settings: Settings,
    args: argparse.Namespace,
    *,
    fallback_name: str = "",
) -> Profile | None:
    reference = str(getattr(args, "profile", "") or fallback_name).strip()
    if not reference:
        return None
    profile_dir = str(getattr(settings, "profile_dir", "") or "").strip()
    base_dir = Path(profile_dir) if profile_dir else None
    return load_profile(reference, base_dir=base_dir)


def _profile_base_dir(settings: Settings) -> Path | None:
    profile_dir = str(getattr(settings, "profile_dir", "") or "").strip()
    return Path(profile_dir) if profile_dir else None


def _build_retriever(
    settings: Settings,
    store_path: Path,
    profile: Profile | None,
) -> tuple[object, VectorStore, RagRetriever]:
    embedder = build_embedder(settings)
    store = initialize_store(store_path, embedder)
    retriever = _configured_retriever(embedder, store, profile)
    return embedder, store, retriever


def _configured_retriever(
    embedder: Callable[[str], Sequence[float]],
    store: VectorStore,
    profile: Profile | None,
) -> RagRetriever:
    retriever = RagRetriever(
        embedder,
        store,
        default_metadata_filters=(
            profile.metadata_filters if profile is not None else None
        ),
        source_weights=(profile.source_weights if profile is not None else None),
    )
    if profile is not None and profile.retrieval_top_k is not None:
        setattr(retriever, "preferred_top_k", profile.retrieval_top_k)
    return retriever


def _route_override(args: argparse.Namespace, profile: Profile | None) -> str | None:
    explicit = str(getattr(args, "route", "") or "").strip()
    if explicit:
        return explicit
    if profile is not None and profile.route_mode is not None:
        return profile.route_mode
    return None


def _effective_constraints(
    args: argparse.Namespace, profile: Profile | None, raw_constraints: str
) -> str:
    constraints = str(raw_constraints or "").strip()
    if constraints:
        return constraints
    if profile is not None:
        return profile.default_constraints
    return ""


def _render_text_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    header_line = "  ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, separator] + body)


def _comparison_targets(
    settings: Settings,
    args: argparse.Namespace,
) -> list[tuple[str, Profile | None]]:
    base_dir = _profile_base_dir(settings)
    requested = [
        str(item).strip()
        for item in list(getattr(args, "profiles", []) or [])
        if str(item).strip()
    ]
    targets: list[tuple[str, Profile | None]] = []
    seen: set[str] = set()
    if getattr(args, "include_default", False):
        targets.append(("default", None))
        seen.add("default")
    if requested:
        for reference in requested:
            profile = load_profile(reference, base_dir=base_dir)
            if profile is None or profile.name in seen:
                continue
            targets.append((profile.name, profile))
            seen.add(profile.name)
    else:
        for summary in list_profiles(base_dir=base_dir):
            if summary.name in seen:
                continue
            profile = load_profile(str(summary.path), base_dir=base_dir)
            if profile is None:
                continue
            targets.append((profile.name, profile))
            seen.add(profile.name)
    if not targets:
        raise ValueError("no profiles available to compare")
    return targets


def _run_single_query(
    *,
    args: argparse.Namespace,
    settings: Settings,
    query: str,
    constraints: str,
    store: VectorStore,
    retriever: RagRetriever,
    profile: Profile | None,
) -> tuple[ChatSessionResult, Path, Path | None]:
    profile_name = profile.name if profile is not None else ""
    result = run_chat_session(
        query,
        constraints,
        retriever,
        model=str(getattr(args, "model", "") or "").strip() or None,
        route_mode=_route_override(args, profile),
        profile_name=profile_name,
        prompt_preamble=(profile.prompt_preamble if profile is not None else ""),
        response_format_guidance=response_format_guidance(profile),
        approve_min_citation_hit_rate=(
            profile.approve_min_citation_hit_rate if profile is not None else None
        ),
        approve_min_answer_support_score=(
            profile.approve_min_answer_support_score if profile is not None else None
        ),
    )
    decision_record_path = _persist_decision_record(args, settings, result)
    artifact_path = persist_run_artifact(
        settings,
        result=result,
        query=query,
        constraints=constraints,
        store_path=args.store,
        store_metadata=store_metadata(store),
        profile_name=profile_name,
        requested_route=_route_override(args, profile) or "",
    )
    return result, artifact_path, decision_record_path


def command_ingest(args: argparse.Namespace) -> int:
    try:
        settings = get_settings()
        if getattr(args, "reset_store", False) and Path(args.store).exists():
            Path(args.store).unlink()
        embedder = build_embedder(settings)
        store = initialize_store(args.store, embedder)

        doc_paths = [Path(p) for p in args.paths]
        documents, skipped_existing = _filter_new_documents(
            ingest_paths(doc_paths),
            _existing_content_hashes(store),
        )
        if not documents:
            print("No new documents to ingest; all content already exists in the store.")
            print('Next: ask a question with `magi ask "what changed?"`')
            return 0

        chunks = _chunk_documents(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        embedded = embed_chunks(chunks, embedder)
        entries = _vector_entries(embedded)
        store.add(entries)
        persist_store(
            args.store,
            store,
            embedder=embedder,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        print(f"Documents processed: {len(documents)}")
        print(f"Chunks added: {len(entries)}")
        if skipped_existing:
            print(f"Duplicates skipped: {skipped_existing}")
        else:
            print("Duplicates skipped: 0")
        print(describe_store_destination(args.store, store))
        print('Next: ask a question with `magi ask "what should i know?"`')
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
    json_output = getattr(args, "json", False)
    try:
        settings = get_settings()
        profile = _load_profile_from_args(settings, args)
        embedder, store, retriever = _build_retriever(settings, args.store, profile)
        constraints = _effective_constraints(args, profile, getattr(args, "constraints", ""))
        entries = getattr(store, "entries", None)
        if isinstance(entries, list) and len(entries) == 0:
            if json_output:
                payload: dict[str, object] = {
                    "error": "empty_store",
                    "message": "No documents are available in the MAGI store.",
                    "next": "Run `magi ingest path/to/file.pdf` before asking questions.",
                    "store": str(args.store),
                }
                print(json.dumps(payload, ensure_ascii=True, indent=2))
            else:
                _print_error("Error: No documents are available in the MAGI store.")
                _print_error(
                    "Run `magi ingest path/to/file.pdf` before asking questions."
                )
                _print_error(f"Store: {args.store}")
            return 1

        result, artifact_path, decision_record_path = _run_single_query(
            args=args,
            settings=settings,
            query=args.query,
            constraints=constraints,
            store=store,
            retriever=retriever,
            profile=profile,
        )

        if json_output:
            payload = decision_payload(result)
            payload["decision_record_path"] = (
                str(decision_record_path) if decision_record_path is not None else ""
            )
            payload["artifact_path"] = str(artifact_path)
            print(json.dumps(payload, ensure_ascii=True, indent=2))
            return 0

        _print_chat_report(result, artifact_path, profile)
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


def _load_batch_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        raise ValueError("batch JSON input must be a list of objects")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            return [dict(row) for row in reader]
    raise ValueError("batch input must be .jsonl, .json, .csv, or .tsv")


def _completed_batch_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                identifier = str(payload.get("id", "")).strip()
                if identifier:
                    completed.add(identifier)
    return completed


def command_batch(args: argparse.Namespace) -> int:
    try:
        settings = get_settings()
        profile = _load_profile_from_args(settings, args)
        embedder, store, retriever = _build_retriever(settings, args.store, profile)
        records = _load_batch_records(args.input)
        completed = (
            _completed_batch_ids(args.out) if getattr(args, "resume", False) and args.out else set()
        )
        processed = 0
        skipped = 0
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
        base_route = str(getattr(args, "route", "") or "").strip()
        for index, record in enumerate(records, start=1):
            identifier = str(record.get("id", "")).strip() or f"row-{index}"
            if identifier in completed:
                skipped += 1
                continue
            query = str(record.get("query", "")).strip()
            if not query:
                skipped += 1
                continue
            per_record_route = str(record.get("route", "")).strip()
            per_record_constraints = _effective_constraints(
                args,
                profile,
                str(record.get("constraints", "")).strip(),
            )
            if per_record_route:
                setattr(args, "route", per_record_route)
            else:
                setattr(args, "route", base_route)
            result, artifact_path, _ = _run_single_query(
                args=args,
                settings=settings,
                query=query,
                constraints=per_record_constraints,
                store=store,
                retriever=retriever,
                profile=profile,
            )
            payload = {
                "id": identifier,
                "query": query,
                "constraints": per_record_constraints,
                "verdict": result.final_decision.verdict,
                "query_mode": result.decision_trace.query_mode,
                "requires_human_review": result.final_decision.requires_human_review,
                "abstained": result.final_decision.abstained,
                "artifact_path": str(artifact_path),
            }
            if args.out:
                with args.out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            processed += 1
            if getattr(args, "limit", 0) and processed >= args.limit:
                break
        print(f"processed\t{processed}")
        print(f"skipped\t{skipped}")
        if args.out:
            print(f"output\t{args.out}")
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


def command_compare(args: argparse.Namespace) -> int:
    try:
        settings = get_settings()
        embedder = build_embedder(settings)
        store = initialize_store(args.store, embedder)
        targets = _comparison_targets(settings, args)
        comparisons: list[
            tuple[str, Profile | None, str, ChatSessionResult, Path]
        ] = []

        for label, profile in targets:
            retriever = _configured_retriever(embedder, store, profile)
            constraints = _effective_constraints(
                args,
                profile,
                getattr(args, "constraints", ""),
            )
            result, artifact_path, _ = _run_single_query(
                args=args,
                settings=settings,
                query=args.query,
                constraints=constraints,
                store=store,
                retriever=retriever,
                profile=profile,
            )
            comparisons.append((label, profile, constraints, result, artifact_path))

        rows: list[list[str]] = []
        for label, profile, _constraints, result, artifact_path in comparisons:
            decision = result.final_decision
            trace = result.decision_trace
            rows.append(
                [
                    label,
                    profile.presentation_style if profile is not None else "standard",
                    trace.query_mode,
                    decision.verdict,
                    decision.residual_risk,
                    f"{trace.citation_hit_rate:.2f}",
                    f"{trace.answer_support_score:.2f}",
                    "yes" if decision.requires_human_review else "no",
                    "yes" if decision.abstained else "no",
                    artifact_path.stem,
                ]
            )

        print("=" * 60)
        print("PROFILE COMPARISON")
        print("=" * 60)
        print(f"Query: {args.query}")
        print(f"Store: {args.store}")
        print(
            _render_text_table(
                [
                    "profile",
                    "style",
                    "mode",
                    "verdict",
                    "risk",
                    "cite",
                    "support",
                    "review",
                    "abstain",
                    "run_id",
                ],
                rows,
            )
        )
        print("")
        print("Use `python -m magi.app.cli explain <run_id>` for any row above.")

        if getattr(args, "full", False):
            for label, profile, constraints, result, artifact_path in comparisons:
                print("")
                print("=" * 60)
                print(f"PROFILE: {label}")
                print(
                    "Presentation Style: "
                    f"{profile.presentation_style if profile is not None else 'standard'}"
                )
                print(f"Applied Constraints: {constraints or 'None'}")
                print("=" * 60)
                print(format_chat_report(result, artifact_path, profile))
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


def command_explain(args: argparse.Namespace) -> int:
    try:
        settings = get_settings()
        path = resolve_artifact_path(args.artifact, settings=settings)
        payload = load_run_artifact(path)
        print(render_run_artifact(payload))
        return 0
    except Exception as e:
        _print_error(f"Error: {e}")
        return 1


def command_profiles(args: argparse.Namespace) -> int:
    try:
        settings = get_settings()
        base_dir = _profile_base_dir(settings)
        name = str(getattr(args, "name", "") or "").strip()
        if name:
            profile = load_profile(name, base_dir=base_dir)
            if profile is None:
                raise FileNotFoundError(f"profile not found: {name}")
            print(f"Name: {profile.name}")
            print(f"Description: {profile.description or 'None'}")
            print(f"Route Mode: {profile.route_mode or 'auto'}")
            print(f"Retrieval Top K: {profile.retrieval_top_k or 'auto'}")
            print(f"Presentation Style: {profile.presentation_style}")
            print(f"Default Constraints: {profile.default_constraints or 'None'}")
            print(f"Prompt Preamble: {profile.prompt_preamble or 'None'}")
            print(
                f"Response Format Guidance: {profile.response_format_guidance or 'default'}"
            )
            print(
                "Approval Thresholds: "
                f"citation={profile.approve_min_citation_hit_rate if profile.approve_min_citation_hit_rate is not None else 'default'}, "
                f"support={profile.approve_min_answer_support_score if profile.approve_min_answer_support_score is not None else 'default'}"
            )
            if profile.source_weights:
                print("Source Weights:")
                for source, weight in sorted(profile.source_weights.items()):
                    print(f"  - {source}: {weight:.2f}")
            return 0

        summaries = list_profiles(base_dir=base_dir)
        for summary in summaries:
            print(
                "\t".join(
                    [
                        summary.name,
                        summary.route_mode or "auto",
                        str(summary.retrieval_top_k or "auto"),
                        summary.presentation_style,
                        summary.source,
                        summary.description,
                    ]
                )
            )
        return 0
    except Exception as e:
        _print_error(f"Error: {e}")
        return 1


def command_replay(args: argparse.Namespace) -> int:
    try:
        settings = get_settings()
        artifact_path = resolve_artifact_path(args.artifact, settings=settings)
        artifact = load_run_artifact(artifact_path)
        input_payload = artifact.get("input", {})
        if not isinstance(input_payload, dict):
            raise ValueError("artifact input payload is invalid")
        artifact_store = artifact.get("store", {})
        store_path = getattr(args, "store", None)
        if store_path is None:
            store_path = Path(
                str(
                    ((artifact_store or {}) if isinstance(artifact_store, dict) else {}).get(
                        "path",
                        DEFAULT_STORE,
                    )
                )
            )
        replay_args = argparse.Namespace(
            query=str(input_payload.get("query", "")),
            constraints=str(input_payload.get("constraints", "")),
            store=store_path,
            verbose=getattr(args, "verbose", False),
            json=getattr(args, "json", False),
            decision_record_out=None,
            profile=str(getattr(args, "profile", "") or input_payload.get("profile", "") or ""),
            route=str(getattr(args, "route", "") or input_payload.get("requested_route", "") or ""),
            model=str(getattr(args, "model", "") or ""),
        )
        return command_chat(replay_args)
    except Exception as e:
        _print_error(f"Error: {e}")
        return 1


def command_diff(args: argparse.Namespace) -> int:
    try:
        settings = get_settings()
        left = load_run_artifact(resolve_artifact_path(args.left, settings=settings))
        right = load_run_artifact(resolve_artifact_path(args.right, settings=settings))
        diff = diff_run_artifacts(left, right)
        print(render_artifact_diff(diff))
        return 0
    except Exception as e:
        _print_error(f"Error: {e}")
        return 1


def _benchmark_report_payload(
    cases_path: Path,
    predictions: list[str],
    gold: list[str],
    rows: list[tuple[str, str, str]],
) -> dict[str, object]:
    score = accuracy(predictions, gold)
    lower, upper = confidence_interval(score, len(gold))
    return {
        "metadata": {
            "suite_type": "benchmark",
            "source": str(cases_path),
        },
        "summary": {
            "accuracy": score,
            "count": len(gold),
            "ci_lower": lower,
            "ci_upper": upper,
        },
        "cases": [
            {"id": case_id, "expected": expected, "predicted": predicted}
            for case_id, expected, predicted in rows
        ],
        "classification_report": classification_report(predictions, gold),
    }


def _scenario_threshold_failures(
    report,
    args: argparse.Namespace,
) -> list[tuple[str, float, float, str]]:
    summary = report.summary
    min_thresholds = {
        "overall_score": getattr(args, "min_overall_score", None),
        "verdict_accuracy": getattr(args, "min_verdict_accuracy", None),
        "requirement_pass_rate": getattr(args, "min_requirement_pass_rate", None),
        "retrieval_hit_rate": getattr(args, "min_retrieval_hit_rate", None),
        "retrieval_top_source_accuracy": getattr(
            args,
            "min_retrieval_top_source_accuracy",
            None,
        ),
        "retrieval_source_recall": getattr(
            args,
            "min_retrieval_source_recall",
            None,
        ),
        "average_citation_hit_rate": getattr(
            args,
            "min_average_citation_hit_rate",
            None,
        ),
        "average_answer_support_score": getattr(
            args,
            "min_average_answer_support_score",
            None,
        ),
        "supported_answer_rate": getattr(args, "min_supported_answer_rate", None),
    }
    max_thresholds = {
        "latency_p50_ms": getattr(args, "max_p50_latency_ms", None),
        "latency_p95_ms": getattr(args, "max_p95_latency_ms", None),
        "latency_max_ms": getattr(args, "max_max_latency_ms", None),
        "average_estimated_cost_usd": getattr(args, "max_average_cost_usd", None),
        "total_estimated_cost_usd": getattr(args, "max_total_cost_usd", None),
    }
    failures: list[tuple[str, float, float, str]] = []
    for field, minimum in min_thresholds.items():
        if minimum is None:
            continue
        actual = float(getattr(summary, field))
        if actual < float(minimum):
            failures.append((field, actual, float(minimum), "minimum"))
    for field, maximum in max_thresholds.items():
        if maximum is None:
            continue
        actual = float(getattr(summary, field))
        if actual > float(maximum):
            failures.append((field, actual, float(maximum), "maximum"))
    return failures


def command_eval_run(args: argparse.Namespace) -> int:
    try:
        if args.kind == "scenario":
            dataset = load_scenario_dataset(args.cases)
            force_stub = None
            if args.mode == "stub":
                force_stub = True
            elif args.mode == "live":
                force_stub = False
            report = run_scenario_suite(
                dataset,
                force_stub=force_stub,
                model=args.model,
                requested_mode=args.mode,
            )
            print(render_scenario_report(report))
            if args.report_out:
                write_scenario_report(report, args.report_out)
                print(f"report_saved\t{args.report_out}")
            failures = _scenario_threshold_failures(report, args)
            if failures:
                for field, actual, threshold, direction in failures:
                    print(
                        "threshold_failed\t"
                        f"{field}\tactual={actual:.4f}\t{direction}={threshold:.4f}",
                        file=sys.stderr,
                    )
                return 1
            return 0

        benchmark_dataset = load_dataset(args.cases)
        predictions, gold, rows, features = evaluate_dataset(benchmark_dataset)
        payload = _benchmark_report_payload(args.cases, predictions, gold, rows)
        for case_id, expected, predicted in rows:
            print(f"{case_id}\t{expected}\t{predicted}")
        summary = cast(dict[str, object], payload["summary"])
        print(f"\naccuracy\t{float(str(summary['accuracy'])):.2%}")
        if args.features_out:
            export_feature_log(features, args.features_out)
            print(f"features_saved\t{args.features_out}")
        if args.report_out:
            save_json_document(args.report_out, payload)
            print(f"report_saved\t{args.report_out}")
        return 0
    except Exception as e:
        _print_error(f"Error: {e}")
        return 1


def command_eval_compare(args: argparse.Namespace) -> int:
    try:
        baseline = load_report(args.baseline)
        candidate = load_report(args.candidate)
        comparison = compare_reports(baseline, candidate)
        print(render_report_comparison(comparison))
        return 0
    except Exception as e:
        _print_error(f"Error: {e}")
        return 1


def command_eval_regressions(args: argparse.Namespace) -> int:
    try:
        baseline = load_report(args.baseline)
        candidate = load_report(args.candidate)
        comparison = compare_reports(baseline, candidate)
        thresholds = {
            "overall_score": args.min_overall_score_delta,
            "verdict_accuracy": args.min_verdict_accuracy_delta,
            "retrieval_hit_rate": args.min_retrieval_hit_rate_delta,
            "average_answer_support_score": args.min_answer_support_score_delta,
        }
        failures = failing_regressions(comparison, thresholds)
        if failures:
            for metric, actual, minimum in failures:
                print(
                    f"regression\t{metric}\tactual={actual:+.4f}\tminimum={minimum:+.4f}",
                    file=sys.stderr,
                )
            return 1
        print("status\tok")
        return 0
    except Exception as e:
        _print_error(f"Error: {e}")
        return 1


_SHELL_EXIT_COMMANDS = {"exit", "quit", ":q", "\\q"}
_SHELL_META_HELP = {"help", "?", ":help"}
_SHELL_COMMANDS = {
    "ask",
    "ingest",
    "docs",
    "chat",
    "batch",
    "compare",
    "explain",
    "runs",
    "profiles",
    "replay",
    "diff",
    "eval",
    "shell",
    "setup",
    "status",
}
_SHELL_STORE_COMMANDS = {"ask", "ingest", "chat", "batch", "compare", "replay"}


def _shell_command_tokens(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return []
    if stripped in _SHELL_EXIT_COMMANDS:
        return ["exit"]
    if stripped in _SHELL_META_HELP:
        return ["help"]
    if stripped.startswith("help "):
        parts = shlex.split(stripped)
        if len(parts) == 2 and parts[1] in _SHELL_COMMANDS:
            return [parts[1], "--help"]
        return ["help"]

    tokens = shlex.split(stripped)
    if not tokens:
        return []
    if tokens[0] in _SHELL_EXIT_COMMANDS:
        return ["exit"]
    if tokens[0] in _SHELL_META_HELP:
        return ["help"]
    if tokens[0] in _SHELL_COMMANDS or tokens[0].startswith("-"):
        return tokens
    return ["chat", stripped]


def _apply_shell_defaults(tokens: list[str], args: argparse.Namespace) -> list[str]:
    if not tokens:
        return tokens
    command = tokens[0]
    if (
        command == "docs"
        and len(tokens) >= 2
        and tokens[1] == "add"
        and "--store" not in tokens
    ):
        store = getattr(args, "store", None)
        if isinstance(store, Path):
            tokens = ["docs", "add", "--store", str(store), *tokens[2:]]
    if (
        command == "docs"
        and len(tokens) >= 2
        and tokens[1] == "add"
        and getattr(args, "verbose", False)
        and "--verbose" not in tokens
        and "-v" not in tokens
    ):
        tokens = ["docs", "add", "-v", *tokens[2:]]
    if command in _SHELL_STORE_COMMANDS and "--store" not in tokens:
        store = getattr(args, "store", None)
        if isinstance(store, Path):
            tokens = [command, "--store", str(store), *tokens[1:]]
    if (
        command in _SHELL_STORE_COMMANDS
        and getattr(args, "verbose", False)
        and "--verbose" not in tokens
        and "-v" not in tokens
    ):
        tokens = [command, "-v", *tokens[1:]]
    return tokens


def _print_shell_help() -> None:
    print("MAGI interactive shell")
    print("Type a command, or type a plain question to run `chat`.")
    print("Commands: status, ask, docs add, ingest, chat, compare, profiles")
    print("Also available: batch, runs show, explain, replay, diff, eval, setup")
    print("Examples:")
    print("  status")
    print('  ask "Summarize MAGI in one sentence."')
    print("  profiles security-review")
    print("  docs add docs/briefing.txt")
    print("  runs show <run-id>")
    print("Type `help <command>` for command help, or `exit` to quit.")


def _print_shell_banner() -> None:
    print(MAGI_ASCII_LOGO)
    print("Type `help` for commands, `exit` to quit.")


def command_shell(args: argparse.Namespace) -> int:
    _print_shell_banner()
    print(_shell_status_summary(args))
    last_status = 0
    while True:
        try:
            line = input("magi> ")
        except EOFError:
            print()
            return last_status
        except KeyboardInterrupt:
            print()
            return last_status

        try:
            tokens = _shell_command_tokens(line)
        except ValueError as exc:
            _print_error(f"Error: {exc}")
            last_status = 1
            continue
        if not tokens:
            continue
        if tokens == ["exit"]:
            return last_status
        if tokens == ["help"]:
            _print_shell_help()
            continue

        tokens = _apply_shell_defaults(tokens, args)
        try:
            last_status = main(tokens)
        except SystemExit as exc:
            code = exc.code
            last_status = int(code) if isinstance(code, int) else 1
    return last_status


def _add_common_options(
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
        help="Accepted for compatibility; debug trace output is not printed.",
    )


def _add_profile_options(target: argparse.ArgumentParser) -> None:
    target.add_argument("--profile", help="Optional domain profile name or path.")
    target.add_argument(
        "--route",
        choices=_ROUTE_CHOICES,
        help="Optional query route override.",
    )
    target.add_argument("--model", help="Optional model override for live runs.")


def _add_ingest_options(target: argparse.ArgumentParser) -> None:
    _add_common_options(target, with_defaults=False)
    target.add_argument("paths", nargs="+", help="Paths to documents for ingestion.")
    target.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        help="Chunk size in characters.",
    )
    target.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks.",
    )
    target.add_argument(
        "--reset-store",
        action="store_true",
        default=False,
        help="Delete the existing local store file before ingesting.",
    )


def _add_chat_options(target: argparse.ArgumentParser) -> None:
    _add_common_options(target, with_defaults=False)
    _add_profile_options(target)
    target.add_argument("query", help="User query to send to the MAGI system.")
    target.add_argument("--constraints", help="Optional constraints for Balthasar.")
    target.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print the chat result as JSON.",
    )
    target.add_argument(
        "--decision-record-out",
        type=Path,
        help="Optional path to persist the structured decision record as JSON.",
    )


def _add_setup_parser(subparsers: Any) -> None:
    setup_parser = subparsers.add_parser(
        "setup",
        help="Save or verify the AI provider key used by MAGI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  magi setup
  magi setup --provider openai
  magi setup --status
  magi setup --reset
""",
    )
    setup_parser.add_argument(
        "--provider",
        choices=tuple(_SETUP_PROVIDERS),
        help="AI provider to configure.",
    )
    setup_parser.add_argument(
        "--api-key",
        help="API key to save. Omit to enter it securely in an interactive terminal.",
    )
    setup_parser.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="Print provider setup status without revealing key values.",
    )
    setup_parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="Exit successfully only when MAGI has a usable provider key.",
    )
    setup_parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Remove saved provider key values from the MAGI user config.",
    )
    setup_parser.set_defaults(handler=command_setup)


def _add_basic_parser(
    subparsers: Any,
    name: str,
    *,
    help_text: str,
    handler: Callable[[argparse.Namespace], int],
) -> None:
    command_parser = subparsers.add_parser(name, help=help_text)
    _add_common_options(command_parser, with_defaults=False)
    command_parser.set_defaults(handler=handler)


def _add_ingest_parsers(subparsers: Any) -> None:
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest one or more documents."
    )
    _add_ingest_options(ingest_parser)
    ingest_parser.set_defaults(handler=command_ingest)

    docs_parser = subparsers.add_parser(
        "docs",
        help="Manage documents in the MAGI store.",
    )
    docs_subparsers = docs_parser.add_subparsers(dest="docs_command")
    docs_add_parser = docs_subparsers.add_parser(
        "add",
        help="Add documents to the MAGI store.",
        description="Friendly alias for `magi ingest`.",
    )
    _add_ingest_options(docs_add_parser)
    docs_add_parser.set_defaults(handler=command_ingest)


def _add_chat_parser(
    subparsers: Any,
    name: str,
    *,
    description: str | None = None,
) -> None:
    command_parser = subparsers.add_parser(
        name,
        help="Ask a question against ingested documents.",
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""examples:
  magi {name} "summarize the uploaded policy"
  magi {name} "should we deploy?" --profile security-review
  magi {name} "what changed?" --json
""",
    )
    _add_chat_options(command_parser)
    command_parser.set_defaults(handler=command_chat)


def _add_batch_parser(subparsers: Any) -> None:
    batch_parser = subparsers.add_parser(
        "batch", help="Run a batch of queries from JSONL, JSON, CSV, or TSV."
    )
    _add_common_options(batch_parser, with_defaults=False)
    _add_profile_options(batch_parser)
    batch_parser.add_argument("input", type=Path, help="Batch input file.")
    batch_parser.add_argument("--out", type=Path, help="Optional JSONL output path.")
    batch_parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip rows whose id already exists in the output JSONL.",
    )
    batch_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of records to process.",
    )
    batch_parser.set_defaults(handler=command_batch)


def _add_compare_parser(subparsers: Any) -> None:
    compare_parser = subparsers.add_parser(
        "compare", help="Run the same query across multiple profiles."
    )
    _add_common_options(compare_parser, with_defaults=False)
    compare_parser.add_argument("query", help="User query to compare across profiles.")
    compare_parser.add_argument(
        "--constraints",
        help="Optional explicit constraints shared across runs.",
    )
    compare_parser.add_argument(
        "--profiles",
        nargs="*",
        help="Optional profile names or paths. Defaults to all discovered profiles.",
    )
    compare_parser.add_argument(
        "--include-default",
        action="store_true",
        default=False,
        help="Also include the unprofiled default run in the comparison.",
    )
    compare_parser.add_argument(
        "--route",
        choices=_ROUTE_CHOICES,
        help="Optional query route override shared across compared runs.",
    )
    compare_parser.add_argument("--model", help="Optional model override for live runs.")
    compare_parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Render the full formatted output for each compared profile.",
    )
    compare_parser.set_defaults(handler=command_compare)


def _add_artifact_parsers(subparsers: Any) -> None:
    explain_parser = subparsers.add_parser(
        "explain", help="Render a saved run artifact."
    )
    explain_parser.add_argument("artifact", help="Artifact path or run id.")
    explain_parser.set_defaults(handler=command_explain)

    runs_parser = subparsers.add_parser("runs", help="Inspect saved MAGI runs.")
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command")
    runs_show_parser = runs_subparsers.add_parser(
        "show",
        help="Render a saved run artifact.",
        description="Friendly alias for `magi explain`.",
    )
    runs_show_parser.add_argument("artifact", help="Artifact path or run id.")
    runs_show_parser.set_defaults(handler=command_explain)

    replay_parser = subparsers.add_parser(
        "replay", help="Replay a saved run artifact against the current code."
    )
    _add_common_options(replay_parser, with_defaults=False)
    _add_profile_options(replay_parser)
    replay_parser.add_argument("artifact", help="Artifact path or run id.")
    replay_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print the replay result as JSON.",
    )
    replay_parser.set_defaults(handler=command_replay)

    diff_parser = subparsers.add_parser("diff", help="Diff two saved run artifacts.")
    diff_parser.add_argument("left", help="Left artifact path or run id.")
    diff_parser.add_argument("right", help="Right artifact path or run id.")
    diff_parser.set_defaults(handler=command_diff)


def _add_profiles_parser(subparsers: Any) -> None:
    profiles_parser = subparsers.add_parser(
        "profiles", help="List or inspect built-in and workspace profiles."
    )
    profiles_parser.add_argument(
        "name",
        nargs="?",
        help="Optional profile name to inspect in detail.",
    )
    profiles_parser.set_defaults(handler=command_profiles)


def _add_eval_run_thresholds(target: argparse.ArgumentParser) -> None:
    target.add_argument("--min-overall-score", type=float)
    target.add_argument("--min-verdict-accuracy", type=float)
    target.add_argument("--min-requirement-pass-rate", type=float)
    target.add_argument("--min-retrieval-hit-rate", type=float)
    target.add_argument("--min-retrieval-top-source-accuracy", type=float)
    target.add_argument("--min-retrieval-source-recall", type=float)
    target.add_argument("--min-average-citation-hit-rate", type=float)
    target.add_argument("--min-average-answer-support-score", type=float)
    target.add_argument("--min-supported-answer-rate", type=float)
    target.add_argument("--max-p50-latency-ms", type=float)
    target.add_argument("--max-p95-latency-ms", type=float)
    target.add_argument("--max-max-latency-ms", type=float)
    target.add_argument("--max-average-cost-usd", type=float)
    target.add_argument("--max-total-cost-usd", type=float)


def _add_eval_parsers(subparsers: Any) -> None:
    eval_parser = subparsers.add_parser("eval", help="Run or compare evaluation suites.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command")

    eval_run_parser = eval_subparsers.add_parser(
        "run", help="Run a scenario or benchmark evaluation suite."
    )
    eval_run_parser.add_argument(
        "--kind",
        choices=("scenario", "benchmark"),
        required=True,
        help="Evaluation suite type.",
    )
    eval_run_parser.add_argument(
        "--cases",
        "--file",
        dest="cases",
        type=Path,
        required=True,
        help="Path to the evaluation dataset.",
    )
    eval_run_parser.add_argument(
        "--mode",
        choices=("auto", "stub", "live"),
        default="auto",
        help="Scenario execution mode.",
    )
    eval_run_parser.add_argument("--model", help="Optional live model override.")
    eval_run_parser.add_argument(
        "--report-out",
        type=Path,
        help="Optional JSON report path.",
    )
    eval_run_parser.add_argument(
        "--features-out",
        type=Path,
        help="Optional benchmark feature log path.",
    )
    _add_eval_run_thresholds(eval_run_parser)
    eval_run_parser.set_defaults(handler=command_eval_run)

    eval_compare_parser = eval_subparsers.add_parser(
        "compare", help="Compare two saved evaluation reports."
    )
    eval_compare_parser.add_argument("baseline", type=Path)
    eval_compare_parser.add_argument("candidate", type=Path)
    eval_compare_parser.set_defaults(handler=command_eval_compare)

    eval_regression_parser = eval_subparsers.add_parser(
        "regressions", help="Fail when key report metrics regress."
    )
    eval_regression_parser.add_argument("baseline", type=Path)
    eval_regression_parser.add_argument("candidate", type=Path)
    eval_regression_parser.add_argument(
        "--min-overall-score-delta",
        type=float,
        default=0.0,
        help="Minimum allowed delta for overall_score.",
    )
    eval_regression_parser.add_argument(
        "--min-verdict-accuracy-delta",
        type=float,
        default=0.0,
        help="Minimum allowed delta for verdict_accuracy.",
    )
    eval_regression_parser.add_argument(
        "--min-retrieval-hit-rate-delta",
        type=float,
        default=0.0,
        help="Minimum allowed delta for retrieval_hit_rate.",
    )
    eval_regression_parser.add_argument(
        "--min-answer-support-score-delta",
        type=float,
        default=0.0,
        help="Minimum allowed delta for average_answer_support_score.",
    )
    eval_regression_parser.set_defaults(handler=command_eval_regressions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MAGI terminal helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""workflow:
  magi setup
  magi status
  magi ingest docs/briefing.pdf
  magi ask "what risks should i consider?"

friendly aliases:
  magi ask ...          same as magi chat ...
  magi docs add ...     same as magi ingest ...
  magi runs show <id>   same as magi explain <id>
""",
    )
    parser.set_defaults(handler=None)
    _add_common_options(parser, with_defaults=True)

    subparsers = parser.add_subparsers(dest="command")

    _add_setup_parser(subparsers)
    _add_basic_parser(
        subparsers,
        "status",
        help_text="Show provider, store, artifact, and profile status.",
        handler=command_status,
    )
    _add_basic_parser(
        subparsers,
        "shell",
        help_text="Open the interactive MAGI shell.",
        handler=command_shell,
    )

    _add_ingest_parsers(subparsers)

    _add_chat_parser(subparsers, "chat")
    _add_chat_parser(
        subparsers,
        "ask",
        description="Friendly alias for `magi chat`.",
    )

    _add_batch_parser(subparsers)

    _add_compare_parser(subparsers)

    _add_artifact_parsers(subparsers)
    _add_profiles_parser(subparsers)

    _add_eval_parsers(subparsers)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        if argv is None and sys.stdin.isatty():
            return command_shell(args)
        parser.print_help()
        return 0
    if _requires_provider_setup(args) and not ensure_provider_setup():
        return 1
    store_path = getattr(args, "store", None)
    if isinstance(store_path, Path):
        store_path.parent.mkdir(parents=True, exist_ok=True)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
