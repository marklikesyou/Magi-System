from __future__ import annotations

import json
import re
from typing import Callable, Iterable, Mapping, Sequence

from .semantic import semantic_similarity
from .vectorstore import RetrievedChunk, VectorStore, metadata_matches_filters

Formatter = Callable[[Iterable[RetrievedChunk]], str]
Embedder = Callable[[str], Sequence[float]]


def default_formatter(chunks: Iterable[RetrievedChunk]) -> str:
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        lines.append(f"[{idx}] ({source}) {chunk.text}")
    return "\n".join(lines)


class RagRetriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        formatter: Formatter = default_formatter,
        default_metadata_filters: Mapping[str, object] | None = None,
        source_weights: Mapping[str, float] | None = None,
    ):
        self.embedder = embedder
        self.store = store
        self.formatter = formatter
        self.default_metadata_filters = dict(default_metadata_filters or {})
        self.source_weights = {
            str(key): float(value)
            for key, value in (source_weights or {}).items()
            if str(key).strip()
        }

    @staticmethod
    def _dedupe_key(chunk: RetrievedChunk) -> tuple[str, str]:
        source = str(chunk.metadata.get("source", chunk.document_id))
        compact = re.sub(r"\s+", " ", chunk.text).strip().lower()
        return source, compact

    def cache_token(self) -> str:
        revision = getattr(self.store, "revision", "unknown")
        token_payload = {
            "store_id": id(self.store),
            "revision": revision,
            "default_metadata_filters": self.default_metadata_filters,
            "source_weights": self.source_weights,
            "preferred_top_k": getattr(self, "preferred_top_k", None),
            "embedder": {
                "class": self.embedder.__class__.__qualname__,
                "dimension": getattr(self.embedder, "dimension", None),
                "model": getattr(self.embedder, "model", None),
            },
        }
        return json.dumps(token_payload, sort_keys=True, default=str)

    def _combined_filters(
        self, metadata_filters: Mapping[str, object] | None
    ) -> Mapping[str, object] | None:
        if not self.default_metadata_filters and not metadata_filters:
            return None
        combined = dict(self.default_metadata_filters)
        if metadata_filters:
            combined.update(metadata_filters)
        return combined

    def _lexical_score(self, query: str, chunk: RetrievedChunk) -> float:
        return semantic_similarity(query, (chunk.text,))

    def _source_weight(self, chunk: RetrievedChunk) -> float:
        source = str(chunk.metadata.get("source", "")).strip()
        if not source:
            return 0.0
        direct = self.source_weights.get(source)
        if direct is not None:
            return direct - 1.0
        basename = source.rsplit("/", 1)[-1]
        mapped = self.source_weights.get(basename)
        if mapped is not None:
            return mapped - 1.0
        return 0.0

    def _section_boost(self, query: str, chunk: RetrievedChunk) -> float:
        title = str(chunk.metadata.get("section_title", "")).strip().lower()
        if not title:
            return 0.0
        return min(0.15, semantic_similarity(query, (title,)) * 0.2)

    def _hybrid_candidates(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filters: Mapping[str, object] | None,
    ) -> list[RetrievedChunk]:
        enriched_query = query
        semantic_top_k = max(top_k * 4, 20)
        embedding = self.embedder(enriched_query)
        semantic_results = self.store.search(
            embedding,
            top_k=semantic_top_k,
            metadata_filters=metadata_filters,
        )
        keyword_search = getattr(self.store, "keyword_search", None)
        lexical_results: list[RetrievedChunk]
        if callable(keyword_search):
            lexical_results = keyword_search(
                query,
                top_k=max(top_k * 8, 40),
                metadata_filters=metadata_filters,
            )
        else:
            lexical_results = []
            for entry in self.store.entries:
                if not metadata_matches_filters(entry.metadata, metadata_filters):
                    continue
                base_chunk = RetrievedChunk(
                    document_id=entry.document_id,
                    text=entry.text,
                    score=0.0,
                    metadata=entry.metadata,
                )
                lexical_score = self._lexical_score(query, base_chunk)
                if lexical_score <= 0.0:
                    continue
                chunk = RetrievedChunk(
                    document_id=entry.document_id,
                    text=entry.text,
                    score=lexical_score,
                    metadata=entry.metadata,
                )
                lexical_results.append(chunk)

        semantic_by_id = {item.document_id: item for item in semantic_results}
        lexical_by_id = {item.document_id: item for item in lexical_results}
        candidate_ids = list(
            dict.fromkeys(
                [item.document_id for item in semantic_results]
                + [item.document_id for item in lexical_results]
            )
        )
        candidates: dict[str, RetrievedChunk] = {}
        for document_id in candidate_ids:
            existing = semantic_by_id.get(document_id)
            lexical_chunk = lexical_by_id.get(document_id)
            candidate = existing or lexical_chunk
            if candidate is None:
                continue
            semantic = 0.0 if existing is None else max(0.0, float(existing.score))
            lexical = (
                max(0.0, float(lexical_chunk.score))
                if lexical_chunk is not None
                else self._lexical_score(query, candidate)
            )
            section_boost = self._section_boost(query, candidate)
            source_boost = self._source_weight(candidate)
            hybrid = (semantic * 0.65) + (lexical * 0.35) + section_boost + source_boost
            metadata = dict(candidate.metadata)
            metadata["semantic_score"] = round(semantic, 4)
            metadata["lexical_score"] = round(lexical, 4)
            metadata["section_boost"] = round(section_boost, 4)
            metadata["source_boost"] = round(source_boost, 4)
            candidates[document_id] = RetrievedChunk(
                document_id=candidate.document_id,
                text=candidate.text,
                score=hybrid,
                metadata=metadata,
            )
        ranked = list(candidates.values())
        ranked.sort(key=lambda chunk: chunk.score, reverse=True)
        return ranked

    def retrieve(
        self,
        query: str,
        *,
        persona: str | None = None,
        top_k: int = 8,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        if not query:
            return []
        combined_filters = self._combined_filters(metadata_filters)
        enriched_query = f"[{persona}] {query}" if persona else query
        results = self._hybrid_candidates(
            enriched_query,
            top_k=top_k,
            metadata_filters=combined_filters,
        )
        page_numbers = {
            match.group(1)
            for match in re.finditer(r"page\s+(\d+)", query, re.IGNORECASE)
        }
        page_tokens = {f"page {value}" for value in page_numbers}
        if page_numbers:
            page_search = getattr(self.store, "page_search", None)
            if callable(page_search):
                matched = page_search(
                    page_numbers,
                    top_k=max(top_k * 4, 20),
                    metadata_filters=combined_filters,
                )
            else:
                page_suffixes = {f"#page-{value}" for value in page_numbers}
                matched = []
                for entry in self.store.entries:
                    if not metadata_matches_filters(entry.metadata, combined_filters):
                        continue
                    entry_id = entry.document_id.lower()
                    lower_text = entry.text.lower()
                    matched_request = False
                    for token in page_tokens:
                        token_lower = token.lower()
                        suffix = token_lower.replace(" ", "-")
                        if (
                            token_lower in lower_text
                            or suffix in entry_id
                            or any(
                                suffix_alt in entry_id
                                for suffix_alt in page_suffixes
                            )
                        ):
                            matched_request = True
                            break
                    if matched_request:
                        matched.append(
                            RetrievedChunk(
                                document_id=entry.document_id,
                                text=entry.text,
                                score=1.2,
                                metadata=entry.metadata,
                            )
                        )
            if matched:
                seen: set[tuple[str, str]] = set()
                combined: list[RetrievedChunk] = []
                for chunk in matched + results:
                    key = self._dedupe_key(chunk)
                    if key in seen:
                        continue
                    seen.add(key)
                    combined.append(chunk)
                results = combined

        if page_numbers:
            page_requests = {token.lower() for token in page_tokens}

            def page_match_score(chunk: RetrievedChunk) -> int:
                text_lower = chunk.text.lower()
                metadata_source = str(chunk.metadata.get("source", "")).lower()
                doc_id_lower = chunk.document_id.lower()
                score = 0
                for label in page_requests:
                    page_suffix = label.replace(" ", "-")
                    if (
                        label in text_lower
                        or label in metadata_source
                        or f"#{page_suffix}" in doc_id_lower
                        or f"/{page_suffix}" in doc_id_lower
                    ):
                        score += 1
                return score

            results.sort(
                key=lambda chunk: (page_match_score(chunk), chunk.score),
                reverse=True,
            )
        unique: list[RetrievedChunk] = []
        seen_keys: set[tuple[str, str]] = set()
        for chunk in results:
            key = self._dedupe_key(chunk)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique.append(chunk)
        return unique[:top_k]

    def __call__(
        self,
        query: str,
        *,
        persona: str | None = None,
        top_k: int = 8,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> str:
        results = self.retrieve(
            query,
            persona=persona,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )
        if not results:
            return ""
        return self.formatter(results)
