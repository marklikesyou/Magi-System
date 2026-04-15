

from __future__ import annotations

import re
from typing import Callable ,Iterable ,Sequence

from .vectorstore import InMemoryVectorStore ,RetrievedChunk

Formatter =Callable [[Iterable [RetrievedChunk ]],str ]
Embedder =Callable [[str ],Sequence [float ]]

def default_formatter (chunks :Iterable [RetrievedChunk ])->str :
    lines =[]
    for idx ,chunk in enumerate (chunks ,start =1 ):
        source =chunk .metadata .get ("source","unknown")
        lines .append (f"[{idx }] ({source }) {chunk .text }")
    return "\n".join (lines )

class RagRetriever :
    def __init__ (
    self ,
    embedder :Embedder ,
    store :InMemoryVectorStore ,
    *,
    formatter :Formatter =default_formatter ,
    ):
        self .embedder =embedder
        self .store =store
        self .formatter =formatter

    @staticmethod
    def _dedupe_key(chunk: RetrievedChunk) -> tuple[str, str]:
        source = str(chunk.metadata.get("source", chunk.document_id))
        return source, chunk.text

    def retrieve(self, query: str, *, persona: str | None = None, top_k: int = 8) -> list[RetrievedChunk]:
        if not query :
            return []
        enriched_query =f"[{persona }] {query }"if persona else query
        embedding =self .embedder (enriched_query )
        results =self .store .search (embedding ,top_k =top_k )
        page_numbers ={
        match .group (1 )
        for match in re .finditer (r"page\s+(\d+)",query ,re .IGNORECASE )
        }
        if page_numbers :
            page_tokens ={f"page {value }"for value in page_numbers }
            page_suffixes ={f"#page-{value }"for value in page_numbers }
            matched :list [RetrievedChunk ]=[]
            for entry in self .store .entries :
                entry_id =entry .document_id .lower ()
                lower_text =entry .text .lower ()
                matched_request =False
                for token in page_tokens :
                    token_lower =token .lower ()
                    suffix =token_lower .replace (" ","-")
                    if (
                    token_lower in lower_text
                    or suffix in entry_id
                    or any (suffix_alt in entry_id for suffix_alt in page_suffixes )
                    ):
                        matched_request =True
                        break
                if matched_request :
                    matched .append (
                    RetrievedChunk (
                    document_id =entry .document_id ,
                    text =entry .text ,
                    score =1.0 ,
                    metadata =entry .metadata ,
                    )
                    )
            if matched :
                seen :set [tuple [str ,str ]]=set ()
                combined :list [RetrievedChunk ]=[]
                for chunk in matched +results :
                    key =self._dedupe_key (chunk )
                    if key in seen :
                        continue
                    seen .add (key )
                    combined .append (chunk )
                results =combined

        if page_numbers :
            page_requests ={token .lower ()for token in page_tokens }

            def page_match_score (chunk :RetrievedChunk )->int :
                text_lower =chunk .text .lower ()
                metadata_source =str (chunk .metadata .get ("source","")).lower ()
                doc_id_lower =chunk .document_id .lower ()
                score =0
                for label in page_requests :
                    page_suffix =label .replace (" ","-")
                    if (
                    label in text_lower
                    or label in metadata_source
                    or f"#{page_suffix}"in doc_id_lower
                    or f"/{page_suffix}"in doc_id_lower
                    ):
                        score +=1
                return score

            results .sort (
            key =lambda chunk :(page_match_score (chunk ),chunk .score ),
            reverse =True ,
            )
        unique :list [RetrievedChunk ]=[]
        seen_keys :set [tuple [str ,str ]]=set ()
        for chunk in results :
            key =self._dedupe_key (chunk )
            if key in seen_keys :
                continue
            seen_keys .add (key )
            unique .append (chunk )
        return unique [:top_k]

    def __call__ (self ,query :str ,*,persona :str |None =None ,top_k :int =8 )->str :
        results =self .retrieve (query ,persona =persona ,top_k =top_k )
        if not results :
            return ""
        return self .formatter (results )
