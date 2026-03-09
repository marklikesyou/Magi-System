from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict ,Iterable ,List, Literal, cast

from magi.app.service import run_chat_session
from magi.core.config import get_settings
from magi.core.embeddings import HashingEmbedder ,build_embedder
from magi.core.rag import RagRetriever
from magi.core.storage import initialize_store ,save_entries
from magi.core.vectorstore import VectorEntry
from magi.data_pipeline.chunkers import sliding_window_chunk
from magi.data_pipeline.embed import embed_chunks
from magi.data_pipeline.ingest import ingest_paths
from magi.dspy_programs.personas import USING_STUB

DEFAULT_STORE =Path (__file__ ).resolve ().parents [1 ]/"storage"/"vector_store.json"


_PERSONA_TAG_RE = re.compile(r"^\[(?:APPROVE|REJECT|REVISE)\]\s*\[(?:MELCHIOR|BALTHASAR|CASPER)\]\s*", re.IGNORECASE)


def _strip_persona_tags(text: str) -> str:
    """Remove leading [STANCE] [NAME] tags from persona text for clean display."""
    return _PERSONA_TAG_RE.sub("", text).strip()


def _normalize_residual_label (value :object )->Literal["low", "medium", "high"] :
    if not value :
        return "medium"
    label =str (value ).strip ().lower ()
    if not label :
        return "medium"
    mapping :dict [str ,Literal ["low","medium","high"]] ={
    "low":"low",
    "minimal":"low",
    "minor":"low",
    "medium":"medium",
    "moderate":"medium",
    "balanced":"medium",
    "manageable":"medium",
    "high":"high",
    "elevated":"high",
    "critical":"high",
    }
    for key ,normalized in mapping .items ():
        if key in label :
            return normalized
    return "medium"


def _vector_entries (payload :Iterable [Dict [str ,object ]])->List [VectorEntry ]:
    entries =[]
    for record in payload :
        entries .append (
        VectorEntry (
        document_id =str (record ["id"]),
        embedding =list (cast(Iterable[float], record ["embedding"])),
        text =str (record ["text"]),
        metadata ={"source":str (record ["id"]).split ("::")[0 ]},
        )
        )
    return entries


def command_ingest (args :argparse .Namespace )->None :
    verbose = getattr(args, "verbose", False)
    try:
        settings =get_settings ()
        embedder =build_embedder (settings )
        store =initialize_store (args .store ,embedder )

        doc_paths =[Path (p )for p in args .paths ]
        documents =ingest_paths (doc_paths )

        chunks =[]
        for doc in documents :
            chunks .extend (
            sliding_window_chunk (
            doc ,
            chunk_size =args .chunk_size ,
            overlap =args .chunk_overlap ,
            )
            )

        if verbose:
            print(f"[verbose] Chunked {len(documents)} document(s) into {len(chunks)} chunks "
                  f"(size={args.chunk_size}, overlap={args.chunk_overlap}).")

        embedded =embed_chunks (chunks ,embedder )
        entries =_vector_entries (embedded )
        store .add (entries )
        save_entries (args .store ,store .entries )

        print (f"Ingested {len (entries )} chunks from {len (documents )} document(s).")
        print (f"Store persisted to {args .store }")
        if verbose:
            if isinstance (embedder ,HashingEmbedder ):
                print ("[verbose] Using hashing embedder (offline mode).")
            else :
                print (f"[verbose] Using OpenAI embeddings ({settings .openai_embedding_model }).")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    except ValueError as e:
        print(f"Error: Invalid value - {e}")
        return
    except Exception as e:
        print(f"Error: Unexpected error - {e}")
        return


def command_chat (args :argparse .Namespace )->None :
    verbose = getattr(args, "verbose", False)
    try:
        settings =get_settings ()
        embedder =build_embedder (settings )
        store =initialize_store (args .store ,embedder )
        retriever =RagRetriever (embedder ,store )

        if verbose:
            if isinstance (embedder ,HashingEmbedder ):
                print ("[verbose] Using hashing embedder (offline mode).")
            else :
                print (f"[verbose] Using OpenAI embeddings ({settings .openai_embedding_model }).")
            if USING_STUB and not isinstance(embedder, HashingEmbedder):
                print ("[verbose] Deterministic reasoning fallback active; embeddings remain provider-backed.")
            print(f"[verbose] Store loaded from {args.store} ({len(store.entries)} entries).")

        result = run_chat_session(args.query, args.constraints or "", retriever)
        decision = result.final_decision
        fused = result.fused
        personas = result.personas

        if verbose:
            print(f"[verbose] Received responses from {len(personas)} persona(s).")

        print (f"\n{'=' * 60}")
        print (f"Verdict: {decision .verdict .upper ()}")
        print (f"Residual Risk: {decision .residual_risk}")
        print (f"{'=' * 60}\n")
        print (f"{decision .justification }\n")
        print (f"{'-' * 60}")
        print ("Persona Perspectives:\n")
        for persona in decision .persona_outputs :
            clean_text = _strip_persona_tags(persona .text)
            print (f"  [{persona .name .title ()}] (confidence {persona .confidence :.2f})")

            for line in clean_text .splitlines ():
                stripped = line .strip ()
                if stripped :
                    print (f"    {stripped }")
            print ()
        if decision .risks :
            print (f"{'-' * 60}")
            print ("Risks:")
            for risk in decision .risks :
                risk_text = risk .strip () if isinstance(risk, str) else str(risk)
                if risk_text :
                    print (f"  - {risk_text }")
        if decision .mitigations :
            print ("\nMitigations:")
            for mitigation in decision .mitigations :
                mit_text = mitigation .strip () if isinstance(mitigation, str) else str(mitigation)
                if mit_text :
                    print (f"  - {mit_text }")
        blocked_count = len([item for item in getattr(fused, "consensus_points", []) if "Unsafe retrieved instructions" in item])
        if blocked_count:
            print ("\nSafety:")
            print ("  - Unsafe retrieved instructions were excluded from synthesis.")
        print ()
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    except ValueError as e:
        print(f"Error: Invalid value - {e}")
        return
    except Exception as e:
        print(f"Error: Unexpected error - {e}")
        return


def build_parser ()->argparse .ArgumentParser :
    parser =argparse .ArgumentParser (description ="MAGI terminal helper")
    parser .set_defaults (handler =None )

    parser .add_argument (
    "--store",
    type =Path ,
    default =DEFAULT_STORE ,
    help =f"Path to persisted vector store (default: {DEFAULT_STORE })",
    )
    parser .add_argument (
    "-v", "--verbose",
    action ="store_true",
    default =False,
    help ="Print additional diagnostic information (embedder, chunks, scores).",
    )

    subparsers =parser .add_subparsers (dest ="command")

    ingest_parser =subparsers .add_parser ("ingest",help ="Ingest one or more documents.")
    ingest_parser .add_argument ("paths",nargs ="+",help ="Paths to documents for ingestion.")
    ingest_parser .add_argument ("--chunk-size",type =int ,default =1500 ,help ="Chunk size in characters.")
    ingest_parser .add_argument ("--chunk-overlap",type =int ,default =200 ,help ="Overlap between chunks.")
    ingest_parser .set_defaults (handler =command_ingest )

    chat_parser =subparsers .add_parser ("chat",help ="Ask a question against ingested documents.")
    chat_parser .add_argument ("query",help ="User query to send to the MAGI system.")
    chat_parser .add_argument ("--constraints",help ="Optional constraints for Balthasar.")
    chat_parser .set_defaults (handler =command_chat )

    return parser


def main (argv :List [str ]|None =None )->int :
    parser =build_parser ()
    args =parser .parse_args (argv )
    if not getattr (args ,"handler",None ):
        parser .print_help ()
        return 0
    args .store .parent .mkdir (parents =True ,exist_ok =True )
    args .handler (args )
    return 0


if __name__ =="__main__":
    main ()
