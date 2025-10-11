from __future__ import annotations 

import argparse 
from pathlib import Path 
from typing import Dict ,Iterable ,List 

from magi.core.config import get_settings 
from magi.core.embeddings import HashingEmbedder ,build_embedder 
from magi.core.rag import RagRetriever 
from magi.core.storage import initialize_store ,save_entries 
from magi.core.vectorstore import VectorEntry 
from magi.data_pipeline.chunkers import sliding_window_chunk 
from magi.data_pipeline.embed import embed_chunks 
from magi.data_pipeline.ingest import ingest_paths 
from magi.decision.aggregator import resolve_verdict 
from magi.decision.schema import FinalDecision ,PersonaOutput 
from magi.dspy_programs.personas import MagiProgram ,USING_STUB 
from magi.dspy_programs.setup import configure_dspy 

DEFAULT_STORE =Path (__file__ ).resolve ().parents [1 ]/"storage"/"vector_store.json"


def _normalize_residual_label (value :object )->str :
    if not value :
        return "medium"
    label =str (value ).strip ().lower ()
    if not label :
        return "medium"
    mapping ={
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
        embedding =list (record ["embedding"]),
        text =str (record ["text"]),
        metadata ={"source":str (record ["id"]).split ("::")[0 ]},
        )
        )
    return entries 


def command_ingest (args :argparse .Namespace )->None :
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

    embedded =embed_chunks (chunks ,embedder )
    entries =_vector_entries (embedded )
    store .add (entries )
    save_entries (args .store ,store .entries )

    print (f"Ingested {len (entries )} chunks from {len (documents )} document(s).")
    print (f"Store persisted to {args .store }")
    if isinstance (embedder ,HashingEmbedder ):
        print ("Using hashing embedder (offline mode).")
    else :
        print (f"Using OpenAI embeddings ({settings .openai_embedding_model }).")


def command_chat (args :argparse .Namespace )->None :
    settings =get_settings ()
    embedder =build_embedder (settings )
    store =initialize_store (args .store ,embedder )
    retriever =RagRetriever (embedder ,store )

    if not USING_STUB :
        configure_dspy (settings )
    else :
        if not isinstance (embedder ,HashingEmbedder ):
            print ("Warning: DSPy stub active; real embeddings will still be used.")

    if isinstance (embedder ,HashingEmbedder ):
        print ("Using hashing embedder (offline mode).")
    else :
        print (f"Using OpenAI embeddings ({settings .openai_embedding_model }).")

    magi =MagiProgram (retriever =retriever )

    fused ,personas =magi .forward (args .query ,constraints =args .constraints or "")

    persona_outputs =[]
    for name ,result in personas .items ():
        persona_outputs .append (
        PersonaOutput (
        name =name .lower (),
        text =str (result ),
        confidence =float (getattr (result ,"confidence",0.0 )or 0.0 ),
        evidence =[],
        )
        )

    fused_final =str (getattr (fused ,"final_answer","")).strip ()
    fused_justification =str (getattr (fused ,"justification",str (fused ))).strip ()
    combined_justification =fused_final if fused_final else fused_justification
    next_steps =getattr (fused ,"next_steps",[])
    if isinstance (next_steps ,str ):
        next_steps =[next_steps ]if next_steps else []
    if next_steps :
        combined_justification =(
        combined_justification +"\n\nNext steps:\n"+"\n".join (f"- {step }"for step in next_steps )
        ).strip ()
    if not combined_justification :
        combined_justification =str (fused )
    residual_risk_value =_normalize_residual_label (getattr (fused ,"residual_risk",None ))

    decision =FinalDecision (
    verdict =resolve_verdict (fused ,personas ,persona_outputs ),
    justification =combined_justification ,
    persona_outputs =persona_outputs ,
    risks =list (getattr (fused ,"risks",[]))if getattr (fused ,"risks",None )else [],
    mitigations =list (getattr (fused ,"mitigations",[]))
    if getattr (fused ,"mitigations",None )
    else [],
    residual_risk =residual_risk_value ,
    )

    print (f"\nVerdict: {decision .verdict .upper ()}")
    print (f"Justification: {decision .justification }\n")
    for persona in decision .persona_outputs :
        print (f"- {persona .name .title ()} (confidence {persona .confidence :.2f}):")
        print (f"  {persona .text }\n")
    if decision .risks :
        print ("Risks:")
        for risk in decision .risks :
            print (f"  - {risk }")
    if decision .mitigations :
        print ("\nMitigations:")
        for mitigation in decision .mitigations :
            print (f"  - {mitigation }")
    print ()


def build_parser ()->argparse .ArgumentParser :
    parser =argparse .ArgumentParser (description ="MAGI terminal helper")
    parser .set_defaults (handler =None )

    parser .add_argument (
    "--store",
    type =Path ,
    default =DEFAULT_STORE ,
    help =f"Path to persisted vector store (default: {DEFAULT_STORE })",
    )

    subparsers =parser .add_subparsers (dest ="command")

    ingest_parser =subparsers .add_parser ("ingest",help ="Ingest one or more documents.")
    ingest_parser .add_argument ("paths",nargs ="+",help ="Paths to documents for ingestion.")
    ingest_parser .add_argument ("--chunk-size",type =int ,default =512 ,help ="Chunk size in characters.")
    ingest_parser .add_argument ("--chunk-overlap",type =int ,default =64 ,help ="Overlap between chunks.")
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
