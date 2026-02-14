

from __future__ import annotations

from pathlib import Path
from typing import Dict ,Iterable ,List


def _extract_pdf_pages (path :Path )->List [Dict [str ,str ]]:
    try :
        from pypdf import PdfReader
    except ImportError as exc :
        raise RuntimeError (
        "pypdf missing"
        )from exc

    reader =PdfReader (str (path ))
    records :List [Dict [str ,str ]]=[]
    for index ,page in enumerate (reader .pages ,start =1 ):
        text =page .extract_text ()or ""
        labeled =f"[Page {index }]\n{text.strip ()}"
        records .append ({"id":f"{path }#page-{index }","text":labeled })
    return records


def load_text (path :Path )->List [Dict [str ,str ]]:
    if not path .exists ():
        raise FileNotFoundError (path )
    try :
        content =path .read_text (encoding ="utf-8")
        return [{"id":str (path ),"text":content }]
    except UnicodeDecodeError :
        if path .suffix .lower ()==".pdf":
            return _extract_pdf_pages (path )
        raise RuntimeError (
        f"cannot decode {path } as UTF-8 text. Provide plain text or install "
        "a parser for this format."
        )


def ingest_paths (paths :Iterable [Path ])->List [Dict [str ,str ]]:
    records =[]
    for path in paths :
        entries =load_text (path )
        added =False
        for entry in entries :
            text =entry .get ("text","")
            if not text .strip ():
                continue
            records .append (entry )
            added =True
        if not added :
            print (f"Warning: {path } produced no extractable text, skipping.")
    return records
