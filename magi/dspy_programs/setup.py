from __future__ import annotations

import os
from pathlib import Path

from magi.core.config import Settings
from .signatures import STUB_MODE

if not STUB_MODE :
    import dspy
    from dspy .clients .lm import LM

_CONFIGURED =False
_CACHE_ROOT =Path (__file__ ).resolve ().parents [1 ]/"storage"/"dspy_cache"
_CACHE_ROOT .mkdir (parents =True ,exist_ok =True )


def _resolve_model_name (model :str )->tuple [str ,str ]:
    if "/"in model :
        canonical =model
    else :
        canonical =f"openai/{model }"

    lower =canonical .lower ()
    reasoning_models =("o1-preview","o1-mini","o1","o3-mini","o3")
    is_reasoning =any (lower .endswith (f"/{token }")or lower .endswith (f"/{token }-")or f"/{token }-"in lower
    for token in reasoning_models )
    model_type ="responses"if is_reasoning else "chat"
    return canonical ,model_type


def configure_dspy (settings :Settings )->None :
    if STUB_MODE :
        return

    global _CONFIGURED
    if _CONFIGURED :
        return

    if not settings .openai_api_key :
        raise RuntimeError (
        "OPENAI_API_KEY must be set (see .env) before running MAGI with DSPy."
        )

    os .environ .setdefault ("OPENAI_API_KEY",settings .openai_api_key )
    if settings .openai_api_base :
        os .environ .setdefault ("OPENAI_API_BASE",settings .openai_api_base )
        os .environ .setdefault ("OPENAI_BASE_URL",settings .openai_api_base )
    if settings .openai_organization :
        os .environ .setdefault ("OPENAI_ORGANIZATION",settings .openai_organization )
    os .environ .setdefault (
    "OPENAI_TIMEOUT",
    str (int (settings .openai_request_timeout )),
    )
    os .environ .setdefault ("DSPY_CACHEDIR",str (_CACHE_ROOT ))
    os .environ .setdefault ("DSPY_CACHE_DIR",str (_CACHE_ROOT ))

    model_name ,model_type =_resolve_model_name (settings .openai_model )
    temperature =1.0 if model_type =="responses"else 0.0

    import litellm
    litellm .drop_params =True

    lm =LM (
    model =model_name ,
    model_type =model_type ,
    temperature =temperature ,
    )

    dspy .settings .configure (lm =lm )
    print (f"DSPy configured with model={model_name }, model_type={model_type }, temperature={temperature }")
    _CONFIGURED =True


__all__ =["configure_dspy"]
