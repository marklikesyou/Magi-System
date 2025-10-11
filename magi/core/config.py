

from functools import lru_cache 
from pydantic import Field 
from pydantic_settings import BaseSettings ,SettingsConfigDict 


class Settings (BaseSettings ):
    """this is the env"""

    openai_api_key :str =Field (default ="",env ="OPENAI_API_KEY")
    google_api_key :str =Field (default ="",env ="GOOGLE_API_KEY")
    openai_api_base :str =Field (default ="",env ="OPENAI_API_BASE")
    openai_organization :str =Field (default ="",env ="OPENAI_ORGANIZATION")
    openai_request_timeout :float =Field (
    default =60.0 ,
    env ="OPENAI_REQUEST_TIMEOUT",
    )
    openai_model :str =Field (default ="gpt-4o-mini",env ="OPENAI_MODEL")
    gemini_model :str =Field (default ="gemini-2.5-flash-lite",env ="GEMINI_MODEL")
    openai_embedding_model :str =Field (
    default ="text-embedding-3-small",env ="OPENAI_EMBEDDING_MODEL"
    )
    force_hash_embeddings :bool =Field (
    default =False ,env ="MAGI_FORCE_HASH_EMBEDDER"
    )
    vector_db_url :str =Field (default ="",env ="DATABASE_URL")
    langfuse_public_key :str =Field (default ="",env ="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key :str =Field (default ="",env ="LANGFUSE_SECRET_KEY")
    google_project :str =Field (default ="",env ="GOOGLE_PROJECT")
    google_location :str =Field (default ="",env ="GOOGLE_LOCATION")
    google_use_vertex :bool =Field (default =False ,env ="GOOGLE_USE_VERTEX")

    model_config =SettingsConfigDict (
    env_file =".env",
    env_file_encoding ="utf-8",
    case_sensitive =False ,
    extra ="ignore",
    )


@lru_cache (maxsize =1 )
def get_settings ()->Settings :

    return Settings ()
