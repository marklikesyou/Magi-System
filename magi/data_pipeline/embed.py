

from __future__ import annotations 

from typing import Callable ,Dict ,Iterable ,List ,Sequence 

EmbedFn =Callable [[str ],Sequence [float ]]


def embed_chunks (chunks :Iterable [Dict [str ,str ]],embed_fn :EmbedFn )->List [Dict [str ,object ]]:


    embedded =[]
    for chunk in chunks :
        vector =list (embed_fn (chunk ["text"]))
        embedded .append ({**chunk ,"embedding":vector })
    return embedded 
