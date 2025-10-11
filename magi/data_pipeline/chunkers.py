

from __future__ import annotations 

from typing import Dict ,Iterable ,List 


def sliding_window_chunk (
document :Dict [str ,str ],
*,
chunk_size :int =512 ,
overlap :int =64 ,
)->List [Dict [str ,str ]]:


    text =document ["text"]
    doc_id =document ["id"]
    if overlap >=chunk_size :
        raise ValueError ("overlap must be smaller than chunk_size")

    chunks =[]
    start =0 
    idx =0 
    while start <len (text ):
        end =start +chunk_size 
        chunk_text =text [start :end ]
        chunks .append ({"id":f"{doc_id }::chunk-{idx }","text":chunk_text })
        if end >=len (text ):
            break 
        start =end -overlap 
        idx +=1 
    return chunks 
