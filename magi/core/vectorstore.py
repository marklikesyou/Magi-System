

from __future__ import annotations 

import math 
from dataclasses import dataclass ,field 
from typing import Any ,Dict ,Iterable ,List ,Sequence 


def cosine_similarity (a :Sequence [float ],b :Sequence [float ])->float :


    if len (a )!=len (b ):
        raise ValueError ("vectors must share the same dimensionality.")
    dot =sum (x *y for x ,y in zip (a ,b ))
    norm_a =math .sqrt (sum (x *x for x in a ))
    norm_b =math .sqrt (sum (y *y for y in b ))
    if norm_a ==0.0 or norm_b ==0.0 :
        return 0.0 
    return dot /(norm_a *norm_b )


@dataclass 
class VectorEntry :


    document_id :str 
    embedding :List [float ]
    text :str 
    metadata :Dict [str ,Any ]=field (default_factory =dict )

    def to_dict (self )->Dict [str ,Any ]:
        return {
        "document_id":self .document_id ,
        "embedding":self .embedding ,
        "text":self .text ,
        "metadata":self .metadata ,
        }

    @classmethod 
    def from_dict (cls ,payload :Dict [str ,Any ])->"VectorEntry":
        return cls (
        document_id =payload ["document_id"],
        embedding =list (payload ["embedding"]),
        text =payload ["text"],
        metadata =dict (payload .get ("metadata",{})),
        )


@dataclass 
class RetrievedChunk :


    document_id :str 
    text :str 
    score :float 
    metadata :Dict [str ,Any ]


class InMemoryVectorStore :
    def __init__ (self ,dim :int ):
        self .dim =dim 
        self ._entries :List [VectorEntry ]=[]

    def add (self ,entries :Iterable [VectorEntry ])->None :
        for entry in entries :
            if len (entry .embedding )!=self .dim :
                raise ValueError ("embedding dimension mismatch.")
            self ._entries .append (entry )

    def load (self ,entries :Iterable [VectorEntry ])->None :
        self ._entries =[]
        self .add (entries )

    def dump (self )->List [Dict [str ,Any ]]:
        return [entry .to_dict ()for entry in self ._entries ]

    @property 
    def entries (self )->List [VectorEntry ]:
        return list (self ._entries )

    def search (self ,query_embedding :Sequence [float ],top_k :int =5 )->List [RetrievedChunk ]:
        if len (query_embedding )!=self .dim :
            raise ValueError ("query dimension mismatch.")

        scored =[
        RetrievedChunk (
        document_id =entry .document_id ,
        text =entry .text ,
        score =cosine_similarity (query_embedding ,entry .embedding ),
        metadata =entry .metadata ,
        )
        for entry in self ._entries 
        ]
        scored .sort (key =lambda chunk :chunk .score ,reverse =True )
        return scored [:top_k ]
