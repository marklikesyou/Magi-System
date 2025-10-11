"""built to evaluate the personas."""

from __future__ import annotations 

from typing import Iterable 


def accuracy (predictions :Iterable [str ],references :Iterable [str ])->float :
    preds =list (predictions )
    refs =list (references )
    if not preds :
        return 0.0 
    matches =sum (1 for pred ,ref in zip (preds ,refs )if pred ==ref )
    return matches /len (preds )
