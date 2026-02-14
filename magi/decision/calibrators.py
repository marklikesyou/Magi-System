

from __future__ import annotations

from typing import Any

try :
    import torch
    import torch .nn as nn
    import torch .nn .functional as F
except ImportError :
    torch =None
    nn =None
    F =None


class PairwiseScorer :


    def __init__ (self ,dim :int =384 ):
        if torch is None or nn is None :
            raise RuntimeError ("PyTorch is required for PairwiseScorer. Install with: pip install 'magi-system[torch]'")
        self ._model =nn .Sequential (nn .Linear (dim ,dim ),nn .ReLU (),nn .Linear (dim ,1 ))

    def __call__ (self ,embeddings :Any )->Any :
        return self .forward (embeddings )

    def forward (self ,embeddings :"torch.Tensor")->"torch.Tensor":
        if torch is None :
            raise RuntimeError ("PyTorch is required for PairwiseScorer. Install with: pip install 'magi-system[torch]'")
        if embeddings .ndim !=2 :
            raise ValueError ("Expected embeddings shaped [batch, dim].")
        return self ._model (embeddings ).squeeze (-1 )


def bradley_terry_loss (s_pos :"torch.Tensor",s_neg :"torch.Tensor")->"torch.Tensor":
    if torch is None or F is None :
        raise RuntimeError ("PyTorch is required for PairwiseScorer. Install with: pip install 'magi-system[torch]'")
    if s_pos .shape !=s_neg .shape :
        raise ValueError ("score tensors must share the same shape")
    logits =s_pos -s_neg
    target =torch .ones_like (logits )
    return F .binary_cross_entropy_with_logits (logits ,target )
