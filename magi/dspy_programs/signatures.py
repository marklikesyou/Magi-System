from __future__ import annotations 

import os 
from pathlib import Path 

_cache_dir =Path (__file__ ).resolve ().parents [1 ]/"storage"/"dspy_cache"
os .environ .setdefault ("DSPY_CACHEDIR",str (_cache_dir ))
os .environ .setdefault ("DSPY_CACHE_DIR",str (_cache_dir ))
_cache_dir .mkdir (parents =True ,exist_ok =True )

_force_stub =os .getenv ("MAGI_FORCE_DSPY_STUB","0")!="0"

try :
    if _force_stub :
        raise ModuleNotFoundError ("forced stub mode for DSPy signatures.")
    import dspy 
    from dspy .signatures import InputField ,OutputField 
except ModuleNotFoundError :
    STUB_MODE =True 

    class _Field :
        def __init__ (self ,desc :str |None =None ):
            self .desc =desc 

    class _Signature :
        pass 

    class _DSPyStub :
        Signature =_Signature 

    dspy =_DSPyStub ()

    def InputField (*_ :object ,**__ :object )->_Field :
        return _Field ()

    def OutputField (*_ :object ,desc :str |None =None ,**__ :object )->_Field :
        return _Field (desc =desc )
else :
    STUB_MODE =False 


class AnalyzeEvidence (dspy .Signature ):
    query :str =InputField ()
    context :str =InputField ()
    analysis :str =OutputField (desc ="Concise, sourced analysis with inline citations")
    confidence :float =OutputField ()
    stance :str =OutputField ()
    actions :str =OutputField ()


class StakeholderPlan (dspy .Signature ):

    query :str =InputField ()
    constraints :str =InputField ()
    context :str =InputField ()
    plan :str =OutputField ()
    cost_estimate :str =OutputField ()
    confidence :float =OutputField ()
    stance :str =OutputField ()
    actions :str =OutputField ()


class EthicalRisk (dspy .Signature ):

    proposal :str =InputField ()
    risks :str =OutputField ()
    mitigations :str =OutputField ()
    residual_risk :str =OutputField ()
    confidence :float =OutputField ()
    stance :str =OutputField ()
    actions :str =OutputField ()
    outstanding_questions :str =OutputField ()


class DecisionProposal (dspy .Signature ):

    query :str =InputField ()
    melchior :str =InputField ()
    balthasar :str =InputField ()
    casper :str =InputField ()
    verdict :str =OutputField (desc ="One of approve, reject, or revise.")
    justification :str =OutputField ()
    confidence :float =OutputField ()
    final_answer :str =OutputField ()
    next_steps :str =OutputField ()
    consensus_points :str =OutputField ()
    disagreements :str =OutputField ()

class ExplanationDraft (dspy .Signature ):

    query :str =InputField ()
    context :str =InputField ()
    dialogue :str =InputField ()
    melchior :str =InputField ()
    balthasar :str =InputField ()
    casper :str =InputField ()
    final_answer :str =OutputField ()
    justification :str =OutputField ()
    next_steps :str =OutputField ()


if STUB_MODE :
    AnalyzeEvidence .output_fields ={"analysis":None ,"answer_outline":None ,"confidence":None ,"stance":None ,"actions":None }
    StakeholderPlan .output_fields ={"plan":None ,"communication_plan":None ,"cost_estimate":None ,"confidence":None ,"stance":None ,"actions":None }
    EthicalRisk .output_fields ={"risks":None ,"mitigations":None ,"residual_risk":None ,"confidence":None ,"stance":None ,"actions":None ,"outstanding_questions":None }
    DecisionProposal .output_fields ={
    "verdict":None ,
    "justification":None ,
    "confidence":None ,
    "final_answer":None ,
    "next_steps":None ,
    "consensus_points":None ,
    "disagreements":None ,
    }
    ExplanationDraft .output_fields ={
    "final_answer":None ,
    "justification":None ,
    "next_steps":None ,
    }


__all__ =[
"AnalyzeEvidence",
"StakeholderPlan",
"EthicalRisk",
"DecisionProposal",
"ExplanationDraft",
"STUB_MODE",
]


