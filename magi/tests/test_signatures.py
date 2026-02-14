import pytest

try :
    from magi.dspy_programs import signatures
except ModuleNotFoundError :
    pytest .skip ("dspy is not installed",allow_module_level =True )


def test_analyze_evidence_outputs_confidence ():
    instance =signatures .AnalyzeEvidence ()
    output_fields =getattr (instance ,"output_fields",{})
    assert "confidence"in output_fields


def test_decision_proposal_has_verdict_field ():
    instance =signatures .DecisionProposal ()
    output_fields =getattr (instance ,"output_fields",{})
    assert "verdict"in output_fields
