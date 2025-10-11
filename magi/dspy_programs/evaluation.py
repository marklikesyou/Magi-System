"""DSPy evaluation metrics and judges for the MAGI system."""

from __future__ import annotations 

from typing import Any ,Dict ,Optional 

from .signatures import STUB_MODE 

if not STUB_MODE :
    import dspy 
    from dspy .evaluate import Evaluate 

    class ConsensusJudge (dspy .Signature ):


        query =dspy .InputField (desc ="The original query posed to the system")
        melchior_analysis =dspy .InputField (desc ="Melchior's scientific analysis")
        balthasar_plan =dspy .InputField (desc ="Balthasar's strategic plan")
        casper_risks =dspy .InputField (desc ="Casper's risk assessment")

        consensus_reached :bool =dspy .OutputField (
        desc ="Did the three personas reach consensus on key points?"
        )
        consensus_score :float =dspy .OutputField (
        desc ="Consensus score from 0.0 (complete disagreement) to 1.0 (full agreement)"
        )
        areas_of_agreement :str =dspy .OutputField (
        desc ="List the specific areas where personas agree"
        )
        areas_of_conflict :str =dspy .OutputField (
        desc ="List the specific areas where personas disagree"
        )

    class DecisionQualityJudge (dspy .Signature ):


        query =dspy .InputField (desc ="The original query")
        verdict =dspy .InputField (desc ="Final verdict: approve/revise/reject")
        justification =dspy .InputField (desc ="Justification for the verdict")
        evidence_quality =dspy .InputField (desc ="Quality of evidence from Melchior")
        plan_feasibility =dspy .InputField (desc ="Feasibility of Balthasar's plan")
        risk_severity =dspy .InputField (desc ="Severity of risks from Casper")

        decision_quality :float =dspy .OutputField (
        desc ="Quality score 0.0-1.0 based on: clarity, justification, and evidence alignment"
        )
        is_well_justified :bool =dspy .OutputField (
        desc ="Is the decision well-justified given the inputs?"
        )
        decision_critique :str =dspy .OutputField (
        desc ="Critique of the decision-making process"
        )

    class PersonaConsistencyJudge (dspy .Signature ):


        persona_name =dspy .InputField (desc ="Name of the persona (Melchior/Balthasar/Casper)")
        expected_traits =dspy .InputField (desc ="Expected personality traits")
        actual_output =dspy .InputField (desc ="The persona's actual output")

        consistency_score :float =dspy .OutputField (
        desc ="How well does the output match expected personality (0.0-1.0)?"
        )
        trait_evidence :str =dspy .OutputField (
        desc ="Evidence of personality traits in the output"
        )
        out_of_character :str =dspy .OutputField (
        desc ="Any behavior that seems out of character"
        )

    class RiskMitigationJudge (dspy .Signature ):


        identified_risks =dspy .InputField (desc ="List of identified risks")
        proposed_mitigations =dspy .InputField (desc ="Proposed mitigation strategies")
        residual_risk =dspy .InputField (desc ="Assessed residual risk level")

        coverage_score :float =dspy .OutputField (
        desc ="How comprehensively are risks covered (0.0-1.0)?"
        )
        mitigation_quality :float =dspy .OutputField (
        desc ="Quality of proposed mitigations (0.0-1.0)"
        )
        gaps_identified :str =dspy .OutputField (
        desc ="Any gaps in risk assessment or mitigation"
        )


    def consensus_metric (gold :Any ,pred :Any ,trace :Optional [Any ]=None )->float :

        judge =dspy .Predict (ConsensusJudge )


        melchior =pred .persona_outputs .get ("melchior",{})
        balthasar =pred .persona_outputs .get ("balthasar",{})
        casper =pred .persona_outputs .get ("casper",{})

        result =judge (
        query =gold .query ,
        melchior_analysis =str (melchior .get ("analysis","")),
        balthasar_plan =str (balthasar .get ("plan","")),
        casper_risks =str (casper .get ("risks",""))
        )

        return float (result .consensus_score )

    def decision_quality_metric (gold :Any ,pred :Any ,trace :Optional [Any ]=None )->float :

        judge =dspy .Predict (DecisionQualityJudge )

        melchior =pred .persona_outputs .get ("melchior",{})
        balthasar =pred .persona_outputs .get ("balthasar",{})
        casper =pred .persona_outputs .get ("casper",{})

        result =judge (
        query =gold .query ,
        verdict =pred .verdict ,
        justification =pred .justification ,
        evidence_quality =f"Confidence: {melchior .get ('confidence',0 )}",
        plan_feasibility =f"Cost: {balthasar .get ('cost_estimate','unknown')}",
        risk_severity =f"Level: {casper .get ('residual_risk','unknown')}"
        )

        return float (result .decision_quality )

    def personality_consistency_metric (gold :Any ,pred :Any ,trace :Optional [Any ]=None )->float :

        judge =dspy .Predict (PersonaConsistencyJudge )

        personas =[
        ("Melchior","Scientific, skeptical, evidence-driven, precise",
        pred .persona_outputs .get ("melchior",{})),
        ("Balthasar","Pragmatic, stakeholder-focused, resource-conscious",
        pred .persona_outputs .get ("balthasar",{})),
        ("Casper","Risk-aware, ethical, conservative, safety-focused",
        pred .persona_outputs .get ("casper",{}))
        ]

        total_score =0.0 
        for name ,traits ,output in personas :
            result =judge (
            persona_name =name ,
            expected_traits =traits ,
            actual_output =str (output .get ("text",""))
            )
            total_score +=float (result .consistency_score )

        return total_score /3.0 

    def risk_coverage_metric (gold :Any ,pred :Any ,trace :Optional [Any ]=None )->float :

        judge =dspy .Predict (RiskMitigationJudge )

        casper =pred .persona_outputs .get ("casper",{})

        result =judge (
        identified_risks =str (casper .get ("risks","")),
        proposed_mitigations =str (casper .get ("mitigations","")),
        residual_risk =str (casper .get ("residual_risk","unknown"))
        )


        return (float (result .coverage_score )+float (result .mitigation_quality ))/2.0 

    def composite_magi_metric (gold :Any ,pred :Any ,trace :Optional [Any ]=None )->float :



        weights ={
        "consensus":0.25 ,
        "decision":0.30 ,
        "personality":0.25 ,
        "risk":0.20 
        }

        scores ={
        "consensus":consensus_metric (gold ,pred ,trace ),
        "decision":decision_quality_metric (gold ,pred ,trace ),
        "personality":personality_consistency_metric (gold ,pred ,trace ),
        "risk":risk_coverage_metric (gold ,pred ,trace )
        }


        total =sum (weights [k ]*scores [k ]for k in weights )


        if trace :
            print (f"Evaluation Breakdown:")
            for key ,score in scores .items ():
                print (f"  {key .capitalize ()}: {score :.2f} (weight: {weights [key ]})")
            print (f"  Total: {total :.2f}")

        return total 



    class MAGISystemJudge (dspy .Signature ):


        query =dspy .InputField (desc ="Original query to the MAGI system")
        context =dspy .InputField (desc ="Retrieved context/evidence")

        melchior_output =dspy .InputField (desc ="Melchior's full analysis")
        melchior_confidence =dspy .InputField (desc ="Melchior's confidence score")

        balthasar_output =dspy .InputField (desc ="Balthasar's strategic plan")
        balthasar_confidence =dspy .InputField (desc ="Balthasar's confidence score")

        casper_output =dspy .InputField (desc ="Casper's risk assessment")
        casper_confidence =dspy .InputField (desc ="Casper's confidence score")

        final_verdict =dspy .InputField (desc ="Final verdict from fusion")
        final_justification =dspy .InputField (desc ="Justification for verdict")


        scientific_rigor :float =dspy .OutputField (
        desc ="Score for scientific analysis quality (0.0-1.0)"
        )
        strategic_feasibility :float =dspy .OutputField (
        desc ="Score for strategic plan feasibility (0.0-1.0)"
        )
        risk_management :float =dspy .OutputField (
        desc ="Score for risk assessment completeness (0.0-1.0)"
        )
        decision_quality :float =dspy .OutputField (
        desc ="Score for final decision quality (0.0-1.0)"
        )
        system_coherence :float =dspy .OutputField (
        desc ="Score for overall system coherence and alignment (0.0-1.0)"
        )


        strengths :str =dspy .OutputField (
        desc ="Key strengths of the MAGI analysis"
        )
        weaknesses :str =dspy .OutputField (
        desc ="Areas needing improvement"
        )
        recommendation :str =dspy .OutputField (
        desc ="Recommendation for decision refinement"
        )

    def comprehensive_judge_metric (gold :Any ,pred :Any ,trace :Optional [Any ]=None )->Dict [str ,float ]:

        judge =dspy .Predict (MAGISystemJudge )

        melchior =pred .persona_outputs .get ("melchior",{})
        balthasar =pred .persona_outputs .get ("balthasar",{})
        casper =pred .persona_outputs .get ("casper",{})

        result =judge (
        query =gold .query ,
        context =gold .context if hasattr (gold ,'context')else "",

        melchior_output =str (melchior .get ("analysis","")),
        melchior_confidence =str (melchior .get ("confidence",0 )),

        balthasar_output =str (balthasar .get ("plan","")),
        balthasar_confidence =str (balthasar .get ("confidence",0 )),

        casper_output =f"Risks: {casper .get ('risks','')}\nMitigations: {casper .get ('mitigations','')}",
        casper_confidence =str (casper .get ("confidence",0 )),

        final_verdict =pred .verdict ,
        final_justification =pred .justification 
        )


        scores ={
        "scientific_rigor":float (result .scientific_rigor ),
        "strategic_feasibility":float (result .strategic_feasibility ),
        "risk_management":float (result .risk_management ),
        "decision_quality":float (result .decision_quality ),
        "system_coherence":float (result .system_coherence ),
        "overall":sum ([
        float (result .scientific_rigor ),
        float (result .strategic_feasibility ),
        float (result .risk_management ),
        float (result .decision_quality ),
        float (result .system_coherence )
        ])/5.0 
        }

        if trace :
            print (f"\n=== MAGI System Evaluation ===")
            print (f"Query: {gold .query }")
            print (f"\nScores:")
            for key ,value in scores .items ():
                if key !="overall":
                    print (f"  {key .replace ('_',' ').title ()}: {value :.2f}")
            print (f"  Overall: {scores ['overall']:.2f}")
            print (f"\nStrengths: {result .strengths }")
            print (f"Weaknesses: {result .weaknesses }")
            print (f"Recommendation: {result .recommendation }")

        return scores 



    def create_magi_evaluator (devset ,metric =None ,num_threads =1 ):

        if metric is None :
            metric =composite_magi_metric 

        return Evaluate (
        devset =devset ,
        metric =metric ,
        num_threads =num_threads ,
        display_progress =True ,
        display_table =True 
        )



    def magi_optimization_metric (gold :Any ,pred :Any ,trace :Optional [Any ]=None )->bool :

        score =composite_magi_metric (gold ,pred ,trace )
        return score >=0.7 

else :

    class _StubJudge :
        def __init__ (self ,*args ,**kwargs ):
            self .args =args 
            self .kwargs =kwargs 

    class ConsensusJudge (_StubJudge ):
        pass 

    class DecisionQualityJudge (_StubJudge ):
        pass 

    class PersonaConsistencyJudge (_StubJudge ):
        pass 

    class RiskMitigationJudge (_StubJudge ):
        pass 

    class MAGISystemJudge (_StubJudge ):
        pass 

    def consensus_metric (*args ,**kwargs ):
        return 0.5 

    def decision_quality_metric (*args ,**kwargs ):
        return 0.5 

    def personality_consistency_metric (*args ,**kwargs ):
        return 0.5 

    def risk_coverage_metric (*args ,**kwargs ):
        return 0.5 

    def composite_magi_metric (*args ,**kwargs ):
        return 0.5 

    def comprehensive_judge_metric (*args ,**kwargs ):
        return {"overall":0.5 }

    def create_magi_evaluator (*args ,**kwargs ):
        raise NotImplementedError ("DSPy required for evaluation")

    def magi_optimization_metric (*args ,**kwargs ):
        return False 


__all__ =[
"ConsensusJudge",
"DecisionQualityJudge",
"PersonaConsistencyJudge",
"RiskMitigationJudge",
"MAGISystemJudge",
"consensus_metric",
"decision_quality_metric",
"personality_consistency_metric",
"risk_coverage_metric",
"composite_magi_metric",
"comprehensive_judge_metric",
"create_magi_evaluator",
"magi_optimization_metric",
]
