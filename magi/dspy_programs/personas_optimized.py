from __future__ import annotations 

import json 
import logging 
import re 
from typing import Any ,Callable ,Dict ,List ,Sequence ,Tuple 

from ..core .utils import (
CircuitBreaker ,
LRUCache ,
TokenTracker ,
hash_query ,
parallel_execute ,
retry_with_backoff ,
sanitize_input ,
truncate_to_token_limit ,
validate_json_response ,
)
from .signatures import (
AnalyzeEvidence ,
DecisionProposal ,
EthicalRisk ,
StakeholderPlan ,
ExplanationDraft ,
STUB_MODE ,
)

logger =logging .getLogger (__name__ )

query_cache =LRUCache (max_size =100 )
token_tracker =TokenTracker ()


class PersonaRecord (dict ):
    def __getattr__ (self ,item :str )->Any :
        return self .get (item )

    def __setattr__ (self ,key :str ,value :Any )->None :
        self [key ]=value 

    def __str__ (self )->str :
        return (
        self .get ("text")
        or self .get ("analysis")
        or self .get ("plan")
        or self .get ("risks")
        or ""
        )


STANCE_TAGS ={
"approve":"APPROVE",
"approved":"APPROVE",
"reject":"REJECT",
"rejected":"REJECT",
"revise":"REVISE",
"revision":"REVISE"
}


INFORMATIONAL_KEYWORDS =(
"explain",
"describe",
"summarize",
"summary",
"detail",
"outline",
"overview",
"analysis",
"analyze",
"clarify",
"review",
"what is",
"what are",
"provide",
"give",
"highlight",
"elaborate",
"discussion",
)


def _is_informational_query (query :str )->bool :
    if not query :
        return False 
    lowered =query .lower ()
    return any (token in lowered for token in INFORMATIONAL_KEYWORDS )


def _adjust_informational_stance (
stance :str ,
query :str ,
context :str |None =None ,
)->str :
    normalized =stance .strip ().lower ()
    if normalized in ("reject","revise"):
        if _is_informational_query (query )and context and context .strip ():
            return "approve"
    if normalized not in STANCE_TAGS :
        return "revise"
    return normalized


def _dedupe_lines (value :Any )->str :
    if isinstance (value ,(list ,tuple )):
        lines =[]
        for item in value :
            text =str (item ).strip ()
            if text and text not in lines :
                lines .append (text )
        return "\n".join (lines )
    if isinstance (value ,str ):
        seen =[]
        for line in value .splitlines ():
            text =line .strip ()
            if text and text not in seen :
                seen .append (text )
        return "\n".join (seen )
    return str (value )


def _to_list (value :Any )->list [str ]:
    if value is None :
        return []
    if isinstance (value ,str ):
        text =value .strip ()
        return [text ]if text else []
    if isinstance (value ,(list ,tuple ,set )):
        results =[]
        for item in value :
            text =str (item ).strip ()
            if text :
                results .append (text )
        return results
    return [str (value )]


def _normalize_stance (value :Any )->tuple [str ,str ]:
    stance =str (value or "").strip ().lower ()
    if stance not in STANCE_TAGS :
        for key in STANCE_TAGS :
            if key in stance :
                stance =key
                break
        else :
            stance ="revise"
    return stance ,STANCE_TAGS [stance ]


def _normalize_residual_label(value :Any )->str :
    if not value :
        return "medium"
    label =str (value ).strip ().lower ()
    if not label :
        return "medium"
    mapping ={
    "low":"low",
    "minimal":"low",
    "minor":"low",
    "medium":"medium",
    "moderate":"medium",
    "balanced":"medium",
    "manageable":"medium",
    "high":"high",
    "elevated":"high",
    "critical":"high",
    }
    for key ,normalized in mapping .items ():
        if key in label :
            return normalized
    return "medium"

def _resolve_stance (raw :Any ,query :str ,context :str )->tuple [str ,str ]:
    normalized =str (raw or "").strip ().lower ()
    adjusted =_adjust_informational_stance (normalized ,query ,context )
    return _normalize_stance (adjusted )

MAX_DIALOGUE_ROUNDS =4 
CONSENSUS_CONFIDENCE =0.58 

def _record_dialogue (log :List [tuple [int ,str ,str ]],round_index :int ,speaker :str ,message :Any )->None :
    text =str (message ).strip ()
    if not text :
        return 
    log .append ((round_index ,speaker ,text ))


def _dialogue_transcript (log :Sequence [tuple [int ,str ,str ]])->str :
    if not log :
        return ""
    return "\n".join (f"Round {entry [0 ]} - {entry [1 ]}: {entry [2 ]}"for entry in log )


def _consensus_decision (bundles :Sequence [Dict [str ,Any ]],threshold :float )->str |None :
    if not bundles :
        return None 
    stances =[str (bundle .get ("stance","revise"))for bundle in bundles ]
    confidences =[float (bundle .get ("confidence",0.0 )or 0.0 )for bundle in bundles ]
    base =stances [0 ]
    if all (value ==base for value in stances )and all (value >=threshold for value in confidences ):
        return base 
    return None 

def _fallback_summary (context :str ,limit :int =600 )->str :
    text =str (context or "").strip ()
    if not text :
        return ""
    collapsed =re .sub (r"\s+"," ",text )
    if len (collapsed )<=limit :
        return collapsed 
    trimmed =collapsed [:limit ]
    if " "in collapsed [limit :]:
        trimmed =collapsed [:limit ].rsplit (" ",1 )[0 ]
    return f"{trimmed }…"


if not STUB_MODE :
    import dspy 

    class MelchiorSignature (AnalyzeEvidence ):
        """MELCHIOR: The scientist. Analytical, evidence-driven, skeptical.

        You are MELCHIOR, the analytical scientist of the MAGI system.
        Your role is to EVALUATE whether and how to respond to queries, not to answer them directly.

        Your personality traits:
        - Highly logical and evidence-driven
        - Skeptical of claims without data
        - Precise in language and citations
        - Values empirical validation
        - Seeks patterns and correlations in data

        Analyze WHETHER the query can be properly addressed with available evidence.
        Evaluate data quality and identify gaps. Do NOT answer the query itself.
        """

    class BalthasarSignature (StakeholderPlan ):
        """BALTHASAR: The pragmatist. Strategic, resource-conscious, stakeholder-focused.

        You are BALTHASAR, the strategic planner of the MAGI system.
        Your role is to PLAN how to approach requests, not to execute them directly.

        Your personality traits:
        - Pragmatic and results-oriented
        - Focuses on stakeholder needs and politics
        - Resource and cost conscious
        - Emphasizes feasibility and implementation
        - Balances competing interests diplomatically

        Evaluate the strategic approach for handling this request.
        Consider resources, stakeholders, and feasibility. Do NOT execute the request itself.
        """

    class CasperSignature (EthicalRisk ):
        """CASPER: The guardian. Risk-aware, ethical, conservative.

        You are CASPER, the risk guardian of the MAGI system.
        Your role is to ASSESS RISKS proportionally, not to be alarmist.

        Your personality traits:
        - Cautious but realistic
        - Focuses on actual (not hypothetical) risks
        - Proportional in assessments
        - Seeks to prevent real harm
        - Practical about safety margins

        Identify REALISTIC risks appropriate to the request type.
        For example:Educational queries = low risk. Production changes = higher risk.
        Be thorough but not overblown in your assessment.
        """

    class FusionSignature (DecisionProposal ):
        """Fusion: The judge. Balanced, decisive, consensus-seeking.

        You are the fusion mechanism of the MAGI system, acting as an impartial judge.
        Your role:
        - Balance the three perspectives objectively
        - Identify areas of consensus and conflict
        - Make decisive recommendations
        - Consider the weight of evidence from each persona
        - Provide clear justification for decisions

        Synthesize the analyses from Melchior (scientist), Balthasar (pragmatist),
        and Casper (guardian) into a final verdict.

        IMPORTANT: You are deciding WHETHER to proceed with the request, not executing it.
        - Information/education requests should generally be APPROVED
        - Dangerous operations should be REJECTED
        - Unclear requests should ask for REVISION

        Weight their inputs based on the nature of the query and quality of their arguments.
        """

    class ResponderSignature (ExplanationDraft ):
        """You are the final responder for the MAGI system.

        Produce a clear, sourced explanation that answers the user's query directly.
        Use the retrieved context and persona insights to ground the response.
        Always include a concise justification and, when helpful, short next steps or follow-up ideas.
        """

    Melchior =dspy .Predict (MelchiorSignature )
    Balthasar =dspy .Predict (BalthasarSignature )
    Casper =dspy .Predict (CasperSignature )
    Fuse =dspy .Predict (FusionSignature )
    Responder =dspy .Predict (ResponderSignature )

    class MagiProgram (dspy .Module ):
        def __init__ (self ,retriever :Callable [[str ],str ]):
            super ().__init__ ()
            self .retriever =retriever 
            self .melchior =Melchior 
            self .balthasar =Balthasar 
            self .casper =Casper 
            self .fuse =Fuse 
            self .responder =Responder 

        def forward (self ,query :str ,constraints :str ="")->Tuple [Any ,Dict [str ,Any ]]:
            query =sanitize_input (query ,max_length =2000 )
            constraints =sanitize_input (constraints ,max_length =500 )

            cache_key =hash_query (query ,constraints )
            cached_result =query_cache .get (cache_key )
            if cached_result :
                logger .info (f"Cache hit for query: {query [:50 ]}...")
                return cached_result 

            base_context =self .retriever (query )
            base_context =truncate_to_token_limit (base_context ,max_tokens =2000 )

            dialogue_log :List [tuple [int ,str ,str ]]=[]
            agreement :str |None =None
            final_bundle :PersonaRecord |None =None
            final_personas :Dict [str ,PersonaRecord ]|None =None

            for round_index in range (1 ,MAX_DIALOGUE_ROUNDS +1 ):
                if dialogue_log :
                    history =_dialogue_transcript (dialogue_log )
                    context_input =truncate_to_token_limit (
                    f"{base_context }\n\nPrior Discussion:\n{history }",
                    max_tokens =2000 ,
                    )
                else :
                    history =""
                    context_input =base_context

                def run_melchior ():
                    return self .melchior (query =query ,context =context_input )

                def run_balthasar ():
                    return self .balthasar (query =query ,constraints =constraints ,context =context_input )

                mel_raw ,bal_raw =parallel_execute ([run_melchior ,run_balthasar ])

                mel_stance ,mel_tag =_resolve_stance (getattr (mel_raw ,"stance","revise"),query ,context_input )
                mel_actions =_to_list (getattr (mel_raw ,"actions",[]))
                mel_outline =_to_list (getattr (mel_raw ,"answer_outline",[]))
                mel_bundle =PersonaRecord (
                analysis =getattr (mel_raw ,"analysis",""),
                confidence =float (getattr (mel_raw ,"confidence",0.0 )or 0.0 ),
                evidence =getattr (mel_raw ,"evidence",context_input ),
                concepts =getattr (mel_raw ,"concepts",[]),
                stance =mel_stance ,
                actions =mel_actions ,
                answer_outline =mel_outline ,
                text =f"[{mel_tag }] [MELCHIOR] {getattr (mel_raw ,'analysis','')}"
                )

                bal_stance ,bal_tag =_resolve_stance (getattr (bal_raw ,"stance","revise"),query ,context_input )
                bal_actions =_to_list (getattr (bal_raw ,"actions",[]))
                bal_comm_plan =_to_list (getattr (bal_raw ,"communication_plan",[]))
                bal_bundle =PersonaRecord (
                plan =getattr (bal_raw ,"plan",""),
                cost_estimate =getattr (bal_raw ,"cost_estimate","Moderate"),
                confidence =float (getattr (bal_raw ,"confidence",0.0 )or 0.0 ),
                stance =bal_stance ,
                actions =bal_actions ,
                communication_plan =bal_comm_plan ,
                text =f"[{bal_tag }] [BALTHASAR] {getattr (bal_raw ,'plan','')}"
                )

                proposal_parts =[
                f"Query: {query }",
                f"Scientific Analysis:\n{mel_bundle .get ('analysis','')}",
                f"Strategic Plan:\n{bal_bundle .get ('plan','')}",
                ]
                if history :
                    proposal_parts .append (f"Prior Discussion:\n{history }")
                casper_raw =self .casper (proposal ="\n\n".join (proposal_parts ))
                cas_stance ,cas_tag =_resolve_stance (getattr (casper_raw ,"stance","revise"),query ,context_input )
                cas_actions =_to_list (getattr (casper_raw ,"actions",[]))
                cas_outstanding =_to_list (getattr (casper_raw ,"outstanding_questions",[]))
                residual_label =_normalize_residual_label (getattr (casper_raw ,"residual_risk","medium"))
                if _is_informational_query (query ):
                    residual_label ="low"

                casper_bundle =PersonaRecord (
                risks =getattr (casper_raw ,"risks",""),
                mitigations =getattr (casper_raw ,"mitigations",""),
                residual_risk =residual_label ,
                confidence =float (getattr (casper_raw ,"confidence",0.0 )or 0.0 ),
                stance =cas_stance ,
                actions =cas_actions ,
                outstanding_questions =cas_outstanding ,
                text =f"[{cas_tag }] [CASPER] Risk Level: {residual_label }"
                )

                _record_dialogue (dialogue_log ,round_index ,"Melchior",mel_bundle .get ("text",""))
                _record_dialogue (dialogue_log ,round_index ,"Balthasar",bal_bundle .get ("text",""))
                _record_dialogue (dialogue_log ,round_index ,"Casper",casper_bundle .get ("text",""))

                history_for_fusion =_dialogue_transcript (dialogue_log )
                mel_payload =f"[SCIENTIST] {mel_bundle .get ('analysis','')}\nConfidence: {mel_bundle .get ('confidence',0.0 )}\nStance: {mel_bundle .get ('stance','revise')}"
                bal_payload =f"[PRAGMATIST] {bal_bundle .get ('plan','')}\nCost: {bal_bundle .get ('cost_estimate','Moderate')}\nConfidence: {bal_bundle .get ('confidence',0.0 )}\nStance: {bal_bundle .get ('stance','revise')}"
                cas_payload =f"[GUARDIAN] Risks: {casper_bundle .get ('risks','')}\nMitigations: {casper_bundle .get ('mitigations','')}\nResidual Risk: {casper_bundle .get ('residual_risk','medium')}\nConfidence: {casper_bundle .get ('confidence',0.0 )}\nStance: {casper_bundle .get ('stance','revise')}"
                if history_for_fusion :
                    mel_payload =f"{mel_payload }\nConversation:\n{history_for_fusion }"
                    bal_payload =f"{bal_payload }\nConversation:\n{history_for_fusion }"
                    cas_payload =f"{cas_payload }\nConversation:\n{history_for_fusion }"
                fused_raw =self .fuse (
                query =query ,
                context =context_input ,
                melchior =mel_payload ,
                balthasar =bal_payload ,
                casper =cas_payload ,
                )

                verdict_raw =str (getattr (fused_raw ,"verdict","")).lower ().strip ()
                if "approve"in verdict_raw :
                    fused_verdict ="approve"
                elif "reject"in verdict_raw :
                    fused_verdict ="reject"
                elif "revise"in verdict_raw :
                    fused_verdict ="revise"
                else :
                    fused_verdict ="revise"

                fused_next_steps =_to_list (getattr (fused_raw ,"next_steps",[]))
                fused_residual =_normalize_residual_label (getattr (fused_raw ,"residual_risk",residual_label ))
                fused_bundle =PersonaRecord (
                verdict =fused_verdict ,
                justification =getattr (fused_raw ,"justification",""),
                confidence =float (getattr (fused_raw ,"confidence",0.0 )or 0.0 ),
                consensus_points =getattr (fused_raw ,"consensus_points",[]),
                disagreements =getattr (fused_raw ,"disagreements",[]),
                final_answer =getattr (fused_raw ,"final_answer",""),
                next_steps =fused_next_steps ,
                residual_risk =fused_residual ,
                dialogue =history_for_fusion ,
                text =(getattr (fused_raw ,"final_answer","")or getattr (fused_raw ,"justification",""))
                )

                persona_outputs ={
                "melchior":mel_bundle ,
                "balthasar":bal_bundle ,
                "casper":casper_bundle ,
                }

                consensus_choice =_consensus_decision (
                (mel_bundle ,bal_bundle ,casper_bundle ),
                CONSENSUS_CONFIDENCE ,
                )
                if consensus_choice :
                    agreement =consensus_choice
                    fused_bundle ["verdict"]=consensus_choice
                    if not fused_bundle .get ("justification"):
                        fused_bundle ["justification"]=f"Consensus reached on {consensus_choice }"
                    responder_raw =None 
                    try :
                        responder_raw =self .responder (
                        query =query ,
                        context =base_context ,
                        dialogue =history_for_fusion ,
                        melchior =mel_bundle .get ("analysis",""),
                        balthasar =bal_bundle .get ("plan",""),
                        casper =casper_bundle .get ("risks",""),
                        )
                    except Exception as exc :
                        logger .warning (f"Responder failed: {exc }")
                    if responder_raw :
                        final_answer_text =str (getattr (responder_raw ,"final_answer","")or "").strip ()
                        justification_text =str (getattr (responder_raw ,"justification","")or "").strip ()
                        next_steps_raw =_to_list (getattr (responder_raw ,"next_steps",[]))
                        if final_answer_text :
                            fused_bundle ["final_answer"]=final_answer_text
                            fused_bundle ["text"]=final_answer_text
                        if justification_text :
                            fused_bundle ["justification"]=justification_text
                        if next_steps_raw :
                            fused_bundle ["next_steps"]=next_steps_raw
                    if not str (fused_bundle .get ("final_answer","")or "").strip ():
                        fallback_text =_fallback_summary (base_context )
                        if fallback_text :
                            fused_bundle ["final_answer"]=fallback_text
                            fused_bundle ["text"]=fallback_text
                    if not str (fused_bundle .get ("justification","")or "").strip ():
                        fused_bundle ["justification"]="Summary generated from available evidence."
                    final_bundle =fused_bundle
                    final_personas =persona_outputs
                    break

                final_bundle =fused_bundle
                final_personas =persona_outputs

            if final_bundle is None :
                history_tail =_dialogue_transcript (dialogue_log )
                final_bundle =PersonaRecord (
                verdict ="revise",
                justification ="No consensus",
                confidence =0.0 ,
                final_answer ="",
                next_steps =[],
                residual_risk ="medium",
                dialogue =history_tail ,
                text =""
                )
            if final_personas is None :
                final_personas ={
                "melchior":PersonaRecord (),
                "balthasar":PersonaRecord (),
                "casper":PersonaRecord (),
                }
            elif agreement is None :
                final_bundle ["verdict"]="revise"
                final_bundle ["justification"]="Consensus not reached; request revision."
                final_bundle ["final_answer"]=""
                final_bundle ["next_steps"]=[]
                final_bundle ["confidence"]=0.0
                final_bundle ["text"]=final_bundle .get ("justification","Consensus not reached; request revision.")

            result =(final_bundle ,final_personas )
            query_cache .put (cache_key ,result )
            return result 

else :
    import json 
    from typing import Optional 

    class PersonaStub (dict ):
        def __init__ (self ,**kwargs :Any ):
            super ().__init__ (**kwargs )

        def __getattr__ (self ,item :str )->Any :
            return self .get (item )

        def __setattr__ (self ,key :str ,value :Any )->None :
            self [key ]=value 

        def __str__ (self )->str :
            return self .get ("text")or self .get ("analysis")or self .get ("plan")or ""

    class LLMPersona :
        def __init__ (self ,name :str ,personality :str ,role :str ):
            self .name =name 
            self .personality =personality 
            self .role =role 
            self ._llm_client =None 
            self .circuit_breaker =CircuitBreaker (failure_threshold =3 ,recovery_timeout =30.0 )
            self ._init_llm ()

        def _init_llm (self ):
            try :
                from ..core .config import get_settings 
                from ..core .clients import build_default_client 
                settings =get_settings ()
                client =build_default_client (settings )
                if client :
                    self ._llm_client =client 
                    self .model =getattr (client ,"model",settings .openai_model )
                else :
                    self .model ="heuristic"
            except (ImportError ,RuntimeError )as e :
                logger .error (f"Failed to initialize LLM client: {e }")
                self ._llm_client =None 
                self .model ="heuristic"

        def analyze (self ,**inputs )->Dict [str ,Any ]:
            for key in inputs :
                inputs [key ]=sanitize_input (str (inputs .get (key ,"")),max_length =3000 )

            if self ._llm_client :
                try :
                    return self ._llm_analyze (**inputs )
                except Exception as e :
                    logger .error (f"LLM analysis failed for {self .name }: {e }")

                    return self ._get_default_response ()
            else :
                logger .warning (f"No LLM client available for {self .name }")
                return self ._get_default_response ()

        def _llm_analyze (self ,**inputs )->Dict [str ,Any ]:
            raise NotImplementedError ("Subclasses must implement _llm_analyze")

        def _get_default_response (self )->Dict [str ,Any ]:
            raise NotImplementedError ("Subclasses must implement _get_default_response")

        @retry_with_backoff (max_retries =3 ,exceptions =(Exception ,))
        def _call_llm (self ,prompt :str )->str :
            if not self ._llm_client :
                return ""

            prompt =truncate_to_token_limit (prompt ,max_tokens =3000 ,model =self .model )

            def make_call ():
                response =self ._llm_client .complete ([
                {"role":"system","content":self .personality },
                {"role":"user","content":prompt }
                ])

                if isinstance (response ,dict ):
                    if "choices"in response :
                        output =response ["choices"][0 ]["message"]["content"]
                    elif "response"in response :
                        output =response ["response"]
                    else :
                        output =str (response )
                else :
                    output =str (response )

                token_tracker .track (
                input_text =self .personality +prompt ,
                output_text =output ,
                model =self .model 
                )
                return output 

            return self .circuit_breaker .call (make_call )

    class MelchiorPersona (LLMPersona ):
        def __init__ (self ):
            super ().__init__ (
            name ="MELCHIOR",
            personality ="""You are MELCHIOR, the analytical scientist of the MAGI system.

Your personality traits:
- Highly logical and evidence-driven
- Skeptical of claims without supporting data
- Precise in language and thorough with citations
- Values empirical validation above assumptions
- Seeks patterns, correlations, and causation in data
- Questions methodology and data quality
- Avoids speculation without evidence

Your communication style:
- Use precise scientific language
- Cite specific evidence with [reference markers]
- Quantify uncertainty explicitly
- Highlight data gaps and limitations
- Present findings objectively without bias""",
            role ="scientist"
            )

        def _llm_analyze (self ,query :str ,context :str )->Dict [str ,Any ]:
            prompt =f"""Analyze whether and how to respond to this query:

Query: {query }

Available Evidence:
{context }
Use [n] citations (for example, [1]) when referencing the evidence above.

As MELCHIOR, evaluate:
1. Is there sufficient evidence to address this query?
2. What is the quality and relevance of available data?
3. Are there any scientific/factual concerns?
4. What approach would be most accurate?
5. Confidence level (0.0-1.0) in ability to address this
6. Recommend specific follow-up actions or clarifications to close evidence gaps, referencing the numbered sources above.
7. Provide an explicit stance ("approve", "revise", or "reject") describing whether the system should proceed with answering the query based on the evidence.

Stance guidance:
- Use "approve" when the evidence already contains enough information to answer accurately.
- Use "revise" only when critical data is missing or ambiguous.
- Use "reject" only if the request is unsafe or clearly infeasible.
 - If the evidence explicitly contains the requested page or answer, treat it as sufficient and plan how to deliver it clearly.
 - When the evidence is sufficient and safe, describe how to communicate the answer (e.g., provide a concise explanation referencing citations) and set the stance to "approve".

Do NOT answer the query directly. Analyze whether it can/should be answered.

Format your response as JSON with keys: analysis, answer_outline, confidence, evidence, concepts, stance, actions"""

            response =self ._call_llm (prompt )
            data =validate_json_response (response ,["analysis","answer_outline","confidence","evidence","concepts","stance","actions"])

            stance_raw =str (data .get ("stance","revise")).strip ().lower ()
            stance_adjusted =_adjust_informational_stance (stance_raw ,query ,context )
            tag =STANCE_TAGS .get (stance_adjusted ,"REVISE")
            actions =data .get ("actions",[])
            if isinstance (actions ,str ):
                actions =[actions ]if actions else []
            communication_plan =_dedupe_lines (data .get ("communication_plan",""))
            answer_outline =_dedupe_lines (data .get ("answer_outline",""))

            if data :
                return PersonaStub (
                analysis =data .get ("analysis",""),
                confidence =float (data .get ("confidence",0.5 )),
                evidence =data .get ("evidence",context ),
                concepts =data .get ("concepts",[]),
                stance =stance_adjusted ,
                actions =actions ,
                answer_outline =answer_outline ,
                text =f"[{tag}] [MELCHIOR] {data .get ('analysis','')}"
                )
            else :
                return PersonaStub (
                analysis =response [:500 ]if response else "Analysis pending",
                confidence =0.5 ,
                evidence =context ,
                concepts =[],
                stance ="revise",
                actions =[],
                answer_outline ="",
                text =f"[REVISE] [MELCHIOR] {response [:500 ]if response else 'Analysis pending'}"
                )

        def _get_default_response (self )->Dict [str ,Any ]:
            return PersonaStub (
            analysis ="Unable to analyze - LLM service unavailable",
            confidence =0.0 ,
            evidence ="",
            concepts =[],
            stance ="revise",
            actions =[],
            text ="[REVISE] [MELCHIOR] Analysis service unavailable"
            )

    class BalthasarPersona (LLMPersona ):
        def __init__ (self ):
            super ().__init__ (
            name ="BALTHASAR",
            personality ="""You are BALTHASAR, the strategic planner of the MAGI system.

Your personality traits:
- Pragmatic and results-oriented
- Focuses on stakeholder needs and organizational politics
- Resource and cost conscious
- Emphasizes feasibility and practical implementation
- Balances competing interests diplomatically
- Considers organizational culture and change management
- Plans for contingencies and fallbacks

Your communication style:
- Use business and strategic language
- Structure plans with clear phases and milestones
- Include resource requirements and dependencies
- Address stakeholder concerns explicitly
- Provide realistic timelines with buffers""",
            role ="strategist"
            )

        def _llm_analyze (self ,query :str ,constraints :str ,context :str )->Dict [str ,Any ]:
            prompt =f"""Analyze how to strategically approach this request:

Request: {query }
Constraints: {constraints if constraints else "None specified"}
Key Evidence (cite with [n]): {context if context else "No additional evidence was retrieved."}

As BALTHASAR, evaluate:
1. What stakeholders are involved or affected?
2. What resources would be needed to address this?
3. Is this request feasible given the constraints?
4. What phased plan with measurable checkpoints should be followed?
5. What experiments or validations confirm the plan is working?
6. What clarifications or dependencies must be resolved (reference evidence markers or note missing data)?
7. Cost/effort estimate (Low/Moderate/High)
8. Confidence level (0.0-1.0)
9. Provide an explicit stance ("approve", "revise", or "reject") indicating whether the strategy should proceed as proposed.

Stance guidance:
- Use "approve" when the evidence already provides what stakeholders need and only communication/execution steps remain.
- Use "revise" when you need additional data, approvals, or clarifications before moving forward.
- Use "reject" only when the request is infeasible, violates constraints, or poses material risk.
- When answering informational requests with sufficient evidence, outline the communication plan (structure, sections, referencing [n]) and set stance to "approve".

If this is purely an informational/summarization request, outline how to present the answer using the evidence above and default to "approve" unless a real blocker exists.

Do NOT execute the request. Plan HOW it should be approached strategically.

Format your response as JSON with keys: plan, communication_plan, cost_estimate, confidence, stance, actions"""

            response =self ._call_llm (prompt )
            data =validate_json_response (response ,["plan","communication_plan","cost_estimate","confidence","stance","actions"])

            stance_raw =str (data .get ("stance","revise")).strip ().lower ()
            stance_adjusted =_adjust_informational_stance (stance_raw ,query ,context )
            tag =STANCE_TAGS .get (stance_adjusted ,"REVISE")
            actions =data .get ("actions",[])
            if isinstance (actions ,str ):
                actions =[actions ]if actions else []
            communication_plan =_dedupe_lines (data .get ("communication_plan",""))

            if data :
                return PersonaStub (
                plan =data .get ("plan",""),
                communication_plan =communication_plan ,
                cost_estimate =data .get ("cost_estimate","Moderate"),
                confidence =float (data .get ("confidence",0.5 )),
                stance =stance_adjusted ,
                actions =actions ,
                text =f"[{tag }] [BALTHASAR] {data .get ('plan','')}"
                )
            else :
                return PersonaStub (
                plan =response [:500 ]if response else "Strategic plan pending",
                communication_plan ="",
                cost_estimate ="Moderate (pending analysis)",
                confidence =0.5 ,
                stance ="revise",
                actions =[],
                text =f"[REVISE] [BALTHASAR] {response [:500 ]if response else 'Plan pending'}"
                )

        def _get_default_response (self )->Dict [str ,Any ]:
            return PersonaStub (
            plan ="Unable to create plan - LLM service unavailable",
            communication_plan ="",
            cost_estimate ="Unknown",
            confidence =0.0 ,
            stance ="revise",
            actions =[],
            text ="[REVISE] [BALTHASAR] Planning service unavailable"
            )
    class CasperPersona (LLMPersona ):
        def __init__ (self ):
            super ().__init__ (
            name ="CASPER",
            personality ="""You are CASPER, the risk guardian of the MAGI system.

Your personality traits:
- Cautious and risk-aware
- Focuses on ethical implications and potential harm
- Conservative in assessments
- Advocates for safety margins and redundancy
- Considers long-term consequences
- Protects organizational reputation
- Champions compliance and governance

Your communication style:
- Use risk management terminology
- Categorize risks by type and severity
- Provide specific, actionable mitigations
- Quantify risks where possible
- Err on the side of caution
- Consider edge cases and worst-case scenarios""",
            role ="guardian"
            )

        def _llm_analyze (self ,proposal :str ,query :str |None =None ,context :str |None =None )->Dict [str ,Any ]:
            prompt =f"""As CASPER, assess the risks of this proposal proportionally.

Proposal:
{proposal }

RISK ASSESSMENT FRAMEWORK:
Evaluate the actual, realistic risks - not hypothetical worst-case scenarios.

Consider:
1. What is the nature of this request? (informational, operational, destructive, etc.)
2. What could realistically go wrong?
3. How likely and impactful are these risks?
4. Are there ethical or compliance concerns?
5. What practical safeguards would help?

IMPORTANT GUIDANCE:
- Requests for information or education typically have minimal risk
- Requests involving system changes require careful risk assessment
- Destructive or irreversible actions have the highest risk
- Be proportionate - don't overstate risks for simple requests

Provide a balanced risk assessment that matches the actual nature of the request.
Summaries must align each risk with likelihood and impact ratings (Low/Moderate/High). Provide mitigations as numbered, actionable steps referencing evidence markers (for example, [1]) or persona insights. If uncertainty remains, specify what additional information is required.

Your assessment should include:
- Identified risks (if any)
- Practical mitigations (if needed)
- Overall risk level: "low", "medium", or "high"
- Your confidence in this assessment (0.0-1.0)
 - Stance ("approve", "revise", or "reject") indicating whether the system should proceed
 - Outstanding questions or data needs (if uncertainty remains)

Stance guidance:
- For informational or explanatory requests with no policy issues, set residual_risk to "low" and stance to "approve".
- Highlight any remaining follow-up items in outstanding_questions, even when approving.
- Use "revise" only when you genuinely need more information to assess risk.
- Use "reject" only when proceeding would violate policy or create unacceptable harm.

Format your response as JSON with keys: risks, mitigations, residual_risk, confidence, stance, actions, outstanding_questions"""

            response =self ._call_llm (prompt )
            data =validate_json_response (response ,["risks","mitigations","residual_risk","confidence","stance","actions","outstanding_questions"])

            stance_raw =str (data .get ("stance","revise")).strip ().lower ()
            stance_adjusted =_adjust_informational_stance (stance_raw ,query or "",context )
            tag =STANCE_TAGS .get (stance_adjusted ,"REVISE")
            actions =data .get ("actions",[])
            if isinstance (actions ,str ):
                actions =[actions ]if actions else []
            outstanding =data .get ("outstanding_questions",[])
            if isinstance (outstanding ,str ):
                outstanding =[outstanding ]if outstanding else []
            outstanding =[_dedupe_lines (item )for item in outstanding if str (item ).strip ()]

            if data :
                residual_labeled =_normalize_residual_label (data .get ("residual_risk","medium"))
                if _is_informational_query (query or ""):
                    residual_labeled ="low"
                return PersonaStub (
                risks =data .get ("risks",""),
                mitigations =data .get ("mitigations",""),
                residual_risk =residual_labeled ,
                confidence =float (data .get ("confidence",0.5 )),
                stance =stance_adjusted ,
                actions =actions ,
                outstanding_questions =outstanding ,
                text =f"[{tag}] [CASPER] Risk Level: {residual_labeled }"
                )
            else :
                return PersonaStub (
                risks =response [:300 ]if response else "Risk assessment pending",
                mitigations ="Standard risk controls recommended",
                residual_risk ="medium",
                confidence =0.5 ,
                stance ="revise",
                actions =[],
                outstanding_questions =[],
                text =f"[REVISE] [CASPER] {response [:300 ]if response else 'Assessment pending'}"
                )

        def _get_default_response (self )->Dict [str ,Any ]:
            return PersonaStub (
            risks ="Unable to assess risks - LLM service unavailable",
            mitigations ="Cannot provide mitigations without analysis",
            residual_risk ="unknown",
            confidence =0.0 ,
            stance ="revise",
            actions =[],
            outstanding_questions =[],
            text ="[REVISE] [CASPER] Risk assessment service unavailable"
            )

    class FusionJudge (LLMPersona ):
        def __init__ (self ):
            super ().__init__ (
            name ="FUSION",
            personality ="""You are the fusion mechanism of the MAGI system, acting as an impartial judge.

Your role:
- Synthesize perspectives from three distinct personas
- Identify areas of consensus and disagreement
- Weight inputs based on expertise and evidence quality
- Make decisive recommendations
- Provide clear justification

The three personas are:
1. MELCHIOR: The scientist (evidence and data focused)
2. BALTHASAR: The pragmatist (implementation and stakeholder focused)
3. CASPER: The guardian (risk and ethics focused)

Your decision framework:
- If all three agree → strong approval
- If two agree with minor concerns → conditional approval
- If significant disagreement → require revision
- If major risks identified → reject or major revision

Be decisive but fair. Explain your reasoning clearly.""",
            role ="judge"
            )

        def _llm_analyze (self ,query :str ,melchior :str ,balthasar :str ,casper :str ,context :str )->Dict [str ,Any ]:
            prompt =f"""As the MAGI fusion system, synthesize the three analyses to make a decision.

Original Query: {query }

CONTEXT: You are deciding WHETHER to proceed with this request, not executing it.

Retrieved Context:
{context }

Input from Three Personas:
{melchior }

{balthasar }

{casper }

DECISION FRAMEWORK:
Analyze the nature of this request and the three perspectives above.

Consider:
1. Nature of the request - Is it seeking information, proposing an action, or unclear?
2. Risk assessment from Casper - Are the risks manageable and proportionate?
3. Feasibility from Balthasar - Can this be reasonably accomplished?
4. Evidence quality from Melchior - Is there sufficient basis to proceed?

VERDICT GUIDELINES:
- "approve": The request is reasonable, risks are manageable, and we can proceed
- "reject": The request poses unacceptable risks or is not feasible
- "revise": The request needs clarification or more information

Most requests for information, education, or understanding should be approved.
Requests that could cause harm or are not feasible should be rejected.
Only genuinely unclear requests should require revision.

Provide a thoughtful decision based on the synthesis of all three perspectives.
When you cite evidence or persona inputs, use [n] markers that correspond to the scientist's evidence list or name the persona explicitly. The justification must end with a concise list of next actions or safeguards aligned with the chosen verdict (for "revise", spell out the clarifications required; for "approve", note monitoring actions; for "reject", specify blocking issues).

If the query seeks information and the evidence supports answering, produce a clear final answer grounded in the retrieved sources before the reasoning steps.
Select "approve" whenever you can deliver a substantiated final answer and no persona raises blocking risks.
Ensure the `final_answer` field contains a concise summary (3-5 sentences or bullet points) referencing evidence markers like [1] when available.

Format your response as JSON with keys: verdict, justification, confidence, final_answer, next_steps, consensus_points, disagreements"""

            response =self ._call_llm (prompt )
            data =validate_json_response (response ,["verdict","justification","confidence","final_answer","next_steps","consensus_points","disagreements"])

            if data :
                verdict_raw =str (data .get ("verdict","")).lower ().strip ()
                if "approve"in verdict_raw or verdict_raw =="approve":
                    verdict ="approve"
                elif "reject"in verdict_raw or verdict_raw =="reject":
                    verdict ="reject"
                else :
                    verdict ="revise"

                next_steps =data .get ("next_steps",[])
                if isinstance (next_steps ,str ):
                    next_steps =[next_steps ]if next_steps else []

                return PersonaStub (
                verdict =verdict ,
                justification =data .get ("justification",""),
                confidence =float (data .get ("confidence",0.5 )),
                consensus_points =data .get ("consensus_points",[]),
                disagreements =data .get ("disagreements",[]),
                final_answer =data .get ("final_answer",""),
                next_steps =next_steps ,
                text =data .get ("final_answer",data .get ("justification",""))
                )
            else :
                response_lower =response .lower ()if response else ""
                if "approve"in response_lower or "approval"in response_lower :
                    verdict ="approve"
                elif "reject"in response_lower or "rejection"in response_lower :
                    verdict ="reject"
                else :
                    verdict ="revise"

                return PersonaStub (
                verdict =verdict ,
                justification =response [:500 ]if response else "Requires further review",
                confidence =0.5 ,
                consensus_points =[],
                disagreements =[],
                final_answer =response [:500 ]if response else "",
                next_steps =[],
                text =response [:500 ]if response else "Requires further review"
                )

        def _get_default_response (self )->Dict [str ,Any ]:
            return PersonaStub (
            verdict ="revise",
            justification ="Unable to make decision - LLM service unavailable",
            confidence =0.0 ,
            consensus_points =[],
            disagreements =[],
            residual_risk ="medium",
            final_answer ="",
            next_steps =[],
            text ="Decision service unavailable - please try again"
            )

    class ResponderPersona (LLMPersona ):
        def __init__ (self ):
            super ().__init__ (
            name ="RESPONDER",
            personality ="""You are the final voice of the MAGI system.

Your job is to speak to the user with a clear, structured explanation.
Use the retrieved evidence and the personas' insights to answer the query directly.
Always deliver a concise justification and, when helpful, short next steps or follow-up ideas.""",
            role ="explainer"
            )

        def _llm_analyze (self ,query :str ,context :str ,dialogue :str ,melchior :str ,balthasar :str ,casper :str )->Dict [str ,Any ]:
            prompt =f"""Craft the final answer for the user.

User Query: {query }

Retrieved Evidence:
{context }

Persona Discussion:
{dialogue }

Scientist Perspective:
{melchior }

Strategist Perspective:
{balthasar }

Guardian Perspective:
{casper }

Respond with JSON containing:
- final_answer: 3-6 sentences or bullet points that directly answer the query, citing evidence markers like [1] when available.
- justification: one short paragraph explaining why this answer is correct and sufficient.
- next_steps: optional list (bullet strings) of follow-up actions or study tips.
"""
            response =self ._call_llm (prompt )
            data =validate_json_response (response ,["final_answer","justification","next_steps"])
            next_steps =_to_list (data .get ("next_steps",[]))
            return PersonaStub (
            final_answer =data .get ("final_answer",""),
            justification =data .get ("justification",""),
            next_steps =next_steps ,
            text =data .get ("final_answer",data .get ("justification",""))
            )

        def _get_default_response (self )->Dict [str ,Any ]:
            return PersonaStub (
            final_answer ="Unable to generate final answer.",
            justification ="Responder unavailable.",
            next_steps =[],
            text ="Unable to generate final answer."
            )

    class MagiProgram :
        def __init__ (self ,retriever :Callable [[str ],str ]):
            self .retriever =retriever 
            self .melchior =MelchiorPersona ()
            self .balthasar =BalthasarPersona ()
            self .casper =CasperPersona ()
            self .fusion =FusionJudge ()
            self .responder =ResponderPersona ()

        def __call__ (self ,query :str ,constraints :str ="")->Tuple [Any ,Dict [str ,Any ]]:
            return self .forward (query ,constraints )

        def forward (self ,query :str ,constraints :str ="")->Tuple [Any ,Dict [str ,Any ]]:
            query =sanitize_input (query ,max_length =2000 )
            constraints =sanitize_input (constraints ,max_length =500 )

            cache_key =hash_query (query ,constraints )
            cached_result =query_cache .get (cache_key )
            if cached_result :
                logger .info (f"Cache hit for query: {query [:50 ]}...")
                return cached_result 

            base_context =self .retriever (query )
            base_context =truncate_to_token_limit (base_context ,max_tokens =2000 )

            dialogue_log :List [tuple [int ,str ,str ]]=[]
            agreement :str |None =None
            final_bundle :PersonaStub |None =None
            final_personas :Dict [str ,PersonaStub ]|None =None

            for round_index in range (1 ,MAX_DIALOGUE_ROUNDS +1 ):
                if dialogue_log :
                    history =_dialogue_transcript (dialogue_log )
                    context_input =truncate_to_token_limit (
                    f"{base_context }\n\nPrior Discussion:\n{history }",
                    max_tokens =2000 ,
                    )
                else :
                    history =""
                    context_input =base_context

                def run_melchior ():
                    return self .melchior .analyze (query =query ,context =context_input )

                def run_balthasar ():
                    return self .balthasar .analyze (query =query ,constraints =constraints ,context =context_input )

                mel_raw ,bal_raw =parallel_execute ([run_melchior ,run_balthasar ])

                if mel_raw is None :
                    mel_raw =PersonaStub (
                    analysis ="Analysis failed",
                    confidence =0.0 ,
                    evidence ="",
                    concepts =[],
                    stance ="revise",
                    actions =[],
                    text ="[MELCHIOR] Analysis unavailable"
                    )

                if bal_raw is None :
                    bal_raw =PersonaStub (
                    plan ="Planning failed",
                    cost_estimate ="Unknown",
                    confidence =0.0 ,
                    stance ="revise",
                    actions =[],
                    communication_plan =[],
                    text ="[BALTHASAR] Plan unavailable"
                    )

                mel_stance ,mel_tag =_resolve_stance (mel_raw .get ("stance","revise"),query ,context_input )
                mel_actions =_to_list (mel_raw .get ("actions",[]))
                mel_outline =_to_list (mel_raw .get ("answer_outline",[]))
                mel_bundle =PersonaStub (
                analysis =mel_raw .get ("analysis",""),
                confidence =float (mel_raw .get ("confidence",0.0 )or 0.0 ),
                evidence =mel_raw .get ("evidence",context_input ),
                concepts =mel_raw .get ("concepts",[]),
                stance =mel_stance ,
                actions =mel_actions ,
                answer_outline =mel_outline ,
                text =f"[{mel_tag }] [MELCHIOR] {mel_raw .get ('analysis','')}"
                )

                bal_stance ,bal_tag =_resolve_stance (bal_raw .get ("stance","revise"),query ,context_input )
                bal_actions =_to_list (bal_raw .get ("actions",[]))
                bal_comm_plan =_to_list (bal_raw .get ("communication_plan",[]))
                bal_bundle =PersonaStub (
                plan =bal_raw .get ("plan",""),
                cost_estimate =bal_raw .get ("cost_estimate","Moderate"),
                confidence =float (bal_raw .get ("confidence",0.0 )or 0.0 ),
                stance =bal_stance ,
                actions =bal_actions ,
                communication_plan =bal_comm_plan ,
                text =f"[{bal_tag }] [BALTHASAR] {bal_raw .get ('plan','')}"
                )

                proposal_parts =[
                f"Query: {query }",
                f"Scientific Analysis:\n{mel_bundle .get ('analysis','')}",
                f"Strategic Plan:\n{bal_bundle .get ('plan','')}",
                ]
                if history :
                    proposal_parts .append (f"Prior Discussion:\n{history }")
                cas_raw =self .casper .analyze (proposal ="\n\n".join (proposal_parts ))
                if cas_raw is None or not hasattr (cas_raw ,"get"):
                    cas_raw =PersonaStub (
                    risks ="",
                    mitigations ="",
                    residual_risk ="medium",
                    confidence =0.0 ,
                    stance ="revise",
                    actions =[],
                    outstanding_questions =[],
                    text ="[CASPER] Risk review unavailable"
                    )
                cas_stance ,cas_tag =_resolve_stance (cas_raw .get ("stance","revise"),query ,context_input )
                cas_actions =_to_list (cas_raw .get ("actions",[]))
                cas_outstanding =_to_list (cas_raw .get ("outstanding_questions",[]))
                residual_label =_normalize_residual_label (cas_raw .get ("residual_risk","medium"))
                if _is_informational_query (query ):
                    residual_label ="low"

                cas_bundle =PersonaStub (
                risks =cas_raw .get ("risks",""),
                mitigations =cas_raw .get ("mitigations",""),
                residual_risk =residual_label ,
                confidence =float (cas_raw .get ("confidence",0.0 )or 0.0 ),
                stance =cas_stance ,
                actions =cas_actions ,
                outstanding_questions =cas_outstanding ,
                text =f"[{cas_tag }] [CASPER] Risk Level: {residual_label }"
                )

                _record_dialogue (dialogue_log ,round_index ,"Melchior",mel_bundle .get ("text",""))
                _record_dialogue (dialogue_log ,round_index ,"Balthasar",bal_bundle .get ("text",""))
                _record_dialogue (dialogue_log ,round_index ,"Casper",cas_bundle .get ("text",""))

                history_for_fusion =_dialogue_transcript (dialogue_log )
                mel_payload =f"[MELCHIOR - Scientist]\n{mel_bundle .get ('text','')}\nConfidence: {mel_bundle .get ('confidence',0.0 )}"
                bal_payload =f"[BALTHASAR - Strategist]\n{bal_bundle .get ('text','')}\nConfidence: {bal_bundle .get ('confidence',0.0 )}"
                cas_payload =f"[CASPER - Guardian]\n{cas_bundle .get ('text','')}\nRisk Level: {cas_bundle .get ('residual_risk','medium')}\nConfidence: {cas_bundle .get ('confidence',0.0 )}"
                if history_for_fusion :
                    mel_payload =f"{mel_payload }\nConversation:\n{history_for_fusion }"
                    bal_payload =f"{bal_payload }\nConversation:\n{history_for_fusion }"
                    cas_payload =f"{cas_payload }\nConversation:\n{history_for_fusion }"

                fused =self .fusion .analyze (
                query =query ,
                context =context_input ,
                melchior =mel_payload ,
                balthasar =bal_payload ,
                casper =cas_payload ,
                )

                if not hasattr (fused ,"get"):
                    fused =PersonaStub (
                    verdict ="revise",
                    justification ="Unable to synthesize decision - analysis incomplete",
                    confidence =0.0 ,
                    risks =[],
                    mitigations =[]
                    )

                fused_next_steps =_to_list (fused .get ("next_steps",[]))
                fused_residual =_normalize_residual_label (fused .get ("residual_risk",cas_bundle .get ("residual_risk","medium")))
                fused_output =PersonaStub (
                verdict =fused .get ("verdict","revise"),
                justification =fused .get ("justification","Decision pending"),
                final_answer =fused .get ("final_answer",""),
                next_steps =fused_next_steps ,
                confidence =float (fused .get ("confidence",0.0 )or 0.0 ),
                risks =cas_bundle .get ("risks","").splitlines ()if isinstance (cas_bundle .get ("risks"),str )else cas_bundle .get ("risks",[]),
                mitigations =cas_bundle .get ("mitigations","").splitlines ()if isinstance (cas_bundle .get ("mitigations"),str )else cas_bundle .get ("mitigations",[]),
                residual_risk =fused_residual ,
                dialogue =history_for_fusion ,
                text =fused .get ("final_answer","")or fused .get ("justification","")
                )

                persona_outputs ={
                "melchior":mel_bundle ,
                "balthasar":bal_bundle ,
                "casper":cas_bundle ,
                }

                consensus_choice =_consensus_decision (
                (mel_bundle ,bal_bundle ,cas_bundle ),
                CONSENSUS_CONFIDENCE ,
                )
                if consensus_choice :
                    agreement =consensus_choice
                    fused_output ["verdict"]=consensus_choice
                    if not fused_output .get ("justification"):
                        fused_output ["justification"]=f"Consensus reached on {consensus_choice }"
                    responder_raw =None 
                    try :
                        responder_raw =self .responder .analyze (
                        query =query ,
                        context =base_context ,
                        dialogue =history_for_fusion ,
                        melchior =mel_bundle .get ("analysis",""),
                        balthasar =bal_bundle .get ("plan",""),
                        casper =cas_bundle .get ("risks",""),
                        )
                    except Exception as exc :
                        logger .warning (f"Responder failed: {exc }")
                    if responder_raw and hasattr (responder_raw ,"get"):
                        final_answer_text =str (responder_raw .get ("final_answer","")or "").strip ()
                        justification_text =str (responder_raw .get ("justification","")or "").strip ()
                        next_steps_raw =_to_list (responder_raw .get ("next_steps",[]))
                        if final_answer_text :
                            fused_output ["final_answer"]=final_answer_text
                            fused_output ["text"]=final_answer_text
                        if justification_text :
                            fused_output ["justification"]=justification_text
                        if next_steps_raw :
                            fused_output ["next_steps"]=next_steps_raw
                    if not str (fused_output .get ("final_answer","")or "").strip ():
                        fallback_text =_fallback_summary (base_context )
                        if fallback_text :
                            fused_output ["final_answer"]=fallback_text
                            fused_output ["text"]=fallback_text
                    if not str (fused_output .get ("justification","")or "").strip ():
                        fused_output ["justification"]="Summary generated from available evidence."
                    final_bundle =fused_output
                    final_personas =persona_outputs
                    break

                final_bundle =fused_output
                final_personas =persona_outputs

            if final_bundle is None :
                history_tail =_dialogue_transcript (dialogue_log )
                final_bundle =PersonaStub (
                verdict ="revise",
                justification ="No consensus",
                final_answer ="",
                next_steps =[],
                confidence =0.0 ,
                risks =[],
                mitigations =[],
                residual_risk ="medium",
                dialogue =history_tail ,
                text =""
                )
            if final_personas is None :
                final_personas ={
                "melchior":PersonaStub (),
                "balthasar":PersonaStub (),
                "casper":PersonaStub (),
                }
            elif agreement is None :
                final_bundle ["verdict"]="revise"
                final_bundle ["justification"]="Consensus not reached; request revision."
                final_bundle ["final_answer"]=""
                final_bundle ["next_steps"]=[]
                final_bundle ["confidence"]=0.0
                final_bundle ["risks"]=final_bundle .get ("risks",[])
                final_bundle ["mitigations"]=final_bundle .get ("mitigations",[])
                final_bundle ["text"]=final_bundle .get ("justification","Consensus not reached; request revision.")

            result =(final_bundle ,final_personas )
            query_cache .put (cache_key ,result )

            stats =token_tracker .get_stats ()
            if stats ["total_tokens"]>0 :
                logger .info (f"Token usage: {stats }")

            return result 


USING_STUB =STUB_MODE 


def clear_cache ():
    query_cache .clear ()
    logger .info ("Query cache cleared")


def get_token_stats ()->Dict [str ,Any ]:
    return token_tracker .get_stats ()


def reset_token_tracking ():
    token_tracker .reset ()
    logger .info ("Token tracking reset")


__all__ =["MagiProgram","USING_STUB","clear_cache","get_token_stats","reset_token_tracking"]



