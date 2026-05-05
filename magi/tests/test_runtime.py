from __future__ import annotations

import json

import pytest

from magi.core.clients import LLMClient, LLMClientError
from magi.core.config import Settings
from magi.core.embeddings import HashingEmbedder
from magi.core.rag import RagRetriever
from magi.core.vectorstore import InMemoryVectorStore, VectorEntry
from magi.dspy_programs import runtime
from magi.dspy_programs.runtime import (
    MagiProgram,
    _StructuredRunner,
    _evidence_directly_addresses_query,
    _json_schema,
    _normalize_balthasar_response,
    _normalize_casper_response,
    _normalize_melchior_response,
    _normalize_persona_stance,
    _normalize_responder_response,
)
from magi.dspy_programs.schemas import (
    BalthasarResponse,
    CasperResponse,
    FusionResponse,
    MelchiorResponse,
    ResponderResponse,
    RetrievedEvidence,
)
from magi.eval.scenario_harness import ScenarioEvidence, ScenarioRetriever


class _MalformedClient(LLMClient):
    def complete(self, messages, *, tools=None, response_format=None):  # type: ignore[override]
        return {"choices": [{"message": {"content": '{"analysis": "ok"}'}}]}


class _ValidClientWithoutText(LLMClient):
    def complete(self, messages, *, tools=None, response_format=None):  # type: ignore[override]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "analysis": "Grounded answer.",
                                "answer_outline": ["Lead with the evidence."],
                                "confidence": 0.8,
                                "evidence_quotes": ['[1] "Source excerpt"'],
                                "stance": "approve",
                                "actions": ["Answer carefully."],
                            }
                        )
                    }
                }
            ]
        }


class _NoopClient(LLMClient):
    model = "gemini-default"

    def complete(self, messages, *, tools=None, response_format=None):  # type: ignore[override]
        return {"choices": [{"message": {"content": "{}"}}]}


class _SequenceClient(LLMClient):
    model = "gpt-4o-mini-2024-07-18"

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self._payloads = list(payloads)
        self.messages: list[list[dict[str, object]]] = []

    def complete(self, messages, *, tools=None, response_format=None):  # type: ignore[override]
        assert messages
        assert response_format is not None
        self.messages.append(list(messages))
        if not self._payloads:
            raise AssertionError("unexpected extra client call")
        payload = self._payloads.pop(0)
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def test_json_schema_excludes_defaulted_fields_for_strict_mode():
    response_format = _json_schema("melchior_response", MelchiorResponse)
    schema = response_format["json_schema"]["schema"]

    assert "text" not in schema["properties"]
    assert "text" not in schema["required"]
    assert set(schema["required"]) == set(schema["properties"])


def test_structured_runner_rejects_partial_json():
    runner = _StructuredRunner(_MalformedClient(), "gpt-4o-mini-2024-07-18")
    with pytest.raises(LLMClientError, match="schema validation"):
        runner.run(
            system_prompt="Return JSON",
            user_prompt="Return JSON",
            schema_name="melchior_response",
            schema_cls=MelchiorResponse,
        )


def test_structured_runner_accepts_missing_defaulted_fields():
    runner = _StructuredRunner(_ValidClientWithoutText(), "gpt-4o-mini-2024-07-18")

    response = runner.run(
        system_prompt="Return JSON",
        user_prompt="Return JSON",
        schema_name="melchior_response",
        schema_cls=MelchiorResponse,
    )

    assert response.analysis == "Grounded answer."
    assert response.text == ""


def test_magi_program_passes_no_openai_default_to_google_only_client(monkeypatch):
    settings = Settings(
        openai_api_key="",
        google_api_key="google-key",
        openai_model="openai-default",
        gemini_model="gemini-default",
    )
    captured: dict[str, object] = {}

    def fake_build_default_client(
        settings_arg: Settings, *, model: str | None = None
    ) -> LLMClient:
        captured["model"] = model
        captured["settings"] = settings_arg
        return _NoopClient()

    monkeypatch.setattr(runtime, "get_settings", lambda: settings)
    monkeypatch.setattr(runtime, "build_default_client", fake_build_default_client)

    program = MagiProgram(retriever=lambda query, **kwargs: "", force_stub=False)

    assert captured["model"] is None
    assert program.model_name == "gemini-default"
    assert program.effective_mode == "live"


def test_live_prompts_omit_routing_debug_signals():
    client = _SequenceClient(
        [
            {
                "analysis": "The evidence supports a guarded pilot.",
                "answer_outline": ["Lead with [1]."],
                "confidence": 0.8,
                "evidence_quotes": ['[1] "The pilot proposal keeps human review."'],
                "stance": "approve",
                "actions": ["Cite [1]."],
            },
            {
                "plan": "Answer with the cited controlled plan.",
                "communication_plan": ["Cite [1]."],
                "cost_estimate": "low",
                "confidence": 0.8,
                "stance": "approve",
                "actions": ["Keep human review."],
            },
            {
                "risks": ["Risk is limited when human review stays in place."],
                "mitigations": ["Keep human review."],
                "residual_risk": "medium",
                "confidence": 0.8,
                "stance": "approve",
                "actions": ["Monitor the pilot."],
                "outstanding_questions": [],
            },
            {
                "verdict": "approve",
                "justification": "The answer is grounded in [1].",
                "confidence": 0.8,
                "final_answer": (
                    "Approve within the cited limits and controls: [1] The pilot "
                    "proposal scopes MAGI to internal triage and keeps human review."
                ),
                "next_steps": ["Keep human review."],
                "consensus_points": ["Cited evidence supports the answer."],
                "disagreements": [],
                "residual_risk": "medium",
                "risks": ["Over-claiming."],
                "mitigations": ["Stay within [1]."],
            },
        ]
    )
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="pilot_brief",
                    text=(
                        "The pilot proposal scopes MAGI to internal triage and keeps "
                        "human review on every decision."
                    ),
                )
            ]
        ),
        force_stub=False,
        client=client,
        enable_live_personas=True,
    )

    program(
        "Should we pilot MAGI next month?",
        constraints="Keep human review in the loop.",
    )

    prompt_text = "\n".join(
        str(message.get("content", ""))
        for call_messages in client.messages
        for message in call_messages
    ).lower()
    assert "routing signals" not in prompt_text
    assert "route scores" not in prompt_text
    assert "scores:" not in prompt_text
    assert "decision markers" not in prompt_text
    assert "explicit constraints were supplied" not in prompt_text


def test_supports_llm_when_google_key_is_configured(monkeypatch):
    settings = Settings(openai_api_key="", google_api_key="google-key")

    monkeypatch.setattr(runtime, "get_settings", lambda: settings)
    monkeypatch.setattr(runtime, "_FORCE_STUB", False)

    assert runtime._supports_llm()


def test_magi_program_cache_invalidates_when_store_changes():
    runtime.clear_cache()
    embedder = HashingEmbedder(dimension=32)
    store = InMemoryVectorStore(dim=32)
    retriever = RagRetriever(embedder, store)
    program = MagiProgram(retriever=retriever, force_stub=True)

    first, _ = program("What does page 1 say?", constraints="")
    store.add(
        [
            VectorEntry(
                document_id="doc#page-1",
                embedding=embedder("[Page 1] MAGI now has supporting evidence."),
                text="[Page 1] MAGI now has supporting evidence.",
                metadata={"source": "doc"},
            )
        ]
    )
    second, _ = program("What does page 1 say?", constraints="")

    assert first.verdict == "revise"
    assert second.verdict == "approve"
    assert "supporting evidence" in second.final_answer


def test_magi_program_records_cache_hit_metadata():
    runtime.clear_cache()
    program = MagiProgram(retriever=lambda query, **kwargs: "", force_stub=True)

    program("Summarize MAGI", constraints="")
    first_run = dict(program.last_run_metadata)
    program("Summarize MAGI", constraints="")
    second_run = dict(program.last_run_metadata)

    assert first_run["cache_hit"] is False
    assert second_run["cache_hit"] is True


def test_magi_program_cache_isolates_retriever_configuration():
    runtime.clear_cache()
    embedder = HashingEmbedder(dimension=32)
    store = InMemoryVectorStore(dim=32)
    store.add(
        [
            VectorEntry(
                document_id="alpha",
                embedding=embedder("MAGI alpha evidence supports a guarded pilot."),
                text="MAGI alpha evidence supports a guarded pilot.",
                metadata={"source": "alpha", "team": "alpha"},
            ),
            VectorEntry(
                document_id="beta",
                embedding=embedder("MAGI beta evidence says the SLA is missing."),
                text="MAGI beta evidence says the SLA is missing.",
                metadata={"source": "beta", "team": "beta"},
            ),
        ]
    )

    alpha_program = MagiProgram(
        retriever=RagRetriever(
            embedder,
            store,
            default_metadata_filters={"team": "alpha"},
        ),
        force_stub=True,
    )
    beta_program = MagiProgram(
        retriever=RagRetriever(
            embedder,
            store,
            default_metadata_filters={"team": "beta"},
        ),
        force_stub=True,
    )

    alpha, _ = alpha_program("Summarize MAGI evidence.", constraints="")
    beta, _ = beta_program("Summarize MAGI evidence.", constraints="")

    assert "alpha evidence" in alpha.final_answer.lower()
    assert "beta evidence" in beta.final_answer.lower()
    assert beta_program.last_run_metadata["cache_hit"] is False


def test_magi_program_cache_isolates_stub_and_live_modes():
    runtime.clear_cache()
    query = "What is MAGI live evidence?"

    def retriever(_query, **_kwargs):
        return "MAGI live evidence is grounded."

    stub_program = MagiProgram(retriever=retriever, force_stub=True)
    stub, _ = stub_program(query, constraints="")

    client = _SequenceClient(
        [
            {
                "analysis": "MAGI live evidence is grounded. [1]",
                "answer_outline": ["Use [1]."],
                "confidence": 0.9,
                "evidence_quotes": ['[1] "MAGI live evidence is grounded."'],
                "stance": "approve",
                "actions": ["Answer with [1]."],
            },
            {
                "plan": "Answer from MAGI live evidence. [1]",
                "communication_plan": ["Cite [1]."],
                "cost_estimate": "low",
                "confidence": 0.9,
                "stance": "approve",
                "actions": ["Cite [1]."],
            },
            {
                "risks": ["Low risk of over-claiming."],
                "mitigations": ["Stay within [1]."],
                "residual_risk": "low",
                "confidence": 0.9,
                "stance": "approve",
                "actions": ["Stay within [1]."],
                "outstanding_questions": [],
            },
            {
                "verdict": "approve",
                "justification": "MAGI live evidence is grounded. [1]",
                "confidence": 0.9,
                "final_answer": "MAGI live evidence is grounded. [1]",
                "next_steps": ["Use the cited answer."],
                "consensus_points": ["Grounded in [1]."],
                "disagreements": [],
                "residual_risk": "low",
                "risks": ["Low risk of over-claiming."],
                "mitigations": ["Stay within [1]."],
            },
            {
                "final_answer": "MAGI live evidence is grounded. [1]",
                "justification": "MAGI live evidence is grounded. [1]",
                "next_steps": ["Use the cited answer."],
            },
        ]
    )
    live_program = MagiProgram(
        retriever=retriever,
        force_stub=False,
        client=client,
        enable_live_personas=True,
    )
    live, _ = live_program(query, constraints="")

    assert stub_program.last_run_metadata["cache_hit"] is False
    assert live_program.last_run_metadata["cache_hit"] is False
    assert live.final_answer == "MAGI live evidence is grounded. [1]"
    assert live.final_answer != stub.final_answer


def test_magi_program_skips_live_responder_by_default():
    client = _SequenceClient(
        [
            {
                "analysis": "MAGI evidence is grounded. [1]",
                "answer_outline": ["Use [1]."],
                "confidence": 0.9,
                "evidence_quotes": ['[1] "MAGI evidence is grounded."'],
                "stance": "approve",
                "actions": ["Answer with [1]."],
            },
            {
                "plan": "Answer from MAGI evidence. [1]",
                "communication_plan": ["Cite [1]."],
                "cost_estimate": "low",
                "confidence": 0.9,
                "stance": "approve",
                "actions": ["Cite [1]."],
            },
            {
                "risks": ["Low risk of over-claiming."],
                "mitigations": ["Stay within [1]."],
                "residual_risk": "low",
                "confidence": 0.9,
                "stance": "approve",
                "actions": ["Stay within [1]."],
                "outstanding_questions": [],
            },
            {
                "verdict": "approve",
                "justification": "MAGI evidence is grounded. [1]",
                "confidence": 0.9,
                "final_answer": "MAGI evidence is grounded. [1]",
                "next_steps": ["Use the cited answer."],
                "consensus_points": ["Grounded in [1]."],
                "disagreements": [],
                "residual_risk": "low",
                "risks": ["Low risk of over-claiming."],
                "mitigations": ["Stay within [1]."],
            },
        ]
    )
    program = MagiProgram(
        retriever=lambda query, **kwargs: "MAGI evidence is grounded.",
        force_stub=False,
        client=client,
        enable_live_personas=True,
    )

    fused, _ = program("Summarize MAGI.", constraints="")

    assert fused.final_answer == "MAGI evidence is grounded. [1]"
    assert program.last_run_metadata["responder_mode"] == "deterministic"


def test_normalize_persona_stance_rewrites_evidence_gap_reject_to_revise():
    stance = _normalize_persona_stance(
        "What is MAGI's guaranteed p95 latency SLA?",
        "reject",
        "The current evidence does not specify MAGI's p95 latency SLA.",
    )

    assert stance == "revise"


def test_normalize_casper_response_relaxes_benign_informational_revision():
    response = CasperResponse(
        text="[REVISE] [CASPER] Risk level: medium. The evidence base may become outdated over time.",
        risks=["The evidence base may become outdated over time."],
        mitigations=["Refresh documents regularly."],
        residual_risk="medium",
        confidence=0.7,
        stance="revise",
        actions=["Refresh documents regularly."],
        outstanding_questions=[],
    )
    normalized = _normalize_casper_response(
        "Summarize MAGI in one sentence.",
        [
            RetrievedEvidence(
                citation="[1]", source="README", text="MAGI overview", score=1.0
            )
        ],
        [],
        response,
    )

    assert normalized.stance == "approve"
    assert normalized.residual_risk == "low"


def test_normalize_summary_synthesis_promotes_melchior_and_balthasar_to_approve():
    evidence = [
        RetrievedEvidence(
            citation="[1]", source="README", text="MAGI overview", score=1.0
        )
    ]
    melchior = MelchiorResponse(
        text="[REVISE] [MELCHIOR] The evidence does not provide a one-sentence summary.",
        analysis="The evidence does not provide a one-sentence summary, but a summary can be derived from the available material.",
        answer_outline=["Summarize the core purpose."],
        confidence=0.7,
        evidence_quotes=['[1] "MAGI overview"'],
        stance="revise",
        actions=["Derive a concise summary from the evidence."],
    )
    balthasar = BalthasarResponse(
        text="[REVISE] [BALTHASAR] Extract key features for a concise summary.",
        plan="Extract key features for a concise summary.",
        communication_plan=["Keep it concise."],
        cost_estimate="low",
        confidence=0.8,
        stance="revise",
        actions=["Summarize the core function."],
    )

    normalized_melchior = _normalize_melchior_response(
        "Summarize MAGI in one sentence.", evidence, melchior
    )
    normalized_balthasar = _normalize_balthasar_response(
        "Summarize MAGI in one sentence.", evidence, balthasar
    )

    assert normalized_melchior.stance == "approve"
    assert normalized_balthasar.stance == "approve"


def test_normalize_guardrailed_recommendation_promotes_revise_to_approve():
    evidence = [
        RetrievedEvidence(
            citation="[1]",
            source="brief",
            text=(
                "The pilot proposal scopes MAGI to internal policy triage with a "
                "human reviewer, weekly refreshes, and explicit rollout controls."
            ),
            score=1.0,
        )
    ]
    melchior = MelchiorResponse(
        text="[REVISE] [MELCHIOR] The proposal includes scope, budget, timeline, and controls.",
        analysis="The proposal includes scope, budget, timeline, controls, and human reviewer guardrails for a pilot.",
        answer_outline=["Approve a bounded pilot."],
        confidence=0.7,
        evidence_quotes=[
            '[1] "The pilot proposal scopes MAGI to internal policy triage with a human reviewer."'
        ],
        stance="revise",
        actions=["Recommend a bounded pilot with safeguards."],
    )
    casper = CasperResponse(
        text="[REVISE] [CASPER] Risk level: medium. Weekly refreshes and a human reviewer mitigate the main risks.",
        risks=["The evidence base can become outdated without weekly refreshes."],
        mitigations=["Keep a human reviewer in the loop.", "Refresh documents weekly."],
        residual_risk="medium",
        confidence=0.75,
        stance="revise",
        actions=["Keep a human reviewer in the loop.", "Refresh documents weekly."],
        outstanding_questions=[],
    )

    normalized_melchior = _normalize_melchior_response(
        "Should we pilot MAGI for internal policy triage next month?",
        evidence,
        melchior,
    )
    normalized_casper = _normalize_casper_response(
        "Should we pilot MAGI for internal policy triage next month?",
        evidence,
        [],
        casper,
    )

    assert normalized_melchior.stance == "approve"
    assert normalized_casper.stance == "approve"


def test_direct_support_uses_fuzzy_matching_for_nearby_wording():
    evidence = [
        RetrievedEvidence(
            citation="[1]",
            source="brief",
            text=(
                "The pilot proposal scopes MAGI to internal policy triage with a "
                "human reviewer, weekly refreshes, and explicit rollout controls."
            ),
            score=0.22,
        )
    ]

    assert _evidence_directly_addresses_query(
        "Should we pilot MAGI for internal policy triaging next month?",
        evidence,
    )


def test_normalize_responder_response_falls_back_when_answer_conflicts_with_approve():
    fusion = FusionResponse(
        text="MAGI is a multi-persona reasoning engine grounded in retrieved evidence.",
        verdict="approve",
        justification="The answer is fully grounded in the available evidence.",
        confidence=0.82,
        final_answer="MAGI is a multi-persona reasoning engine grounded in retrieved evidence.",
        next_steps=["Answer directly."],
        consensus_points=["The answer is grounded in retrieved evidence."],
        disagreements=[],
        residual_risk="medium",
        risks=["Operational drift is possible if documents are stale."],
        mitigations=["Refresh documents regularly."],
    )
    response = ResponderResponse(
        text="Revise the implementation plan before proceeding.",
        final_answer="Revise the implementation plan before proceeding.",
        justification="This needs more work first.",
        next_steps=["Revise the implementation plan."],
    )

    normalized = _normalize_responder_response(
        "Should we pilot MAGI for internal policy triage next month?",
        fusion,
        [
            RetrievedEvidence(
                citation="[1]",
                source="brief",
                text=(
                    "The pilot proposal scopes MAGI to internal policy triage with a "
                    "human reviewer, weekly refreshes, and explicit rollout controls."
                ),
                score=1.0,
            )
        ],
        response,
    )

    assert normalized.final_answer != response.final_answer
    assert "[1]" in normalized.final_answer
    assert "human reviewer" in normalized.final_answer


def test_live_program_falls_back_to_grounded_approve_answer_when_model_is_uncited():
    runtime.clear_cache()
    client = _SequenceClient(
        [
            {
                "analysis": "The evidence defines MAGI.",
                "answer_outline": ["Summarize MAGI."],
                "confidence": 0.8,
                "evidence_quotes": ['[1] "MAGI is a multi persona reasoning engine."'],
                "stance": "approve",
                "actions": ["Answer directly."],
            },
            {
                "plan": "Answer directly.",
                "communication_plan": ["Keep it short."],
                "cost_estimate": "low",
                "confidence": 0.8,
                "stance": "approve",
                "actions": ["Answer directly."],
            },
            {
                "risks": ["Low risk of over-claiming."],
                "mitigations": ["Stay within the evidence."],
                "residual_risk": "low",
                "confidence": 0.8,
                "stance": "approve",
                "actions": ["Stay within the evidence."],
                "outstanding_questions": [],
            },
            {
                "verdict": "approve",
                "justification": "This is enough to answer.",
                "confidence": 0.8,
                "final_answer": "MAGI means Modified Adjusted Gross Income.",
                "next_steps": ["Answer directly."],
                "consensus_points": ["Enough evidence."],
                "disagreements": [],
                "residual_risk": "low",
                "risks": ["Low risk of over-claiming."],
                "mitigations": ["Stay within the evidence."],
            },
            {
                "final_answer": "MAGI means Modified Adjusted Gross Income.",
                "justification": "This is enough to answer.",
                "next_steps": ["Answer directly."],
            },
        ]
    )

    program = MagiProgram(
        retriever=lambda query, **kwargs: "MAGI is a multi persona reasoning engine.",
        force_stub=False,
        client=client,
        enable_live_personas=True,
    )
    fused, _ = program("What is MAGI?", constraints="")

    assert fused.verdict == "approve"
    assert "[1]" in fused.final_answer
    assert "multi persona reasoning engine" in fused.final_answer.lower()
    assert "modified adjusted gross income" not in fused.final_answer.lower()


def test_human_like_summary_uses_relevant_evidence_not_distractor():
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="incident-review_notes",
                    text=(
                        "The incident review process collects customer impact, timeline, "
                        "root cause, mitigations, and follow-up owners before leadership sign-off."
                    ),
                ),
                ScenarioEvidence(
                    source="unrelated_calendar",
                    text="The team lunch calendar lists birthdays, office snacks, and optional Friday demos.",
                ),
            ]
        ),
        force_stub=True,
    )

    fused, _ = program(
        "Can you summarize the incident review notes in one concise sentence?",
        constraints="",
    )

    assert fused.verdict == "approve"
    assert "[1]" in fused.final_answer
    assert "birthday" not in fused.final_answer.lower()


def test_informational_answer_allows_relevant_calendar_source():
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="calendar",
                    text=(
                        "The release calendar says the launch readiness review is scheduled "
                        "for Thursday with operations and support owners."
                    ),
                )
            ]
        ),
        force_stub=True,
    )

    fused, personas = program(
        "What does the release calendar say about the readiness review?",
        constraints="",
    )

    assert fused.verdict == "approve"
    assert {persona.stance for persona in personas.values()} == {"approve"}
    assert "readiness review is scheduled" in fused.final_answer.lower()


def test_source_qualified_key_points_prefer_matching_source_label():
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="rollout_status",
                    text=(
                        "The rollout status is green. The release is approved and monitored by ops. "
                        "No production incidents were recorded during the latest review window."
                    ),
                ),
                ScenarioEvidence(
                    source="pilot_brief",
                    text=(
                        "The pilot proposal scopes MAGI to internal policy triage for four weeks. "
                        "Every decision keeps a human reviewer in the loop. Weekly evidence refreshes, "
                        "rollback criteria, and audit logs are required guardrails before launch."
                    ),
                ),
                ScenarioEvidence(
                    source="team_social",
                    text=(
                        "The team social agenda covers lunch, demos, office logistics, and a Friday photo booth."
                    ),
                ),
                ScenarioEvidence(
                    source="magi_overview",
                    text=(
                        "MAGI is a multi persona reasoning engine for assessing user requests against an "
                        "evidence base. It retrieves context from a vector store, convenes three specialized "
                        "personas, and returns a fused verdict with citations."
                    ),
                ),
            ]
        ),
        force_stub=True,
    )

    fused, _ = program(
        "Give me the key points from the MAGI overview.",
        constraints="",
    )

    assert fused.verdict == "approve"
    assert "multi persona reasoning engine" in fused.final_answer.lower()
    assert "pilot proposal" not in fused.final_answer.lower()


def test_guarded_decision_uses_control_evidence_and_skips_distractor():
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="proposal_brief",
                    text=(
                        "The pilot proposal scopes the work to internal policy triage for four weeks, "
                        "caps budget at 10k, and keeps a human reviewer on every decision."
                    ),
                ),
                ScenarioEvidence(
                    source="control_plan",
                    text=(
                        "Weekly evidence refreshes, rollback criteria, and audit logs are required "
                        "guardrails before launch."
                    ),
                ),
                ScenarioEvidence(
                    source="travel_policy",
                    text="The travel policy covers meal limits, hotel booking rules, and receipt deadlines.",
                ),
            ]
        ),
        force_stub=True,
    )

    fused, personas = program(
        "Should we move forward with the internal policy triage pilot next month?",
        constraints="Keep human review in the loop and stay under the approved budget.",
    )

    assert fused.verdict == "approve"
    assert {persona.stance for persona in personas.values()} == {"approve"}
    assert "travel policy" not in fused.final_answer.lower()
    assert "[1]" in fused.final_answer and "[3]" in fused.final_answer


def test_non_pilot_decision_approval_uses_generic_wording():
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="renewal_brief",
                    text=(
                        "The renewal proposal limits access to read-only data, assigns an owner, "
                        "keeps audit logging enabled, and includes rollback criteria."
                    ),
                )
            ]
        ),
        force_stub=True,
    )

    fused, _ = program(
        "Should we renew the vendor contract next month?",
        constraints="Keep audit logging enabled.",
    )

    assert fused.verdict == "approve"
    assert "renewal proposal" in fused.final_answer.lower()
    assert "bounded internal pilot" not in fused.final_answer.lower()
    assert "operational trial" not in fused.final_answer.lower()
    assert "pilot" not in fused.final_answer.lower()


def test_unsupported_detail_question_revises_instead_of_approving_related_evidence():
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="pilot_brief",
                    text=(
                        "The pilot proposal scopes MAGI to internal policy triage for four weeks. "
                        "Every decision keeps a human reviewer in the loop. Weekly evidence refreshes, "
                        "rollback criteria, and audit logs are required guardrails before launch."
                    ),
                ),
                ScenarioEvidence(
                    source="rollout_status",
                    text=(
                        "The rollout status is green. The release is approved and monitored by ops. "
                        "No production incidents were recorded during the latest review window."
                    ),
                ),
            ]
        ),
        force_stub=True,
    )

    fused, personas = program("What is MAGI's Q4 procurement budget?", constraints="")

    assert fused.verdict == "revise"
    assert {persona.stance for persona in personas.values()} == {"revise"}
    assert "not sufficient" in fused.final_answer.lower()
    assert "pilot proposal scopes" not in fused.final_answer.lower()


def test_yes_no_evidence_question_gets_fact_check_answer_not_decision_approval():
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="pilot_brief",
                    text=(
                        "The pilot proposal scopes MAGI to internal policy triage for four weeks. "
                        "Every decision keeps a human reviewer in the loop."
                    ),
                ),
                ScenarioEvidence(
                    source="rollout_status",
                    text=(
                        "The rollout status is green. The release is approved and monitored by ops. "
                        "No production incidents were recorded during the latest review window."
                    ),
                ),
            ]
        ),
        force_stub=True,
    )

    fused, _ = program(
        "Does the evidence say the rollout status is green?",
        constraints="",
    )

    assert fused.verdict == "approve"
    assert fused.final_answer.startswith("Yes,")
    assert "bounded internal pilot" not in fused.final_answer.lower()
    assert "rollout status is green" in fused.final_answer.lower()


def test_paraphrased_rollout_information_question_avoids_decision_approval():
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="rollout_status",
                    text=(
                        "The rollout status is green. The release is approved and monitored by ops. "
                        "No production incidents were recorded during the latest review window."
                    ),
                ),
                ScenarioEvidence(
                    source="pilot_brief",
                    text=(
                        "The pilot proposal scopes MAGI to internal policy triage for four weeks. "
                        "Every decision keeps a human reviewer in the loop."
                    ),
                ),
            ]
        ),
        force_stub=True,
    )

    fused, personas = program("How is the MAGI rollout going right now?", constraints="")

    assert fused.verdict == "approve"
    assert {persona.stance for persona in personas.values()} == {"approve"}
    assert "rollout status is green" in fused.final_answer.lower()
    assert "bounded internal pilot" not in fused.final_answer.lower()
    assert "pilot proposal" not in fused.final_answer.lower()


def test_negative_fact_check_preserves_claim_polarity():
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="rollout_status",
                    text=(
                        "The rollout status is green. The release is approved and monitored by ops. "
                        "No production incidents were recorded during the latest review window."
                    ),
                ),
            ]
        ),
        force_stub=True,
    )

    fused, _ = program(
        "Does the evidence say production incidents happened?",
        constraints="",
    )

    assert fused.verdict == "approve"
    assert fused.final_answer.startswith("No.")
    assert "no production incidents" in fused.final_answer.lower()


def test_negative_fact_check_preserves_generic_negated_claim_polarity():
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="support_review",
                    text=(
                        "No customer escalations were logged during the latest support review window. "
                        "The support trial remained monitored by QA."
                    ),
                ),
            ]
        ),
        force_stub=True,
    )

    fused, _ = program(
        "Did customer escalations happen during the latest support review window?",
        constraints="",
    )

    assert fused.verdict == "approve"
    assert fused.final_answer.startswith("No.")
    assert "no customer escalations were logged" in fused.final_answer.lower()


def test_decision_approval_does_not_invent_budget_support():
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="pilot_brief",
                    text=(
                        "The pilot proposal scopes MAGI to internal policy triage for four weeks. "
                        "Every decision keeps a human reviewer in the loop. Weekly evidence refreshes, "
                        "rollback criteria, and audit logs are required guardrails before launch."
                    ),
                ),
            ]
        ),
        force_stub=True,
    )

    fused, _ = program(
        "Should we proceed with the internal policy triage pilot next month?",
        constraints="keep human review and rollback guardrails",
    )

    assert fused.verdict == "approve"
    assert "budget" not in fused.final_answer.lower()
    assert "human reviewer" in fused.final_answer.lower()


def test_guarded_decision_skips_unnamed_low_control_distractor():
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="pilot_brief",
                    text=(
                        "The pilot proposal scopes MAGI to internal policy triage for four weeks. "
                        "Every decision keeps a human reviewer in the loop."
                    ),
                ),
                ScenarioEvidence(
                    source="event_agenda",
                    text=(
                        "The internal policy triage celebration next month includes lunch, demos, "
                        "office logistics, and a photo booth."
                    ),
                ),
                ScenarioEvidence(
                    source="control_plan",
                    text=(
                        "Weekly evidence refreshes, rollback criteria, and audit logs are required "
                        "guardrails before launch."
                    ),
                ),
            ]
        ),
        force_stub=True,
    )

    fused, _ = program(
        "Should we pilot MAGI for internal policy triage next month?",
        constraints="Keep human review in the loop.",
    )

    assert fused.verdict == "approve"
    assert "human reviewer" in fused.final_answer.lower()
    assert "rollback criteria" in fused.final_answer.lower()
    assert "event agenda" not in fused.final_answer.lower()
    assert "photo booth" not in fused.final_answer.lower()


def test_incomplete_decision_revises_instead_of_cautious_approval():
    program = MagiProgram(
        retriever=ScenarioRetriever(
            [
                ScenarioEvidence(
                    source="proposal_note",
                    text=(
                        "The note says the current approval workflow is slow, but it does not include "
                        "budget, owner, risk controls, or rollback plan."
                    ),
                ),
                ScenarioEvidence(
                    source="calendar_note",
                    text="The calendar note lists planning meetings and social events.",
                ),
            ]
        ),
        force_stub=True,
    )

    fused, personas = program("Should we replace the approval workflow?", constraints="")

    assert fused.verdict == "revise"
    assert {persona.stance for persona in personas.values()} == {"revise"}
    assert "calendar note" not in fused.final_answer.lower()
    assert "proceeding cautiously" not in fused.final_answer.lower()
