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
            citation="[1]", source="brief", text="Pilot proposal", score=1.0
        )
    ]
    melchior = MelchiorResponse(
        text="[REVISE] [MELCHIOR] The proposal includes scope, budget, timeline, and controls.",
        analysis="The proposal includes scope, budget, timeline, controls, and human reviewer guardrails for a pilot.",
        answer_outline=["Approve a bounded pilot."],
        confidence=0.7,
        evidence_quotes=['[1] "Pilot proposal"'],
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
                citation="[1]", source="brief", text="Pilot proposal", score=1.0
            )
        ],
        response,
    )

    assert normalized.final_answer == fusion.final_answer
