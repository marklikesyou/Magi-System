from __future__ import annotations

from magi.core.routing import mode_prompt_brief, route_query


def test_route_query_selects_summarize_for_summary_request() -> None:
    route = route_query("Summarize MAGI in one sentence.")

    assert route.mode == "summarize"
    assert route.scores["summarize"] > route.scores["decision"]


def test_route_query_selects_extract_for_source_location_request() -> None:
    route = route_query("What page and section mention rollback conditions?")

    assert route.mode == "extract"
    assert route.retrieval_top_k >= 10


def test_route_query_selects_fact_check_for_verification_request() -> None:
    route = route_query("Verify whether the brief guarantees SOC 2 compliance.")

    assert route.mode == "fact_check"
    assert any(signal.startswith("fact_check:") for signal in route.signals)


def test_route_query_keeps_evidence_yes_no_above_rollout_decision_marker() -> None:
    route = route_query("Does the evidence say the rollout status is green?")

    assert route.mode == "fact_check"
    assert route.scores["fact_check"] > route.scores["decision"]


def test_route_query_treats_rollout_status_question_as_information() -> None:
    route = route_query("What is the MAGI rollout status right now?")

    assert route.mode == "summarize"
    assert route.scores["summarize"] > route.scores["decision"]


def test_route_query_treats_paraphrased_rollout_question_as_information() -> None:
    route = route_query("How is the MAGI rollout going right now?")

    assert route.mode == "summarize"
    assert route.scores["summarize"] > route.scores["decision"]


def test_route_query_selects_recommend_for_tradeoff_request() -> None:
    route = route_query("Compare the options and recommend the best rollout plan.")

    assert route.mode == "recommend"
    assert route.scores["recommend"] >= route.scores["summarize"]


def test_route_query_selects_decision_when_constraints_are_present() -> None:
    route = route_query(
        "Should we deploy this pilot next week?",
        constraints="Budget <= 50k; must keep human review",
    )

    assert route.mode == "decision"
    assert route.scores["decision"] > route.scores["recommend"]


def test_route_query_honors_forced_mode() -> None:
    route = route_query("Summarize this", forced_mode="fact_check")

    assert route.mode == "fact_check"
    assert route.rationale == "Explicit route override was provided."


def test_mode_prompt_brief_omits_routing_debug_signals() -> None:
    route = route_query(
        "Should we pilot MAGI next month?",
        constraints="Keep human review in the loop.",
    )

    brief = mode_prompt_brief(route)

    assert brief == (
        "Route mode: decision. Expected answer style: Return a clear decision "
        "with cited support, risks, mitigations, and explicit unknowns."
    )
    assert "should we" not in brief.lower()
    assert "pilot" not in brief.lower()
    assert "routing signals" not in brief.lower()
    assert "route scores" not in brief.lower()
