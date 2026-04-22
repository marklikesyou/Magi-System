from __future__ import annotations

from magi.core.routing import route_query


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
