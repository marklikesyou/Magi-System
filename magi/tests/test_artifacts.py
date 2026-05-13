from __future__ import annotations

from magi.app.artifacts import diff_run_artifacts, render_artifact_diff, render_run_artifact


def _payload(
    *,
    run_id: str,
    profile: str,
    requested_route: str,
    verdict: str,
    query_mode: str,
    effective_mode: str,
    requires_human_review: bool,
    abstained: bool,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "created_at": "2026-01-01T00:00:00Z",
        "artifact_path": f"/tmp/{run_id}.json",
        "input": {
            "query": "Should we deploy the pilot?",
            "constraints": "",
            "profile": profile,
            "requested_route": requested_route,
        },
        "store": {"path": "/tmp/store.json"},
        "summary": {
            "verdict": verdict,
            "query_mode": query_mode,
            "citation_hit_rate": 1.0,
            "answer_support_score": 0.4,
            "requires_human_review": requires_human_review,
            "abstained": abstained,
            "abstention_reason": "Need more evidence." if abstained else "",
            "effective_mode": effective_mode,
            "model": "stub",
        },
        "decision": {
            "residual_risk": "medium",
            "justification": "Grounded answer with caveats.",
            "risks": ["Residual monitoring risk."],
            "mitigations": ["Keep rollback ready."],
        },
        "fused": {
            "final_answer": "Proceed with a guarded rollout.",
            "next_steps": ["Confirm rollback owner."],
        },
        "decision_trace": {
            "trace_id": f"trace-{run_id}",
            "end_to_end_ms": 42.5,
            "retrieved_evidence_ids": ["doc::1", "doc::2"],
            "used_evidence_ids": ["doc::1"],
            "cited_evidence_ids": ["doc::1"],
            "blocked_evidence_ids": ["doc::2"],
            "cache_hit": True,
            "persona_mode": "live",
            "responder_mode": "deterministic",
            "live_fallback_count": 1,
            "live_fallback_labels": ["fusion"],
            "decision_features": {"approve_guardrail_triggered": False},
            "spans": [
                {
                    "name": "program.run",
                    "start_ms": 1.0,
                    "duration_ms": 40.0,
                    "attributes": {"effective_mode": effective_mode},
                },
                {
                    "name": "grounding.verify",
                    "start_ms": 41.0,
                    "duration_ms": 1.0,
                    "attributes": {"citation_hit_rate": 1.0},
                },
            ],
            "routing_rationale": "Decision markers outweighed summary markers.",
            "routing_scores": {"decision": 4, "summarize": 1},
            "routing_signals": ["contains should we", "contains deploy"],
        },
    }


def test_render_run_artifact_uses_readable_sections() -> None:
    report = render_run_artifact(
        _payload(
            run_id="demo",
            profile="exec-brief",
            requested_route="summarize",
            verdict="approve",
            query_mode="summarize",
            effective_mode="stub",
            requires_human_review=True,
            abstained=False,
        )
    )

    assert "RUN ARTIFACT" in report
    assert "Input:" in report
    assert "Outcome:" in report
    assert "Grounding:" in report
    assert "Routing:" in report
    assert "Execution:" in report
    assert "Trace Spans:" in report
    assert "program.run" in report
    assert "Live Fallback Labels: fusion" in report
    assert "Next Steps:" in report
    assert "Mitigations:" in report


def test_render_artifact_diff_shows_left_and_right_values() -> None:
    left = _payload(
        run_id="run-left",
        profile="",
        requested_route="",
        verdict="revise",
        query_mode="decision",
        effective_mode="stub",
        requires_human_review=False,
        abstained=False,
    )
    right = _payload(
        run_id="run-right",
        profile="exec-brief",
        requested_route="summarize",
        verdict="approve",
        query_mode="summarize",
        effective_mode="live",
        requires_human_review=True,
        abstained=True,
    )

    report = render_artifact_diff(diff_run_artifacts(left, right))

    assert "ARTIFACT DIFF" in report
    assert "Profile: default -> exec-brief" in report
    assert "Requested Route: auto -> summarize" in report
    assert "Verdict: revise -> approve" in report
    assert "Resolved Mode: decision -> summarize" in report
    assert "Effective Mode: stub -> live" in report
    assert "Human Review Changed: yes" in report
    assert "Abstained Changed: yes" in report
