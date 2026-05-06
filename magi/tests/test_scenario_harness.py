from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from magi.core.clients import LLMClient, LLMClientError
from magi.eval.run_scenarios import _threshold_failures
from magi.eval.scenario_harness import (
    ScenarioDataset,
    ScenarioEvidence,
    ScenarioRetriever,
    load_scenario_dataset,
    run_scenario_suite,
)


class _SequenceClient(LLMClient):
    model = "gpt-4o-mini-2024-07-18"

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self._payloads = list(payloads)

    def complete(self, messages, *, tools=None, response_format=None):  # type: ignore[override]
        assert messages
        assert response_format is not None
        if not self._payloads:
            raise AssertionError("unexpected extra client call")
        payload = self._payloads.pop(0)
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}


class _FailingClient(LLMClient):
    model = "gpt-4o-mini-2024-07-18"

    def complete(self, messages, *, tools=None, response_format=None):  # type: ignore[override]
        del messages, tools, response_format
        raise LLMClientError("provider unavailable")


def test_live_scenarios_suite_passes_in_stub_mode() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "eval" / "live_scenarios.yaml"
    dataset = load_scenario_dataset(dataset_path)

    report = run_scenario_suite(dataset, force_stub=True, requested_mode="stub")

    assert report.summary.effective_mode == "stub"
    assert report.summary.verdict_accuracy == 1.0
    assert report.summary.overall_score == 1.0
    assert report.summary.requirement_pass_rate == 1.0
    assert report.summary.latency_p50_ms >= 0.0
    assert report.summary.latency_p95_ms >= report.summary.latency_p50_ms
    assert report.summary.latency_max_ms >= report.summary.latency_p50_ms
    assert report.summary.average_estimated_cost_usd == 0.0
    assert report.cases[0].latency_ms >= 0.0


def test_production_scenarios_suite_passes_in_stub_mode() -> None:
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "production_scenarios.yaml"
    )
    dataset = load_scenario_dataset(dataset_path)

    report = run_scenario_suite(dataset, force_stub=True, requested_mode="stub")

    assert report.summary.effective_mode == "stub"
    assert report.summary.overall_score == 1.0
    assert report.summary.verdict_accuracy == 1.0
    assert report.summary.requirement_pass_rate == 1.0


def test_run_scenarios_cli_writes_json_report(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "eval" / "run_scenarios.py"
    dataset = Path(__file__).resolve().parents[1] / "eval" / "live_scenarios.yaml"
    report_out = tmp_path / "scenario_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cases",
            str(dataset),
            "--mode",
            "stub",
            "--report-out",
            str(report_out),
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["summary"]["overall_score"] == 1.0
    assert payload["summary"]["effective_mode"] == "stub"
    assert "report_saved" in completed.stdout


def test_live_scenarios_suite_passes_with_fake_live_client() -> None:
    dataset = ScenarioDataset.model_validate(
        {
            "cases": [
                {
                    "id": "live_grounded_summary",
                    "query": "Summarize MAGI in one sentence.",
                    "expected_verdict": "approve",
                    "expected_residual_risk": "low",
                    "evidence": [
                        {
                            "source": "README",
                            "text": "MAGI is a multi persona reasoning engine for assessing user requests against an evidence base.",
                        }
                    ],
                    "checks": {
                        "required_terms_any": ["magi", "reasoning engine"],
                        "min_citations": 1,
                        "min_citation_hit_rate": 1.0,
                        "min_answer_support_score": 0.2,
                        "require_human_review": True,
                    },
                }
            ]
        }
    )
    client = _SequenceClient(
        [
            {
                "analysis": "The evidence defines MAGI directly.",
                "answer_outline": ["Summarize MAGI from [1]."],
                "confidence": 0.8,
                "evidence_quotes": ['[1] "MAGI is a multi persona reasoning engine."'],
                "stance": "approve",
                "actions": ["Answer with citation [1]."],
            },
            {
                "plan": "Answer in one sentence with citation [1].",
                "communication_plan": ["Keep it concise.", "Cite [1]."],
                "cost_estimate": "low",
                "confidence": 0.8,
                "stance": "approve",
                "actions": ["Keep it concise.", "Cite [1]."],
            },
            {
                "risks": ["Low risk of over-claiming beyond [1]."],
                "mitigations": ["Stay within [1]."],
                "residual_risk": "low",
                "confidence": 0.8,
                "stance": "approve",
                "actions": ["Stay within [1]."],
                "outstanding_questions": [],
            },
            {
                "verdict": "approve",
                "justification": "The evidence in [1] is enough for a one-sentence summary.",
                "confidence": 0.8,
                "final_answer": "MAGI is a multi persona reasoning engine for assessing user requests against an evidence base [1].",
                "next_steps": ["Answer directly."],
                "consensus_points": ["[1] supports the summary."],
                "disagreements": [],
                "residual_risk": "low",
                "risks": ["Low risk of over-claiming beyond [1]."],
                "mitigations": ["Stay within [1]."],
            },
            {
                "final_answer": "MAGI is a multi persona reasoning engine for assessing user requests against an evidence base [1].",
                "justification": "The evidence in [1] is enough for a one-sentence summary.",
                "next_steps": ["Answer directly."],
            },
        ]
    )

    report = run_scenario_suite(
        dataset,
        force_stub=False,
        client=client,
        requested_mode="live",
    )

    assert report.summary.effective_mode == "live"
    assert report.summary.overall_score == 1.0
    assert report.summary.verdict_accuracy == 1.0
    assert report.cases[0].citation_count >= 1
    assert report.cases[0].requires_human_review is True


def test_live_scenario_gate_fails_when_provider_fallback_occurs() -> None:
    dataset = ScenarioDataset.model_validate(
        {
            "cases": [
                {
                    "id": "live_fallback_summary",
                    "query": "Summarize MAGI in one sentence.",
                    "expected_verdict": "approve",
                    "expected_residual_risk": "low",
                    "evidence": [
                        {
                            "source": "README",
                            "text": "MAGI is a multi persona reasoning engine for assessing user requests against an evidence base.",
                        }
                    ],
                    "checks": {
                        "required_terms_any": ["magi", "reasoning engine"],
                        "min_citations": 1,
                        "min_citation_hit_rate": 1.0,
                        "require_human_review": True,
                    },
                }
            ]
        }
    )

    report = run_scenario_suite(
        dataset,
        force_stub=False,
        client=_FailingClient(),
        requested_mode="live",
    )

    assert report.summary.effective_mode == "live"
    assert report.summary.live_fallback_count == 1
    assert report.cases[0].live_fallback_count == 1
    failures = _threshold_failures(
        report,
        Namespace(
            min_overall_score=None,
            min_verdict_accuracy=None,
            min_requirement_pass_rate=None,
            min_retrieval_hit_rate=None,
            min_retrieval_top_source_accuracy=None,
            min_retrieval_source_recall=None,
            min_average_citation_hit_rate=None,
            min_average_answer_support_score=None,
            min_supported_answer_rate=None,
            max_p50_latency_ms=None,
            max_p95_latency_ms=None,
            max_max_latency_ms=None,
            max_average_cost_usd=None,
            max_total_cost_usd=None,
            max_live_fallbacks=None,
            allow_live_fallbacks=False,
            mode="live",
        ),
    )
    assert ("live_fallback_count", 1.0, 0.0, "maximum") in failures

    allowed_failures = _threshold_failures(
        report,
        Namespace(
            min_overall_score=None,
            min_verdict_accuracy=None,
            min_requirement_pass_rate=None,
            min_retrieval_hit_rate=None,
            min_retrieval_top_source_accuracy=None,
            min_retrieval_source_recall=None,
            min_average_citation_hit_rate=None,
            min_average_answer_support_score=None,
            min_supported_answer_rate=None,
            max_p50_latency_ms=None,
            max_p95_latency_ms=None,
            max_max_latency_ms=None,
            max_average_cost_usd=None,
            max_total_cost_usd=None,
            max_live_fallbacks=None,
            allow_live_fallbacks=True,
            mode="live",
        ),
    )
    assert not any(failure[0] == "live_fallback_count" for failure in allowed_failures)


def test_scenario_retriever_ranks_query_matches_above_distractors() -> None:
    retriever = ScenarioRetriever(
        [
            ScenarioEvidence(
                source="distractor",
                text="The office snack budget was approved for the summer social.",
            ),
            ScenarioEvidence(
                source="pilot_brief",
                text="The pilot proposal keeps a human reviewer on every internal policy triage decision.",
            ),
            ScenarioEvidence(
                source="notes",
                text="Rollback criteria and weekly audits are required before launch.",
            ),
        ]
    )

    results = retriever.retrieve("Should we pilot internal policy triage with a human reviewer?")

    assert results[0].metadata["source"] == "pilot_brief"
    assert "policy triage" in results[0].text.lower()
    assert results[0].score >= results[1].score


def test_scenario_retriever_respects_persona_filter_after_ranking() -> None:
    retriever = ScenarioRetriever(
        [
            ScenarioEvidence(
                source="general",
                text="General rollout status is green and monitored.",
            ),
            ScenarioEvidence(
                source="melchior_notes",
                text="The retrieved evidence supports the rollout summary.",
                persona="melchior",
            ),
            ScenarioEvidence(
                source="casper_notes",
                text="Residual risk stays medium without weekly audits.",
                persona="casper",
            ),
        ]
    )

    results = retriever.retrieve("Explain the rollout summary", persona="melchior")

    assert [item.metadata["source"] for item in results] == [
        "melchior_notes",
        "general",
    ]


def test_run_scenario_suite_tracks_retrieved_sources_and_hits() -> None:
    dataset = ScenarioDataset.model_validate(
        {
            "cases": [
                {
                    "id": "retrieval_metrics",
                    "query": "Summarize MAGI in one sentence.",
                    "expected_verdict": "approve",
                    "evidence": [
                        {
                            "source": "distractor",
                            "text": "The office snack budget was approved for the summer social.",
                        },
                        {
                            "source": "README",
                            "text": "MAGI is a multi persona reasoning engine for assessing user requests against an evidence base.",
                        },
                    ],
                    "checks": {
                        "required_sources_any": ["README"],
                        "min_citation_hit_rate": 1.0,
                        "min_answer_support_score": 0.2,
                        "require_human_review": True,
                    },
                }
            ]
        }
    )

    report = run_scenario_suite(dataset, force_stub=True, requested_mode="stub")

    source_check = next(
        check for check in report.cases[0].checks if check.name == "required_sources_any"
    )
    assert source_check.passed is True
    assert report.cases[0].expected_retrieval_sources == ["README"]
    assert report.cases[0].retrieved_sources[0] == "README"
    assert report.cases[0].retrieved_relevant_chunk_count >= 1
    assert report.cases[0].first_expected_source_rank == 1
    assert report.cases[0].retrieval_source_recall == 1.0
    assert report.summary.retrieval_evaluable_cases == 1
    assert report.summary.cases_with_retrieval_hits == 1
    assert report.summary.retrieval_hit_rate == 1.0
    assert report.summary.retrieval_ranked_cases == 1
    assert report.summary.retrieval_top_source_accuracy == 1.0
    assert report.summary.retrieval_mrr == 1.0
    assert report.summary.retrieval_source_recall == 1.0
    assert report.cases[0].citation_hit_rate >= 1.0
    assert report.cases[0].answer_support_score > 0.2
    assert report.cases[0].answer_supported is True
    assert report.cases[0].requires_human_review is True
    assert report.summary.average_citation_hit_rate >= 1.0
    assert report.summary.average_answer_support_score > 0.2
    assert report.summary.supported_answer_rate == 1.0
    assert report.summary.latency_p50_ms >= 0.0
    assert report.summary.total_estimated_cost_usd == 0.0


def test_run_scenario_suite_counts_cited_abstention_as_supported_generically() -> None:
    dataset = ScenarioDataset.model_validate(
        {
            "cases": [
                {
                    "id": "paraphrased_missing_service_level",
                    "query": "Can MAGI guarantee a ninety fifth percentile response-time service level for customers?",
                    "expected_verdict": "abstain",
                    "expected_residual_risk": "medium",
                    "evidence": [
                        {
                            "source": "operations_summary",
                            "text": "MAGI tracks latency during internal pilots and reports observed response times to operators.",
                        },
                        {
                            "source": "release_note",
                            "text": "The current release is an internal alpha with no customer-facing service commitment.",
                        },
                    ],
                    "checks": {
                        "min_citations": 1,
                        "min_citation_hit_rate": 1.0,
                    },
                }
            ]
        }
    )

    report = run_scenario_suite(dataset, force_stub=True, requested_mode="stub")

    assert report.cases[0].predicted_verdict == "abstain"
    assert report.cases[0].citation_hit_rate == 1.0
    assert report.cases[0].answer_supported is True
    assert report.summary.supported_answer_rate == 1.0


def test_run_scenario_suite_fails_when_required_source_is_not_retrieved() -> None:
    dataset = ScenarioDataset.model_validate(
        {
            "cases": [
                {
                    "id": "missing_source",
                    "query": "Summarize MAGI in one sentence.",
                    "expected_verdict": "approve",
                    "evidence": [
                        {
                            "source": "README",
                            "text": "MAGI is a multi persona reasoning engine for assessing user requests against an evidence base.",
                        }
                    ],
                    "checks": {"required_sources_all": ["README", "pilot_brief"]},
                }
            ]
        }
    )

    report = run_scenario_suite(dataset, force_stub=True, requested_mode="stub")

    source_check = next(
        check for check in report.cases[0].checks if check.name == "required_sources_all"
    )
    assert source_check.passed is False
    assert "pilot_brief" in source_check.details
    assert report.summary.requirement_pass_rate < 1.0


def test_run_scenario_suite_reports_rank_when_expected_source_is_not_first() -> None:
    dataset = ScenarioDataset.model_validate(
        {
            "cases": [
                {
                    "id": "ranked_source",
                    "query": "Explain rollout status and monitoring cadence.",
                    "expected_verdict": "approve",
                    "evidence": [
                        {
                            "source": "weekly_report",
                            "text": "The rollout status and monitoring cadence are tracked in a weekly report.",
                        },
                        {
                            "source": "rollout_notes",
                            "text": "The rollout status is green and monitored by ops.",
                        },
                    ],
                    "checks": {"required_sources_any": ["rollout_notes"]},
                }
            ]
        }
    )

    report = run_scenario_suite(dataset, force_stub=True, requested_mode="stub")

    assert report.cases[0].retrieved_sources[:2] == ["weekly_report", "rollout_notes"]
    assert report.cases[0].first_expected_source_rank == 2
    assert report.cases[0].retrieval_source_recall == 1.0
    assert report.summary.retrieval_top_source_accuracy == 0.0
    assert report.summary.retrieval_mrr == 0.5
