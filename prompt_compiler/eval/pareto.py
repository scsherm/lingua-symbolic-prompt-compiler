from __future__ import annotations

from prompt_compiler.eval.evaluator import CandidateReport


def compute_pareto_frontier(reports: list[CandidateReport]) -> list[CandidateReport]:
    frontier: list[CandidateReport] = []
    for report in reports:
        if any(_dominates(other, report) for other in reports if other is not report):
            continue
        frontier.append(report)
    return sorted(
        frontier,
        key=lambda item: (
            item.format_failure_rate,
            item.task_failure_rate,
            -float((item.candidate_extraction or {}).get("schema_valid_rate", 1.0)),
            item.objective_score,
            -item.token_reduction,
        ),
    )


def _dominates(a: CandidateReport, b: CandidateReport) -> bool:
    a_semantic = _semantic_loss(a)
    b_semantic = _semantic_loss(b)
    better_or_equal = (
        a.token_reduction >= b.token_reduction
        and a_semantic <= b_semantic
    )
    strictly_better = (
        a.token_reduction > b.token_reduction
        or a_semantic < b_semantic
    )
    return better_or_equal and strictly_better


def _semantic_loss(report: CandidateReport) -> float:
    if report.evaluation_profile == "extraction" and report.task_loss is not None:
        return report.task_loss
    if report.avg_judge_loss is not None or report.avg_normalized_semantic_drift != 0.0:
        return report.avg_combined_semantic_loss
    return report.avg_semantic_drift
