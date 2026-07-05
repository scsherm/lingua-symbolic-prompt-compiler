from __future__ import annotations

from dataclasses import dataclass

from prompt_compiler.eval.evaluator import CandidateReport


@dataclass(frozen=True)
class OperatorDiagnostic:
    operator_key: str
    uses: int
    avg_loss: float
    failure_rate: float


def summarize_operator_diagnostics(reports: list[CandidateReport]) -> list[OperatorDiagnostic]:
    totals: dict[str, dict[str, float]] = {}
    for report in reports:
        failure_rate = max(report.format_failure_rate, report.task_failure_rate, report.language_failure_rate)
        for operator_key, uses in report.operator_summary.items():
            row = totals.setdefault(operator_key, {"uses": 0.0, "loss": 0.0, "failures": 0.0})
            row["uses"] += uses
            row["loss"] += report.avg_loss * uses
            row["failures"] += failure_rate * uses
    diagnostics = [
        OperatorDiagnostic(
            operator_key=key,
            uses=int(value["uses"]),
            avg_loss=value["loss"] / max(value["uses"], 1),
            failure_rate=value["failures"] / max(value["uses"], 1),
        )
        for key, value in totals.items()
    ]
    return sorted(diagnostics, key=lambda item: (item.failure_rate, item.avg_loss))

