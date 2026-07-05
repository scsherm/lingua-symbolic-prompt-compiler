from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from prompt_compiler.candidates.generation import seed_population
from prompt_compiler.data.dataset_builder import InputExample, ReferenceExample, build_reference_dataset
from prompt_compiler.data.splits import split_dataset
from prompt_compiler.eval.contract_checks import OutputContract
from prompt_compiler.eval.embedding_distance import DriftScorer
from prompt_compiler.eval.evaluator import CandidateReport, Evaluator
from prompt_compiler.eval.pareto import compute_pareto_frontier
from prompt_compiler.models.base import GenerateParams, ModelClient
from prompt_compiler.optimize.credit_assignment import OperatorDiagnostic, summarize_operator_diagnostics
from prompt_compiler.optimize.curriculum import curriculum_subset
from prompt_compiler.prompt.template import PromptTemplate
from prompt_compiler.reports.writer import write_run_artifacts
from prompt_compiler.tokenizer import ApproxTokenizer, Tokenizer


@dataclass(frozen=True)
class EvaluationReport:
    original_instruction_tokens: int
    best_instruction_tokens: int
    token_reduction: float
    validation_semantic_drift: float
    format_failure_rate: float
    task_failure_rate: float
    diagnostics: list[OperatorDiagnostic]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["diagnostics"] = [asdict(item) for item in self.diagnostics]
        return data


@dataclass(frozen=True)
class CompressionRunResult:
    best_prompt_template: str
    pareto_frontier: list[CandidateReport]
    evaluation_report: EvaluationReport
    reference_dataset: list[ReferenceExample]
    all_reports: list[CandidateReport]


def optimize_prompt(
    *,
    target_model: ModelClient,
    original_prompt: PromptTemplate,
    inputs: Iterable[InputExample | Mapping[str, object] | str],
    output_dir: Path,
    epochs: int = 3,
    population_size: int = 32,
    tokenizer: Tokenizer | None = None,
    output_contract: OutputContract | None = None,
    drift_scorer: DriftScorer | None = None,
    params: GenerateParams | None = None,
) -> CompressionRunResult:
    tokenizer = tokenizer or ApproxTokenizer()
    params = params or GenerateParams()
    output_contract = output_contract or OutputContract()
    output_dir.mkdir(parents=True, exist_ok=True)

    references = build_reference_dataset(
        model=target_model,
        prompt=original_prompt,
        inputs=inputs,
        tokenizer=tokenizer,
        params=params,
    )
    split = split_dataset(references)
    original_instruction_tokens = tokenizer.count(original_prompt.instruction_text())
    evaluator = Evaluator(tokenizer=tokenizer, output_contract=output_contract, drift_scorer=drift_scorer)
    population = seed_population(original_prompt.text, tokenizer=tokenizer, population_size=population_size)

    epoch_reports: list[CandidateReport] = []
    all_evaluated_reports: list[CandidateReport] = []
    for epoch in range(max(epochs, 1)):
        subset = curriculum_subset(split.train, epoch)
        epoch_reports = [
            evaluator.evaluate_candidate(
                candidate=candidate,
                model=target_model,
                references=subset,
                original_instruction_tokens=original_instruction_tokens,
                params=params,
            )
            for candidate in population
        ]
        all_evaluated_reports.extend(epoch_reports)
        frontier = compute_pareto_frontier(epoch_reports)
        population = _keep_frontier_candidates(population, frontier, population_size)

    finalists = compute_pareto_frontier(epoch_reports)
    validation_set = split.validation or split.train
    validation_reports = [
        evaluator.evaluate_candidate(
            candidate=_candidate_by_id(population, report.candidate_id),
            model=target_model,
            references=validation_set,
            original_instruction_tokens=original_instruction_tokens,
            params=params,
        )
        for report in finalists
        if _has_candidate(population, report.candidate_id)
    ]
    final_reports = validation_reports or finalists
    final_frontier = compute_pareto_frontier(final_reports)
    best = _choose_best(final_frontier or final_reports)
    diagnostics = summarize_operator_diagnostics(final_reports)

    evaluation_report = EvaluationReport(
        original_instruction_tokens=original_instruction_tokens,
        best_instruction_tokens=best.instruction_tokens,
        token_reduction=best.token_reduction,
        validation_semantic_drift=best.avg_semantic_drift,
        format_failure_rate=best.format_failure_rate,
        task_failure_rate=best.task_failure_rate,
        diagnostics=diagnostics,
    )
    result = CompressionRunResult(
        best_prompt_template=best.prompt_template,
        pareto_frontier=final_frontier,
        evaluation_report=evaluation_report,
        reference_dataset=references,
        all_reports=all_evaluated_reports + final_reports,
    )
    write_run_artifacts(output_dir, result)
    return result


def _choose_best(reports: list[CandidateReport]) -> CandidateReport:
    if not reports:
        raise ValueError("No candidate reports were produced")
    return sorted(
        reports,
        key=lambda item: (
            item.format_failure_rate,
            item.task_failure_rate,
            item.language_failure_rate,
            item.avg_loss,
            -item.token_reduction,
        ),
    )[0]


def _keep_frontier_candidates(population, frontier: list[CandidateReport], population_size: int):
    frontier_ids = {report.candidate_id for report in frontier}
    kept = [candidate for candidate in population if candidate.id in frontier_ids]
    return kept[:population_size] or population[:population_size]


def _has_candidate(population, candidate_id: str) -> bool:
    return any(candidate.id == candidate_id for candidate in population)


def _candidate_by_id(population, candidate_id: str):
    for candidate in population:
        if candidate.id == candidate_id:
            return candidate
    raise KeyError(candidate_id)
