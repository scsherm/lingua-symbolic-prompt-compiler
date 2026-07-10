from __future__ import annotations

import argparse
import json
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prompt_compiler.env import load_env_file
from prompt_compiler.eval.embedding_distance import make_drift_scorer, normalize_distance
from prompt_compiler.eval.extraction_metrics import ExtractionMetrics, score_extraction
from prompt_compiler.eval.llm_judge import LLMSemanticJudge
from prompt_compiler.models.base import GenerateParams
from prompt_compiler.models.openai_client import OpenAIResponsesModel
from prompt_compiler.prompt.template import PromptTemplate
from prompt_compiler.tokenizer import make_tokenizer


TARGET_INPUT_PRICE = 0.05
TARGET_OUTPUT_PRICE = 0.40
JUDGE_INPUT_PRICE = 0.75
JUDGE_OUTPUT_PRICE = 4.50


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen compressed SOC extraction prompt")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--original-prompt", required=True)
    parser.add_argument("--candidate-prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--offset", type=int, default=10)
    parser.add_argument("--limit", type=int, default=90)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--target-model", default="gpt-5-nano-2025-08-07")
    parser.add_argument("--judge-model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--judge-repetition", type=int, default=0)
    parser.add_argument("--judge-limit", type=int, default=None)
    parser.add_argument("--semantic-drift-normalization", type=float, default=10.578125)
    parser.add_argument("--judge-calibration-floor", type=float, default=0.0)
    parser.add_argument("--judge-calibration-ceiling", type=float, default=0.6944444444444443)
    parser.add_argument("--env-file", default=".env.local")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(Path(args.dataset))[args.offset : args.offset + args.limit]
    original_prompt = PromptTemplate(Path(args.original_prompt).read_text(encoding="utf-8"))
    candidate_prompt = PromptTemplate(Path(args.candidate_prompt).read_text(encoding="utf-8"))
    tokenizer = make_tokenizer(f"model:{args.target_model}")
    manifest = {
        "dataset": args.dataset,
        "offset": args.offset,
        "row_count": len(rows),
        "repetitions": args.repetitions,
        "target_model": args.target_model,
        "judge_model": args.judge_model,
        "original_instruction_tokens": tokenizer.count(original_prompt.instruction_text()),
        "candidate_instruction_tokens": tokenizer.count(candidate_prompt.instruction_text()),
        "judge_repetition": args.judge_repetition,
    }
    _write_json(output_dir / "manifest.json", manifest)

    target_path = output_dir / "paired_outputs.jsonl"
    completed = {(row["input_id"], row["repetition"]) for row in _read_jsonl(target_path)}
    jobs = [
        (row, repetition)
        for row in rows
        for repetition in range(args.repetitions)
        if (row["id"], repetition) not in completed
    ]
    target_model = OpenAIResponsesModel(name=args.target_model)
    target_params = GenerateParams(max_tokens=2048, reasoning_effort="minimal")
    write_lock = threading.Lock()

    def evaluate_pair(row: dict, repetition: int) -> dict:
        original_rendered = original_prompt.render({"input": row["input"]})
        candidate_rendered = candidate_prompt.render({"input": row["input"]})
        original_response = target_model.generate(original_rendered, target_params)
        candidate_response = target_model.generate(candidate_rendered, target_params)
        original_score = score_extraction(original_response.text, row["expected"])
        candidate_score = score_extraction(candidate_response.text, row["expected"])
        return {
            "input_id": row["id"],
            "repetition": repetition,
            "expected": row["expected"],
            "original_output": original_response.text,
            "candidate_output": candidate_response.text,
            "original_extraction": original_score.to_dict(),
            "candidate_extraction": candidate_score.to_dict(),
            "f1_delta": candidate_score.f1 - original_score.f1,
            "original_usage": original_response.usage,
            "candidate_usage": candidate_response.usage,
        }

    with ThreadPoolExecutor(max_workers=max(1, args.max_concurrency)) as executor:
        futures = {executor.submit(evaluate_pair, row, repetition): (row["id"], repetition) for row, repetition in jobs}
        finished = len(completed)
        for future in as_completed(futures):
            result = future.result()
            with write_lock:
                _append_jsonl(target_path, result)
            finished += 1
            if finished % 10 == 0 or finished == len(rows) * args.repetitions:
                print(f"target_pairs={finished}/{len(rows) * args.repetitions}", flush=True)

    paired_rows = _read_jsonl(target_path)
    target_summary = _target_summary(paired_rows, args.repetitions)
    _write_json(output_dir / "target_summary.json", target_summary)

    diagnostic_path = output_dir / "diagnostics.jsonl"
    diagnostic_completed = {row["input_id"] for row in _read_jsonl(diagnostic_path)}
    diagnostic_rows = [
        row
        for row in paired_rows
        if row["repetition"] == args.judge_repetition and row["input_id"] not in diagnostic_completed
    ]
    if args.judge_limit is not None:
        diagnostic_rows = diagnostic_rows[: max(0, args.judge_limit - len(diagnostic_completed))]
    drift_scorer = make_drift_scorer(
        "sentence-transformers",
        model_name="mixedbread-ai/mxbai-embed-large-v1",
    )
    judge_model = OpenAIResponsesModel(name=args.judge_model)

    def diagnose(row: dict) -> dict:
        source = next(item for item in rows if item["id"] == row["input_id"])
        raw_distance = drift_scorer.distance(row["candidate_output"], row["original_output"])
        judge = LLMSemanticJudge(
            model=judge_model,
            params=GenerateParams(max_tokens=4096, reasoning_effort="medium"),
            trace_path=output_dir / "judge_traces.jsonl",
        )
        judge.calibration_floor = args.judge_calibration_floor
        judge.calibration_ceiling = args.judge_calibration_ceiling
        judgment = judge.judge(
            original_prompt=original_prompt.text,
            input_text=source["input"],
            reference_output=row["original_output"],
            candidate_output=row["candidate_output"],
            input_id=row["input_id"],
        )
        return {
            "input_id": row["input_id"],
            "repetition": row["repetition"],
            "f1_delta": row["f1_delta"],
            "embedding_distance": raw_distance,
            "normalized_embedding_distance": normalize_distance(raw_distance, args.semantic_drift_normalization),
            "judge": judgment.to_dict(),
        }

    with ThreadPoolExecutor(max_workers=max(1, args.judge_concurrency)) as executor:
        futures = {executor.submit(diagnose, row): row["input_id"] for row in diagnostic_rows}
        finished = len(diagnostic_completed)
        for future in as_completed(futures):
            result = future.result()
            with write_lock:
                _append_jsonl(diagnostic_path, result)
            finished += 1
            diagnostic_total = min(len(rows), args.judge_limit) if args.judge_limit is not None else len(rows)
            if finished % 10 == 0 or finished == diagnostic_total:
                print(f"diagnostics={finished}/{diagnostic_total}", flush=True)

    diagnostics = _read_jsonl(diagnostic_path)
    judge_traces = _read_jsonl(output_dir / "judge_traces.jsonl")
    final_summary = {
        **target_summary,
        "diagnostics": _diagnostic_summary(diagnostics),
        "estimated_cost_usd": _estimated_cost(paired_rows, judge_traces),
    }
    _write_json(output_dir / "final_summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def _target_summary(rows: list[dict], repetitions: int) -> dict:
    original = _aggregate([row["original_extraction"] for row in rows])
    candidate = _aggregate([row["candidate_extraction"] for row in rows])
    by_input: dict[str, list[float]] = {}
    for row in rows:
        by_input.setdefault(row["input_id"], []).append(float(row["f1_delta"]))
    paired_deltas = [sum(values) / len(values) for values in by_input.values()]
    wins = sum(delta > 1e-12 for delta in paired_deltas)
    losses = sum(delta < -1e-12 for delta in paired_deltas)
    ties = len(paired_deltas) - wins - losses
    ci_low, ci_high = _bootstrap_ci(paired_deltas)
    return {
        "pair_count": len(rows),
        "report_count": len(by_input),
        "repetitions": repetitions,
        "original": original,
        "candidate": candidate,
        "f1_delta": float(candidate["f1"]) - float(original["f1"]),
        "per_report_mean_f1_delta": sum(paired_deltas) / max(len(paired_deltas), 1),
        "bootstrap_95_ci": [ci_low, ci_high],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "by_repetition": _by_repetition(rows, repetitions),
    }


def _aggregate(rows: list[dict]) -> dict[str, float | int]:
    true_positives = sum(int(row["true_positives"]) for row in rows)
    false_positives = sum(int(row["false_positives"]) for row in rows)
    false_negatives = sum(int(row["false_negatives"]) for row in rows)
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": sum(bool(row["exact_match"]) for row in rows) / max(len(rows), 1),
        "valid_json_rate": sum(bool(row["valid_json"]) for row in rows) / max(len(rows), 1),
        "schema_valid_rate": sum(bool(row["schema_valid"]) for row in rows) / max(len(rows), 1),
    }


def _by_repetition(rows: list[dict], repetitions: int) -> list[dict]:
    result = []
    for repetition in range(repetitions):
        subset = [row for row in rows if row["repetition"] == repetition]
        original = _aggregate([row["original_extraction"] for row in subset])
        candidate = _aggregate([row["candidate_extraction"] for row in subset])
        result.append(
            {
                "repetition": repetition,
                "original_f1": original["f1"],
                "candidate_f1": candidate["f1"],
                "f1_delta": float(candidate["f1"]) - float(original["f1"]),
            }
        )
    return result


def _bootstrap_ci(values: list[float], samples: int = 10000, seed: int = 17) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return means[int(samples * 0.025)], means[min(int(samples * 0.975), samples - 1)]


def _diagnostic_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    task_losses = [max(-float(row["f1_delta"]), 0.0) for row in rows]
    embeddings = [float(row["normalized_embedding_distance"]) for row in rows]
    judges = [float(row["judge"]["loss"]) for row in rows]
    return {
        "examples": len(rows),
        "mean_normalized_embedding_distance": sum(embeddings) / len(embeddings),
        "mean_judge_loss": sum(judges) / len(judges),
        "mean_judge_position_disagreement": sum(float(row["judge"]["position_disagreement"]) for row in rows) / len(rows),
        "embedding_vs_f1_regression_pearson": _correlation(embeddings, task_losses),
        "judge_vs_f1_regression_pearson": _correlation(judges, task_losses),
    }


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def _estimated_cost(paired_rows: list[dict], judge_traces: list[dict]) -> dict[str, float]:
    target_input = sum(_usage(row, "original_usage", "input_tokens") + _usage(row, "candidate_usage", "input_tokens") for row in paired_rows)
    target_output = sum(_usage(row, "original_usage", "output_tokens") + _usage(row, "candidate_usage", "output_tokens") for row in paired_rows)
    judge_input = sum(int((row.get("usage") or {}).get("input_tokens", 0)) for row in judge_traces)
    judge_output = sum(int((row.get("usage") or {}).get("output_tokens", 0)) for row in judge_traces)
    target_cost = (target_input * TARGET_INPUT_PRICE + target_output * TARGET_OUTPUT_PRICE) / 1_000_000
    judge_cost = (judge_input * JUDGE_INPUT_PRICE + judge_output * JUDGE_OUTPUT_PRICE) / 1_000_000
    return {
        "target": target_cost,
        "judge": judge_cost,
        "total": target_cost + judge_cost,
    }


def _usage(row: dict, section: str, key: str) -> int:
    return int((row.get(section) or {}).get(key, 0))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
