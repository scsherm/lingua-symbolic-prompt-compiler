from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SOC prompt-compression generalization suite")
    parser.add_argument("--manifest", default="data/hf/soc_generalization/manifest.json")
    parser.add_argument("--output-root", default="runs/soc_generalization_v1")
    parser.add_argument("--selection-rows", type=int, default=20)
    parser.add_argument("--validation-repetitions", type=int, default=2)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--judge-limit", type=int, default=20)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, task in enumerate(manifest["tasks"], start=1):
        problem_id = f"{task['task']}__{task['prompt_variant']}"
        problem_root = output_root / problem_id
        selection_dir = problem_root / "selection"
        validation_dir = problem_root / "validation"
        print(f"[{index}/{len(manifest['tasks'])}] {problem_id}", flush=True)
        if not (selection_dir / "compression_report.json").exists():
            _run(
                [
                    sys.executable,
                    "-m",
                    "prompt_compiler.cli",
                    "--provider",
                    "openai",
                    "--model",
                    "gpt-5-nano-2025-08-07",
                    "--proposer",
                    "openai",
                    "--proposer-model",
                    "gpt-5.4-mini-2026-03-17",
                    "--proposer-max-output-tokens",
                    "4096",
                    "--proposer-reasoning-effort",
                    "none",
                    "--evaluation-profile",
                    "extraction",
                    "--task-weight",
                    "0.5",
                    "--token-weight",
                    "0.5",
                    "--prompt",
                    task["prompt"],
                    "--inputs",
                    task["dataset"],
                    "--output-dir",
                    str(selection_dir),
                    "--example-limit",
                    str(args.selection_rows),
                    "--epochs",
                    "1",
                    "--population-size",
                    str(args.population_size),
                    "--require-json",
                    "--max-output-tokens",
                    "2048",
                    "--reasoning-effort",
                    "minimal",
                    "--tokenizer",
                    "model:gpt-5-nano-2025-08-07",
                    "--embedding-provider",
                    "sentence-transformers",
                    "--embedding-model",
                    "mixedbread-ai/mxbai-embed-large-v1",
                    "--semantic-drift-normalization",
                    "10.578125",
                    "--chunkers",
                    "obligation",
                    "--prompt-perturbations",
                    "2",
                    "--rewrites-per-prompt",
                    "4",
                    "--proposal-pool-limit",
                    "12",
                    "--proposal-jitter-seed",
                    str(100 + index),
                    "--max-concurrency",
                    "8",
                    "--quiet",
                    "--live-log-file",
                    "",
                ]
            )
        if not (validation_dir / "final_summary.json").exists():
            _run(
                [
                    sys.executable,
                    "-m",
                    "scripts.evaluate_frozen_soc_candidate",
                    "--dataset",
                    task["dataset"],
                    "--original-prompt",
                    task["prompt"],
                    "--candidate-prompt",
                    str(selection_dir / "best_prompt.txt"),
                    "--output-dir",
                    str(validation_dir),
                    "--offset",
                    str(args.selection_rows),
                    "--limit",
                    str(100 - args.selection_rows),
                    "--repetitions",
                    str(args.validation_repetitions),
                    "--max-concurrency",
                    "10",
                    "--judge-concurrency",
                    "8",
                    "--judge-limit",
                    str(args.judge_limit),
                ]
            )
        selection = json.loads((selection_dir / "compression_report.json").read_text(encoding="utf-8"))
        validation = json.loads((validation_dir / "final_summary.json").read_text(encoding="utf-8"))
        results.append(
            {
                "problem_id": problem_id,
                "task": task["task"],
                "prompt_variant": task["prompt_variant"],
                "token_reduction": selection["token_reduction"],
                "original_f1": validation["original"]["f1"],
                "candidate_f1": validation["candidate"]["f1"],
                "f1_delta": validation["f1_delta"],
                "bootstrap_95_ci": validation["bootstrap_95_ci"],
                "wins": validation["wins"],
                "ties": validation["ties"],
                "losses": validation["losses"],
                "original_exact_match": validation["original"]["exact_match_rate"],
                "candidate_exact_match": validation["candidate"]["exact_match_rate"],
                "original_schema_valid": validation["original"]["schema_valid_rate"],
                "candidate_schema_valid": validation["candidate"]["schema_valid_rate"],
                "estimated_cost_usd": validation["estimated_cost_usd"],
            }
        )
        _write_summary(output_root, results)
    return 0


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _write_summary(output_root: Path, results: list[dict]) -> None:
    payload = {
        "completed_problems": len(results),
        "results": results,
        "aggregate": _aggregate(results),
    }
    (output_root / "suite_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    reductions = sorted(float(row["token_reduction"]) for row in results)
    deltas = [float(row["f1_delta"]) for row in results]
    return {
        "median_token_reduction": reductions[len(reductions) // 2],
        "mean_f1_delta": sum(deltas) / len(deltas),
        "improved": sum(delta > 1e-12 for delta in deltas),
        "tied": sum(abs(delta) <= 1e-12 for delta in deltas),
        "regressed": sum(delta < -1e-12 for delta in deltas),
        "noninferior_at_zero": sum(row["bootstrap_95_ci"][0] >= 0 for row in results),
        "schema_regressions": sum(row["candidate_schema_valid"] < row["original_schema_valid"] for row in results),
        "worst_f1_delta": min(deltas),
        "best_f1_delta": max(deltas),
    }


if __name__ == "__main__":
    raise SystemExit(main())
