from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from prompt_compiler.eval.contract_checks import OutputContract
from prompt_compiler.eval.embedding_distance import DEFAULT_EMBEDDING_MODEL, make_drift_scorer
from prompt_compiler.models.mock import RuleBasedMockModel
from prompt_compiler.optimize.optimizer import optimize_prompt
from prompt_compiler.prompt.template import PromptTemplate
from prompt_compiler.tokenizer import make_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Behavioral prompt compression compiler")
    parser.add_argument("--model", default="mock", help="Model adapter name. Only 'mock' is built in.")
    parser.add_argument("--prompt", required=True, help="Path to original prompt template")
    parser.add_argument("--inputs", required=True, help="JSONL file with {'id','input'} rows")
    parser.add_argument("--output-dir", required=True, help="Directory for run artifacts")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--require-json", action="store_true")
    parser.add_argument(
        "--tokenizer",
        default="approx",
        help="Tokenizer spec: approx, tiktoken:<encoding>, or model:<model-name>",
    )
    parser.add_argument(
        "--embedding-provider",
        default="lexical",
        choices=("lexical", "sentence-transformers", "hf-inference"),
        help="Drift scorer backend. Use hf-inference or sentence-transformers for Mixedbread embeddings.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--hf-provider", default=None, help="Optional Hugging Face Inference Provider name")
    args = parser.parse_args()

    if args.model != "mock":
        raise SystemExit("Only the local mock model is built in. Pass a custom ModelClient from Python for real models.")

    prompt = PromptTemplate(Path(args.prompt).read_text(encoding="utf-8"))
    inputs = _read_jsonl(Path(args.inputs))
    result = optimize_prompt(
        target_model=RuleBasedMockModel(),
        original_prompt=prompt,
        inputs=inputs,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        population_size=args.population_size,
        tokenizer=make_tokenizer(args.tokenizer),
        drift_scorer=make_drift_scorer(
            args.embedding_provider,
            model_name=args.embedding_model,
            api_key=os.environ.get("HF_TOKEN"),
            hf_provider=args.hf_provider,
        ),
        output_contract=OutputContract(require_json=args.require_json),
    )
    print(json.dumps(result.evaluation_report.to_dict(), ensure_ascii=False, indent=2))
    print(f"best_prompt={Path(args.output_dir) / 'best_prompt.txt'}")
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
