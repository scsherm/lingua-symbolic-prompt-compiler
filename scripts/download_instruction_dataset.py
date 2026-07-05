from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


DATASET_SERVER = "https://datasets-server.huggingface.co"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download instruction/completion rows from Hugging Face.")
    parser.add_argument("--dataset", default="HuggingFaceH4/no_robots")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = list(fetch_rows(args.dataset, args.config, args.split, args.limit))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            normalized = normalize_row(args.dataset, row)
            if normalized:
                handle.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {output}")
    return 0


def fetch_rows(dataset: str, config: str, split: str, limit: int):
    offset = 0
    remaining = limit
    while remaining > 0:
        length = min(remaining, 100)
        query = urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        with urlopen(f"{DATASET_SERVER}/rows?{query}", timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        page = payload.get("rows", [])
        if not page:
            break
        for item in page:
            yield item["row_idx"], item["row"]
        offset += len(page)
        remaining -= len(page)
        if len(page) < length:
            break


def normalize_row(dataset: str, indexed_row: tuple[int, dict]) -> dict | None:
    row_idx, row = indexed_row
    if dataset == "HuggingFaceH4/no_robots":
        assistant = first_assistant_message(row.get("messages", []))
        if not row.get("prompt") or not assistant:
            return None
        return {
            "id": row.get("prompt_id") or f"row_{row_idx:06d}",
            "input": row["prompt"],
            "reference_output": assistant,
            "dataset": dataset,
            "category": row.get("category", ""),
            "source_row": row_idx,
        }
    if dataset == "databricks/databricks-dolly-15k":
        instruction = row.get("instruction", "")
        context = row.get("context", "")
        response = row.get("response", "")
        if not instruction or not response:
            return None
        input_text = instruction if not context else f"{instruction}\n\nContext:\n{context}"
        return {
            "id": f"dolly_{row_idx:06d}",
            "input": input_text,
            "reference_output": response,
            "dataset": dataset,
            "category": row.get("category", ""),
            "source_row": row_idx,
        }
    return generic_normalize(dataset, row_idx, row)


def first_assistant_message(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "assistant" and message.get("content"):
            return message["content"]
    return ""


def generic_normalize(dataset: str, row_idx: int, row: dict) -> dict | None:
    input_text = row.get("prompt") or row.get("instruction") or row.get("input")
    output_text = row.get("response") or row.get("output") or row.get("completion") or row.get("answer")
    if not input_text or not output_text:
        print(f"skipping row {row_idx}: no recognizable input/output fields", file=sys.stderr)
        return None
    return {
        "id": row.get("id") or f"row_{row_idx:06d}",
        "input": input_text,
        "reference_output": output_text,
        "dataset": dataset,
        "category": row.get("category", ""),
        "source_row": row_idx,
    }


if __name__ == "__main__":
    raise SystemExit(main())

