from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a SOC extraction benchmark from WitFoo attack reports")
    parser.add_argument("--input", default="data/raw/witfoo-precinct6/graph/attack_reports.jsonl")
    parser.add_argument("--output", default="data/hf/witfoo_soc_extraction_100.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--confirmed-only", action="store_true")
    args = parser.parse_args()

    rows = []
    with Path(args.input).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if args.confirmed_only and row.get("disposition_category") != "confirmed-malicious":
                continue
            rows.append(row)

    selected = _diverse_sample(rows, max(0, args.limit), args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            expected = {
                "incident_id": row["incident_id"],
                "classification": row["mo_name"],
                "suspicion_score": round(float(row["suspicion_score"]), 2),
                "disposition": row["disposition"],
                "disposition_category": row["disposition_category"],
                "first_observed_at": _iso8601(row["first_observed_at"]),
                "last_observed_at": _iso8601(row["last_observed_at"]),
                "lead_count": row["lead_count"],
                "node_count": row["node_count"],
                "edge_count": row["edge_count"],
                "set_role_names": row["set_role_names"],
                "matched_rules": row["matched_rules"],
                "products_observed": row["products_observed"],
                "attack_tactics": row["attack_tactics"],
                "attack_techniques": row["attack_techniques"],
                "lifecycle_stage": row["lifecycle_stage"],
            }
            normalized = {
                "id": row["incident_id"],
                "input": row["report_text"],
                "expected": expected,
                "dataset": "witfoo/precinct6-cybersecurity",
                "source_file": "graph/attack_reports.jsonl",
                "report_source": row["report_source"],
            }
            handle.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(selected)} rows to {output}")
    return 0


def _iso8601(timestamp: int | float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _diverse_sample(rows: list[dict], limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        products = row.get("products_observed") or [""]
        buckets[(str(row.get("mo_name", "")), str(products[0]))].append(row)
    for values in buckets.values():
        rng.shuffle(values)
    ordered_keys = sorted(buckets)
    selected: list[dict] = []
    while len(selected) < limit:
        added = False
        for key in ordered_keys:
            if not buckets[key]:
                continue
            selected.append(buckets[key].pop())
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
