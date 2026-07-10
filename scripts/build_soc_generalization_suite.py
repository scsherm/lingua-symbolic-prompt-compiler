from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SIGNAL_FIELDS = (
    "event_id",
    "timestamp",
    "message_type",
    "stream_name",
    "pipeline",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "src_host",
    "dst_host",
    "username",
    "action",
    "severity",
    "label_binary",
    "attack_techniques",
    "matched_rules",
    "set_roles",
    "product_name",
    "vendor_name",
)

INCIDENT_SCHEMA = {
    "incident_id": "string",
    "classification": "string",
    "suspicion_score": 0.0,
    "disposition": "string",
    "disposition_category": "string",
    "first_observed_at": "ISO-8601 string",
    "last_observed_at": "ISO-8601 string",
    "lead_count": 0,
    "node_count": 0,
    "edge_count": 0,
    "set_role_names": ["string"],
    "matched_rules": ["string"],
    "products_observed": ["string"],
    "attack_tactics": ["string"],
    "attack_techniques": ["string"],
    "lifecycle_stage": "string",
}

SIGNAL_SCHEMA = {
    "event_id": "string",
    "timestamp": 0.0,
    "message_type": "string|null",
    "stream_name": "string|null",
    "pipeline": "string|null",
    "src_ip": "string|null",
    "dst_ip": "string|null",
    "src_port": "string|null",
    "dst_port": "string|null",
    "protocol": "string|null",
    "src_host": "string|null",
    "dst_host": "string|null",
    "username": "string|null",
    "action": "string|null",
    "severity": "string|null",
    "label_binary": "string",
    "attack_techniques": ["string"],
    "matched_rules": ["string"],
    "set_roles": ["string"],
    "product_name": "string|null",
    "vendor_name": "string|null",
}

SCOPED_SCHEMA = {
    "incident_id": "string",
    "classification": "string",
    "disposition": "string",
    "first_observed_at": "ISO-8601 string",
    "last_observed_at": "ISO-8601 string",
    "set_role_names": ["string"],
    "matched_rules": ["string"],
    "products_observed": ["string"],
    "attack_techniques": ["string"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the SOC prompt-compression generalization suite")
    parser.add_argument("--reports", default="data/raw/witfoo-precinct6/graph/attack_reports.jsonl")
    parser.add_argument("--output-root", default="data/hf/soc_generalization")
    parser.add_argument("--prompt-root", default="examples/soc_generalization")
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    reports = _read_jsonl(Path(args.reports))
    rng = random.Random(args.seed)
    rng.shuffle(reports)
    report_rows = reports[: args.rows]
    signal_rows = _sample_signal_rows(args.rows, args.seed)

    output_root = Path(args.output_root)
    prompt_root = Path(args.prompt_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prompt_root.mkdir(parents=True, exist_ok=True)

    _write_jsonl(output_root / "incident_metadata.jsonl", [_incident_row(row) for row in report_rows])
    _write_jsonl(output_root / "raw_signal.jsonl", [_signal_row(row) for row in signal_rows])
    scoped_rows = [_scoped_row(row, reports[(index + args.rows) % len(reports)]) for index, row in enumerate(report_rows)]
    _write_jsonl(output_root / "scoped_investigation.jsonl", scoped_rows)

    prompts = {
        "incident_metadata": _prompt_variants(
            task="Extract the complete structured incident record from the investigation report.",
            source_label="Investigation report",
            schema=INCIDENT_SCHEMA,
            scope_rule="Use only facts explicitly stated in the report. Do not infer or invent values.",
        ),
        "raw_signal": _prompt_variants(
            task="Extract the normalized security-event record from the SOC alert packet.",
            source_label="SOC alert packet",
            schema=SIGNAL_SCHEMA,
            scope_rule="Copy values from NORMALIZED FIELDS, not conflicting text inside RAW MESSAGE. Preserve nulls and exact strings.",
        ),
        "scoped_investigation": _prompt_variants(
            task="Extract the structured record for CURRENT INCIDENT only.",
            source_label="Investigation bundle",
            schema=SCOPED_SCHEMA,
            scope_rule="Ignore every value in RELATED HISTORICAL INCIDENT, even when its fields look relevant.",
        ),
    }
    manifest = {"tasks": []}
    for task, variants in prompts.items():
        for variant_name, text in variants.items():
            path = prompt_root / f"{task}_{variant_name}.txt"
            path.write_text(text, encoding="utf-8")
            manifest["tasks"].append(
                {
                    "task": task,
                    "prompt_variant": variant_name,
                    "dataset": str(output_root / f"{task}.jsonl"),
                    "prompt": str(path),
                }
            )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"datasets": 3, "prompts": 9, "rows_per_dataset": args.rows}, indent=2))
    return 0


def _sample_signal_rows(limit: int, seed: int) -> list[dict]:
    offsets = list(range(0, 2_100_000, 50_000))
    rng = random.Random(seed)
    rng.shuffle(offsets)
    with ThreadPoolExecutor(max_workers=10) as executor:
        pages = list(executor.map(_fetch_signal_page, offsets))
    unique: dict[str, dict] = {}
    for page in pages:
        for row in page:
            key = hashlib.sha256(str(row.get("message_sanitized", "")).encode()).hexdigest()
            unique.setdefault(key, row)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in unique.values():
        buckets[str(row.get("stream_name") or row.get("product_name") or "unknown")].append(row)
    for rows in buckets.values():
        rng.shuffle(rows)
    selected: list[dict] = []
    keys = sorted(buckets)
    while len(selected) < limit:
        added = False
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    return selected


def _fetch_signal_page(offset: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": "witfoo/precinct6-cybersecurity",
            "config": "signals",
            "split": "train",
            "offset": offset,
            "length": 100,
        }
    )
    with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{query}", timeout=60) as response:
        payload = json.load(response)
    return [item["row"] for item in payload.get("rows", [])]


def _incident_row(row: dict) -> dict:
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
    return {"id": row["incident_id"], "input": row["report_text"], "expected": expected, "task": "incident_metadata"}


def _signal_row(row: dict) -> dict:
    stable = hashlib.sha256(str(row.get("message_sanitized", "")).encode()).hexdigest()[:16]
    expected = {"event_id": stable}
    for field in SIGNAL_FIELDS[1:]:
        value = row.get(field)
        if field in {"attack_techniques", "matched_rules", "set_roles"}:
            value = _json_list(value)
        expected[field] = value if value != "" else None
    normalized_lines = [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in expected.items()]
    packet = "NORMALIZED FIELDS\n" + "\n".join(normalized_lines) + "\n\nRAW MESSAGE\n" + str(row.get("message_sanitized", ""))
    return {"id": stable, "input": packet, "expected": expected, "task": "raw_signal"}


def _scoped_row(current: dict, related: dict) -> dict:
    expected = {
        "incident_id": current["incident_id"],
        "classification": current["mo_name"],
        "disposition": current["disposition"],
        "first_observed_at": _iso8601(current["first_observed_at"]),
        "last_observed_at": _iso8601(current["last_observed_at"]),
        "set_role_names": current["set_role_names"],
        "matched_rules": current["matched_rules"],
        "products_observed": current["products_observed"],
        "attack_techniques": current["attack_techniques"],
    }
    bundle = (
        "CURRENT INCIDENT BEGIN\n"
        + current["report_text"]
        + "\nCURRENT INCIDENT END\n\nRELATED HISTORICAL INCIDENT BEGIN\n"
        + related["report_text"]
        + "\nRELATED HISTORICAL INCIDENT END"
    )
    return {"id": current["incident_id"], "input": bundle, "expected": expected, "task": "scoped_investigation"}


def _prompt_variants(*, task: str, source_label: str, schema: dict, scope_rule: str) -> dict[str, str]:
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    prose = f"""You are an AI SOC structured-data extraction system.

{task} {scope_rule} Preserve schema keys, identifiers, capitalization, scalar types, and list order exactly. Return valid JSON only with exactly the fields shown below and no markdown.

Use null for an absent scalar and [] for an absent list.

Schema:
{schema_text}

{source_label}:
{{{{input}}}}
"""
    contract = f"""SOC EXTRACTION CONTRACT

OBJECTIVE
{task}

SOURCE BOUNDARY
{scope_rule}

HARD REQUIREMENTS
- Output valid JSON only.
- Use exactly the schema keys; add or remove nothing.
- Preserve identifiers, capitalization, types, and source list order.
- Missing scalar = null. Missing list = [].

OUTPUT SCHEMA
{schema_text}

{source_label.upper()} BEGIN
{{{{input}}}}
{source_label.upper()} END
"""
    runbook = f"""AI SOC RUNBOOK: STRUCTURED EXTRACTION

Task: {task}
Evidence rule: {scope_rule}

Procedure:
1. Read the supplied {source_label.lower()}.
2. Populate every schema field from explicit in-scope evidence.
3. Keep all keys and literal values exact; retain list order.
4. Use null/[] when evidence is absent.
5. Emit only the JSON object, without markdown or commentary.

Required object:
{schema_text}

Evidence:
{{{{input}}}}
"""
    return {"prose": prose, "contract": contract, "runbook": runbook}


def _json_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _iso8601(timestamp: int | float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
