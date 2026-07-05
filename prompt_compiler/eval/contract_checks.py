from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContractCheckResult:
    ok: bool
    failures: tuple[str, ...] = ()
    parsed_json: Any | None = None


@dataclass(frozen=True)
class OutputContract:
    require_json: bool = False
    required_fields: tuple[str, ...] = ()
    status_values: tuple[str, ...] = ("OPEN", "CLOSED")
    output_language: str | None = "EN"
    hard_penalty: float = 10.0
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self, output: str) -> ContractCheckResult:
        failures: list[str] = []
        parsed: Any | None = None
        if self.require_json or self.required_fields:
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                failures.append("invalid_json")
                return ContractCheckResult(ok=False, failures=tuple(failures), parsed_json=None)
            if not isinstance(parsed, dict):
                failures.append("json_not_object")
            for field_name in self.required_fields:
                if not isinstance(parsed, dict) or field_name not in parsed:
                    failures.append(f"missing_field:{field_name}")
            if isinstance(parsed, dict) and "status" in parsed and parsed["status"] not in self.status_values:
                failures.append("invalid_status")
        if self.output_language == "EN" and _cjk_ratio(output) > 0.2:
            failures.append("language_drift")
        return ContractCheckResult(ok=not failures, failures=tuple(failures), parsed_json=parsed)


def _cjk_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    cjk = [char for char in chars if re.match(r"[\u3400-\u9fff]", char)]
    return len(cjk) / len(chars)

