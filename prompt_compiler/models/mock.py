from __future__ import annotations

import json
from dataclasses import dataclass

from prompt_compiler.models.base import GenerateParams, ModelResponse


@dataclass
class RuleBasedMockModel:
    """Deterministic model client for local compiler checks and examples."""

    name: str = "rule-based-mock"

    def config(self) -> dict:
        return {"name": self.name, "type": "deterministic-mock"}

    def generate(self, prompt: str, params: GenerateParams) -> ModelResponse:
        lowered = prompt.lower()
        risky_terms = ("malware", "credential", "theft", "beacon", "exfil", "ransom")
        status = "OPEN" if any(term in lowered for term in risky_terms) else "CLOSED"

        wants_json = "json" in lowered or "status" in lowered or "状" in prompt or "出" in prompt
        if wants_json:
            text = json.dumps(
                {
                    "status": status,
                    "summary": "Risk signal found." if status == "OPEN" else "No strong risk signal.",
                    "rationale": ["matched deterministic mock rule"],
                },
                sort_keys=True,
            )
        else:
            text = f"status={status}"

        return ModelResponse(text=text, model=self.name, params=params, usage=None, metadata={"mock": True})

