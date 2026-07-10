from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from prompt_compiler.models.base import GenerateParams, ModelClient


DIMENSIONS = (
    "required_information",
    "instruction_compliance",
    "contradiction",
    "unsupported_addition",
    "format_language_style",
    "functional_equivalence",
)
SEVERITY_COSTS = {"none": 0.0, "minor": 0.25, "major": 0.70, "critical": 1.0}


def _dimension_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "none": {"type": "number", "minimum": 0, "maximum": 100},
            "minor": {"type": "number", "minimum": 0, "maximum": 100},
            "major": {"type": "number", "minimum": 0, "maximum": 100},
            "critical": {"type": "number", "minimum": 0, "maximum": 100},
            "finding": {"type": "string"},
        },
        "required": ["none", "minor", "major", "critical", "finding"],
    }


JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {name: _dimension_schema() for name in DIMENSIONS},
    "required": list(DIMENSIONS),
}


@dataclass(frozen=True)
class JudgeDimension:
    probabilities: dict[str, float]
    expected_severity: float
    finding: str


@dataclass(frozen=True)
class JudgePass:
    order: str
    loss: float
    dimensions: dict[str, JudgeDimension]


@dataclass(frozen=True)
class JudgeResult:
    loss: float
    raw_loss: float
    position_disagreement: float
    passes: tuple[JudgePass, JudgePass]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LLMSemanticJudge:
    model: ModelClient
    params: GenerateParams = field(default_factory=lambda: GenerateParams(max_tokens=4096, reasoning_effort="medium"))
    trace_path: Path | None = None
    calibration_floor: float = field(default=0.0, init=False)
    calibration_ceiling: float = field(default=1.0, init=False)

    def calibrate(self, *, original_prompt: str, references) -> dict[str, float]:
        floor_losses: list[float] = []
        ceiling_losses: list[float] = []
        calibration_references = list(references)[:3]
        for reference in calibration_references:
            floor_losses.append(
                self._judge_pair(
                    original_prompt=original_prompt,
                    input_text=reference.input_text,
                    reference_output=reference.reference_output,
                    candidate_output=reference.reference_output,
                    input_id=f"calibration:identical:{reference.id}",
                )[0]
            )
            ceiling_losses.append(
                self._judge_pair(
                    original_prompt=original_prompt,
                    input_text=reference.input_text,
                    reference_output=reference.reference_output,
                    candidate_output="",
                    input_id=f"calibration:empty:{reference.id}",
                )[0]
            )
        self.calibration_floor = sum(floor_losses) / max(len(floor_losses), 1)
        self.calibration_ceiling = sum(ceiling_losses) / max(len(ceiling_losses), 1)
        if self.calibration_ceiling <= self.calibration_floor:
            self.calibration_floor = 0.0
            self.calibration_ceiling = 1.0
        return {
            "floor": self.calibration_floor,
            "ceiling": self.calibration_ceiling,
            "span": self.calibration_ceiling - self.calibration_floor,
        }

    def judge(
        self,
        *,
        original_prompt: str,
        input_text: str,
        reference_output: str,
        candidate_output: str,
        input_id: str,
    ) -> JudgeResult:
        raw_loss, disagreement, passes = self._judge_pair(
            original_prompt=original_prompt,
            input_text=input_text,
            reference_output=reference_output,
            candidate_output=candidate_output,
            input_id=input_id,
        )
        calibrated = _clamp(
            (raw_loss - self.calibration_floor)
            / max(self.calibration_ceiling - self.calibration_floor, 1e-9)
        )
        return JudgeResult(
            loss=calibrated,
            raw_loss=raw_loss,
            position_disagreement=disagreement,
            passes=passes,
        )

    def _judge_pair(self, *, original_prompt, input_text, reference_output, candidate_output, input_id):
        passes = (
            self._judge_once(
                order="reference_first",
                original_prompt=original_prompt,
                input_text=input_text,
                first_label="REFERENCE OUTPUT",
                first_text=reference_output,
                second_label="CANDIDATE OUTPUT",
                second_text=candidate_output,
                input_id=input_id,
            ),
            self._judge_once(
                order="candidate_first",
                original_prompt=original_prompt,
                input_text=input_text,
                first_label="CANDIDATE OUTPUT",
                first_text=candidate_output,
                second_label="REFERENCE OUTPUT",
                second_text=reference_output,
                input_id=input_id,
            ),
        )
        raw_loss = sum(item.loss for item in passes) / 2.0
        return raw_loss, abs(passes[0].loss - passes[1].loss), passes

    def _judge_once(
        self,
        *,
        order: str,
        original_prompt: str,
        input_text: str,
        first_label: str,
        first_text: str,
        second_label: str,
        second_text: str,
        input_id: str,
    ) -> JudgePass:
        prompt = _judge_prompt(
            original_prompt=original_prompt,
            input_text=input_text,
            first_label=first_label,
            first_text=first_text,
            second_label=second_label,
            second_text=second_text,
        )
        params = replace(
            self.params,
            response_json_schema=JUDGE_SCHEMA,
            response_json_schema_name="semantic_equivalence_judgment",
        )
        response = self.model.generate(prompt, params)
        parsed = _extract_json(response.text)
        dimensions = {name: _parse_dimension(parsed.get(name, {})) for name in DIMENSIONS}
        loss = sum(item.expected_severity for item in dimensions.values()) / len(dimensions)
        result = JudgePass(order=order, loss=_clamp(loss), dimensions=dimensions)
        self._write_trace(input_id, prompt, response.text, parsed, result, response.usage, response.metadata)
        return result

    def _write_trace(self, input_id, prompt, response, parsed, result, usage, metadata) -> None:
        if not self.trace_path:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "input_id": input_id,
            "judge_model": self.model.name,
            "judge_prompt": prompt,
            "judge_response": response,
            "parsed_response": parsed,
            "result": asdict(result),
            "usage": usage,
            "response_metadata": metadata,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _judge_prompt(*, original_prompt, input_text, first_label, first_text, second_label, second_text) -> str:
    return f"""Evaluate only semantic and behavioral divergence between the candidate output and reference output for this exact prompt and input.

Do not evaluate prompt length, token reduction, compression style, or which output sounds more polished. The reference is the behavioral anchor, but do not assume surface wording must match.

Reason carefully before assigning probabilities. For each dimension, return percentages for none/minor/major/critical that sum to 100 and one concise evidence-based finding.

Severity anchors:
- none: functionally equivalent; wording-only differences.
- minor: limited difference unlikely to change the central answer or required behavior.
- major: meaningful omission, unsupported addition, instruction failure, or behavior change.
- critical: contradiction, unusable output, central-answer reversal, or loss of a mandatory contract.

Dimensions:
- required_information: required content missing or materially altered.
- instruction_compliance: divergence from explicit prompt requirements.
- contradiction: claims or decisions conflicting with the reference.
- unsupported_addition: consequential content absent from the reference or unsupported by the input.
- format_language_style: only requirements explicitly imposed by the original prompt.
- functional_equivalence: whether a downstream user receives materially equivalent utility and behavior.

ORIGINAL PROMPT BEGIN
{original_prompt}
ORIGINAL PROMPT END

INPUT BEGIN
{input_text}
INPUT END

{first_label} BEGIN
{first_text}
{first_label} END

{second_label} BEGIN
{second_text}
{second_label} END
"""


def _parse_dimension(value: object) -> JudgeDimension:
    row = value if isinstance(value, dict) else {}
    raw = {key: max(float(row.get(key, 0.0)), 0.0) for key in SEVERITY_COSTS}
    total = sum(raw.values())
    probabilities = (
        {key: amount / total for key, amount in raw.items()}
        if total > 0
        else {"none": 0.0, "minor": 0.0, "major": 0.0, "critical": 1.0}
    )
    expected = sum(probabilities[key] * SEVERITY_COSTS[key] for key in SEVERITY_COSTS)
    return JudgeDimension(
        probabilities=probabilities,
        expected_severity=_clamp(expected),
        finding=str(row.get("finding", "")).strip(),
    )


def _extract_json(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)
