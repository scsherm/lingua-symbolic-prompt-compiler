from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from prompt_compiler.hashing import stable_hash, stable_json
from prompt_compiler.models.base import GenerateParams, ModelResponse


@dataclass(frozen=True)
class ModelCallKey:
    model_config_hash: str
    prompt_hash: str
    input_id: str
    params_hash: str

    @classmethod
    def from_call(
        cls,
        *,
        model_config: dict,
        prompt: str,
        input_id: str,
        params: GenerateParams,
    ) -> "ModelCallKey":
        return cls(
            model_config_hash=stable_hash(model_config),
            prompt_hash=stable_hash(prompt),
            input_id=input_id,
            params_hash=stable_hash(asdict(params)),
        )

    def digest(self) -> str:
        return stable_hash(asdict(self))


class ModelCallCache:
    def __init__(self, path: Path | None = None):
        self.path = path
        self._records: dict[str, dict] = {}
        if path and path.exists():
            self._records = json.loads(path.read_text(encoding="utf-8"))

    def get(self, key: ModelCallKey) -> ModelResponse | None:
        record = self._records.get(key.digest())
        if not record:
            return None
        return ModelResponse(**record)

    def set(self, key: ModelCallKey, response: ModelResponse) -> None:
        self._records[key.digest()] = asdict(response)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(stable_json(self._records), encoding="utf-8")

    def get_or_generate(self, key: ModelCallKey, generate: Callable[[], ModelResponse]) -> ModelResponse:
        cached = self.get(key)
        if cached is not None:
            return cached
        response = generate()
        self.set(key, response)
        return response

