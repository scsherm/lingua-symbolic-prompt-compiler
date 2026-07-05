from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_-]*)\s*\}\}")


@dataclass(frozen=True)
class PromptTemplate:
    text: str

    def variables(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(match.group(1) for match in _PLACEHOLDER_RE.finditer(self.text)))

    def render(self, values: Mapping[str, object]) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in values:
                raise KeyError(f"Missing prompt variable: {name}")
            return str(values[name])

        return _PLACEHOLDER_RE.sub(replace, self.text)

    def instruction_text(self) -> str:
        return _PLACEHOLDER_RE.sub("", self.text)

