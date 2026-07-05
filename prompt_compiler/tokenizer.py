from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol


class Tokenizer(Protocol):
    name: str

    def tokens(self, text: str) -> list[str]:
        ...

    def count(self, text: str) -> int:
        ...


@dataclass(frozen=True)
class TokenProfile:
    text: str
    token_count: int
    tokenizer: str


class ApproxTokenizer:
    """Small local tokenizer used when no model tokenizer adapter is supplied.

    It is not a substitute for the target model tokenizer. It exists so the
    compiler can run locally and so tokenizer-aware code has a stable protocol.
    """

    name = "approx"

    _token_re = re.compile(
        r"[\u3400-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z0-9]",
        re.UNICODE,
    )

    def tokens(self, text: str) -> list[str]:
        return self._token_re.findall(text)

    def count(self, text: str) -> int:
        return len(self.tokens(text))

    def profile(self, text: str) -> TokenProfile:
        return TokenProfile(text=text, token_count=self.count(text), tokenizer=self.name)


class TiktokenTokenizer:
    """Optional adapter for OpenAI/tiktoken-compatible token counting."""

    def __init__(self, *, encoding_name: str | None = None, model_name: str | None = None):
        if not encoding_name and not model_name:
            raise ValueError("Provide encoding_name or model_name")
        try:
            tiktoken = import_module("tiktoken")
        except ModuleNotFoundError as exc:
            raise RuntimeError("tiktoken is not installed; install it or use ApproxTokenizer") from exc
        if model_name:
            self._encoding = tiktoken.encoding_for_model(model_name)
            self.name = f"tiktoken:model:{model_name}"
        else:
            self._encoding = tiktoken.get_encoding(encoding_name)
            self.name = f"tiktoken:{encoding_name}"

    def tokens(self, text: str) -> list[str]:
        return [str(token) for token in self._encoding.encode(text)]

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))


def make_tokenizer(spec: str | None) -> Tokenizer:
    if not spec or spec == "approx":
        return ApproxTokenizer()
    if spec.startswith("tiktoken:"):
        return TiktokenTokenizer(encoding_name=spec.removeprefix("tiktoken:"))
    if spec.startswith("model:"):
        return TiktokenTokenizer(model_name=spec.removeprefix("model:"))
    raise ValueError(f"Unknown tokenizer spec: {spec}")
