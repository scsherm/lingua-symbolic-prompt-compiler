from __future__ import annotations

from typing import Iterable

from prompt_compiler.candidates.candidate import CompressedChunk


def assemble_candidate(chunks: Iterable[CompressedChunk], strategy: str = "newline") -> str:
    texts = [chunk.text.strip() for chunk in chunks if chunk.text.strip()]
    if strategy == "newline":
        return "\n".join(texts)
    if strategy == "compact_dsl":
        return "; ".join(texts)
    if strategy == "sectioned":
        return "\n".join(f"[{chunk.chunk_type.value}] {chunk.text.strip()}" for chunk in chunks if chunk.text.strip())
    raise ValueError(f"Unknown assembly strategy: {strategy}")

