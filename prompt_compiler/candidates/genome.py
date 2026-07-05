from __future__ import annotations

from dataclasses import dataclass

from prompt_compiler.operators.rewrite_ops import RewriteOperator


@dataclass(frozen=True)
class CandidateGenome:
    chunker_name: str
    chunk_operator_map: dict[str, RewriteOperator]
    assembly_strategy: str = "newline"
    global_language_policy: str = "out_lang=EN"
    symbol_density: float = 0.5
    preserve_negations: bool = True
    preserve_output_language: bool = True

