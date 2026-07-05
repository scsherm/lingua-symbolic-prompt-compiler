from __future__ import annotations

from dataclasses import dataclass, field

from prompt_compiler.candidates.genome import CandidateGenome
from prompt_compiler.hashing import stable_hash
from prompt_compiler.operators.rewrite_ops import RewriteOperator
from prompt_compiler.prompt.chunk import ChunkType


@dataclass(frozen=True)
class CompressedChunk:
    id: str
    original_text: str
    text: str
    chunk_type: ChunkType
    operator: RewriteOperator
    original_tokens: int
    compressed_tokens: int
    gloss: str = ""
    protected: bool = False

    @property
    def token_delta(self) -> int:
        return self.original_tokens - self.compressed_tokens


@dataclass(frozen=True)
class Candidate:
    genome: CandidateGenome
    chunks: tuple[CompressedChunk, ...]
    prompt_template: str
    id: str = field(default="")

    def __post_init__(self) -> None:
        if self.id:
            return
        object.__setattr__(
            self,
            "id",
            stable_hash(
                {
                    "chunker": self.genome.chunker_name,
                    "assembly": self.genome.assembly_strategy,
                    "prompt": self.prompt_template,
                    "operators": {key: value.value for key, value in self.genome.chunk_operator_map.items()},
                }
            )[:12],
        )

