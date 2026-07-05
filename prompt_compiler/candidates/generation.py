from __future__ import annotations

from prompt_compiler.candidates.candidate import Candidate, CompressedChunk
from prompt_compiler.candidates.genome import CandidateGenome
from prompt_compiler.operators.proposer import TokenizerAwareRewritePlanner
from prompt_compiler.operators.rewrite_ops import RewriteOperator
from prompt_compiler.prompt.assembly import assemble_candidate
from prompt_compiler.prompt.chunk import PromptChunk
from prompt_compiler.prompt.chunkers import generate_chunkings
from prompt_compiler.tokenizer import Tokenizer


BROAD_OPERATORS = (
    RewriteOperator.KEEP,
    RewriteOperator.SHORT_ENGLISH,
    RewriteOperator.TELEGRAPH_ENGLISH,
    RewriteOperator.SYMBOLIC_DSL,
    RewriteOperator.SCHEMA_ABBREVIATION,
    RewriteOperator.HYBRID_SYMBOLIC_ENGLISH,
    RewriteOperator.SHORT_MANDARIN,
    RewriteOperator.FORMAL_CHINESE,
    RewriteOperator.CLASSICAL_CHINESE_LIKE,
    RewriteOperator.MANDARIN_SYMBOLIC,
    RewriteOperator.BILINGUAL_DSL,
    RewriteOperator.MIXED_MIN_TOKEN_FORM,
)


def seed_population(prompt_template: str, tokenizer: Tokenizer, population_size: int) -> list[Candidate]:
    planner = TokenizerAwareRewritePlanner(tokenizer)
    population: list[Candidate] = []
    seen_prompts: set[str] = set()
    for chunker_name, chunks in generate_chunkings(prompt_template, tokenizer).items():
        _append_unique(
            population,
            seen_prompts,
            _candidate_from_uniform_operator(chunker_name, chunks, RewriteOperator.KEEP, tokenizer, planner),
        )
        if len(population) >= population_size:
            return population
        for operator in BROAD_OPERATORS[1:]:
            _append_unique(
                population,
                seen_prompts,
                _candidate_from_uniform_operator(chunker_name, chunks, operator, tokenizer, planner),
            )
            if len(population) >= population_size:
                return population
        _append_unique(population, seen_prompts, _candidate_from_min_tokens(chunker_name, chunks, tokenizer, planner))
        if len(population) >= population_size:
            return population
    return population[:population_size]


def _candidate_from_uniform_operator(
    chunker_name: str,
    chunks: list[PromptChunk],
    operator: RewriteOperator,
    tokenizer: Tokenizer,
    planner: TokenizerAwareRewritePlanner,
) -> Candidate:
    compressed: list[CompressedChunk] = []
    operator_map: dict[str, RewriteOperator] = {}
    for chunk in chunks:
        variant = next(item for item in planner.plan(chunk, (operator,)) if item.operator == operator)
        operator_map[chunk.id] = operator
        compressed.append(_compressed_chunk(chunk, variant.operator, variant.text, variant.token_count, variant.gloss, tokenizer))
    return _build_candidate(chunker_name, operator_map, compressed)


def _candidate_from_min_tokens(
    chunker_name: str,
    chunks: list[PromptChunk],
    tokenizer: Tokenizer,
    planner: TokenizerAwareRewritePlanner,
) -> Candidate:
    compressed: list[CompressedChunk] = []
    operator_map: dict[str, RewriteOperator] = {}
    for chunk in chunks:
        variants = [variant for variant in planner.plan(chunk, BROAD_OPERATORS) if variant.text.strip() or chunk.protected]
        if chunk.protected:
            variants = [variant for variant in variants if variant.text == chunk.text]
        variant = variants[0]
        operator_map[chunk.id] = variant.operator
        compressed.append(_compressed_chunk(chunk, variant.operator, variant.text, variant.token_count, variant.gloss, tokenizer))
    return _build_candidate(chunker_name, operator_map, compressed)


def _compressed_chunk(
    chunk: PromptChunk,
    operator: RewriteOperator,
    text: str,
    token_count: int,
    gloss: str,
    tokenizer: Tokenizer,
) -> CompressedChunk:
    return CompressedChunk(
        id=chunk.id,
        original_text=chunk.text,
        text=text,
        chunk_type=chunk.chunk_type,
        operator=operator,
        original_tokens=tokenizer.count(chunk.text),
        compressed_tokens=token_count,
        gloss=gloss,
        protected=chunk.protected,
    )


def _build_candidate(
    chunker_name: str,
    operator_map: dict[str, RewriteOperator],
    chunks: list[CompressedChunk],
    assembly_strategy: str = "newline",
) -> Candidate:
    prompt = assemble_candidate(chunks, strategy=assembly_strategy)
    if "{{input}}" not in prompt:
        prompt = f"{prompt}\nInput:\n{{{{input}}}}"
    genome = CandidateGenome(
        chunker_name=chunker_name,
        chunk_operator_map=operator_map,
        assembly_strategy=assembly_strategy,
    )
    return Candidate(genome=genome, chunks=tuple(chunks), prompt_template=prompt)


def _append_unique(population: list[Candidate], seen_prompts: set[str], candidate: Candidate) -> None:
    if candidate.prompt_template in seen_prompts:
        return
    seen_prompts.add(candidate.prompt_template)
    population.append(candidate)
