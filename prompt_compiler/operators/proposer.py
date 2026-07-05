from __future__ import annotations

import re
from dataclasses import dataclass

from prompt_compiler.operators.rewrite_ops import RewriteOperator
from prompt_compiler.prompt.chunk import ChunkType, PromptChunk
from prompt_compiler.tokenizer import Tokenizer


@dataclass(frozen=True)
class RewriteVariant:
    operator: RewriteOperator
    text: str
    token_count: int
    gloss: str


class RuleRewriteProposer:
    """Deterministic baseline proposer.

    Real deployments can replace this with an LLM proposer. Keeping this local
    proposer makes the compiler runnable and gives the optimizer broad initial
    search coverage.
    """

    def rewrite(self, chunk: PromptChunk, operator: RewriteOperator) -> tuple[str, str]:
        if chunk.protected:
            return chunk.text, "protected input slot preserved verbatim"
        if operator == RewriteOperator.KEEP:
            return chunk.text, "original chunk"
        if operator == RewriteOperator.DELETE:
            return "", "deleted chunk"
        if operator == RewriteOperator.SHORT_ENGLISH:
            return self._short_english(chunk), "compact English rewrite"
        if operator == RewriteOperator.TELEGRAPH_ENGLISH:
            return self._telegraph(chunk.text), "telegraphic English rewrite"
        if operator == RewriteOperator.SYMBOLIC_DSL:
            return self._symbolic(chunk), "symbolic DSL rewrite"
        if operator == RewriteOperator.SCHEMA_ABBREVIATION:
            return self._schema_abbrev(chunk), "schema abbreviation"
        if operator == RewriteOperator.HYBRID_SYMBOLIC_ENGLISH:
            return self._hybrid(chunk), "hybrid symbolic English"
        if operator == RewriteOperator.SHORT_MANDARIN:
            return self._short_mandarin(chunk), "short Mandarin instruction"
        if operator == RewriteOperator.FORMAL_CHINESE:
            return self._formal_chinese(chunk), "formal Chinese instruction"
        if operator == RewriteOperator.CLASSICAL_CHINESE_LIKE:
            return self._classical_like(chunk), "classical-Chinese-like compression"
        if operator == RewriteOperator.MANDARIN_SYMBOLIC:
            return self._mandarin_symbolic(chunk), "Mandarin-symbolic compression"
        if operator == RewriteOperator.BILINGUAL_DSL:
            return self._bilingual_dsl(chunk), "bilingual DSL compression"
        if operator == RewriteOperator.MIXED_MIN_TOKEN_FORM:
            return self._mixed_min(chunk), "mixed minimum-token form"
        if operator == RewriteOperator.EXAMPLE_DISTILLATION:
            return self._short_english(chunk), "example distilled to compact rule"
        if operator == RewriteOperator.RULE_EXTRACTION:
            return self._hybrid(chunk), "rule extraction"
        if operator == RewriteOperator.MERGE_WITH_PREVIOUS:
            return self._telegraph(chunk.text), "merge marker represented as compact text"
        raise ValueError(operator)

    def _short_english(self, chunk: PromptChunk) -> str:
        text = chunk.text
        replacements = {
            "You must return only valid JSON": "Return valid JSON only",
            "Do not include markdown": "No markdown",
            "The status field must be either OPEN or CLOSED": "status in {OPEN,CLOSED}",
            "You are doing": "Task:",
            "Return only": "Return",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return re.sub(r"\s+", " ", text).strip()

    def _telegraph(self, text: str) -> str:
        stop_words = {
            "you",
            "are",
            "the",
            "a",
            "an",
            "must",
            "should",
            "please",
            "only",
            "either",
            "be",
            "doing",
        }
        words = re.findall(r"\w+|[{}|,:;=∈]", text)
        kept = [word for word in words if word.lower() not in stop_words]
        return " ".join(kept)

    def _symbolic(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return "out=JSON; no_md; status∈{OPEN,CLOSED}"
        if chunk.chunk_type == ChunkType.NEGATIVE_CONSTRAINT:
            return "neg: preserve; no_unsupported_claims"
        if chunk.chunk_type == ChunkType.TASK:
            return "T=triage"
        return self._telegraph(chunk.text)

    def _schema_abbrev(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return "JSON{status:OPEN|CLOSED,summary,rationale[]}; md=0"
        return self._symbolic(chunk)

    def _hybrid(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return "OUT: JSON only; no_md; status∈{OPEN,CLOSED}"
        if chunk.chunk_type == ChunkType.TASK:
            return "Task=alert triage"
        return self._short_english(chunk)

    def _short_mandarin(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return "只返有效JSON；禁markdown；status取OPEN或CLOSED"
        if chunk.chunk_type == ChunkType.NEGATIVE_CONSTRAINT:
            return "禁无据断言"
        if chunk.chunk_type == ChunkType.TASK:
            return "任务：告警分流"
        return f"压缩义：{self._telegraph(chunk.text)}"

    def _formal_chinese(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return "输出须为有效JSON，不得含markdown；status字段限OPEN或CLOSED"
        if chunk.chunk_type == ChunkType.TASK:
            return "职责：执行告警分流"
        return f"须保持原义：{self._telegraph(chunk.text)}"

    def _classical_like(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return "须JSON；禁md；status∈OPEN|CLOSED"
        if chunk.chunk_type == ChunkType.NEGATIVE_CONSTRAINT:
            return "无据勿断"
        if chunk.chunk_type == ChunkType.TASK:
            return "任=警分"
        return f"守义：{self._telegraph(chunk.text)}"

    def _mandarin_symbolic(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return "出=EN/JSON; 禁md; status∈{OPEN,CLOSED}"
        if chunk.chunk_type == ChunkType.TASK:
            return "任=alert_triage"
        return f"守义; {self._symbolic(chunk)}"

    def _bilingual_dsl(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return "out=EN/json; 仅JSON; no_md; status=OPEN|CLOSED"
        if chunk.chunk_type == ChunkType.TASK:
            return "T=告警triage"
        return f"keep_meaning; {self._short_mandarin(chunk)}"

    def _mixed_min(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return "out=EN/json;只JSON;禁md;status∈OPEN|CLOSED"
        if chunk.chunk_type == ChunkType.NEGATIVE_CONSTRAINT:
            return "无证不claim"
        if chunk.chunk_type == ChunkType.TASK:
            return "T=triage"
        return self._symbolic(chunk)


class TokenizerAwareRewritePlanner:
    def __init__(self, tokenizer: Tokenizer, proposer: RuleRewriteProposer | None = None):
        self.tokenizer = tokenizer
        self.proposer = proposer or RuleRewriteProposer()

    def plan(
        self,
        chunk: PromptChunk,
        operators: tuple[RewriteOperator, ...] | None = None,
    ) -> list[RewriteVariant]:
        operators = operators or tuple(RewriteOperator)
        variants: list[RewriteVariant] = []
        seen: set[str] = set()
        for operator in operators:
            text, gloss = self.proposer.rewrite(chunk, operator)
            if text in seen:
                continue
            seen.add(text)
            variants.append(
                RewriteVariant(
                    operator=operator,
                    text=text,
                    token_count=self.tokenizer.count(text),
                    gloss=gloss,
                )
            )
        return sorted(variants, key=lambda item: (item.token_count, item.operator.value))

