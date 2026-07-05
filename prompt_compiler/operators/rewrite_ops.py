from __future__ import annotations

from enum import Enum


class RewriteOperator(Enum):
    KEEP = "keep"
    DELETE = "delete"
    SHORT_ENGLISH = "short_english"
    TELEGRAPH_ENGLISH = "telegraph_english"
    SYMBOLIC_DSL = "symbolic_dsl"
    SCHEMA_ABBREVIATION = "schema_abbreviation"
    FORMAL_CHINESE = "formal_chinese"
    CLASSICAL_CHINESE_LIKE = "classical_chinese_like"
    SHORT_MANDARIN = "short_mandarin"
    MANDARIN_SYMBOLIC = "mandarin_symbolic"
    BILINGUAL_DSL = "bilingual_dsl"
    HYBRID_SYMBOLIC_ENGLISH = "hybrid_symbolic_english"
    MIXED_MIN_TOKEN_FORM = "mixed_min_token_form"
    EXAMPLE_DISTILLATION = "example_distillation"
    RULE_EXTRACTION = "rule_extraction"
    MERGE_WITH_PREVIOUS = "merge_with_previous"

