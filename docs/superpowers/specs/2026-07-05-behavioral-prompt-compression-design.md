# Behavioral Prompt Compression Compiler Design

## Goal

Build a tokenizer-aware exploration engine that takes a target model interface, an original prompt template with `{{input}}`, and input examples, then searches for shorter prompt templates that preserve the target model's behavior.

## Design Position

The system treats original model outputs as behavioral references, not objective labels. It optimizes `M(P', x) ~= M(P, x)` on held-out inputs while minimizing instruction tokens under the target tokenizer. Prompt similarity is not an objective.

The search space includes compact English, symbolic DSL, schema abbreviations, short Mandarin, classical-Chinese-like directives, Mandarin-symbolic hybrids, bilingual DSL, and mixed minimum-token forms from the first implementation. These are evaluated empirically rather than assumed to be good or bad.

## Architecture

The implementation is a small Python package with clear interfaces:

- model clients generate outputs under frozen generation parameters.
- tokenizers count prompt cost and make multilingual compression measurable.
- prompt templates render examples while preserving input placeholders.
- chunkers split invariant instructions into units that can be rewritten independently.
- rewrite operators produce compressed chunk variants.
- candidate generation assembles broad prompt candidates without a combinatorial explosion.
- evaluators compare candidate outputs to references through deterministic checks, lexical/embedding-like drift, optional judge hooks, language checks, and hard contract failures.
- Pareto selection keeps trade-offs visible instead of choosing a single winner too early.
- reports write prompts, frontier data, failures, reference traces, and candidate traces.

## Verification Strategy

Use lightweight deterministic checks for framework behavior: template rendering, placeholder protection, tokenizer-aware operator selection, contract penalties, Pareto logic, optimizer orchestration with a mock model, and artifact writing. These checks are not the center of the development process. Probabilistic live-model quality is handled by cached traces, frozen configs, exploratory runs, and integration traces, not brittle TDD assertions.
