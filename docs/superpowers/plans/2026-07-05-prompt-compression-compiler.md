# Prompt Compression Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first working tokenizer-aware behavioral prompt compression compiler.

**Architecture:** A small Python package with protocols for models/tokenizers, deterministic chunkers and rewrite operators, candidate generation, evaluation, Pareto selection, optimization orchestration, and report writing. The first runnable system uses a deterministic mock model and adapter interfaces for real model clients.

**Tech Stack:** Python 3.11+, standard library only for the core package, `unittest` only for lightweight deterministic framework checks.

---

### Task 1: Repository Skeleton and Deterministic Checks

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `tests/test_template_and_chunking.py`
- Create: `tests/test_rewrite_and_pareto.py`
- Create: `tests/test_evaluator_and_optimizer.py`

- [ ] Add executable checks for template rendering, placeholder protection, multilingual operator generation, Pareto filtering, hard contract penalties, and optimizer artifacts.
- [ ] Use these checks as regression/smoke coverage for deterministic plumbing, not as a test-first gate for the experimental optimizer.

### Task 2: Core Interfaces

**Files:**
- Create: `prompt_compiler/models/base.py`
- Create: `prompt_compiler/models/mock.py`
- Create: `prompt_compiler/tokenizer.py`
- Create: `prompt_compiler/prompt/template.py`
- Create: `prompt_compiler/cache.py`
- Create: `prompt_compiler/hashing.py`

- [ ] Implement frozen generation parameters, model response/config dataclasses, tokenizer protocol, approximate tokenizer, prompt rendering, stable hashing, and JSON-backed model call cache.
- [ ] Run targeted tests for template and tokenizer behavior.

### Task 3: Chunking and Rewrite Operators

**Files:**
- Create: `prompt_compiler/prompt/chunk.py`
- Create: `prompt_compiler/prompt/chunkers.py`
- Create: `prompt_compiler/prompt/assembly.py`
- Create: `prompt_compiler/operators/rewrite_ops.py`
- Create: `prompt_compiler/operators/proposer.py`

- [ ] Implement chunk metadata, paragraph/sentence/markdown/schema/role/token-window chunkers, assembly strategies, broad rewrite operator enum, deterministic rule proposer, and tokenizer-aware rewrite planner.
- [ ] Run rewrite and chunking tests.

### Task 4: Candidates, Evaluation, and Pareto Frontier

**Files:**
- Create: `prompt_compiler/candidates/genome.py`
- Create: `prompt_compiler/candidates/candidate.py`
- Create: `prompt_compiler/candidates/generation.py`
- Create: `prompt_compiler/eval/contract_checks.py`
- Create: `prompt_compiler/eval/embedding_distance.py`
- Create: `prompt_compiler/eval/evaluator.py`
- Create: `prompt_compiler/eval/pareto.py`

- [ ] Implement candidate reports, seed population generation, contract checks, lexical drift fallback, evaluator loss, failure cases, and frontier filtering.
- [ ] Run evaluator and Pareto tests.

### Task 5: Optimizer, Reports, and CLI

**Files:**
- Create: `prompt_compiler/data/dataset_builder.py`
- Create: `prompt_compiler/data/splits.py`
- Create: `prompt_compiler/optimize/curriculum.py`
- Create: `prompt_compiler/optimize/credit_assignment.py`
- Create: `prompt_compiler/optimize/optimizer.py`
- Create: `prompt_compiler/reports/writer.py`
- Create: `prompt_compiler/cli.py`
- Create: `README.md`

- [ ] Implement reference building, deterministic splits, progressive subsets, basic operator diagnostics, optimization loop, artifact writing, and a CLI that can run with the mock model.
- [ ] Run the full test suite.
- [ ] Commit the working implementation.
