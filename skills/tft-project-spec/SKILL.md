---
name: tft-project-spec
description: Turn the weekly City forecasting requirements into implementation scopes, acceptance criteria, review checklists, and delivery slices. Use when Codex needs to interpret the stored technical assignment, verify a plan against project rules, or answer requirement questions before coding.
---

# TFT Project Spec

## Overview

Use this skill to ground work in the repository spec before implementation starts. Read the stored technical assignment and project spec first, then translate them into decisions, constraints, and handoffs without changing the business contract.

## Required Sources

- Read `docs/technical-assignment.md`.
- Read `docs/project-spec.md`.
- Read `references/spec-workflow.md` for the compact checklist.

## Workflow

1. Extract invariant rules before proposing work.
2. State the user goal in project terms: data prep, training, inference, monitoring, or documentation.
3. Map the request to acceptance criteria and constraints.
4. Split the work into delivery slices only after the invariants are explicit.
5. If a request conflicts with the stored spec, call out the conflict and propose the smallest compliant path.

## What To Protect

- The only valid target is weekly `sum(revenue)` by `City` for future ISO weeks.
- Leakage is never acceptable in target construction or feature engineering.
- Rolling weekly backtesting is mandatory.
- TFT must beat the best baseline by at least 5 percent on SMAPE.
- Inference must stay batch-oriented and non-autoregressive.

## Output Expectations

When planning or reviewing work, produce:

- objective
- scope boundaries
- invariants
- acceptance criteria
- assumptions and handoffs

## Typical Triggers

- "Turn the technical assignment into implementation tasks."
- "Check whether this plan violates the forecasting spec."
- "List acceptance criteria for the weekly City pipeline."
- "Review whether a proposed refactor still matches the project contract."
