# Agents Guide

## Purpose

This file defines the shared expectations for any agent working in this repository.

The current source of truth is:
- `docs/project-spec.md`
- `docs/technical-assignment.md`

If older agent notes or implementation comments contradict these documents, follow the docs above.

## Current Project Contract

The repository is now specified as a weekly sales forecasting pipeline at `City` level on a fixed `8`-week horizon.

Core target:
- `target(city, W, h) = weekly_revenue(city, W+h)` for `h in {1..8}`
- `weekly_revenue` is weekly `sum(revenue)` on ISO weeks (`Mon-Sun`)
- `W` is the last fully closed week

Required forecast contract:
- one output row per `City` and `target_week`
- quantiles `q0.1`, `q0.5`, `q0.9`
- fields: `forecast_week`, `target_week`, `horizon_week`, `City`, `q0.1`, `q0.5`, `q0.9`, `model_version`, `feature_version`, `generated_at`

## Cross-Cutting Rules

- Do not reintroduce monthly remainder logic into specs, reviews, or new designs.
- Do not use future-aware features.
- Every lag or rolling feature based on `weekly_revenue` must be built only after `shift(1)`.
- Validation must use rolling weekly backtests, not a single random or temporal split.
- Baselines must be evaluated on the same `8`-week weekly horizon as TFT.
- Inference must stay batch-oriented and should produce a direct multi-horizon forecast.
- Negative revenue values are valid; do not require log-transform.
- Use a fixed history window of `60` weeks.
- For library, framework, and API usage questions, use Context7 documentation by default when available.

## Agent Roles

### Architect Project Agent

**Ownership:** Spec + Data + Feature + Code Validation

**Responsibilities:**
- Requirement decomposition and acceptance criteria
- Data normalization contract and target definitions
- Feature engineering definitions
- Output schema and public contracts
- All documentation in `docs/`
- Validate that implementation code matches the specification

**Hard rules:**
- `8`-week weekly forecast: `target(city, W, h) = weekly_revenue(city, W+h)` for `h in {1..8}`
- Use ISO weeks and only the last fully closed week as cutoff
- Reject future-aware features or target leakage
- Require rolling weekly backtesting
- Enforce output contract: one row per `City` and `target_week` with quantiles `q0.1`, `q0.5`, `q0.9`
- Only Architect can modify feature definitions
- Quality gates are defined in `docs/project-spec.md` and must stay aligned

**Handoff to ML Agent:**
1. Objective
2. Invariants (from spec)
3. Data contract (input/output schema)
4. Feature definitions
5. Acceptance criteria (reference to `docs/project-spec.md`)
6. Implementation notes

### ML Agent

**Ownership:** Training + Backtest + Inference + Monitoring + Results Validation

**Responsibilities:**
- Baseline evaluation and TFT training design
- Rolling backtest execution and metric calculation
- Batch inference path and runtime artifact usage
- Monitoring for drift, SMAPE degradation, and quantile coverage
- Request feature changes from Architect Project Agent when needed

**Hard rules:**
- Compare TFT against both baselines: `seasonal_naive` and `rolling_mean`
- Use rolling backtest with `6` weekly test windows
- Keep `max_encoder_length = 60`, `max_prediction_length = 8`
- Use quantile loss with outputs for `0.1`, `0.5`, and `0.9`
- Enforce quality gates from `docs/project-spec.md`: `SMAPE <= 15%` and at least `5%` improvement over the best baseline
- Run inference in batch across all `City`, under `10` seconds CPU SLA
- Do not modify feature definitions without Architect approval

**Handoff format:**
1. Baselines
2. Dataset config
3. Model config
4. Backtest design
5. Metrics and gates (reference to `docs/project-spec.md`)
6. Artifact expectations
7. Monitoring results

## Working Agreement

- Prefer updating docs first when changing public contracts.
- If code still reflects an older contract, say so explicitly in docs and reviews.
- Treat the files in `agents/` as helpful role notes, but not as the final authority if they conflict with the current spec.
- When handing work to another agent or engineer, state assumptions, invariants, and the exact output schema.

## Known Repository State

- The documentation contract is now aligned to weekly `City` forecasting on an `8`-week horizon.
- Parts of the runtime code and legacy notes may still reflect older daily or monthly logic.
- Any future implementation work should align code and tests to the current weekly spec instead of preserving older targets.
