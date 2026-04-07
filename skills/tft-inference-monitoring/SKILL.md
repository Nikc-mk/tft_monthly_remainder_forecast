---
name: tft-inference-monitoring
description: Design and review batch inference, runtime artifacts, SLA checks, and monitoring for weekly sales forecasts by City. Use when Codex needs to specify prediction outputs, load model artifacts safely, handle cold-start cases, or verify monitoring and runtime behavior against the project rules.
---

# TFT Inference Monitoring

## Overview

Use this skill for the runtime path after model training. Optimize for direct batch prediction of weekly revenue, explicit artifact contracts, and operational checks that detect quality or distribution drift.

## Required Sources

- Read `docs/project-spec.md`.
- Read `docs/technical-assignment.md`.
- Read `references/ops-checklist.md`.

## Workflow

1. Confirm the runtime artifacts: model checkpoint, scaler config, feature pipeline config, and inference entrypoint.
2. Design batch inference across all `City` values at once.
3. Keep prediction direct on the weekly target, not autoregressive over future weeks.
4. Define the output contract as one row per `City + target_week` with `forecast_week`, `target_week`, `horizon_week`, `City`, `q0.1`, `q0.5`, `q0.9`, `model_version`, `feature_version`, and `generated_at`.
5. Confirm the required runtime outputs: `forecast.csv`, `holdout_predictions.csv`, `monitoring_summary.json`, and the monitoring PNG artifacts from the stored spec.
6. Add handling for negative values, missing weeks, and cold-start cities.
7. Define monitoring for drift, SMAPE degradation, and quantile coverage.

## Runtime Rules

- Use batch inference only.
- Do not loop city by city as the core design.
- Do not rebuild the dataset separately for every `City` in production-style flows.
- Keep CPU runtime under 10 seconds for all `City`.
- Preserve support for negative revenue values.
- Keep cold-start fallback explicit and documented.

## Monitoring Rules

- track drift on weekly revenue
- track drift on City distribution
- track SMAPE degradation
- track quantile coverage

## Output Expectations

Return:

- runtime inputs
- batch flow
- output schema
- fallback behavior
- SLA checks
- monitoring checks
- operational risks
