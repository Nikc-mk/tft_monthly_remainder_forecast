---
name: tft-model-training
description: Design and review baseline models, TFT training, and rolling backtests for weekly sales forecasting by City. Use when Codex needs to configure the dataset, choose TFT hyperparameters within the project contract, evaluate metrics, or check whether model quality gates are satisfied.
---

# TFT Model Training

## Overview

Use this skill for model design and evaluation after the data contract is fixed. Keep the focus on baseline comparison, rolling weekly validation, and release gates instead of one-off leaderboard improvements.

## Required Sources

- Read `docs/project-spec.md`.
- Read `docs/technical-assignment.md`.
- Read `references/backtest-gates.md`.

## Workflow

1. Confirm that the input dataset matches the normalized weekly contract.
2. Define and evaluate the required baselines.
3. Configure the training dataset using the specified target, group id, encoder length, and feature groups.
4. Configure TFT with quantile outputs for `0.1`, `0.5`, and `0.9`.
5. Run rolling backtests on the last six weekly windows.
6. Report `SMAPE`, `MAE`, and coverage metrics against the acceptance gates.

## Required Training Rules

- Keep `max_encoder_length = 60`.
- Keep `max_prediction_length = 8`.
- Use `GroupNormalizer(groups=["City"])` unless the spec changes.
- Use quantile loss and `output_size = 3`.
- Preserve support for negative values with robust scaling behavior.
- Do not replace rolling backtests with a single validation split.

## Quality Gate

- `SMAPE <= 15%`
- at least `5%` SMAPE improvement over the best baseline
- quantile outputs available for `q0.1`, `q0.5`, and `q0.9`
- coverage metrics reported

## Output Expectations

Return:

- baseline design
- dataset config
- model config
- backtest design
- metrics
- gate decision
- artifact expectations
