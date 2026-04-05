# TFT Scaling Runtime Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore per-city TFT forecast scale in the weekly runtime so forecasts no longer collapse to near-constant values across cities.

**Architecture:** Keep the existing `pipeline/` flow, but wrap TFT training and prediction with explicit scaling and inverse-transform logic. Preserve the current city-keyed series contract so backtest, holdout, and forecast code can keep calling `predict()` through a runtime wrapper instead of learning about scaler internals.

**Tech Stack:** Python, Darts TFTModel, pandas, numpy, scikit-learn-compatible scaling via Darts transformers

---

### Task 1: Lock in the regression

**Files:**
- Modify: `tests/test_inference.py`
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write the failing test**

Add a test that trains the lightweight TFT on `tests/fixtures/raw_sales_small.csv`, generates the forecast, compares mean `q0.5` forecast magnitude per city to the mean of the latest observed weeks, and asserts the ratio stays above a small floor such as `0.1`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_inference.InferenceTests.test_generate_forecast_preserves_city_scale`
Expected: FAIL because the current runtime TFT predicts values near zero relative to recent history.

### Task 2: Fix TFT runtime scaling

**Files:**
- Modify: `pipeline/training.py`
- Modify: `pipeline/inference.py`
- Test: `tests/test_training.py`

- [ ] **Step 1: Add scaling helpers and wrapper state**

Introduce focused helpers for scaling target, past covariates, future covariates, and static city codes before fitting TFT, then store the fitted transformers alongside the trained model or wrapper.

- [ ] **Step 2: Keep inference/backtest on original scale**

Ensure `predict()` returns inverse-transformed forecasts so `run_backtest()`, `evaluate_holdout()`, and `generate_forecast()` continue operating in original revenue units without changing their external contract.

- [ ] **Step 3: Add a unit-level regression guard**

Add a test in `tests/test_training.py` that verifies runtime TFT predictions stay in the same order of magnitude as recent history on the small fixture.

### Task 3: Verify end to end

**Files:**
- Test: `tests/test_training.py`
- Test: `tests/test_inference.py`

- [ ] **Step 1: Run targeted tests**

Run: `python -m unittest tests.test_training tests.test_inference`
Expected: PASS

- [ ] **Step 2: Re-check the original regression**

Run the single regression test again to confirm the red-green cycle completed with the intended behavior.
