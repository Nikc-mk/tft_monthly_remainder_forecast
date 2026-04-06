# TFT Without Past Covariates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove lag and rolling features from the TFT pipeline while keeping weekly baselines unchanged.

**Architecture:** Keep `City` target history plus known future covariates as the only TFT inputs. Remove generated lag/rolling columns, stop building `past_covariates_by_city`, and update training, inference, monitoring, and tests to treat past covariates as optional/absent for TFT.

**Tech Stack:** Python, pandas, Darts TFTModel, unittest

---

### Task 1: Lock the new contract with failing tests

**Files:**
- Modify: `tests/test_data_pipeline.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Write the failing tests**

Add assertions that training frames no longer contain `lag_*` or `rolling_*`, `build_darts_series()` returns `past_covariates_by_city=None`, and TFT prediction paths do not pass `past_covariates`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_data_pipeline tests.test_training`
Expected: FAIL because code still creates lag/rolling features and training still expects `past_covariates`.

### Task 2: Remove past features from the data pipeline

**Files:**
- Modify: `pipeline/data_pipeline.py`
- Modify: `docs/project-spec.md`
- Modify: `docs/technical-assignment.md`

- [ ] **Step 1: Write the minimal implementation**

Remove lag/rolling feature generation, make `PAST_FEATURES` empty, stop materializing `past_covariates` when no past features exist, and document that TFT now uses only target history plus known future covariates.

- [ ] **Step 2: Run focused tests**

Run: `python -m unittest tests.test_data_pipeline`
Expected: PASS

### Task 3: Make TFT runtime paths work without past covariates

**Files:**
- Modify: `pipeline/training.py`
- Modify: `pipeline/inference.py`
- Modify: `pipeline/monitoring.py`
- Modify: `tests/test_inference.py`
- Modify: `tests/test_training.py`
- Modify: `tests/test_monitoring.py`

- [ ] **Step 1: Write the minimal implementation**

Treat `past_covariates` as optional in training, prediction, backtest, holdout evaluation, runtime scaling, inference, and residual monitoring.

- [ ] **Step 2: Run verification**

Run: `python -m unittest tests.test_training tests.test_inference tests.test_monitoring`
Expected: PASS

### Task 4: Final regression verification

**Files:**
- Modify: `pipeline/data_pipeline.py`
- Modify: `pipeline/training.py`
- Modify: `pipeline/inference.py`
- Modify: `pipeline/monitoring.py`
- Modify: `tests/test_data_pipeline.py`
- Modify: `tests/test_training.py`
- Modify: `tests/test_inference.py`
- Modify: `tests/test_monitoring.py`
- Modify: `docs/project-spec.md`
- Modify: `docs/technical-assignment.md`

- [ ] **Step 1: Run full targeted regression suite**

Run: `python -m unittest tests.test_data_pipeline tests.test_training tests.test_inference tests.test_monitoring`
Expected: PASS
