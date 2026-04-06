# Training Device Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move TFT training device selection into YAML config and make `gpu` requests fall back to `cpu` when CUDA is unavailable.

**Architecture:** Extend `training` config with `accelerator` and `devices`, validate the values in config loading, and resolve the actual Lightning trainer kwargs inside `pipeline/training.py`. Keep the runtime resilient by downgrading to `cpu` when `gpu` is requested but not available.

**Tech Stack:** Python, PyYAML, PyTorch, Darts, unittest

---

### Task 1: Cover Config Inputs

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/fixtures/pipeline_valid.yaml`
- Modify: `configs/pipeline.yaml`

- [ ] **Step 1: Write the failing test**

```python
def test_load_pipeline_config_reads_training_device_settings(self):
    path = Path("tests/fixtures/pipeline_valid.yaml")
    config = load_pipeline_config(path)
    self.assertEqual(config["training"]["accelerator"], "auto")
    self.assertEqual(config["training"]["devices"], 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_config.ConfigTests.test_load_pipeline_config_reads_training_device_settings -v`
Expected: FAIL because the fixture does not yet define the new keys.

- [ ] **Step 3: Write minimal implementation**

```yaml
training:
  accelerator: auto
  devices: 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_config.ConfigTests.test_load_pipeline_config_reads_training_device_settings -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_config.py tests/fixtures/pipeline_valid.yaml configs/pipeline.yaml
git commit -m "test: cover training device config"
```

### Task 2: Cover Trainer Fallback Logic

**Files:**
- Modify: `tests/test_training.py`
- Modify: `pipeline/training.py`
- Modify: `pipeline/config.py`

- [ ] **Step 1: Write the failing tests**

```python
@patch("pipeline.training.torch.cuda.is_available", return_value=False)
def test_tft_trainer_kwargs_fall_back_to_cpu_when_gpu_unavailable(self, _mock_cuda):
    config = _lightweight_tft_config()
    config["training"]["accelerator"] = "gpu"
    config["training"]["devices"] = 1

    trainer_kwargs = _tft_trainer_kwargs(config)

    self.assertEqual(trainer_kwargs["accelerator"], "cpu")
    self.assertEqual(trainer_kwargs["devices"], 1)
```

```python
@patch("pipeline.training.torch.cuda.is_available", return_value=True)
def test_tft_trainer_kwargs_use_gpu_when_available(self, _mock_cuda):
    config = _lightweight_tft_config()
    config["training"]["accelerator"] = "auto"
    config["training"]["devices"] = 1

    trainer_kwargs = _tft_trainer_kwargs(config)

    self.assertEqual(trainer_kwargs["accelerator"], "gpu")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_training.TrainingTests.test_tft_trainer_kwargs_fall_back_to_cpu_when_gpu_unavailable tests.test_training.TrainingTests.test_tft_trainer_kwargs_use_gpu_when_available -v`
Expected: FAIL because `_tft_trainer_kwargs()` does not yet read config or inspect CUDA availability.

- [ ] **Step 3: Write minimal implementation**

```python
requested_accelerator = str(config["training"].get("accelerator", "auto")).lower()
requested_devices = int(config["training"].get("devices", 1))

if requested_accelerator == "gpu" and not torch.cuda.is_available():
    accelerator = "cpu"
    devices = 1
elif requested_accelerator == "auto":
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = requested_devices if accelerator == "gpu" else 1
else:
    accelerator = requested_accelerator
    devices = requested_devices if accelerator == "gpu" else 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_training.TrainingTests.test_tft_trainer_kwargs_fall_back_to_cpu_when_gpu_unavailable tests.test_training.TrainingTests.test_tft_trainer_kwargs_use_gpu_when_available -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_training.py pipeline/training.py pipeline/config.py
git commit -m "feat: add configurable training device fallback"
```

### Task 3: Verify End-to-End Training Wiring

**Files:**
- Modify: `tests/test_training.py`
- Modify: `pipeline/training.py`

- [ ] **Step 1: Write the failing test**

```python
@patch("pipeline.training.TFTModel")
def test_train_tft_passes_resolved_trainer_kwargs(self, mock_tft_model):
    model_instance = MagicMock()
    mock_tft_model.return_value = model_instance
    config = _lightweight_tft_config()
    config["training"]["accelerator"] = "cpu"
    config["training"]["devices"] = 1

    train_tft(self.series_data, config)

    self.assertEqual(mock_tft_model.call_args.kwargs["pl_trainer_kwargs"]["accelerator"], "cpu")
    self.assertEqual(mock_tft_model.call_args.kwargs["pl_trainer_kwargs"]["devices"], 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_training.TrainingTests.test_train_tft_passes_resolved_trainer_kwargs -v`
Expected: FAIL if trainer kwargs are not threaded from config.

- [ ] **Step 3: Write minimal implementation**

```python
pl_trainer_kwargs=_tft_trainer_kwargs(config),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_training.TrainingTests.test_train_tft_passes_resolved_trainer_kwargs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_training.py pipeline/training.py
git commit -m "test: verify resolved trainer kwargs are wired into TFT"
```
