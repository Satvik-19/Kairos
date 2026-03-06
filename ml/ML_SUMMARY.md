# KAIROS Entropy Engine — ML Pipeline

> A three-model ensemble that monitors, scores, and classifies the quality of entropy
> produced by live Python chaos engines (double-pendulum + Lorenz attractor).

---

## Why Machine Learning on an Entropy Engine?

KAIROS generates cryptographic-quality entropy by mixing output from physically-chaotic
simulations. The ML pipeline answers three distinct questions in real time:

| Question | Model | Output |
|---|---|---|
| *Does the current byte stream look anomalous?* | Hybrid Anomaly Detector (Model 1) | `anomaly_score`, `is_anomaly` |
| *Can the hash output be predicted?* | LSTM Predictor (Model 2) | `prediction_resistance` |
| *What quality grade is the entropy right now?* | MLP Classifier (Model 4) | `health_status`, `health_confidence` |

All three models run as a single `KairosMLInference` object, called every 50 ms from the
engine feed loop. The system degrades gracefully to a rule-based fallback when models are
not yet trained.

---

## Dataset

Training data was collected **live** from a running KAIROS instance. Crucially the data
includes **chaos engine state features** (pendulum angles/velocities, Lorenz coordinates)
alongside entropy output metrics — this is what allows discrimination of failure modes
whose entropy *output* looks indistinguishable from normal.

| Split | Rows | Description |
|---|---|---|
| `normal_metrics.csv` | **2,889** | Healthy engine — 13 features per row (entropy + states) |
| `degraded_metrics.csv` | **3,849** | Four induced fault modes (see below) |
| `entropy_sequences.npz` | **4,813** rows × (7 state + 32 hash dims) | Chaos states + hash pairs for LSTM |
| `normal_enriched.csv` | **2,830** | Normal data with rolling & delta features (post-EDA) |
| `combined_clean.csv` | **6,738** | All labelled rows merged |

### Degradation Modes Collected

| Label | How it was induced | Key discriminating feature |
|---|---|---|
| `frozen_pendulum` | `pendulum.tick = lambda: None` | `mean(|diff(omega1)|) = 0.0` (pendulum never moves) |
| `lorenz_runaway` | `lorenz.rho = 200.0` | `lorenz_z ≈ 188.8` vs normal `≈ 23.9` (20-sigma outlier) |
| `rd_uniform` | `rd._V[:] = 0; rd._U[:] = 1` | No strong individual feature signal |
| `pool_starvation` | Pool zeroed + `pool.feed` monkey-patched | `entropy_score = 0.000`, `duplicate_rate ≈ 1.0` |

### EDA Key Findings

| Metric | Value | Interpretation |
|---|---|---|
| Degraded overlap within normal 2σ band | **70.8%** | Three of four fault modes produce entropy scores indistinguishable from normal |
| Lag-1 autocorrelation (entropy_score) | **+0.9862** | Health monitor caches for 5 s, sampled at 62 ms — same value repeats ~80× |
| Hash χ² statistic | **287.24** | Expected ≈ 255 for a uniform distribution; confirms hash output is uniform |
| Enriched feature rows | **2,830** | 13-feature windows with rolling + delta features for autoencoder |

**Key insight from EDA:** Entropy output metrics alone cannot distinguish 3 of 4 failure
modes from normal operation. The chaos engine *states* (particularly `lorenz_z` and
`omega1` velocity) are the discriminating signal. This drove the decision to re-collect
data with state features and redesign both models accordingly.

---

## Model 1 — Hybrid Anomaly Detector

This model uses **two complementary sub-detectors** to achieve coverage across all four
failure modes. A single autoencoder approach failed for `frozen_pendulum` (constant inputs
reconstruct *better* than normal, producing lower error — not higher).

### Sub-detector A: Autoencoder on Entropy + Lorenz Features

```
Input (780) = window of 60 time-steps x 13 features
  --> Linear(780, h=97) + ReLU      [encoder]
  --> Linear(97, 32)    + ReLU
  --> Linear(32, 16)    + ReLU
  --> Linear(16, 8)                  [bottleneck]
  --> Linear(8, 16)     + ReLU      [decoder]
  --> Linear(16, 32)    + ReLU
  --> Linear(32, 97)    + ReLU
  --> Linear(97, 780)               [reconstruction]
```

**Features (13 total):**
- Entropy: `entropy_score`, `distribution_uniformity`, `duplicate_rate`, `pool_fill_percent`
- Rolling (window=60): `entropy_score_roll_mean/std`, `distribution_uniformity_roll_mean/std`
- Delta: `entropy_score_delta`, `distribution_uniformity_delta`
- Lorenz state: `lorenz_z`, `lorenz_r`, `lorenz_r_roll_mean`

### Sub-detector B: Pendulum Motion Check (analytical, no model)

```
motion_mean = mean(|diff(omega1)|)  over rolling 60-sample window
is_motion_anomaly = motion_mean < motion_threshold (p5 of normal)
```

- Normal: `omega1` oscillates chaotically → `motion_mean ≈ 0.41 ± 0.24`
- `frozen_pendulum`: `omega1` is constant → `motion_mean = 0.000` exactly
- `motion_threshold = 0.126` (p5 of normal calibration data)

### Combined Detection

```
is_anomaly = ae_reconstruction_error > ae_threshold  OR  motion_mean < motion_threshold
```

### Training

| Parameter | Value |
|---|---|
| Epochs | 60 |
| Batch size | 256 |
| Optimiser | Adam (lr=1e-3) |
| Train windows | 2,216 |
| Validation windows | 554 |
| Temporal split | 80/20 (no shuffle) |

### Training Metrics

| Epoch | Train MSE | Val MSE |
|---|---|---|
| 0 | 0.760322 | 0.548636 |
| 10 | 0.467747 | 0.360701 |
| 20 | 0.377178 | 0.276331 |
| 30 | 0.349338 | 0.261066 |
| 40 | 0.330577 | 0.249660 |
| 50 | 0.297792 | 0.226858 |
| **Best** | — | **0.218183** |

### Evaluation — Per-Mode Detection Rates

| Failure Mode | Detection | Sub-detector responsible |
|---|---|---|
| `frozen_pendulum` | **100.0%** | Motion check (omega1 stops oscillating) |
| `lorenz_runaway` | **100.0%** | Autoencoder (lorenz_z is 20σ outlier) |
| `pool_starvation` | **100.0%** | Autoencoder (entropy_score = 0) |
| `rd_uniform` | 7.2% | Limited — no strong signal in either sub-detector |
| **Overall** | **76.8%** | Meets the 70% target |

| Metric | Value |
|---|---|
| Anomaly threshold (p95) | **0.6270** |
| Motion threshold (p5) | **0.1261** |
| Overall detection rate | **76.8%** |

**Why `rd_uniform` is hard:** When the reaction-diffusion layer is made uniform, the
pendulum and Lorenz attractors continue operating normally. Only the RD contribution to
the hash is degraded — but since the hash is a mix of all three sources, the entropy
output and state features remain statistically normal. This is an architectural property
of the multi-source mixer.

---

## Model 2 — LSTM Prediction Resistance Scorer

### Architecture

```
Input (batch, 30, 7)    # 30 time-steps of (theta1, theta2, omega1, omega2, lorenz_x, y, z)
--> LSTM(input=7, hidden=64, layers=2, dropout=0.2, batch_first=True)
--> Linear(64, 64) + ReLU
--> Linear(64, 32)      # predict next 32-byte hash output
```

- **Input:** 30 consecutive chaos-state vectors (7 dimensions each, normalised)
- **Target:** Next entropy hash output (32 bytes, normalised to [0, 1])
- **Framework:** PyTorch
- **Loss:** MSE

### Training

| Parameter | Value |
|---|---|
| Sequence length | 30 |
| Epochs | 40 |
| Batch size | 128 |
| Optimiser | Adam (lr=1e-3) + gradient clipping (max norm=1.0) |
| Sequence pairs | 4,813 (temporal 80/20 split) |

### Training Metrics

| Epoch | Train MSE | Val MSE |
|---|---|---|
| 0 | 0.214227 | 0.148295 |
| 10 | 0.084653 | 0.090053 |
| 20 | 0.084458 | 0.085939 |
| 30 | 0.084436 | 0.084913 |
| **Best** | — | **0.084799** |

### Evaluation

| Metric | Value | Notes |
|---|---|---|
| Naive baseline MSE | 0.084380 | Mean-prediction over uniform [0,1] |
| LSTM best val MSE | **0.084799** | Essentially identical to baseline |
| Theoretical ceiling | ≈ 0.0833 | `1/12` = Var(Uniform[0,1]) — maximum for random bytes |
| Prediction resistance mean | **0.0848** | σ = 0.0138 |
| LSTM advantage | **≈ 0 bits** | Cannot predict better than chance |

**Interpretation:** The LSTM converges to MSE at the theoretical ceiling for uniformly
random data. This is a **positive result** — it confirms the entropy hash output carries
no exploitable structure. Any meaningful drop below the 1/12 ceiling would indicate a
predictability vulnerability in the mixer.

---

## Model 4 — MLP Entropy Quality Classifier

### Architecture

```
Input (8 features) --> Linear(8, 32) + ReLU
                   --> Linear(32, 16) + ReLU
                   --> Linear(16, 4)   [softmax -> 4 classes]
```

**Feature set (8 total):**
- Entropy: `entropy_score`, `distribution_uniformity`, `duplicate_rate`, `pool_fill_percent`
- Chaos state: `lorenz_z`, `lorenz_r`, `omega1_roll_std`, `omega2_roll_std`

**Classes:** `excellent`, `good`, `degraded`, `critical`

**Framework:** scikit-learn `MLPClassifier`

### Why State Features Transformed the Classifier

| Feature | Normal (mean) | frozen_pendulum | lorenz_runaway |
|---|---|---|---|
| `entropy_score` | 0.976 | 0.976 | 0.975 |
| `lorenz_z` | 23.9 | 23.9 | **188.8** |
| `omega1_roll_std` | 3.55 | **0.000** | 3.55 |

Without state features: `degraded` recall = **0%** (could not be distinguished from `excellent`).
With state features: `degraded` recall = **71%**, `critical` recall = **100%**.

### Class Construction

The `good` class does not naturally occur in collection data. It is synthesised from
normal rows by scaling `entropy_score` into the 0.91–0.95 range.

| Class | Source | Size |
|---|---|---|
| `excellent` | Normal collection | 2,889 |
| `degraded` | `frozen_pendulum` + `rd_uniform` | 1,925 |
| `critical` | `lorenz_runaway` + `pool_starvation` | 1,924 |
| `good` | Synthesised from normal (entropy 0.91–0.95) | 1,000 |

### Training Metrics

| Metric | Value |
|---|---|
| 5-fold CV accuracy | **78.6% ± 12.7%** |
| Train accuracy | **90%** |
| Hidden layers | (32, 16), ReLU |
| Max iterations | 500 |

### Per-Class Classification Report (train set)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `critical` | **1.00** | **1.00** | **1.00** | 1,924 |
| `degraded` | 0.87 | 0.71 | 0.78 | 1,925 |
| `excellent` | 0.83 | **0.93** | 0.88 | 2,889 |
| `good` | **1.00** | **1.00** | **1.00** | 1,000 |
| **Overall** | 0.92 | 0.91 | **0.91** | 7,738 |

**Note on `degraded` recall (71%):** The `frozen_pendulum` and `rd_uniform` modes are
mapped to `degraded`. The `rd_uniform` mode produces no discriminating feature signal
(same issue as the anomaly detector), contributing the remaining 29% misclassification.
`frozen_pendulum` is correctly classified via `omega1_roll_std = 0`.

---

## Runtime Integration

```
kairos/engine.py
    _feed_loop()  [every 50 ms]
        |-- mixer.mix() -> 32-byte hash
        |-- pool.feed(mixed)
        +-- health_monitor.ml_inference.update_state(chaos_states, hash_bytes)
                                    |
                    updates two sliding windows:
                    _anomaly_window (maxlen=60) -- for autoencoder + motion check
                    _state_window   (maxlen=30) -- for LSTM

kairos/entropy/health.py
    HealthMonitor._run()  [every 5 s]
        |-- evaluate(pool_bytes)  ->  base metrics dict
        +-- ml_inference.evaluate(metrics)
                |-- Model 4: MLP class + confidence (primary status)
                |-- Model 1: hybrid anomaly score (AE + motion check)
                +-- Model 2: LSTM prediction error (when window full)
                    |
                    returns: anomaly_score, is_anomaly, prediction_resistance,
                             health_status, health_confidence, ml_active
```

### API Response Fields

**`GET /health`** and **`WS /ws/entropy`** — both expose:

```json
{
  "ml_active":             true,
  "health_status":         "excellent",
  "health_confidence":     0.93,
  "anomaly_score":         0.12,
  "is_anomaly":            false,
  "prediction_resistance": 0.085
}
```

---

## File Structure

```
ml/
├── collect_data.py          # Phase 1 — live data collection (entropy + chaos states)
├── eda.py                   # Phase 2 — EDA, rolling/delta feature engineering
├── model1_anomaly.py        # Phase 3 — hybrid autoencoder + motion threshold training
├── model2_predictor.py      # Phase 4 — LSTM training
├── model4_classifier.py     # Phase 5 — MLP classifier training
├── inference.py             # Phase 6 — KairosMLInference runtime module
├── generate_metrics_report.py  # Generates KAIROS_ML_Metrics.png
├── KAIROS_ML_Metrics.png    # Metrics visualisation
├── ML_SUMMARY.md            # This document
├── data/
│   ├── normal_metrics.csv       2,889 rows (13 cols: entropy + states)
│   ├── degraded_metrics.csv     3,849 rows (same schema)
│   ├── entropy_sequences.npz    4,813 x (7 + 32) dims
│   ├── normal_enriched.csv      2,830 rows x 16+ features
│   └── combined_clean.csv       6,738 rows
└── models/
    ├── model1_autoencoder.pt    PyTorch state dict (60-epoch best)
    ├── model1_scaler.json       Feature stats + ae_threshold + motion_threshold
    ├── model2_predictor.pt      PyTorch state dict
    ├── model2_norm.json         State normalisation + resistance stats
    └── model4_classifier.pkl    sklearn MLP + StandardScaler + feature list
```

---

## Stack

| Component | Library | Version |
|---|---|---|
| Autoencoder / LSTM | PyTorch | 2.10.0+cpu |
| MLP Classifier | scikit-learn | 1.7.2 |
| Data wrangling | pandas | 2.3.3 |
| Numerics | numpy | 2.2.6 |
| Visualisation | matplotlib | 3.10.8 |
| Runtime | Python | 3.10.11 |
