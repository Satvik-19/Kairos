"""
Phase 3 (final): Hybrid anomaly detector.

Two complementary sub-detectors:
  A) Autoencoder (PyTorch) on entropy output + state features.
     Excellent for lorenz_runaway (features 20 sigma from normal)
     and pool_starvation (entropy_score = 0).

  B) Pendulum-motion check: mean(|delta omega1|) over a rolling window.
     For frozen_pendulum: omega1 never changes -> motion_mean = 0.0.
     For normal: omega1 oscillates -> motion_mean is robustly > 0.
     This check is calibrated on normal data only (no degraded labels needed).

Anomaly is flagged if EITHER sub-detector fires.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json

DATA_DIR  = Path('ml/data')
MODEL_DIR = Path('ml/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
WINDOW     = 60
EPOCHS     = 60
BATCH_SIZE = 256
LR         = 1e-3
AE_THRESHOLD_PERCENTILE   = 95   # top 5% reconstruction errors on normal data
MOTION_THRESHOLD_PERCENTILE = 5  # bottom 5% motion scores on normal data

# Features for the autoencoder
FEATURES_BASE = [
    'entropy_score', 'distribution_uniformity',
    'duplicate_rate', 'pool_fill_percent',
    'entropy_score_roll_mean', 'entropy_score_roll_std',
    'distribution_uniformity_roll_mean', 'distribution_uniformity_roll_std',
    'entropy_score_delta', 'distribution_uniformity_delta',
]
FEATURES_STATE = [
    'lorenz_z',           # 20-sigma outlier for lorenz_runaway (rho=200)
    'lorenz_r',           # Lorenz magnitude reinforcer
    'lorenz_r_roll_mean', # Stable running mean of magnitude
]


# ── Autoencoder architecture ───────────────────────────────────────────────────

class EntropyAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        h = max(64, input_dim // 8)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h), nn.ReLU(),
            nn.Linear(h, 32),       nn.ReLU(),
            nn.Linear(32, 16),      nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),  nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, h),  nn.ReLU(),
            nn.Linear(h, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def prepare_windows(df, feature_cols, window):
    data = df[feature_cols].values.astype(np.float32)
    X = [data[i - window: i].flatten() for i in range(window, len(data))]
    return np.array(X, dtype=np.float32)


def _enrich(df, window):
    df = df.copy()
    for col in ['entropy_score', 'distribution_uniformity']:
        df[f'{col}_delta']     = df[col].diff().fillna(0)
        df[f'{col}_roll_mean'] = df[col].rolling(window).mean()
        df[f'{col}_roll_std']  = df[col].rolling(window).std()
    if all(c in df.columns for c in ['lorenz_x', 'lorenz_y', 'lorenz_z']):
        df['lorenz_r'] = np.sqrt(
            df['lorenz_x']**2 + df['lorenz_y']**2 + df['lorenz_z']**2)
        df['lorenz_r_roll_mean'] = df['lorenz_r'].rolling(window).mean()
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    normal = pd.read_csv(DATA_DIR / 'normal_metrics.csv')
    normal = _enrich(normal, WINDOW)

    state_ok  = all(c in normal.columns for c in FEATURES_STATE)
    all_feats = FEATURES_BASE + (FEATURES_STATE if state_ok else [])

    normal = (normal
              .replace([np.inf, -np.inf], np.nan)
              .dropna(subset=all_feats)
              .reset_index(drop=True))

    print(f"Features: {len(all_feats)}  (state={'YES' if state_ok else 'NO'})")
    print(f"Normal training rows: {len(normal)}")

    # ── Part A: Autoencoder ────────────────────────────────────────────────
    means = normal[all_feats].mean()
    stds  = normal[all_feats].std().replace(0, 1)

    normal_norm = (normal[all_feats] - means) / stds
    X = prepare_windows(normal_norm, all_feats, WINDOW)
    input_dim = X.shape[1]

    split   = int(0.8 * len(X))
    X_train = torch.tensor(X[:split])
    X_val   = torch.tensor(X[split:])

    print(f"Autoencoder: {len(X_train)} train / {len(X_val)} val  "
          f"| input_dim={input_dim}")

    model     = EntropyAutoencoder(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    loader    = DataLoader(TensorDataset(X_train, X_train),
                           batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, _ in loader:
            pred  = model(xb)
            loss  = criterion(pred, xb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(xb)
        total_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), X_val).item()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train={total_loss:.6f} | val={val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / 'model1_autoencoder.pt')

    model.load_state_dict(
        torch.load(MODEL_DIR / 'model1_autoencoder.pt', weights_only=True))
    model.eval()
    with torch.no_grad():
        val_errors = ((model(X_val) - X_val) ** 2).mean(dim=1).numpy()

    ae_threshold = float(np.percentile(val_errors,
                                       AE_THRESHOLD_PERCENTILE))
    print(f"\nAutoencoder threshold (p{AE_THRESHOLD_PERCENTILE}): {ae_threshold:.6f}")
    print(f"Best val loss: {best_val_loss:.6f}")

    # ── Part B: Pendulum motion calibration ───────────────────────────────
    motion_threshold = None
    if 'omega1' in normal.columns:
        # mean(|diff(omega1)|) over rolling window — 0 when pendulum frozen
        motion_series = normal['omega1'].diff().abs().rolling(WINDOW).mean()
        motion_clean  = motion_series.dropna()
        motion_threshold = float(np.percentile(
            motion_clean, MOTION_THRESHOLD_PERCENTILE))
        print(f"\nPendulum motion threshold (p{MOTION_THRESHOLD_PERCENTILE}): "
              f"{motion_threshold:.6f}")
        print(f"  Normal motion_mean: {motion_clean.mean():.4f} ± {motion_clean.std():.4f}")
        print(f"  Frozen motion_mean: 0.0 (omega1 constant)")

    # ── Persist scaler + thresholds ───────────────────────────────────────
    scaler_params = {
        'mean':             means.to_dict(),
        'std':              stds.to_dict(),
        'features':         all_feats,
        'anomaly_threshold': ae_threshold,
        'motion_threshold':  motion_threshold,
        'input_dim':         input_dim,
        'window':            WINDOW,
        'backend':           'autoencoder+motion',
    }
    with open(MODEL_DIR / 'model1_scaler.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)

    # ── Evaluate on degraded data ─────────────────────────────────────────
    print("\n--- Evaluation on degraded data (per-group rolling) ---")
    deg_raw  = pd.read_csv(DATA_DIR / 'degraded_metrics.csv')
    det_rates = []

    for lbl in deg_raw['label'].unique():
        sub = deg_raw[deg_raw['label'] == lbl].copy().reset_index(drop=True)
        sub = _enrich(sub, WINDOW)
        sub = (sub.replace([np.inf, -np.inf], np.nan)
                  .dropna(subset=all_feats)
                  .reset_index(drop=True))
        if len(sub) <= WINDOW:
            print(f"  {lbl:20s}: SKIPPED"); continue

        sub_norm  = (sub[all_feats] - means) / stds
        X_d       = torch.tensor(
            prepare_windows(sub_norm, all_feats, WINDOW))
        with torch.no_grad():
            ae_err = ((model(X_d) - X_d) ** 2).mean(dim=1).numpy()
        ae_flags = ae_err > ae_threshold

        motion_flags = np.zeros(len(ae_flags), dtype=bool)
        if motion_threshold is not None and 'omega1' in sub.columns:
            ms = sub['omega1'].diff().abs().rolling(WINDOW).mean()
            motion_vals = ms.values[WINDOW:]  # align with windows
            if len(motion_vals) == len(ae_flags):
                motion_flags = np.nan_to_num(motion_vals, nan=999.0) < motion_threshold

        combined   = ae_flags | motion_flags
        det_rate   = combined.mean()
        det_rates.append(det_rate)
        ae_only    = ae_flags.mean()
        motion_only = motion_flags.mean()
        ok = "OK" if det_rate >= 0.7 else "WARN"
        print(f"  {lbl:20s}: {det_rate:.1%}  "
              f"(AE={ae_only:.1%} | motion={motion_only:.1%})  [{ok}]")

    if det_rates:
        print(f"\n  Overall detection rate: {np.mean(det_rates):.1%}")
        if np.mean(det_rates) >= 0.7:
            print("  OK: Hybrid detector meets 70% target.")

    print(f"\nAutoencoder: ml/models/model1_autoencoder.pt")
    print(f"Scaler+thresholds: ml/models/model1_scaler.json")


if __name__ == '__main__':
    train()
