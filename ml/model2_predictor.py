"""
Phase 4: Prediction Resistance Scorer — 2-layer LSTM.
The model is trained to predict the next entropy hash from chaos state history.
A HIGH prediction error means the hash is unpredictable → good entropy quality.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json

DATA_DIR  = Path('ml/data')
MODEL_DIR = Path('ml/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────
SEQUENCE_LEN  = 30    # 30 timesteps of chaos-state history → predict next hash
STATE_DIM     = 7     # theta1, theta2, omega1, omega2, lorenz_x, y, z
HASH_DIM      = 32    # 32 bytes of hash output
HIDDEN_DIM    = 64
EPOCHS        = 40
BATCH_SIZE    = 128
LEARNING_RATE = 1e-3


# ── Architecture ─────────────────────────────────────────────────────────────

class EntropyPredictor(nn.Module):
    """
    2-layer LSTM that takes a window of chaos-state vectors and attempts to
    predict the next entropy hash.  High prediction error = high unpredictability
    = strong entropy quality signal.
    """
    def __init__(self, state_dim: int, hash_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, hash_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, state_dim)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # use last timestep


# ── Data preparation ─────────────────────────────────────────────────────────

def prepare_sequences(states: np.ndarray, hashes: np.ndarray, seq_len: int):
    """
    Build (X, y) pairs:
      X[i]: states[i : i+seq_len]     shape (seq_len, STATE_DIM)
      y[i]: hashes[i+seq_len]         shape (HASH_DIM,)
    """
    X, y = [], []
    for i in range(len(states) - seq_len):
        X.append(states[i: i + seq_len])
        y.append(hashes[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Training ─────────────────────────────────────────────────────────────────

def train():
    data   = np.load(DATA_DIR / 'entropy_sequences.npz')
    states = data['states']   # (N, 7)
    hashes = data['hashes']   # (N, 32)

    print(f"Sequence data: {states.shape[0]} samples")

    # Normalise states (per feature, fit on full dataset)
    state_mean = states.mean(axis=0)
    state_std  = states.std(axis=0)
    state_std[state_std == 0] = 1.0
    states_norm = (states - state_mean) / state_std

    # Normalise hash bytes to [0, 1]
    hashes_norm = hashes / 255.0

    # Persist normalisation params
    norm_params = {
        'state_mean':   state_mean.tolist(),
        'state_std':    state_std.tolist(),
        'sequence_len': SEQUENCE_LEN,
        'state_dim':    STATE_DIM,
        'hash_dim':     HASH_DIM,
    }
    with open(MODEL_DIR / 'model2_norm.json', 'w') as f:
        json.dump(norm_params, f, indent=2)

    X, y = prepare_sequences(states_norm, hashes_norm, SEQUENCE_LEN)
    print(f"Sequences: X={X.shape}, y={y.shape}")

    # Temporal train/val split (no shuffle — preserve causal order)
    split   = int(0.8 * len(X))
    X_train = torch.tensor(X[:split])
    y_train = torch.tensor(y[:split])
    X_val   = torch.tensor(X[split:])
    y_val   = torch.tensor(y[split:])

    model     = EntropyPredictor(STATE_DIM, HASH_DIM, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    loader    = DataLoader(TensorDataset(X_train, y_train),
                           batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            pred  = model(xb)
            loss  = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       MODEL_DIR / 'model2_predictor.pt')

    # ── Baseline comparison ──────────────────────────────────────────────────
    naive_error = float(
        ((hashes_norm[SEQUENCE_LEN: split] -
          hashes_norm[SEQUENCE_LEN: split].mean()) ** 2).mean()
    )
    print(f"\nNaive (mean) prediction MSE:  {naive_error:.6f}")
    print(f"LSTM best val MSE:            {best_val_loss:.6f}")

    if best_val_loss >= naive_error * 0.95:
        print("  INFO: LSTM barely improves on naive baseline.")
        print("  This is EXPECTED and GOOD — entropy outputs are genuinely unpredictable.")
    else:
        improvement = abs(best_val_loss - naive_error) / naive_error * 100
        print(f"  WARN: LSTM is {improvement:.1f}% better than naive. "
              "Review mixer.py for potential predictability.")

    # ── Prediction-resistance score on validation set ────────────────────────
    model.load_state_dict(
        torch.load(MODEL_DIR / 'model2_predictor.pt', weights_only=True))
    model.eval()
    with torch.no_grad():
        val_errors = ((model(X_val) - y_val) ** 2).mean(dim=1).numpy()

    # High error → high resistance → good entropy
    resistance_scores = np.clip(val_errors, 0.0, 1.0)
    print(f"\nPrediction Resistance Score (val set):")
    print(f"  Mean:   {resistance_scores.mean():.4f}")
    print(f"  Std:    {resistance_scores.std():.4f}")
    print(f"  Min:    {resistance_scores.min():.4f}")
    print(f"  Target: > 0.85 for high-quality entropy")

    if resistance_scores.mean() < 0.85:
        print("  WARN: Mean resistance below 0.85.  Hash stream may be partially predictable.")
    else:
        print("  OK: Entropy stream is highly resistant to prediction.")

    # Persist resistance stats for runtime use
    norm_params.update({
        'resistance_mean': float(resistance_scores.mean()),
        'resistance_std':  float(resistance_scores.std()),
    })
    with open(MODEL_DIR / 'model2_norm.json', 'w') as f:
        json.dump(norm_params, f, indent=2)

    print(f"\nModel saved:      ml/models/model2_predictor.pt")
    print(f"Norm params saved: ml/models/model2_norm.json")


if __name__ == '__main__':
    train()
