"""
Phase 5: Entropy Quality Classifier — sklearn MLP.
Given a snapshot of four health metrics, predict the quality class:
  excellent | good | degraded | critical
This replaces the hand-coded threshold table in health.py with a learned boundary.
"""
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

DATA_DIR  = Path('ml/data')
MODEL_DIR = Path('ml/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_BASE = [
    'entropy_score', 'distribution_uniformity',
    'duplicate_rate', 'pool_fill_percent',
]

# State-derived features — strong discriminators:
#   lorenz_z         → ~199 for lorenz_runaway (rho=200) vs ~27 for normal
#   lorenz_r         → sqrt(x^2+y^2+z^2) — large for runaway
#   omega1_roll_std  → ~0 for frozen_pendulum (state stops changing)
#   omega2_roll_std  → ~0 for frozen_pendulum
FEATURES_STATE = [
    'lorenz_z',
    'lorenz_r',
    'omega1_roll_std',
    'omega2_roll_std',
]

ROLL_WINDOW = 60   # 3 s at 20 Hz

# Map collection labels → quality class names
LABEL_MAP = {
    'normal':           'excellent',
    'frozen_pendulum':  'degraded',
    'lorenz_runaway':   'critical',
    'rd_uniform':       'degraded',
    'pool_starvation':  'critical',
}
CLASS_ORDER = ['excellent', 'good', 'degraded', 'critical']


def _compute_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lorenz_r and within-group rolling std of omega1/omega2.
    Groups preserve per-degradation-mode temporal ordering.
    """
    df = df.copy()
    df['lorenz_r'] = np.sqrt(
        df['lorenz_x']**2 + df['lorenz_y']**2 + df['lorenz_z']**2)

    # Rolling std within each label group to avoid cross-label contamination
    parts = []
    for _, grp in df.groupby('label', sort=False):
        grp = grp.copy()
        grp['omega1_roll_std'] = grp['omega1'].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0)
        grp['omega2_roll_std'] = grp['omega2'].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0)
        parts.append(grp)
    df = pd.concat(parts).sort_index()
    return df


def train():
    combined = pd.read_csv(DATA_DIR / 'combined_clean.csv')
    combined['quality_class'] = combined['label'].map(LABEL_MAP)
    combined = combined.dropna(subset=['quality_class'])

    # Determine if state features are available
    state_cols_present = all(c in combined.columns
                             for c in ['omega1', 'omega2', 'lorenz_x', 'lorenz_y', 'lorenz_z'])
    if state_cols_present:
        combined = _compute_state_features(combined)
        features = FEATURES_BASE + FEATURES_STATE
        print("INFO: State features detected — using enriched feature set.")
    else:
        features = FEATURES_BASE
        print("WARN: State features not found — falling back to entropy-only features.")

    # Synthesise a 'good' class: take normal rows, apply a downward
    # noise on entropy_score to put them in the 0.91–0.95 band.
    rng         = np.random.default_rng(42)
    n_good      = min(1000, len(combined) // 5)
    good_idx    = (combined['quality_class'] == 'excellent').values.nonzero()[0]
    good_sample = combined.iloc[rng.choice(good_idx, size=n_good, replace=False)].copy()
    good_sample['entropy_score'] *= rng.uniform(0.91, 0.95, size=n_good)
    good_sample['quality_class']  = 'good'
    combined = pd.concat([combined, good_sample], ignore_index=True)

    print(f"Class distribution:\n{combined['quality_class'].value_counts()}\n")

    X = combined[features].values
    y = combined['quality_class'].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        max_iter=500,
        random_state=42,
    )

    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    if cv_scores.mean() < 0.80:
        print("WARN: CV accuracy below 80 %.  "
              "Features may not separate all four classes well.")

    # Final fit on full dataset
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    print("\n--- Classification Report (train set) ---")
    # Only show classes that appear in this run
    present = sorted(set(y) | set(y_pred))
    print(classification_report(y, y_pred, labels=present, zero_division=0))

    # Save model bundle
    bundle = {
        'model':    model,
        'scaler':   scaler,
        'features': features,
        'classes':  CLASS_ORDER,
    }
    with open(MODEL_DIR / 'model4_classifier.pkl', 'wb') as f:
        pickle.dump(bundle, f)

    print(f"Model saved: ml/models/model4_classifier.pkl")


if __name__ == '__main__':
    train()
