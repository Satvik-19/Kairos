"""
Phase 2: Exploratory data analysis for KAIROS entropy training data.
Run after collect_data.py has produced all three dataset files.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('ml/data')


def run_eda():
    print("=" * 60)
    print("KAIROS ENTROPY DATA — EDA REPORT")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------------
    normal   = pd.read_csv(DATA_DIR / 'normal_metrics.csv')
    degraded = pd.read_csv(DATA_DIR / 'degraded_metrics.csv')
    combined = pd.concat([normal, degraded], ignore_index=True)

    print(f"\nNormal samples:   {len(normal)}")
    print(f"Degraded samples: {len(degraded)}")
    print(f"Label distribution:\n{combined['label'].value_counts()}")

    # ------------------------------------------------------------------
    # 2. Statistical summary per label
    # ------------------------------------------------------------------
    features = ['entropy_score', 'distribution_uniformity',
                'duplicate_rate', 'pool_fill_percent']
    print("\n--- Statistical Summary per Label ---")
    print(combined.groupby('label')[features]
          .agg(['mean', 'std', 'min', 'max'])
          .round(4)
          .to_string())

    # ------------------------------------------------------------------
    # 3. Data quality — NaN / Inf
    # ------------------------------------------------------------------
    print("\n--- Data Quality Checks ---")
    for col in features:
        nan_c = combined[col].isna().sum()
        inf_c = np.isinf(combined[col]).sum()
        print(f"  {col}: {nan_c} NaN, {inf_c} Inf")
        if nan_c > 0 or inf_c > 0:
            print(f"  WARNING: dropping {nan_c + inf_c} bad rows in {col}")

    combined = (combined
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=features))
    print(f"  Clean rows remaining: {len(combined)}")

    # ------------------------------------------------------------------
    # 4. Class separability
    # ------------------------------------------------------------------
    print("\n--- Class Separability (normal vs degraded) ---")
    norm_scores = normal['entropy_score']
    deg_scores  = degraded['entropy_score']
    print(f"  Normal entropy_score:   mean={norm_scores.mean():.4f}, "
          f"std={norm_scores.std():.4f}")
    print(f"  Degraded entropy_score: mean={deg_scores.mean():.4f}, "
          f"std={deg_scores.std():.4f}")
    overlap = (
        (deg_scores > norm_scores.mean() - 2 * norm_scores.std()) &
        (deg_scores < norm_scores.mean() + 2 * norm_scores.std())
    ).mean()
    print(f"  Degraded samples within normal 2-sigma band: {overlap:.1%}")
    if overlap > 0.3:
        print("  WARN: High overlap — features may not separate well.  "
              "Rolling features will be added below.")
    else:
        print("  OK: Classes are separable.")

    # ------------------------------------------------------------------
    # 5. Autocorrelation (lag-1) per feature
    # ------------------------------------------------------------------
    print("\n--- Autocorrelation (lag-1) per feature ---")
    for col in features:
        lag1 = normal[col].autocorr(lag=1)
        tag  = ("HIGH — window model recommended" if abs(lag1) > 0.9
                else "LOW  — tabular model sufficient")
        print(f"  {col:38s}: {lag1:+.4f}  {tag}")

    # ------------------------------------------------------------------
    # 6. Sequence data
    # ------------------------------------------------------------------
    print("\n--- Sequence Data (entropy_sequences.npz) ---")
    seq    = np.load(DATA_DIR / 'entropy_sequences.npz')
    states = seq['states']
    hashes = seq['hashes']
    print(f"  States shape: {states.shape}")
    print(f"  Hashes shape: {hashes.shape}")
    print(f"  States — any NaN: {np.isnan(states).any()}")
    print(f"  States — any Inf: {np.isinf(states).any()}")
    print("  States per feature (min / max):")
    feat_names = ['theta1', 'theta2', 'omega1', 'omega2',
                  'lorenz_x', 'lorenz_y', 'lorenz_z']
    for i, name in enumerate(feat_names):
        print(f"    {name:12s}: [{states[:, i].min():8.3f}, "
              f"{states[:, i].max():8.3f}]")

    # ------------------------------------------------------------------
    # 7. Hash byte distribution (chi-squared)
    # ------------------------------------------------------------------
    print("\n--- Hash Output Distribution ---")
    flat       = hashes.flatten().astype(np.int32).clip(0, 255)
    byte_cnt   = np.bincount(flat, minlength=256)
    expected   = flat.size / 256
    chi_sq     = float(((byte_cnt - expected) ** 2 / expected).sum())
    print(f"  Chi-squared statistic: {chi_sq:.2f}  "
          f"(expected ~255 for perfectly uniform)")
    if chi_sq > 512:
        print("  WARN: Hash outputs are not uniform — entropy mixing may be flawed")
    else:
        print("  OK: Hash outputs are approximately uniform")

    # ------------------------------------------------------------------
    # 8. Feature engineering — rolling stats and deltas
    # ------------------------------------------------------------------
    print("\n--- Feature Engineering (window=60 samples = 3 s at 20 Hz) ---")
    WINDOW = 60

    STATE_FEATURES = ['theta1', 'theta2', 'omega1', 'omega2',
                      'lorenz_x', 'lorenz_y', 'lorenz_z']
    has_states = all(c in normal.columns for c in STATE_FEATURES)

    for col in features:
        normal[f'{col}_roll_mean'] = normal[col].rolling(WINDOW).mean()
        normal[f'{col}_roll_std']  = normal[col].rolling(WINDOW).std()
    normal['entropy_score_delta']           = normal['entropy_score'].diff().fillna(0)
    normal['distribution_uniformity_delta'] = normal['distribution_uniformity'].diff().fillna(0)

    if has_states:
        # Derived physics features
        normal['pendulum_ke'] = normal['omega1'] ** 2 + normal['omega2'] ** 2
        normal['lorenz_r']    = np.sqrt(
            normal['lorenz_x'] ** 2 + normal['lorenz_y'] ** 2 + normal['lorenz_z'] ** 2)
        # Rolling stats on state features — key signals for degradation modes
        for col in ['omega1', 'omega2', 'pendulum_ke', 'lorenz_z', 'lorenz_r']:
            normal[f'{col}_roll_mean'] = normal[col].rolling(WINDOW).mean()
            normal[f'{col}_roll_std']  = normal[col].rolling(WINDOW).std()

        # Show separability with new features in combined set
        if all(c in combined.columns for c in STATE_FEATURES):
            combined['pendulum_ke'] = combined['omega1']**2 + combined['omega2']**2
            combined['lorenz_r']    = np.sqrt(
                combined['lorenz_x']**2 + combined['lorenz_y']**2 + combined['lorenz_z']**2)
            print("\n--- State-Feature Separability per Label ---")
            for label in combined['label'].unique():
                sub = combined[combined['label'] == label]
                print(f"  {label:20s} | pendulum_ke mean={sub['pendulum_ke'].mean():8.3f} "
                      f"std={sub['pendulum_ke'].std():.3f} | "
                      f"lorenz_r mean={sub['lorenz_r'].mean():8.3f} "
                      f"std={sub['lorenz_r'].std():.3f}")

    normal_enriched = normal.dropna()
    print(f"  Enriched dataset shape: {normal_enriched.shape}")

    # Persist clean datasets
    normal_enriched.to_csv(DATA_DIR / 'normal_enriched.csv', index=False)
    combined.to_csv(DATA_DIR / 'combined_clean.csv', index=False)
    print(f"\n  Saved: ml/data/normal_enriched.csv  ({len(normal_enriched)} rows)")
    print(f"  Saved: ml/data/combined_clean.csv   ({len(combined)} rows)")

    print("\n" + "=" * 60)
    print("EDA COMPLETE.")
    print("=" * 60)


if __name__ == '__main__':
    run_eda()
