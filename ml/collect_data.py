"""
Phase 1: KAIROS entropy data collection pipeline.
Produces three datasets:
  ml/data/normal_metrics.csv       — healthy engine metrics
  ml/data/degraded_metrics.csv     — four induced failure modes
  ml/data/entropy_sequences.npz   — (chaos_state, hash_output) pairs for LSTM training
"""
import time
import csv
import os

import numpy as np
from pathlib import Path

from kairos import EntropyEngine

DATA_DIR = Path('ml/data')
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Normal collection
# ---------------------------------------------------------------------------

def _sample_row(engine, label):
    """Return one merged metrics + chaos-state row dict."""
    h      = engine.health()
    states = engine.get_engine_states()
    p      = states['pendulum']
    l      = states['lorenz']
    return {
        'timestamp':               time.time(),
        'entropy_score':           h.get('entropy_score', 0.0),
        'distribution_uniformity': h.get('distribution_uniformity', 0.0),
        'duplicate_rate':          h.get('duplicate_rate', 0.0),
        'pool_fill_percent':       engine.pool.fill_percent(),
        # Chaos engine states — key discriminators between degradation modes
        'theta1':                  p['theta1'],
        'theta2':                  p['theta2'],
        'omega1':                  p['omega1'],
        'omega2':                  p['omega2'],
        'lorenz_x':                l['x'],
        'lorenz_y':                l['y'],
        'lorenz_z':                l['z'],
        'label':                   label,
    }


def collect_normal(duration_seconds=600, sample_interval=0.05):
    """
    Run engine in healthy mode.  Sample at 20 Hz by default.
    Writes: ml/data/normal_metrics.csv  (metrics + chaos states, unified)
    """
    engine = EntropyEngine()
    time.sleep(5)          # warmup — discard first 5 s

    rows  = []
    start = time.time()
    while time.time() - start < duration_seconds:
        rows.append(_sample_row(engine, 'normal'))
        time.sleep(sample_interval)

    engine.shutdown()
    _write_csv(DATA_DIR / 'normal_metrics.csv', rows)
    print(f"Collected {len(rows)} normal samples")


# ---------------------------------------------------------------------------
# Degraded collection — four failure modes
# ---------------------------------------------------------------------------

def collect_degraded(duration_per_mode=120, sample_interval=0.05):
    """
    Induce four degradation modes, one engine instance per mode.
    Writes: ml/data/degraded_metrics.csv
    """
    all_rows = []

    modes = [
        ('frozen_pendulum',  _degrade_frozen_pendulum),
        ('lorenz_runaway',   _degrade_lorenz_runaway),
        ('rd_uniform',       _degrade_rd_uniform),
        ('pool_starvation',  _degrade_pool_starvation),
    ]

    for mode_name, degrade_fn in modes:
        engine = EntropyEngine()
        time.sleep(5)          # warmup before inducing fault
        degrade_fn(engine)

        start = time.time()
        while time.time() - start < duration_per_mode:
            all_rows.append(_sample_row(engine, mode_name))
            time.sleep(sample_interval)

        engine.shutdown()
        print(f"Collected ~{int(duration_per_mode / sample_interval)} {mode_name} samples")

    _write_csv(DATA_DIR / 'degraded_metrics.csv', all_rows)


# ---------------------------------------------------------------------------
# Sequence collection — (state_vector, hash_bytes) pairs
# ---------------------------------------------------------------------------

def collect_entropy_sequences(duration_seconds=300, sample_interval=0.05):
    """
    Collect (chaos_state_vector → next_entropy_hash) pairs for LSTM training.
    Writes: ml/data/entropy_sequences.npz
    """
    engine = EntropyEngine()
    time.sleep(5)

    states_list = []
    hashes_list = []

    start = time.time()
    while time.time() - start < duration_seconds:
        s = engine.get_engine_states()
        p = s['pendulum']
        l = s['lorenz']

        state_vec = [
            p['theta1'], p['theta2'], p['omega1'], p['omega2'],
            l['x'],      l['y'],      l['z'],
        ]
        hash_bytes = engine.pool.read(32)
        hash_vec   = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)

        states_list.append(state_vec)
        hashes_list.append(hash_vec)
        time.sleep(sample_interval)

    engine.shutdown()

    states_arr = np.array(states_list, dtype=np.float32)
    hashes_arr = np.array(hashes_list, dtype=np.float32)
    np.savez(DATA_DIR / 'entropy_sequences.npz',
             states=states_arr, hashes=hashes_arr)
    print(f"Collected {len(states_list)} sequence samples. "
          f"States shape: {states_arr.shape}")


# ---------------------------------------------------------------------------
# Degradation helpers
# ---------------------------------------------------------------------------

def _degrade_frozen_pendulum(engine):
    """Monkey-patch tick() to no-op — pendulum state freezes."""
    engine.pendulum.tick = lambda: None


def _degrade_lorenz_runaway(engine):
    """Set rho far outside normal attractor — diverging trajectories."""
    engine.lorenz.rho = 200.0


def _degrade_rd_uniform(engine):
    """
    Reset V to all-zeros and U to all-ones.
    With V=0 everywhere, Gray-Scott autocatalysis is zero — V stays flat.
    """
    with engine.rd._lock:
        engine.rd._V[:] = 0.0
        engine.rd._U[:] = 1.0


def _degrade_pool_starvation(engine):
    """
    Clear the entropy pool to all-zeros, then block the feed loop from
    writing new bytes.  Health monitor will see near-zero entropy.
    """
    with engine.pool._lock:
        engine.pool._buffer = bytearray(engine.pool._size)  # zero out
    # Monkey-patch feed to no-op so pool stays stale
    engine.pool.feed = lambda data: None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    print("=" * 60)
    print("KAIROS ML DATA COLLECTION PIPELINE  (enriched – with states)")
    print("=" * 60)

    step = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if step in ('all', 'normal'):
        print("\n[1] Collecting NORMAL data (180 s)...")
        collect_normal(duration_seconds=180)

    if step in ('all', 'degraded'):
        print("\n[2] Collecting DEGRADED data (4 modes x 60 s)...")
        collect_degraded(duration_per_mode=60)

    print("\nDone.  Data written to ml/data/")
