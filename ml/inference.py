"""
Phase 6: Runtime inference module.
Loaded once at server startup — exposes .evaluate() and .update_state().
Falls back to rule-based scoring if model files are not yet trained.
"""
from __future__ import annotations

import json
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / 'models'


class KairosMLInference:
    """
    Loads all three trained models and provides a unified .evaluate() method.
    Thread-safe for concurrent calls from the health-monitor background thread.
    """

    def __init__(self):
        # Internal state windows
        self._anomaly_window: deque = deque(maxlen=60)
        self._state_window:   deque = deque(maxlen=30)
        self._last_hash:      np.ndarray | None = None

        # Model placeholders
        self._anomaly_bundle    = None
        self._anomaly_backend   = None
        self._anomaly_threshold = None
        self._anomaly_features  = None

        self._predictor_model = None
        self._predictor_norm  = None    # dict loaded from JSON

        self._classifier_bundle = None   # dict with 'model', 'scaler', etc.

        self._models_loaded = False
        self._load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        try:
            import torch
            from ml.model1_anomaly import EntropyAutoencoder
            from ml.model2_predictor import EntropyPredictor

            # ── Model 1: Autoencoder + motion check ───────────────────────
            with open(MODEL_DIR / 'model1_scaler.json') as f:
                scaler_data = json.load(f)

            self._anomaly_features   = scaler_data['features']
            self._anomaly_threshold  = scaler_data['anomaly_threshold']
            self._motion_threshold   = scaler_data.get('motion_threshold')
            self._anomaly_window     = deque(maxlen=scaler_data.get('window', 60))

            m1 = EntropyAutoencoder(scaler_data['input_dim'])
            m1.load_state_dict(
                torch.load(MODEL_DIR / 'model1_autoencoder.pt',
                           map_location='cpu', weights_only=True))
            m1.eval()
            self._anomaly_bundle  = {'model': m1, 'scaler_data': scaler_data}
            self._anomaly_backend = 'autoencoder+motion'

            # ── Model 2: LSTM Predictor ───────────────────────────────────
            with open(MODEL_DIR / 'model2_norm.json') as f:
                self._predictor_norm = json.load(f)
            m2 = EntropyPredictor(
                self._predictor_norm['state_dim'],
                self._predictor_norm['hash_dim'],
                64,
            )
            m2.load_state_dict(
                torch.load(MODEL_DIR / 'model2_predictor.pt',
                           map_location='cpu', weights_only=True))
            m2.eval()
            self._predictor_model = m2

            # ── Model 4: Classifier ───────────────────────────────────────
            with open(MODEL_DIR / 'model4_classifier.pkl', 'rb') as f:
                self._classifier_bundle = pickle.load(f)

            self._models_loaded = True
            print(f"[ML] Models loaded successfully "
                  f"(anomaly backend: {self._anomaly_backend}).")

        except FileNotFoundError:
            print("[ML] Model files not found — "
                  "rule-based health scoring active until training completes.")
        except Exception as exc:
            print(f"[ML] Load error ({exc}) — falling back to rule-based scoring.")

    # ------------------------------------------------------------------
    # State update (called every 50 ms from _feed_loop)
    # ------------------------------------------------------------------

    def update_state(self, chaos_states: dict, hash_bytes: bytes):
        """
        Buffer the latest chaos state and hash output for model inference.
        chaos_states must contain 'pendulum' and 'lorenz' sub-dicts.
        """
        p = chaos_states.get('pendulum', {})
        l = chaos_states.get('lorenz',   {})

        state_vec = [
            p.get('theta1', 0.0), p.get('theta2', 0.0),
            p.get('omega1', 0.0), p.get('omega2', 0.0),
            l.get('x',      0.0), l.get('y',      0.0), l.get('z', 0.0),
        ]
        self._state_window.append(state_vec)
        self._last_hash = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------

    def evaluate(self, metrics: dict) -> dict:
        """
        Augments the metrics dict with ML-derived fields.
        Input keys expected: entropy_score, distribution_uniformity,
                             duplicate_rate, pool_fill_percent.
        Returns a copy of metrics extended with:
          ml_active, health_status (overridden when ML active),
          health_confidence, anomaly_score, is_anomaly, prediction_resistance.
        """
        result = dict(metrics)

        if not self._models_loaded:
            result = self._rule_based_fallback(result)
            result['ml_active'] = False
            return result

        result['ml_active'] = True

        # ── Model 4: Classifier → primary health_status ───────────────
        result = self._run_classifier(result)

        # ── Model 1: Autoencoder → anomaly score ─────────────────────
        result = self._run_anomaly(result, metrics)

        # ── Model 2: LSTM → prediction resistance ────────────────────
        result = self._run_predictor(result)

        return result

    # ------------------------------------------------------------------
    # Rule-based fallback (replicates health.py thresholds)
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_based_fallback(result: dict) -> dict:
        score = result.get('entropy_score', 0.0)
        if score >= 0.99:
            result['health_status'] = 'excellent'
        elif score >= 0.95:
            result['health_status'] = 'good'
        elif score >= 0.90:
            result['health_status'] = 'degraded'
        else:
            result['health_status'] = 'critical'
        return result

    # ------------------------------------------------------------------
    # Model 4 — Classifier
    # ------------------------------------------------------------------

    def _run_classifier(self, result: dict) -> dict:
        try:
            bundle  = self._classifier_bundle
            feats   = bundle['features']
            X       = np.array([[result.get(f, 0.0) for f in feats]])
            X_sc    = bundle['scaler'].transform(X)
            cls     = bundle['model'].predict(X_sc)[0]
            proba   = bundle['model'].predict_proba(X_sc)[0]
            result['health_status']      = cls
            result['health_confidence']  = float(proba.max())
        except Exception:
            result['health_confidence'] = 0.0
        return result

    # ------------------------------------------------------------------
    # Model 1 — Hybrid: Autoencoder + pendulum motion check
    # ------------------------------------------------------------------

    def _run_anomaly(self, result: dict, metrics: dict) -> dict:
        import torch
        try:
            scaler_data = self._anomaly_bundle['scaler_data']
            feats       = self._anomaly_features
            means       = np.array([scaler_data['mean'].get(f, 0.0) for f in feats],
                                    dtype=np.float32)
            stds        = np.array([scaler_data['std'].get(f, 1.0)  for f in feats],
                                    dtype=np.float32)

            # Snapshot: base metrics + any state values passed via result
            p = result.get('_pendulum_state', {})
            l = result.get('_lorenz_state',   {})
            lx = l.get('x', 0.0); ly = l.get('y', 0.0); lz = l.get('z', 0.0)
            # Build dict — not all features are live metrics; unknowns stay 0
            snap = dict(metrics)
            snap.update({
                'lorenz_x': lx, 'lorenz_y': ly, 'lorenz_z': lz,
                'lorenz_r': np.sqrt(lx**2 + ly**2 + lz**2),
                '_omega1': p.get('omega1', None),  # stored for motion check
            })
            self._anomaly_window.append(snap)

            window_len = self._anomaly_window.maxlen
            if len(self._anomaly_window) < window_len:
                result['anomaly_score'] = 0.0
                result['is_anomaly']    = False
                return result

            win_dicts = list(self._anomaly_window)

            # ── A) Autoencoder reconstruction ─────────────────────────────
            win_arr  = np.array([[d.get(f, 0.0) for f in feats]
                                 for d in win_dicts], dtype=np.float32)
            # Compute rolling mean/std features for the window
            win_df = pd.DataFrame({f: win_arr[:, i] for i, f in enumerate(feats)})
            for col in ['entropy_score', 'distribution_uniformity']:
                if f'{col}_roll_mean' in feats:
                    win_df[f'{col}_roll_mean'] = win_df[col].rolling(window_len).mean().bfill()
                if f'{col}_roll_std' in feats:
                    win_df[f'{col}_roll_std'] = win_df[col].rolling(window_len).std().bfill()
                if f'{col}_delta' in feats:
                    win_df[f'{col}_delta'] = win_df[col].diff().fillna(0)
            if 'lorenz_r_roll_mean' in feats:
                win_df['lorenz_r_roll_mean'] = win_df['lorenz_r'].rolling(window_len).mean().bfill()

            win_final  = win_df[feats].values.astype(np.float32)
            win_norm   = (win_final - means) / stds
            x_t        = torch.tensor(win_norm.flatten(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                recon  = self._anomaly_bundle['model'](x_t)
                ae_err = float(((recon - x_t) ** 2).mean())

            ae_anomaly = ae_err > self._anomaly_threshold

            # ── B) Pendulum motion check for frozen pendulum ───────────────
            motion_anomaly = False
            if self._motion_threshold is not None:
                omega1_series = [d.get('_omega1') for d in win_dicts
                                 if d.get('_omega1') is not None]
                if len(omega1_series) >= 2:
                    diffs       = np.abs(np.diff(omega1_series))
                    motion_mean = float(np.mean(diffs)) if len(diffs) > 0 else 1.0
                    motion_anomaly = motion_mean < self._motion_threshold

            is_anomaly = bool(ae_anomaly or motion_anomaly)

            # Normalise AE error to [0, 1] for display
            norm_score = float(np.clip(
                ae_err / max(self._anomaly_threshold * 2, 1e-8), 0.0, 1.0))
            if motion_anomaly and not ae_anomaly:
                norm_score = 0.8   # motion-only anomaly → fixed indicator score

            result['anomaly_score'] = round(norm_score, 4)
            result['is_anomaly']    = is_anomaly
        except Exception:
            result['anomaly_score'] = 0.0
            result['is_anomaly']    = False
        return result

    # ------------------------------------------------------------------
    # Model 2 — LSTM prediction resistance
    # ------------------------------------------------------------------

    def _run_predictor(self, result: dict) -> dict:
        import torch
        try:
            if len(self._state_window) < 30 or self._last_hash is None:
                result['prediction_resistance'] = None
                return result

            norm        = self._predictor_norm
            state_mean  = np.array(norm['state_mean'], dtype=np.float32)
            state_std   = np.array(norm['state_std'],  dtype=np.float32)
            state_arr   = np.array(list(self._state_window), dtype=np.float32)
            state_norm  = (state_arr - state_mean) / state_std

            x_t = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = self._predictor_model(x_t).squeeze(0).numpy()

            actual     = self._last_hash / 255.0
            error      = float(((pred - actual) ** 2).mean())
            resistance = float(np.clip(error, 0.0, 1.0))
            result['prediction_resistance'] = round(resistance, 4)
        except Exception:
            result['prediction_resistance'] = None
        return result
