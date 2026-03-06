import { useEffect, useRef, useState } from 'react';

export interface PendulumState {
  theta1: number;
  theta2: number;
  omega1: number;
  omega2: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface LorenzState {
  x: number;
  y: number;
  z: number;
}

export interface ReactionDiffusionState {
  grid_b64: string;
}

export interface ChaosData {
  t: number;
  pendulum: PendulumState;
  lorenz: LorenzState;
  reaction_diffusion: ReactionDiffusionState;
}

export interface EntropyData {
  t: number;
  pool_fill_percent: number;
  entropy_score: number;
  distribution_uniformity: number;
  duplicate_rate: number;
  health_status: string;
  tokens_generated_total: number;
  hashes_per_second: number;
  entropy_rate_bps: number;
}

// In production set VITE_BACKEND_URL=https://your-api.example.com
// http → ws and https → wss is handled automatically by the replace below
const _backendBase = (import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8001').replace(/\/$/, '');
const _wsBase       = _backendBase.replace(/^http/, 'ws');
const CHAOS_WS_URL   = `${_wsBase}/ws/chaos`;
const ENTROPY_WS_URL = `${_wsBase}/ws/entropy`;
const MAX_BACKOFF_MS = 5000;

function useReconnectingWebSocket<T>(
  url: string,
  onMessage: (data: T) => void,
  onDisconnect?: () => void,
) {
  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(500);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;

    const connect = () => {
      if (!mountedRef.current) return;

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        backoffRef.current = 500; // reset backoff on successful connection
      };

      ws.onmessage = (event) => {
        try {
          const data: T = JSON.parse(event.data);
          onMessage(data);
        } catch {
          // ignore parse errors
        }
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        // Clear stale data so components fall back to demo/idle mode
        onDisconnect?.();
        // Exponential backoff reconnect, capped at MAX_BACKOFF_MS
        const delay = backoffRef.current;
        backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
        setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connect();

    return () => {
      mountedRef.current = false;
      if (wsRef.current) {
        wsRef.current.onclose = null; // prevent reconnect on unmount
        wsRef.current.close();
      }
    };
  }, [url]);
}

export function useKairosSocket(): {
  chaosData: ChaosData | null;
  entropyData: EntropyData | null;
} {
  const [chaosData, setChaosData] = useState<ChaosData | null>(null);
  const [entropyData, setEntropyData] = useState<EntropyData | null>(null);

  useReconnectingWebSocket<ChaosData>(CHAOS_WS_URL, setChaosData, () => setChaosData(null));
  useReconnectingWebSocket<EntropyData>(ENTROPY_WS_URL, setEntropyData, () => setEntropyData(null));

  return { chaosData, entropyData };
}
