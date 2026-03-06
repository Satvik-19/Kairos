import { useEffect, useState } from 'react';
import { EntropyData, ChaosData } from '../hooks/useKairosSocket';

interface Props {
  entropyData?: EntropyData | null;
  chaosData?: ChaosData | null;
}

export function SeedParameters({ entropyData, chaosData }: Props) {
  // Demo simulation state — animated while backend is offline
  const [simChaos, setSimChaos] = useState(0.4492);
  const [simHashRate, setSimHashRate] = useState(138.4);
  const [simDamping, setSimDamping] = useState(0.00021);
  const [simPhase, setSimPhase] = useState(0);

  useEffect(() => {
    if (entropyData != null) return; // yield to live data
    const interval = setInterval(() => {
      setSimPhase(p => p + 0.04);
      setSimChaos(v => {
        const next = v + (Math.random() - 0.48) * 0.003;
        return Math.min(0.72, Math.max(0.38, next));
      });
      setSimHashRate(v => {
        const next = v + (Math.random() - 0.5) * 4;
        return Math.min(180, Math.max(90, next));
      });
      setSimDamping(v => {
        const next = v + (Math.random() - 0.5) * 0.00004;
        return Math.min(0.0008, Math.max(0.00005, next));
      });
    }, 400);
    return () => clearInterval(interval);
  }, [entropyData]);

  const isLive = entropyData != null;

  // System Chaos: entropy_score (0–1 normalized score from health monitor)
  const chaos = isLive ? (entropyData.entropy_score ?? simChaos) : simChaos;

  // Hash Rate: hashes_per_second in kH/s
  const hashRateRaw = isLive ? (entropyData.hashes_per_second ?? simHashRate * 1000) : simHashRate * 1000;
  const hashRateKh = hashRateRaw / 1000;

  // Damping Factor: duplicate_rate (low = good)
  const damping = isLive ? (entropyData.duplicate_rate ?? simDamping) : simDamping;

  // Pendulum jitter from live omega values (or simulated)
  const pendulumJitter = chaosData?.pendulum
    ? Math.abs(chaosData.pendulum.omega1).toFixed(3)
    : (2.1 + Math.sin(simPhase) * 1.4).toFixed(3);

  return (
    <div className="border border-[#D8D4CC] bg-white mt-6">
      <div className="p-4 border-b border-[#D8D4CC] flex justify-between items-center">
        <h3 className="font-display text-lg">Seed Parameters</h3>
        <span className={`text-[9px] font-mono border px-1 py-0.5 ${isLive ? 'text-green-700 border-green-200' : 'text-gray-500 border-gray-200'}`}>
          {isLive ? 'LIVE' : 'SIM'}
        </span>
      </div>
      <div className="p-4 space-y-4">
        <div>
          <div className="flex justify-between text-[10px] font-mono mb-1">
            <span className="text-gray-500 uppercase">System Chaos</span>
            <span>{chaos.toFixed(4)}</span>
          </div>
          <div className="w-full h-1.5 bg-[#EBE8E0]">
            <div
              className="h-full bg-[#1A1A1A] transition-all duration-300"
              style={{ width: `${Math.min(100, chaos * 100).toFixed(1)}%` }}
            />
          </div>
        </div>
        <div className="flex justify-between text-[10px] font-mono border-t border-[#EBE8E0] pt-4">
          <span className="text-gray-500 uppercase">Hash Rate</span>
          <span>{hashRateKh.toFixed(1)} kH/s</span>
        </div>
        <div className="flex justify-between text-[10px] font-mono border-t border-[#EBE8E0] pt-4">
          <span className="text-gray-500 uppercase">Damping Factor</span>
          <span>{damping.toFixed(5)}</span>
        </div>
        <div className="flex justify-between text-[10px] font-mono border-t border-[#EBE8E0] pt-4">
          <span className="text-gray-500 uppercase">Pendulum ω₁</span>
          <span>{pendulumJitter} rad/s</span>
        </div>
      </div>
    </div>
  );
}
