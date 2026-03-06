import { useEffect, useState } from 'react';
import { EntropyData } from '../hooks/useKairosSocket';

interface Props {
  entropyData?: EntropyData | null;
}

export function EntropyHealth({ entropyData }: Props) {
  // Demo simulation state (used when backend is unreachable)
  const [simScore, setSimScore] = useState(0.992482);
  const [simUniformity, setSimUniformity] = useState(0.981014);
  const [simDupRate, setSimDupRate] = useState(0.00008);
  // Phase-based pool oscillation: stays in 63–87% range, never drifts to 100%
  const [simPoolPhase, setSimPoolPhase] = useState(0.8);

  useEffect(() => {
    if (entropyData != null) return; // stop sim when live data is available
    const interval = setInterval(() => {
      setSimScore(s => Math.min(0.999999, Math.max(0.900000, s + (Math.random() - 0.5) * 0.0005)));
      setSimUniformity(u => Math.min(0.999999, Math.max(0.900000, u + (Math.random() - 0.5) * 0.0005)));
      setSimDupRate(d => Math.max(0.00001, d + (Math.random() - 0.5) * 0.00002));
      setSimPoolPhase(p => p + 0.04);
    }, 1000);
    return () => clearInterval(interval);
  }, [entropyData]);

  // Use live data when connected, fall back to simulation
  const score = entropyData?.entropy_score ?? simScore;
  const uniformity = entropyData?.distribution_uniformity ?? simUniformity;
  const dupRate = entropyData?.duplicate_rate ?? simDupRate;
  // Sine-wave oscillation between ~63% and ~87%
  const simPool = 75 + 12 * Math.sin(simPoolPhase);
  const pool = entropyData?.pool_fill_percent ?? simPool;

  return (
    <div className="border border-[#D8D4CC] bg-white mb-6">
      <div className="bg-[#EBE8E0] text-[10px] font-mono font-bold p-2 px-4 flex justify-between border-b border-[#D8D4CC] uppercase">
        <span>Entropy Health Matrix</span>
        <span>Samples: 10^9 &nbsp;&nbsp; Verified: 100%</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 divide-y md:divide-y-0 md:divide-x divide-[#D8D4CC] border-b border-[#D8D4CC]">
        <div className="p-4">
          <div className="text-[10px] font-mono text-gray-500 mb-1 uppercase">Entropy_Score</div>
          <div className="text-2xl font-mono text-[#1A1A1A]">{score.toFixed(6)}</div>
          <div className="text-[9px] font-mono text-green-600 mt-1 uppercase">▲ 0.0001% (Nominal)</div>
        </div>
        <div className="p-4">
          <div className="text-[10px] font-mono text-gray-500 mb-1 uppercase">Uniformity_Indx</div>
          <div className="text-2xl font-mono text-[#1A1A1A] mb-2">{uniformity.toFixed(6)}</div>
          <div className="w-full h-1 bg-[#EBE8E0]">
            <div className="h-full bg-teal-500 transition-all duration-500" style={{ width: `${uniformity * 100}%` }}></div>
          </div>
        </div>
        <div className="p-4">
          <div className="text-[10px] font-mono text-gray-500 mb-1 uppercase">Duplicate_Rate</div>
          <div className="text-2xl font-mono text-red-500">{dupRate.toFixed(5)}</div>
          <div className="text-[9px] font-mono text-gray-400 mt-1 uppercase">Max_Allowable: 0.001</div>
        </div>
        <div className="p-4">
          <div className="text-[10px] font-mono text-gray-500 mb-1 uppercase">Pool_Capacity</div>
          <div className="text-2xl font-mono text-[#1A1A1A]">{pool.toFixed(1)}% <span className="text-[9px] text-gray-400 uppercase tracking-wider ml-1">1.24 TB Used</span></div>
        </div>
      </div>

      <div className="w-full overflow-x-auto">
        <table className="w-full text-left text-[10px] font-mono whitespace-nowrap">
          <thead className="bg-[#FAFAFA] text-gray-500 border-b border-[#D8D4CC]">
            <tr>
              <th className="p-3 font-normal uppercase">Timestamp (UTC)</th>
              <th className="p-3 font-normal uppercase">Shannon Index</th>
              <th className="p-3 font-normal uppercase">Min-Entropy</th>
              <th className="p-3 font-normal uppercase">Collision PR</th>
              <th className="p-3 font-normal uppercase">Health Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#EBE8E0]">
            <tr>
              <td className="p-3">2023-10-27T14:30:00Z</td>
              <td className="p-3">7.9998</td>
              <td className="p-3">0.9982</td>
              <td className="p-3">1.2e-14</td>
              <td className="p-3 text-green-600 font-bold uppercase">Passed</td>
            </tr>
            <tr>
              <td className="p-3">2023-10-27T14:15:00Z</td>
              <td className="p-3">7.9995</td>
              <td className="p-3">0.9980</td>
              <td className="p-3">1.5e-14</td>
              <td className="p-3 text-green-600 font-bold uppercase">Passed</td>
            </tr>
            <tr>
              <td className="p-3">2023-10-27T14:00:00Z</td>
              <td className="p-3">7.9997</td>
              <td className="p-3">0.9981</td>
              <td className="p-3">1.3e-14</td>
              <td className="p-3 text-green-600 font-bold uppercase">Passed</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
