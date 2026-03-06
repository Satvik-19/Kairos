import { Pendulum } from './components/Pendulum';
import { Lorenz } from './components/Lorenz';
import { ReactionDiff } from './components/ReactionDiff';
import { EntropyHealth } from './components/EntropyHealth';
import { TokenGenerator } from './components/TokenGenerator';
import { SeedParameters } from './components/SeedParameters';
import { useKairosSocket } from './hooks/useKairosSocket';

export default function App() {
  const { chaosData, entropyData } = useKairosSocket();

  return (
    <div className="min-h-screen bg-[#F4F2EC] text-[#1A1A1A] font-sans p-4 md:p-8">
      <div className="max-w-[1400px] mx-auto">
        <header className="flex flex-col md:flex-row justify-between items-start md:items-end border-b border-[#D8D4CC] pb-4 mb-6">
          <div className="flex items-baseline space-x-4">
            <h1 className="text-4xl font-display tracking-tight">KAIROS</h1>
            <span className="text-[10px] font-mono text-gray-500 uppercase tracking-widest">// ENTROPY ENGINE V1.2</span>
          </div>
          <div className="mt-4 md:mt-0 flex items-center space-x-6 text-[10px] font-mono text-gray-500">
            <div className="flex items-center bg-white border border-[#D8D4CC] px-3 py-1 rounded-full">
              <span className="w-2 h-2 rounded-full bg-green-500 mr-2"></span>
              SYSTEM_ACTIVE: 442.12.08
            </div>
            <span className="uppercase">COORD: 45.32 N, 122.67 W</span>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Sidebar */}
          <div className="lg:col-span-3 flex flex-col">
            <div className="border border-[#D8D4CC] bg-white">
              <div className="bg-[#1A1A1A] text-white text-[10px] font-mono font-bold p-2 px-4 tracking-wider uppercase">
                Live Nodes
              </div>
              <div className="p-4 space-y-4">
                <Pendulum pendulumData={chaosData?.pendulum ?? null} />
                <Lorenz lorenzData={chaosData?.lorenz ?? null} />
                <ReactionDiff gridData={chaosData?.reaction_diffusion?.grid_b64 ?? null} />
              </div>
            </div>
            <SeedParameters entropyData={entropyData} chaosData={chaosData} />
          </div>

          {/* Main Content */}
          <div className="lg:col-span-9 flex flex-col">
            <EntropyHealth entropyData={entropyData} />
            <TokenGenerator />
          </div>
        </div>

        <footer className="border-t border-[#D8D4CC] pt-4 mt-8 flex flex-col md:flex-row justify-between items-center text-[9px] font-mono text-gray-400 uppercase">
          <div className="mb-4 md:mb-0">
            © 2026 KAIROS
          </div>
          <div className="flex space-x-6">
            <span>Total Accumulated Entropy: 1,422.84 GB</span>
            <span>Network Latency: 4.2ms</span>
            <span className="text-green-600">Integrity: Verified</span>
          </div>
        </footer>
      </div>
    </div>
  );
}
