import { useState } from 'react';
import { Copy, RefreshCw, Check } from 'lucide-react';

// In production set VITE_BACKEND_URL=https://your-api.example.com
const BACKEND = (import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8001').replace(/\/$/, '');

function browserFallback(format: string, length: number): string {
  if (format === 'HEXADECIMAL_X64') {
    const bytes = new Uint8Array(length);
    crypto.getRandomValues(bytes);
    return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
  } else if (format === 'BASE64_STD') {
    const bytes = new Uint8Array(length);
    crypto.getRandomValues(bytes);
    return btoa(String.fromCharCode(...bytes));
  } else if (format === 'UUID_V4_RFC') {
    return crypto.randomUUID();
  }
  return '';
}

export function TokenGenerator() {
  const [format, setFormat] = useState('HEXADECIMAL_X64');
  const [length, setLength] = useState(32);
  const [token, setToken] = useState('');
  const [copied, setCopied] = useState(false);
  const [generatedAt, setGeneratedAt] = useState('');
  const [isDemo, setIsDemo] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const nowStamp = () => {
    const now = new Date();
    return `${now.getUTCFullYear()}-${String(now.getUTCMonth() + 1).padStart(2, '0')}-${String(now.getUTCDate()).padStart(2, '0')}_${String(now.getUTCHours()).padStart(2, '0')}:${String(now.getUTCMinutes()).padStart(2, '0')}:${String(now.getUTCSeconds()).padStart(2, '0')}`;
  };

  const generateToken = async () => {
    setIsLoading(true);
    setCopied(false);

    try {
      let result = '';
      let usedDemo = false;

      if (format === 'UUID_V4_RFC') {
        // UUID format — use /token endpoint with uuid format
        const res = await fetch(`${BACKEND}/token?length=16&format=uuid`);
        if (res.ok) {
          const data = await res.json();
          result = data.token;
        } else {
          result = browserFallback(format, length);
          usedDemo = true;
        }
      } else if (format === 'BASE64_STD') {
        const res = await fetch(`${BACKEND}/token?length=${length}&format=base64`);
        if (res.ok) {
          const data = await res.json();
          result = data.token;
        } else {
          result = browserFallback(format, length);
          usedDemo = true;
        }
      } else {
        // HEXADECIMAL_X64 — default hex token
        const res = await fetch(`${BACKEND}/token?length=${length}&format=hex`);
        if (res.ok) {
          const data = await res.json();
          result = data.token;
        } else {
          result = browserFallback(format, length);
          usedDemo = true;
        }
      }

      setToken(result);
      setIsDemo(usedDemo);
      setGeneratedAt(nowStamp());
    } catch {
      // Network error — fall back to browser crypto
      setToken(browserFallback(format, length));
      setIsDemo(true);
      setGeneratedAt(nowStamp());
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(token);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="border border-[#D8D4CC] bg-white">
      <div className="bg-[#EBE8E0] text-[10px] font-mono font-bold p-2 px-4 border-b border-[#D8D4CC] uppercase">
        Generator Configuration
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-12 divide-y lg:divide-y-0 lg:divide-x divide-[#D8D4CC]">
        <div className="lg:col-span-5 p-6 space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-[10px] font-mono text-gray-600 mb-2 uppercase">Encoding Protocol</label>
              <select
                value={format}
                onChange={e => setFormat(e.target.value)}
                className="w-full bg-[#F4F2EC] border-none rounded-none p-2 text-[10px] font-mono focus:ring-1 focus:ring-black outline-none"
              >
                <option>HEXADECIMAL_X64</option>
                <option>BASE64_STD</option>
                <option>UUID_V4_RFC</option>
              </select>
            </div>
            <div>
              <label className="block text-[10px] font-mono text-gray-600 mb-2 uppercase">Hash Iterations</label>
              <input
                type="text"
                defaultValue="1000"
                className="w-full bg-[#F4F2EC] border-none rounded-none p-2 text-[10px] font-mono focus:ring-1 focus:ring-black outline-none"
              />
            </div>
            <div>
              <label className="block text-[10px] font-mono text-gray-600 mb-2 uppercase">Length (Octets)</label>
              <input
                type="number"
                value={length}
                onChange={e => setLength(parseInt(e.target.value) || 0)}
                disabled={format === 'UUID_V4_RFC'}
                className="w-full bg-[#F4F2EC] border-none rounded-none p-2 text-[10px] font-mono focus:ring-1 focus:ring-black disabled:opacity-50 outline-none"
              />
            </div>
            <div>
              <label className="block text-[10px] font-mono text-gray-600 mb-2 uppercase">Entropy Source</label>
              <select
                className="w-full bg-[#F4F2EC] border-none rounded-none p-2 text-[10px] font-mono focus:ring-1 focus:ring-black outline-none"
              >
                <option>HYBRID_ALL_NODES</option>
                <option>NODE_01_ONLY</option>
              </select>
            </div>
          </div>
          <button
            onClick={generateToken}
            disabled={isLoading}
            className="w-full bg-[#1A1A1A] text-white hover:bg-black transition-colors duration-200 font-mono text-[10px] tracking-wider py-3 px-4 flex justify-center items-center cursor-pointer uppercase disabled:opacity-60"
          >
            <RefreshCw className={`w-3.5 h-3.5 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            {isLoading ? 'Synthesizing...' : 'Execute_Synthesis'}
          </button>
        </div>
        <div className="lg:col-span-7 p-6 bg-[#FAFAFA] flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <span className="text-[10px] font-mono text-gray-500 font-bold uppercase">
              OUTPUT STREAM // {format === 'UUID_V4_RFC' ? '128' : length * 8}-BIT
              {isDemo && <span className="ml-2 text-amber-500">(demo)</span>}
            </span>
            <button
              onClick={copyToClipboard}
              disabled={!token}
              className="text-[10px] font-mono text-gray-600 hover:text-black flex items-center transition-colors cursor-pointer border border-[#D8D4CC] bg-white px-2 py-1 uppercase disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {copied ? <Check className="w-3 h-3 mr-1 text-green-600" /> : <Copy className="w-3 h-3 mr-1" />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
          <div className="flex-grow flex items-center justify-center bg-[#F4F2EC] border border-[#D8D4CC] p-6 min-h-[120px]">
            {token ? (
              <p className="font-mono text-lg text-[#1A1A1A] break-all leading-relaxed tracking-widest selection:bg-gray-300 whitespace-pre-wrap">
                {token}
              </p>
            ) : (
              <p className="font-mono text-[11px] text-gray-400 tracking-wider uppercase text-center">
                — awaiting synthesis —<br />
                <span className="text-[9px]">Press Execute_Synthesis to generate</span>
              </p>
            )}
          </div>
          <div className="mt-4 flex justify-between items-center text-[9px] font-mono text-gray-400 uppercase">
            <span>{generatedAt ? `Created: ${generatedAt}` : 'No output yet'}</span>
            <span>Sig: SHA256_RSA</span>
          </div>
        </div>
      </div>
    </div>
  );
}
