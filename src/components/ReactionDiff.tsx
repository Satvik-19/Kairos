import { useEffect, useRef, useState } from 'react';

interface Props {
  gridData?: string | null;
}

// High-contrast colormap: cream (V=0) → amber → deep red → near-black
// Auto-normalized values ensure full color range regardless of V magnitude
function valueToRGB(v: number): [number, number, number] {
  if (v <= 0.25) {
    const t = v / 0.25;
    return [
      Math.round(245 + t * (240 - 245)),
      Math.round(240 + t * (160 - 240)),
      Math.round(232 + t * (80 - 232)),
    ];
  } else if (v <= 0.55) {
    const t = (v - 0.25) / 0.30;
    return [
      Math.round(240 + t * (200 - 240)),
      Math.round(160 + t * (50 - 160)),
      Math.round(80 + t * (20 - 80)),
    ];
  } else if (v <= 0.80) {
    const t = (v - 0.55) / 0.25;
    return [
      Math.round(200 + t * (120 - 200)),
      Math.round(50 + t * (10 - 50)),
      Math.round(20 + t * (10 - 20)),
    ];
  } else {
    const t = (v - 0.80) / 0.20;
    return [
      Math.round(120 + t * (30 - 120)),
      Math.round(10 + t * (5 - 10)),
      Math.round(10 + t * (5 - 10)),
    ];
  }
}

export function ReactionDiff({ gridData }: Props) {
  const [act, setAct] = useState(42.8);
  const [pos, setPos] = useState({ x1: 50, y1: 50, x2: 30, y2: 70, x3: 70, y3: 30 });
  const [vStats, setVStats] = useState<{ min: number; max: number } | null>(null);

  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const displayCanvasRef = useRef<HTMLCanvasElement>(null);

  // CSS animation for demo mode (runs only when no live data)
  useEffect(() => {
    if (gridData != null) return;
    let t = 0;
    const interval = setInterval(() => {
      t += 0.05;
      setAct(40 + Math.sin(t) * 15 + Math.random() * 2);
      setPos({
        x1: 50 + Math.sin(t * 0.8) * 30,
        y1: 50 + Math.cos(t * 1.1) * 30,
        x2: 50 + Math.cos(t * 0.5) * 40,
        y2: 50 + Math.sin(t * 0.9) * 40,
        x3: 50 + Math.sin(t * 1.3) * 20,
        y3: 50 + Math.cos(t * 0.6) * 20,
      });
    }, 50);
    return () => clearInterval(interval);
  }, [gridData]);

  // Render grid_b64 onto display canvas when live data arrives
  useEffect(() => {
    if (gridData == null) return;
    const displayCanvas = displayCanvasRef.current;
    if (!displayCanvas) return;

    try {
      const binaryStr = atob(gridData);
      const bytes = new Uint8Array(binaryStr.length);
      for (let i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
      }
      const floats = new Float32Array(bytes.buffer);
      const N = 64;

      if (!offscreenRef.current) {
        offscreenRef.current = document.createElement('canvas');
        offscreenRef.current.width = N;
        offscreenRef.current.height = N;
      }
      const offscreen = offscreenRef.current;
      const octx = offscreen.getContext('2d');
      if (!octx) return;

      const imageData = octx.createImageData(N, N);

      // Auto-normalize: find actual min/max of V this frame
      let vMin = Infinity;
      let vMax = -Infinity;
      for (let i = 0; i < floats.length; i++) {
        const f = floats[i];
        if (isFinite(f)) {
          if (f < vMin) vMin = f;
          if (f > vMax) vMax = f;
        }
      }
      // Clamp range to avoid single-color frame when V is near-uniform
      const vRange = vMax - vMin > 1e-6 ? vMax - vMin : 1;

      for (let i = 0; i < N * N; i++) {
        const v = Math.max(0, Math.min(1, (floats[i] - vMin) / vRange));
        const [r, g, b] = valueToRGB(v);
        imageData.data[i * 4 + 0] = r;
        imageData.data[i * 4 + 1] = g;
        imageData.data[i * 4 + 2] = b;
        imageData.data[i * 4 + 3] = 255;
      }
      octx.putImageData(imageData, 0, 0);

      const dctx = displayCanvas.getContext('2d');
      if (!dctx) return;
      dctx.imageSmoothingEnabled = false;
      dctx.drawImage(offscreen, 0, 0, displayCanvas.width, displayCanvas.height);

      setVStats({ min: vMin, max: vMax });
    } catch {
      // Ignore decode/render errors — stay in last valid state
    }
  }, [gridData]);

  // Demo mode — CSS gradient animation
  if (gridData == null) {
    return (
      <div className="pb-4 border-b border-[#EBE8E0] last:border-0 last:pb-0">
        <div className="flex justify-between items-center mb-2">
          <span className="text-[11px] font-bold uppercase">Node 03: Reaction</span>
          <span className="text-[9px] font-mono text-gray-800 border border-gray-300 px-1 py-0.5">GS_MODEL</span>
        </div>
        <div
          className="w-full h-36 border border-dashed border-[#D8D4CC] relative flex items-center justify-center overflow-hidden bg-[#FAFAFA]"
          style={{
            background: `
              radial-gradient(circle at ${pos.x1}% ${pos.y1}%, #F43F5E66 0%, transparent 50%),
              radial-gradient(circle at ${pos.x2}% ${pos.y2}%, #F43F5E88 0%, transparent 40%),
              radial-gradient(circle at ${pos.x3}% ${pos.y3}%, #F43F5E55 0%, transparent 60%)
            `
          }}
        >
          <div
            className="w-full h-full opacity-50 mix-blend-multiply"
            style={{
              backgroundImage: 'radial-gradient(#F43F5E 1px, transparent 1px)',
              backgroundSize: '4px 4px'
            }}
          />
          <div className="absolute bottom-1 right-1 text-[9px] font-mono text-gray-400 bg-white/80 px-1 border border-[#EBE8E0]">
            ACT: {act.toFixed(1)}%
          </div>
        </div>
      </div>
    );
  }

  // Live mode — canvas rendering of real Gray-Scott V grid
  return (
    <div className="pb-4 border-b border-[#EBE8E0] last:border-0 last:pb-0">
      <div className="flex justify-between items-center mb-2">
        <span className="text-[11px] font-bold uppercase">Node 03: Reaction</span>
        <span className="text-[9px] font-mono text-green-700 border border-green-200 px-1 py-0.5">GS_LIVE</span>
      </div>
      <div className="w-full h-36 border border-dashed border-[#D8D4CC] relative overflow-hidden bg-[#FAFAFA]">
        <canvas
          ref={displayCanvasRef}
          width={250}
          height={144}
          className="w-full h-full object-contain"
        />
        <div className="absolute bottom-1 right-1 text-[9px] font-mono text-gray-400 bg-white/80 px-1 border border-[#EBE8E0]">
          {vStats ? `V: ${vStats.min.toFixed(3)}–${vStats.max.toFixed(3)}` : 'GS 64×64'}
        </div>
      </div>
    </div>
  );
}
