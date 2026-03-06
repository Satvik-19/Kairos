import { useEffect, useRef } from 'react';
import { LorenzState } from '../hooks/useKairosSocket';

interface Props {
  lorenzData?: LorenzState | null;
}

type Point3 = { x: number; y: number; z: number };

/** Batch-render a Lorenz trail in chunks — ~20 draw calls instead of 500. */
function drawLorenzTrail(
  ctx: CanvasRenderingContext2D,
  trail: Point3[],
  cx: number,
  cy: number,
  scale: number,
) {
  if (trail.length < 2) return;
  const CHUNK = 20;
  for (let cs = 1; cs < trail.length; cs += CHUNK) {
    const ce = Math.min(cs + CHUNK, trail.length);
    const mid = cs + Math.floor((ce - cs) / 2);
    const p = trail[mid] ?? trail[cs];
    const alpha = (mid / trail.length) * 0.85 + 0.15;
    const t = Math.max(0, Math.min(1, p.z / 50));
    const r = Math.round(20 + t * 225);
    const g = Math.round(184 - t * 100);
    const b = Math.round(166 - t * 155);

    ctx.beginPath();
    const first = trail[cs - 1];
    ctx.moveTo(cx + first.x * scale, cy - (first.z - 25) * scale);
    for (let i = cs; i < ce; i++) {
      ctx.lineTo(cx + trail[i].x * scale, cy - (trail[i].z - 25) * scale);
    }
    ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }
}

export function Lorenz({ lorenzData }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  // Rolling trail — 300 points is plenty; keeps GPU load low
  const trailRef = useRef<Point3[]>([]);

  // ── LIVE MODE: draw from backend Lorenz state ────────────────────────────
  useEffect(() => {
    if (!lorenzData) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const scale = 2.2;

    trailRef.current.push({ x: lorenzData.x, y: lorenzData.y, z: lorenzData.z });
    if (trailRef.current.length > 300) trailRef.current.shift();

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawLorenzTrail(ctx, trailRef.current, cx, cy, scale);

    if (textRef.current) {
      textRef.current.innerText = `x: ${lorenzData.x.toFixed(1)} z: ${lorenzData.z.toFixed(1)}`;
    }
  }, [lorenzData]);

  // ── DEMO MODE: clean Lorenz simulation ────────────────────────────────────
  useEffect(() => {
    if (lorenzData != null) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let x = 0.1, y = 0, z = 0;
    const sigma = 10, rho = 28, beta = 8 / 3, dt = 0.008;
    const points: Point3[] = [];
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const scale = 2.2;
    let frameCount = 0;

    const render = () => {
      frameCount++;
      // Run 12 integration steps per frame for visible movement speed
      for (let i = 0; i < 12; i++) {
        const dx = sigma * (y - x) * dt;
        const dy = (x * (rho - z) - y) * dt;
        const dz = (x * y - beta * z) * dt;
        x += dx; y += dy; z += dz;
        points.push({ x, y, z });
        if (points.length > 300) points.shift();
      }

      if (frameCount % 4 === 0 && textRef.current) {
        textRef.current.innerText = `x: ${x.toFixed(1)} z: ${z.toFixed(1)}`;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawLorenzTrail(ctx, points, cx, cy, scale);

      animationFrameId = requestAnimationFrame(render);
    };
    render();
    return () => cancelAnimationFrame(animationFrameId);
  }, [lorenzData]);

  return (
    <div className="pb-4 border-b border-[#EBE8E0] last:border-0 last:pb-0">
      <div className="flex justify-between items-center mb-2">
        <span className="text-[11px] font-bold uppercase">Node 02: Lorenz</span>
        <span className={`text-[9px] font-mono border px-1 py-0.5 ${lorenzData ? 'text-green-700 border-green-200' : 'text-teal-600 border-teal-200'}`}>
          {lorenzData ? 'LORENZ_LIVE' : 'LORENZ_DEMO'}
        </span>
      </div>
      <div className="w-full h-36 border border-dashed border-[#D8D4CC] relative flex items-center justify-center bg-[#FAFAFA] overflow-hidden">
        <canvas ref={canvasRef} width={250} height={144} className="w-full h-full object-contain" />
        <div ref={textRef} className="absolute bottom-1 right-1 text-[9px] font-mono text-gray-400 bg-white/80 px-1 border border-[#EBE8E0]">
          x: 0.0 z: 0.0
        </div>
      </div>
    </div>
  );
}
