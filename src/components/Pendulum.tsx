import { useEffect, useRef } from 'react';
import { PendulumState } from '../hooks/useKairosSocket';

interface Props {
  pendulumData?: PendulumState | null;
}

type Point2 = { x: number; y: number };

/** Batch the fading amber trail into 8 alpha buckets — 8 draw calls instead of 200. */
function drawPendulumTrail(ctx: CanvasRenderingContext2D, trail: Point2[]) {
  if (trail.length < 2) return;
  const BUCKETS = 8;
  for (let b = 0; b < BUCKETS; b++) {
    const start = Math.floor(b * trail.length / BUCKETS);
    // +1 overlap so no gap between buckets
    const end = Math.min(Math.floor((b + 1) * trail.length / BUCKETS) + 1, trail.length);
    if (end <= start + 1) continue;
    const alpha = ((b + 0.5) / BUCKETS) * 0.6;
    ctx.beginPath();
    ctx.moveTo(trail[start].x, trail[start].y);
    for (let i = start + 1; i < end; i++) {
      ctx.lineTo(trail[i].x, trail[i].y);
    }
    ctx.strokeStyle = `rgba(245, 158, 11, ${alpha})`;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

export function Pendulum({ pendulumData }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  // Trail of second bob — 100 points keeps rendering fast
  const trailRef = useRef<Point2[]>([]);

  // ── LIVE MODE: draw from backend state ──────────────────────────────────
  useEffect(() => {
    if (!pendulumData) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const scale = 35;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2 - 20;

    // Backend y-axis: y1 = -L1*cos(θ1) — negative means downward in physics.
    // Canvas y-axis is inverted (positive = down), so we negate.
    const x1c = cx + pendulumData.x1 * scale;
    const y1c = cy - pendulumData.y1 * scale;
    const x2c = cx + pendulumData.x2 * scale;
    const y2c = cy - pendulumData.y2 * scale;

    trailRef.current.push({ x: x2c, y: y2c });
    if (trailRef.current.length > 100) trailRef.current.shift();

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Dashed guide circle
    ctx.beginPath();
    ctx.arc(cx, cy, scale * 2, 0, 2 * Math.PI);
    ctx.strokeStyle = '#D8D4CC';
    ctx.setLineDash([2, 2]);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);

    drawPendulumTrail(ctx, trailRef.current);

    // Arms
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x1c, y1c);
    ctx.lineTo(x2c, y2c);
    ctx.strokeStyle = '#A3A3A3';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Pivot + bobs
    [[cx, cy, 2], [x1c, y1c, 3], [x2c, y2c, 3]].forEach(([px, py, r]) => {
      ctx.beginPath();
      ctx.arc(px, py, r, 0, 2 * Math.PI);
      ctx.fillStyle = '#F59E0B';
      ctx.fill();
    });

    if (textRef.current) {
      textRef.current.innerText = `θ₁: ${pendulumData.theta1.toFixed(2)} rad`;
    }
  }, [pendulumData]);

  // ── DEMO MODE: frontend simulation ───────────────────────────────────────
  useEffect(() => {
    if (pendulumData != null) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    const r1 = 35, r2 = 35;
    const m1 = 1, m2 = 1;
    // Chaotic initial conditions (not symmetric)
    let a1 = 2.1, a2 = 1.8;
    let a1_v = 0, a2_v = 0;
    const g = 9.81;
    // 4 sub-steps per frame with smaller dt → better numerical accuracy + same speed
    const dt = 0.004;
    const SUBSTEPS = 4;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2 - 20;
    let frameCount = 0;
    const trail: Point2[] = [];

    const render = () => {
      frameCount++;

      // Sub-stepped Euler for numerical stability with fewer blow-ups
      for (let s = 0; s < SUBSTEPS; s++) {
        let num1 = -g * (2 * m1 + m2) * Math.sin(a1);
        let num2 = -m2 * g * Math.sin(a1 - 2 * a2);
        let num3 = -2 * Math.sin(a1 - a2) * m2;
        let num4 = a2_v * a2_v * r2 + a1_v * a1_v * r1 * Math.cos(a1 - a2);
        let den = r1 * (2 * m1 + m2 - m2 * Math.cos(2 * a1 - 2 * a2));
        const a1_a = (num1 + num2 + num3 * num4) / den;

        num1 = 2 * Math.sin(a1 - a2);
        num2 = a1_v * a1_v * r1 * (m1 + m2);
        num3 = g * (m1 + m2) * Math.cos(a1);
        num4 = a2_v * a2_v * r2 * m2 * Math.cos(a1 - a2);
        den = r2 * (2 * m1 + m2 - m2 * Math.cos(2 * a1 - 2 * a2));
        const a2_a = (num1 * (num2 + num3 + num4)) / den;

        a1_v += a1_a * dt;
        a2_v += a2_a * dt;
        a1 += a1_v * dt;
        a2 += a2_v * dt;
      }

      // Guard against NaN — reset with slightly varied angles
      if (!isFinite(a1) || !isFinite(a2)) {
        a1 = 2.1 + Math.random() * 0.5;
        a2 = 1.8 + Math.random() * 0.5;
        a1_v = 0; a2_v = 0;
      }

      const x1 = r1 * Math.sin(a1);
      const y1 = r1 * Math.cos(a1);
      const x2 = x1 + r2 * Math.sin(a2);
      const y2 = y1 + r2 * Math.cos(a2);

      trail.push({ x: cx + x2, y: cy + y2 });
      if (trail.length > 100) trail.shift();

      if (frameCount % 5 === 0 && textRef.current) {
        textRef.current.innerText = `θ₁: ${a1.toFixed(2)} rad`;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Guide circle
      ctx.beginPath();
      ctx.arc(cx, cy, r1 + r2, 0, 2 * Math.PI);
      ctx.strokeStyle = '#D8D4CC';
      ctx.setLineDash([2, 2]);
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);

      drawPendulumTrail(ctx, trail);

      // Arms
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + x1, cy + y1);
      ctx.lineTo(cx + x2, cy + y2);
      ctx.strokeStyle = '#A3A3A3';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Bobs
      [[cx, cy, 2], [cx + x1, cy + y1, 3], [cx + x2, cy + y2, 3]].forEach(([px, py, r]) => {
        ctx.beginPath();
        ctx.arc(px, py, r, 0, 2 * Math.PI);
        ctx.fillStyle = '#F59E0B';
        ctx.fill();
      });

      animationFrameId = requestAnimationFrame(render);
    };
    render();
    return () => cancelAnimationFrame(animationFrameId);
  }, [pendulumData]);

  return (
    <div className="pb-4 border-b border-[#EBE8E0] last:border-0 last:pb-0">
      <div className="flex justify-between items-center mb-2">
        <span className="text-[11px] font-bold uppercase">Node 01: Pendulum</span>
        <span className={`text-[9px] font-mono border px-1 py-0.5 ${pendulumData ? 'text-green-700 border-green-200' : 'text-amber-600 border-amber-200'}`}>
          {pendulumData ? 'RK4_LIVE' : 'RK4_DEMO'}
        </span>
      </div>
      <div className="w-full h-36 border border-dashed border-[#D8D4CC] relative flex items-center justify-center bg-[#FAFAFA] overflow-hidden">
        <canvas ref={canvasRef} width={250} height={144} className="w-full h-full object-contain" />
        <div ref={textRef} className="absolute bottom-1 right-1 text-[9px] font-mono text-gray-400 bg-white/80 px-1 border border-[#EBE8E0]">
          θ₁: 0.00 rad
        </div>
      </div>
    </div>
  );
}
