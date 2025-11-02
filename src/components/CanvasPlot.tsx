"use client";
import { useEffect, useRef } from "react";
import type { Point2D } from "@/lib/datasets";

export type CanvasPlotProps = {
  width?: number;
  height?: number;
  points?: Point2D[];
  showAxis?: boolean;
  line?: { m: number; b: number } | null;
  decision?: (x: number, y: number) => number;
};

export default function CanvasPlot({
  width = 480,
  height = 360,
  points = [],
  showAxis = true,
  line = null,
  decision,
}: CanvasPlotProps) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    // decision boundary heatmap
    if (decision) {
      const step = 4;
      const img = ctx.createImageData(width, height);
      for (let y = 0; y < height; y += step) {
        for (let x = 0; x < width; x += step) {
          const nx = pxToCoordX(x, width);
          const ny = pxToCoordY(y, height);
          const p = decision(nx, ny); // 0..1
          const [r, g, b] = lerpColor([230, 240, 255], [255, 230, 230], p);
          for (let dy = 0; dy < step; dy++) {
            for (let dx = 0; dx < step; dx++) {
              const idx = ((y + dy) * width + (x + dx)) * 4;
              img.data[idx + 0] = r;
              img.data[idx + 1] = g;
              img.data[idx + 2] = b;
              img.data[idx + 3] = 255;
            }
          }
        }
      }
      ctx.putImageData(img, 0, 0);
    }

    // axis
    if (showAxis) {
      ctx.strokeStyle = "#e4e4e7";
      ctx.lineWidth = 1;
      // x=0
      const x0 = coordToPxX(0, width);
      ctx.beginPath();
      ctx.moveTo(x0, 0);
      ctx.lineTo(x0, height);
      ctx.stroke();
      // y=0
      const y0 = coordToPxY(0, height);
      ctx.beginPath();
      ctx.moveTo(0, y0);
      ctx.lineTo(width, y0);
      ctx.stroke();
    }

    // line y = m x + b
    if (line) {
      const x1 = -1;
      const y1 = line.m * x1 + line.b;
      const x2 = 1;
      const y2 = line.m * x2 + line.b;
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(coordToPxX(x1, width), coordToPxY(y1, height));
      ctx.lineTo(coordToPxX(x2, width), coordToPxY(y2, height));
      ctx.stroke();
    }

    // points
    for (const p of points) {
      ctx.fillStyle = p.label === undefined ? "#2563eb" : p.label ? "#dc2626" : "#2563eb";
      const r = 3;
      ctx.beginPath();
      ctx.arc(coordToPxX(p.x, width), coordToPxY(p.y, height), r, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [width, height, points, showAxis, line, decision]);

  return <canvas ref={ref} className="rounded-md border border-zinc-200 dark:border-zinc-800" />;
}

export function coordToPxX(x: number, width: number) {
  // domain [-1,1] to [0,width]
  return ((x + 1) / 2) * width;
}
export function coordToPxY(y: number, height: number) {
  // domain [-1,1] to [height,0]
  return height - ((y + 1) / 2) * height;
}
export function pxToCoordX(px: number, width: number) {
  return (px / width) * 2 - 1;
}
export function pxToCoordY(py: number, height: number) {
  return (height - py) / height * 2 - 1;
}

function lerpColor(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  return [
    Math.round(a[0] + (b[0] - a[0]) * clamped),
    Math.round(a[1] + (b[1] - a[1]) * clamped),
    Math.round(a[2] + (b[2] - a[2]) * clamped),
  ];
}
