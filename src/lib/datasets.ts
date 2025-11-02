export type Point2D = { x: number; y: number; label?: number };

export function generateLinearData(n: number, a = 2, b = -1, noise = 0.1): Point2D[] {
  const data: Point2D[] = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 2 - 1; // [-1,1]
    const y = a * x + b + randn(0, noise);
    data.push({ x, y });
  }
  return data;
}

export function generateLogisticGaussians(nPerClass = 100): Point2D[] {
  const data: Point2D[] = [];
  for (let i = 0; i < nPerClass; i++) {
    const [x, y] = gaussian2D(-0.5, -0.2, 0.2);
    data.push({ x, y, label: 0 });
  }
  for (let i = 0; i < nPerClass; i++) {
    const [x, y] = gaussian2D(0.6, 0.3, 0.25);
    data.push({ x, y, label: 1 });
  }
  return data;
}

export function generateXOR(nPerQuadrant = 30): Point2D[] {
  const data: Point2D[] = [];
  const jitter = 0.15;
  const quadrants = [
    { x: -0.5, y: -0.5, label: 0 },
    { x: -0.5, y: 0.5, label: 1 },
    { x: 0.5, y: -0.5, label: 1 },
    { x: 0.5, y: 0.5, label: 0 },
  ];
  for (const q of quadrants) {
    for (let i = 0; i < nPerQuadrant; i++) {
      data.push({
        x: q.x + randn(0, jitter),
        y: q.y + randn(0, jitter),
        label: q.label,
      });
    }
  }
  return data;
}

export function generateSpiral(nPerClass = 100, turns = 2): Point2D[] {
  const data: Point2D[] = [];
  for (let label = 0; label < 2; label++) {
    for (let i = 0; i < nPerClass; i++) {
      const r = i / nPerClass;
      const t = turns * Math.PI * r + (label === 0 ? 0 : Math.PI);
      const x = r * Math.cos(t) + randn(0, 0.05);
      const y = r * Math.sin(t) + randn(0, 0.05);
      data.push({ x, y, label });
    }
  }
  return data;
}

export function normalizePoints(points: Point2D[]): Point2D[] {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  return points.map((p) => ({
    x: (p.x - minX) / (maxX - minX) * 2 - 1,
    y: (p.y - minY) / (maxY - minY) * 2 - 1,
    label: p.label,
  }));
}

function randn(mu = 0, sigma = 1): number {
  // Box-Muller transform
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return mu + z * sigma;
}

function gaussian2D(mx: number, my: number, s = 0.2): [number, number] {
  return [mx + randn(0, s), my + randn(0, s)];
}
