"use client";
import * as tf from "@tensorflow/tfjs";
import { useEffect, useMemo, useRef, useState } from "react";
import CanvasPlot from "@/components/CanvasPlot";
import { generateSpiral, normalizePoints, type Point2D } from "@/lib/datasets";

export default function SpiralPage() {
  const [points, setPoints] = useState<Point2D[]>([]);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0);
  const [learningRate, setLearningRate] = useState(0.01);
  const [running, setRunning] = useState(false);

  const model = useRef<tf.LayersModel | null>(null);
  const compiled = useRef(false);

  useEffect(() => {
    setPoints(normalizePoints(generateSpiral(150, 2)));
  }, []);

  const X = useMemo(
    () => (points.length ? tf.tensor2d(points.map((p) => [p.x, p.y])) : tf.tensor2d([], [0, 2])),
    [points]
  );
  const y = useMemo(
    () => (points.length ? tf.tensor2d(points.map((p) => [p.label ?? 0])) : tf.tensor2d([], [0, 1])),
    [points]
  );

  function ensureModel() {
    if (model.current) return;
    const m = tf.sequential();
    m.add(tf.layers.dense({ units: 16, inputShape: [2], activation: "tanh" }));
    m.add(tf.layers.dense({ units: 16, activation: "tanh" }));
    m.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
    model.current = m;
  }

  useEffect(() => {
    ensureModel();
    compiled.current = false;
  }, [learningRate]);

  async function trainStep() {
    ensureModel();
    if (!compiled.current) {
      model.current!.compile({ optimizer: tf.train.adam(learningRate), loss: "binaryCrossentropy" });
      compiled.current = true;
    }
    const h = await model.current!.fit(X, y, { epochs: 1, batchSize: 64, verbose: 0 });
    const l = h.history.loss?.[0] as number;
    setEpoch((e) => e + 1);
    setLoss(l);
  }

  useEffect(() => {
    if (!running) return;
    let cancelled = false;
    async function loop() {
      if (cancelled) return;
      await trainStep();
      requestAnimationFrame(loop);
    }
    loop();
    return () => {
      cancelled = true;
    };
  }, [running]);

  const decision = (x: number, yv: number) => {
    ensureModel();
    const out = tf.tidy(() => {
      const p = model.current!.predict(tf.tensor2d([[x, yv]])) as tf.Tensor;
      return p.dataSync()[0] as number;
    });
    return out;
  };

  function reset() {
    model.current = null;
    compiled.current = false;
    setEpoch(0);
    setLoss(0);
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold">Spiral Classification (MLP)</h2>
      <p className="text-sm text-zinc-600 dark:text-zinc-400">Two hidden layers with tanh; trained with Adam on log-loss.</p>
      <div className="flex flex-wrap gap-6">
        <CanvasPlot points={points} decision={decision} />
        <div className="min-w-[260px] space-y-3">
          <div className="rounded-md border border-zinc-200 p-3 text-sm dark:border-zinc-800">
            <div><span className="font-medium">Epoch:</span> {epoch}</div>
            <div><span className="font-medium">Loss:</span> {loss.toFixed(6)}</div>
            <div><span className="font-medium">LR:</span> {learningRate}</div>
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium">Learning rate: {learningRate}</label>
            <input
              type="range"
              min={0.001}
              max={0.1}
              step={0.001}
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setRunning((v) => !v)}
              className="rounded-md bg-zinc-900 px-3 py-1.5 text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-black dark:hover:bg-zinc-200"
            >
              {running ? "Pause" : "Start"}
            </button>
            <button
              onClick={reset}
              className="rounded-md border border-zinc-300 px-3 py-1.5 hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-900"
            >
              Reset
            </button>
            <button
              onClick={() => setPoints(normalizePoints(generateSpiral(150, 2)))}
              className="rounded-md border border-zinc-300 px-3 py-1.5 hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-900"
            >
              New Data
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
