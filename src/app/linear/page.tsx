"use client";
import * as tf from "@tensorflow/tfjs";
import { useEffect, useMemo, useRef, useState } from "react";
import CanvasPlot from "@/components/CanvasPlot";
import { generateLinearData, normalizePoints, type Point2D } from "@/lib/datasets";

export default function LinearPage() {
  const [points, setPoints] = useState<Point2D[]>([]);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0);
  const [learningRate, setLearningRate] = useState(0.1);
  const [running, setRunning] = useState(false);

  const slope = useRef(tf.variable(tf.scalar(Math.random() * 2 - 1)));
  const intercept = useRef(tf.variable(tf.scalar(Math.random() * 2 - 1)));
  const optimizer = useRef(tf.train.sgd(learningRate));

  useEffect(() => {
    setPoints(normalizePoints(generateLinearData(100, 1.8, -0.3, 0.05)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    optimizer.current = tf.train.sgd(learningRate);
  }, [learningRate]);

  function predict(xs: tf.Tensor1D) {
    return tf.add(tf.mul(xs, slope.current), intercept.current) as tf.Tensor1D;
  }

  function step(batchX: tf.Tensor1D, batchY: tf.Tensor1D) {
    const lossFn = (): tf.Scalar =>
      (tf.mean(tf.square(tf.sub(batchY, predict(batchX)))) as unknown) as tf.Scalar;
    const value = optimizer.current.minimize(lossFn, true) as tf.Scalar;
    const l = value.dataSync()[0];
    value.dispose();
    return l;
  }

  const xs = useMemo(() => tf.tensor1d(points.map((p) => p.x)), [points]);
  const ys = useMemo(() => tf.tensor1d(points.map((p) => p.y)), [points]);

  useEffect(() => {
    if (!running) return;
    let cancelled = false;
    function loop() {
      if (cancelled) return;
      const l = step(xs as tf.Tensor1D, ys as tf.Tensor1D);
      setEpoch((e) => e + 1);
      setLoss(l);
      requestAnimationFrame(loop);
    }
    loop();
    return () => {
      cancelled = true;
    };
  }, [running, xs, ys]);

  const m = useMemo(() => slope.current.dataSync()[0], [epoch]);
  const b = useMemo(() => intercept.current.dataSync()[0], [epoch]);

  function reset() {
    slope.current.assign(tf.scalar(Math.random() * 2 - 1));
    intercept.current.assign(tf.scalar(Math.random() * 2 - 1));
    setEpoch(0);
    setLoss(0);
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold">Linear Regression (Backprop through MSE)</h2>
      <p className="text-sm text-zinc-600 dark:text-zinc-400">Model: y = m x + b. Optimized with SGD on mean squared error.</p>
      <div className="flex flex-wrap gap-6">
        <CanvasPlot points={points} line={{ m, b }} />
        <div className="min-w-[260px] space-y-3">
          <div className="rounded-md border border-zinc-200 p-3 text-sm dark:border-zinc-800">
            <div><span className="font-medium">Epoch:</span> {epoch}</div>
            <div><span className="font-medium">Loss:</span> {loss.toFixed(6)}</div>
            <div><span className="font-medium">m:</span> {m.toFixed(4)}</div>
            <div><span className="font-medium">b:</span> {b.toFixed(4)}</div>
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium">Learning rate: {learningRate}</label>
            <input
              type="range"
              min={0.001}
              max={0.5}
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
              onClick={() => setPoints(normalizePoints(generateLinearData(100, 1.8, -0.3, 0.05)))}
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
