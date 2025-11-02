"use client";
import * as tf from "@tensorflow/tfjs";
import { useEffect, useMemo, useRef, useState } from "react";
import CanvasPlot from "@/components/CanvasPlot";
import { generateLogisticGaussians, normalizePoints, type Point2D } from "@/lib/datasets";

export default function LogisticPage() {
  const [points, setPoints] = useState<Point2D[]>([]);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0);
  const [learningRate, setLearningRate] = useState(0.1);
  const [running, setRunning] = useState(false);

  // Model: p = sigmoid(w1*x + w2*y + b)
  const w = useRef(tf.variable(tf.randomNormal([2, 1], 0, 0.5)));
  const b = useRef(tf.variable(tf.scalar(0)));
  const optimizer = useRef(tf.train.adam(learningRate));

  useEffect(() => {
    setPoints(normalizePoints(generateLogisticGaussians(120)));
  }, []);

  useEffect(() => {
    optimizer.current = tf.train.adam(learningRate);
  }, [learningRate]);

  const X = useMemo(
    () => (points.length ? tf.tensor2d(points.map((p) => [p.x, p.y])) : tf.tensor2d([], [0, 2])),
    [points]
  );
  const y = useMemo(
    () => (points.length ? tf.tensor2d(points.map((p) => [p.label ?? 0])) : tf.tensor2d([], [0, 1])),
    [points]
  );

  function predict(X: tf.Tensor2D) {
    return tf.sigmoid(tf.add(tf.matMul(X, w.current), b.current)) as tf.Tensor2D;
  }

  function step() {
    const lossFn = (): tf.Scalar =>
      (tf.mean(tf.metrics.binaryCrossentropy(y, predict(X))) as unknown) as tf.Scalar;
    const value = optimizer.current.minimize(lossFn, true) as tf.Scalar;
    const l = value.dataSync()[0];
    value.dispose();
    return l;
  }

  useEffect(() => {
    if (!running) return;
    let cancelled = false;
    function loop() {
      if (cancelled) return;
      const l = step();
      setEpoch((e) => e + 1);
      setLoss(l);
      requestAnimationFrame(loop);
    }
    loop();
    return () => {
      cancelled = true;
    };
  }, [running]);

  const decision = (x: number, yv: number) => {
    const out = tf.tidy(() => {
      const p = predict(tf.tensor2d([[x, yv]]));
      return p.dataSync()[0];
    });
    return out;
  };

  function reset() {
    w.current.assign(tf.randomNormal([2, 1], 0, 0.5));
    b.current.assign(tf.scalar(0));
    setEpoch(0);
    setLoss(0);
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold">Logistic Regression (Binary Classification)</h2>
      <p className="text-sm text-zinc-600 dark:text-zinc-400">Model: p(y=1|x) = ?(w?x + b). Optimized with Adam on log-loss.</p>
      <div className="flex flex-wrap gap-6">
        <CanvasPlot points={points} decision={decision} />
        <div className="min-w-[260px] space-y-3">
          <div className="rounded-md border border-zinc-200 p-3 text-sm dark:border-zinc-800">
            <div><span className="font-medium">Epoch:</span> {epoch}</div>
            <div><span className="font-medium">Loss:</span> {loss.toFixed(6)}</div>
            <div><span className="font-medium">||w||:</span> {norm(w.current).toFixed(4)}</div>
            <div><span className="font-medium">b:</span> {b.current.dataSync()[0].toFixed(4)}</div>
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
              onClick={() => setPoints(normalizePoints(generateLogisticGaussians(120)))}
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

function norm(t: tf.Tensor) {
  const v = t.flatten() as tf.Tensor1D;
  const arr = v.dataSync();
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i] * arr[i];
  v.dispose();
  return Math.sqrt(sum);
}
