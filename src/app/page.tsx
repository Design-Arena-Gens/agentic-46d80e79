import Link from "next/link";

export default function Home() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Backpropagation Lab</h1>
      <p className="max-w-2xl text-zinc-600 dark:text-zinc-400">
        Interactive, in-browser demos that show how gradient descent and backpropagation
        learn parameters in simple and multi-layer neural networks. Explore linear and
        logistic regression, XOR with a two-layer MLP, and a spiral classification demo.
      </p>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Card title="Linear Regression" href="/linear" desc="Fit y = ax + b with MSE" />
        <Card title="Logistic Regression" href="/logistic" desc="2D binary classification" />
        <Card title="XOR (2-layer MLP)" href="/xor" desc="Non-linear decision boundary" />
        <Card title="Spiral (MLP)" href="/spiral" desc="Challenging toy dataset" />
      </div>
      <div className="text-sm text-zinc-500">
        Built with Next.js, Tailwind, and TensorFlow.js. All training runs locally in your browser.
      </div>
    </div>
  );
}

function Card({ title, href, desc }: { title: string; href: string; desc: string }) {
  return (
    <Link
      href={href}
      className="block rounded-lg border border-zinc-200 p-4 transition hover:border-zinc-300 hover:bg-zinc-50 dark:border-zinc-800 dark:hover:border-zinc-700 dark:hover:bg-zinc-900"
    >
      <div className="text-lg font-semibold">{title}</div>
      <div className="text-sm text-zinc-600 dark:text-zinc-400">{desc}</div>
    </Link>
  );
}
