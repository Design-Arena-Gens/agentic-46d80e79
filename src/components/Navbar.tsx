"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Home" },
  { href: "/linear", label: "Linear" },
  { href: "/logistic", label: "Logistic" },
  { href: "/xor", label: "XOR" },
  { href: "/spiral", label: "Spiral" },
];

export default function Navbar() {
  const pathname = usePathname();
  return (
    <nav className="w-full border-b border-zinc-200 bg-white/80 backdrop-blur dark:border-zinc-800 dark:bg-black/50">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <div className="text-lg font-semibold tracking-tight">
          <Link href="/">Backpropagation Lab</Link>
        </div>
        <ul className="flex gap-2 text-sm">
          {links.map((l) => {
            const active = pathname === l.href;
            return (
              <li key={l.href}>
                <Link
                  href={l.href}
                  className={
                    "rounded-md px-3 py-1.5 transition-colors " +
                    (active
                      ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-black"
                      : "text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-900")
                  }
                >
                  {l.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </div>
    </nav>
  );
}
