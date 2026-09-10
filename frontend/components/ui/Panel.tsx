"use client";

import { useState, type ReactNode } from "react";

/**
 * A chart annotation box floating over the map. Collapsible, so the rewind can
 * be played with the panels out of the way (§12 — the particle cloud is the
 * thing the room is watching).
 */
export function Panel({
  title,
  meta,
  children,
  className = "",
  defaultOpen = true,
}: {
  title: string;
  meta?: ReactNode;
  children: ReactNode;
  className?: string;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className={`panel flex min-h-0 flex-col ${className}`}>
      <header className="flex shrink-0 items-center gap-2 border-b border-chart-line/60 px-3 py-2">
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          className="flex items-center gap-1.5 text-ink-bright transition-colors hover:text-survey"
          aria-expanded={open}
        >
          <svg
            viewBox="0 0 10 10"
            className={`h-2 w-2 shrink-0 transition-transform ${open ? "" : "-rotate-90"}`}
            aria-hidden="true"
          >
            <path d="M1 3l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" />
          </svg>
          <h2 className="text-small font-semibold tracking-tight">{title}</h2>
        </button>
        <div className="ml-auto flex items-center gap-1.5">{meta}</div>
      </header>
      {open ? (
        <div className="scrollable min-h-0 flex-1 overflow-y-auto px-3 py-2.5">
          {children}
        </div>
      ) : null}
    </section>
  );
}

/** Label/value row for instrument readouts. */
export function Row({
  label,
  value,
  hint,
  tone = "",
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-[3px]">
      <span className="shrink-0 text-tiny text-ink-dim" title={hint}>
        {label}
      </span>
      <span className={`tnum text-right text-tiny ${tone || "text-ink"}`}>{value}</span>
    </div>
  );
}

/**
 * What a missing measurement means. The fixture set records why each value is
 * absent, and the console shows that reason rather than an empty cell (§2.2).
 */
export function Absent({ reason }: { reason: string }) {
  return (
    <span
      className="cursor-help text-tiny italic text-ink-faint underline decoration-ink-faint/40 decoration-dotted underline-offset-2"
      title={reason}
    >
      not measured
    </span>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-4 text-tiny leading-relaxed text-ink-dim">{children}</p>;
}
