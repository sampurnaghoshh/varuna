import type { ReactNode } from "react";

import type { Source, Verdict } from "@/types/domain";

type Tone = "neutral" | "hazard" | "caution" | "survey" | "clear" | "simulated";

const TONES: Record<Tone, string> = {
  neutral: "border-chart-edge/70 text-ink-dim",
  hazard: "border-hazard/60 bg-hazard/12 text-hazard",
  caution: "border-caution/60 bg-caution/12 text-caution",
  survey: "border-survey/50 bg-survey/10 text-survey",
  clear: "border-clear/50 bg-clear/10 text-clear",
  simulated: "border-caution/45 bg-caution/8 text-caution/90",
};

export function Badge({
  children,
  tone = "neutral",
  title,
}: {
  children: ReactNode;
  tone?: Tone;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 whitespace-nowrap rounded-chart border px-1.5 py-0.5 text-micro ${TONES[tone]}`}
    >
      {children}
    </span>
  );
}

/**
 * Renders provenance badges exactly as the data carries them (§2.3).
 *
 * Nothing here matches on badge text, so when Tier 1 or Tier 2 data lands the
 * "GEOREFERENCE ASSIGNED" badge appears on its own without a code change (§0).
 */
export function ProvenanceBadges({ badges }: { badges: readonly string[] }) {
  if (badges.length === 0) {
    return null;
  }
  return (
    <div className="flex flex-wrap gap-1">
      {badges.map((badge) => (
        <Badge key={badge} tone="simulated">
          {badge}
        </Badge>
      ))}
    </div>
  );
}

/** §5.4 decision bands. UNATTRIBUTED reads as a decision, not a failure. */
export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  const tone: Tone =
    verdict === "STRONG" ? "hazard" : verdict === "MODERATE" ? "caution" : "neutral";
  return <Badge tone={tone}>{verdict}</Badge>;
}

/** §7 envelope. A fixture-backed answer is never shown as a live one (§2.2). */
export function SourceBadge({ source }: { source: Source }) {
  return (
    <Badge
      tone={source === "live" ? "clear" : "neutral"}
      title={
        source === "fixture"
          ? "Served from data/fixtures — real engine output captured at build time (§15)."
          : "Served from the live backend."
      }
    >
      {source === "live" ? "LIVE" : "FIXTURE"}
    </Badge>
  );
}

/**
 * The §2.2 guard: a score never appears without how it was produced.
 * Used everywhere p_oil is rendered.
 */
export function MethodNote({ method, note }: { method: string; note: string }) {
  return (
    <div className="mt-2 border-l-2 border-caution/50 pl-2.5">
      <div className="text-tiny font-medium text-caution/90">
        Scored by {method.replace(/_/g, "-")}
      </div>
      <p className="mt-1 text-tiny leading-relaxed text-ink-dim">{note}</p>
    </div>
  );
}
