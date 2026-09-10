"use client";

import { useState } from "react";

import { Badge, MethodNote, VerdictBadge } from "@/components/ui/Badge";
import { Absent, Empty, Panel, Row } from "@/components/ui/Panel";
import { clock, lrLabel, nats, num, pct } from "@/lib/format";
import type {
  AttributionResponse,
  Candidate,
  Channels,
  Unrankable,
  Vessel,
} from "@/types/domain";

const CHANNEL_TITLES: Record<keyof Candidate["terms_nats"], string> = {
  s1: "E1 · spatiotemporal mass",
  s2: "E2 · axial coherence",
  s3: "E3 · kinematic consistency",
  s4: "E4 · dark-gap coincidence",
  prior: "Vessel-type prior",
};

const CHANNEL_WHY: Record<keyof Candidate["terms_nats"], string> = {
  s1: "How much of the reconstructed origin probability the vessel's corridor swept up, against what an average vessel in this frame swept up.",
  s2: "Deliberate discharge while underway lays oil along the track, so the slick's axis should line up with the vessel's course. Independent of the current field.",
  s3: "The slick's released length divided by the vessel's speed gives an implied discharge duration. Scored against a 90-minute median. Independent of the current field.",
  s4: "A boost when the transponder went quiet around t★, over and above the feed's own 15-minute cadence. Never a penalty for transmitting cleanly.",
  prior: "Tanker 3.0, bulk/cargo 1.5, fishing 1.0, passenger 0.5, lifted by prior detections.",
};

export function AttributionPanel({
  attribution,
  vessels,
  culpritIgnited,
  className = "",
}: {
  attribution: AttributionResponse | null;
  vessels: Vessel[];
  culpritIgnited: boolean;
  className?: string;
}) {
  if (!attribution) {
    return (
      <Panel title="Attribution" className={className}>
        <Empty>
          No attribution ran for this scene. There was no detection to trace back.
        </Empty>
      </Panel>
    );
  }

  if (!attribution.issued) {
    return (
      <Panel
        title="Attribution"
        className={className}
        meta={<VerdictBadge verdict={attribution.verdict} />}
      >
        <WithheldAttribution attribution={attribution} />
      </Panel>
    );
  }

  const { candidates, unrankable, background, none_of_the_above_posterior } = attribution;

  return (
    <Panel
      title="Attribution"
      className={className}
      meta={<VerdictBadge verdict={attribution.verdict} />}
    >
      <div className="space-y-3">
        <div className="flex items-baseline justify-between gap-2">
          <div>
            <div className="text-micro text-ink-dim">Vessels scored</div>
            <div className="tnum text-readout text-ink-bright">
              {attribution.n_ranked}
              <span className="text-tiny text-ink-dim"> of {attribution.n_vessels_in_frame}</span>
            </div>
          </div>
          <div className="text-right">
            <div className="text-micro text-ink-dim">Windage α</div>
            <div className="tnum text-readout text-ink-bright">
              {num(attribution.windage, 3)}
            </div>
          </div>
        </div>

        <p className="border-l-2 border-chart-edge/60 pl-2.5 text-micro leading-relaxed text-ink-dim">
          {attribution.frame_size_note}
        </p>

        <div className="space-y-1.5">
          {candidates.map((candidate) => (
            <CandidateRow
              key={candidate.mmsi}
              candidate={candidate}
              vessel={vessels.find((v) => v.mmsi === candidate.mmsi) ?? null}
              background={background}
              isCulprit={candidate.mmsi === attribution.culprit_mmsi}
              ignited={culpritIgnited && candidate.mmsi === attribution.culprit_mmsi}
            />
          ))}
        </div>

        {/* The system's own escape hatch, kept visible beside the ranking. */}
        <div className="rounded-chart border border-chart-line/70 bg-sea-mid/50 px-2.5 py-2">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-tiny text-ink">None of the above</span>
            <span className="tnum text-tiny text-ink-bright">
              {pct(none_of_the_above_posterior)}
            </span>
          </div>
          <p className="mt-1 text-micro leading-relaxed text-ink-dim">
            An explicit hypothesis in the posterior, pinned at LR 10, so the ranking always
            competes against the possibility that the responsible vessel is not in the frame.
          </p>
        </div>

        <UnrankableList unrankable={unrankable} />

        <div className="border-t border-chart-line/50 pt-2">
          <div className="text-micro text-ink-dim">Frame background (§5.4)</div>
          <Row label="s1 background" value={num(background.s1_bg, 4)} />
          <Row label="s2 background" value={num(background.s2_bg, 4)} />
          <Row label="s3 background" value={num(background.s3_bg, 4)} />
          <p className="mt-1 text-micro leading-relaxed text-ink-faint">
            Each channel is divided by the mean score across vessels in frame. That division
            is what makes the output a likelihood ratio rather than an arbitrary score.
          </p>
        </div>
      </div>
    </Panel>
  );
}

function WithheldAttribution({
  attribution,
}: {
  attribution: Extract<AttributionResponse, { issued: false }>;
}) {
  return (
    <div className="space-y-3">
      <div className="rounded-chart border border-caution/50 bg-caution/8 px-2.5 py-2">
        <div className="text-tiny font-semibold text-caution">No vessel is named</div>
        <p className="mt-1 text-tiny leading-relaxed text-ink">{attribution.reason}</p>
      </div>

      <div>
        <div className="text-micro text-ink-dim">Recommended next step</div>
        <p className="mt-1 text-tiny leading-relaxed text-ink">
          {attribution.recommendation}
        </p>
      </div>

      <Row
        label="Vessels in frame"
        value={attribution.n_vessels_in_frame}
        hint="AIS-visible vessels intersecting the scene during the acquisition window."
      />

      <p className="text-micro leading-relaxed text-ink-faint">
        Withholding is a result, not a failure. The pipeline ran, found the evidence
        untrustworthy, and stopped before it could accuse anyone.
      </p>
    </div>
  );
}

function CandidateRow({
  candidate,
  vessel,
  isCulprit,
  ignited,
}: {
  candidate: Candidate;
  vessel: Vessel | null;
  background: { s1_bg: number; s2_bg: number; s3_bg: number };
  isCulprit: boolean;
  ignited: boolean;
}) {
  const [open, setOpen] = useState(isCulprit);
  const accused = candidate.verdict !== "UNATTRIBUTED";

  return (
    <div
      className={`rounded-chart border transition-colors ${
        accused
          ? "border-hazard/50 bg-hazard/8"
          : "border-chart-line/60 bg-sea-mid/30"
      } ${ignited ? "igniting" : ""}`}
    >
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left"
        aria-expanded={open}
      >
        <span className="tnum w-4 shrink-0 text-tiny text-ink-faint">{candidate.rank}</span>
        <span className="min-w-0 flex-1">
          <span
            className={`block truncate text-tiny font-semibold ${
              accused ? "text-hazard" : "text-ink"
            }`}
          >
            {candidate.name}
          </span>
          <span className="tnum block text-micro text-ink-dim">
            MMSI {candidate.mmsi}
            {vessel ? ` · ${vessel.type}` : ""}
          </span>
        </span>
        <span className="shrink-0 text-right">
          <span className="tnum block text-tiny font-semibold text-ink-bright">
            LR {lrLabel(candidate.lr)}
          </span>
          <span className="tnum block text-micro text-ink-dim">
            {pct(candidate.posterior)}
          </span>
        </span>
      </button>

      <div className="flex flex-wrap items-center gap-1 px-2.5 pb-1.5">
        <VerdictBadge verdict={candidate.verdict} />
        {vessel?.badge ? <Badge tone="simulated">{vessel.badge}</Badge> : null}
      </div>

      {open ? (
        <div className="border-t border-chart-line/50 px-2.5 py-2">
          <ChannelBreakdown candidate={candidate} />
          <ChannelDetail channels={candidate.channels} />
        </div>
      ) : null}
    </div>
  );
}

/** §5.4 log LR terms as a diverging bar. The parts sum to the whole. */
function ChannelBreakdown({ candidate }: { candidate: Candidate }) {
  const terms = candidate.terms_nats;
  const keys = Object.keys(terms) as (keyof typeof terms)[];
  const span = Math.max(...keys.map((key) => Math.abs(terms[key])), 0.5);

  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-micro text-ink-dim">Evidence, in nats</span>
        <span className="tnum text-micro text-ink">
          log LR {nats(candidate.log_lr)}
        </span>
      </div>

      <div className="mt-1.5 space-y-1">
        {keys.map((key) => {
          const value = terms[key];
          const width = (Math.abs(value) / span) * 50;
          return (
            <div key={key} className="group" title={CHANNEL_WHY[key]}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="truncate text-micro text-ink-dim">
                  {CHANNEL_TITLES[key]}
                </span>
                <span
                  className={`tnum shrink-0 text-micro ${
                    value >= 0 ? "text-survey" : "text-ink-faint"
                  }`}
                >
                  {nats(value)}
                </span>
              </div>
              <div className="relative mt-0.5 h-1 bg-chart-line/50">
                <div className="absolute inset-y-0 left-1/2 w-px bg-chart-edge" />
                <div
                  className={`absolute inset-y-0 ${value >= 0 ? "bg-survey/70" : "bg-caution/70"}`}
                  style={
                    value >= 0
                      ? { left: "50%", width: `${width}%` }
                      : { right: "50%", width: `${width}%` }
                  }
                />
              </div>
            </div>
          );
        })}
      </div>

      <p className="mt-1.5 text-micro leading-relaxed text-ink-faint">
        E2 and E3 never touch the current field. That independence is why a coarse ocean
        model does not sink the result.
      </p>
    </div>
  );
}

function ChannelDetail({ channels }: { channels: Channels }) {
  const entries: [string, number | null, string][] = [
    ["s1 · mass overlap", channels.s1, "s1"],
    ["s2 · axial", channels.s2, "s2"],
    ["s3 · kinematic", channels.s3, "s3"],
    ["s4 · dark gap", channels.s4, "s4"],
  ];

  return (
    <div className="mt-2 border-t border-chart-line/50 pt-2">
      {entries.map(([label, value, key]) => (
        <Row
          key={key}
          label={label}
          value={
            value === null ? (
              <Absent reason={channels.unavailable[key] ?? "not scored"} />
            ) : (
              num(value, 4)
            )
          }
        />
      ))}
      <Row
        label="t★"
        value={
          channels.t_star ? (
            clock(channels.t_star)
          ) : (
            <Absent reason={channels.unavailable.t_star ?? "not scored"} />
          )
        }
        hint="The snapshot time contributing the most origin probability to this vessel."
      />
      <Row
        label="Origin mass"
        value={channels.mass === null ? <Absent reason="not scored" /> : channels.mass.toExponential(2)}
      />
    </div>
  );
}

function UnrankableList({ unrankable }: { unrankable: Unrankable[] }) {
  const [open, setOpen] = useState(false);
  if (unrankable.length === 0) {
    return null;
  }

  return (
    <div className="rounded-chart border border-chart-line/70 bg-sea-mid/30">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-baseline justify-between gap-2 px-2.5 py-2 text-left"
        aria-expanded={open}
      >
        <span className="text-tiny text-ink">
          {unrankable.length} vessels could not be scored
        </span>
        <span className="text-micro text-ink-dim">{open ? "hide" : "show"}</span>
      </button>
      <p className="px-2.5 pb-2 text-micro leading-relaxed text-ink-dim">
        Present in the frame, but with no overlap with the reconstructed origin. They are
        listed rather than dropped, so the count on screen matches the traffic in the scene.
      </p>
      {open ? (
        <div className="border-t border-chart-line/50 px-2.5 py-1.5">
          {unrankable.map((vessel) => (
            <div key={vessel.mmsi} className="py-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className="truncate text-micro text-ink">{vessel.name}</span>
                <span className="tnum shrink-0 text-micro text-ink-faint">{vessel.mmsi}</span>
              </div>
              <div className="text-micro text-ink-faint">{vessel.reason}</div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export { MethodNote };
