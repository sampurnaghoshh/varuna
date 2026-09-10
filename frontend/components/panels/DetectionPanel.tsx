"use client";

import { Badge, MethodNote, ProvenanceBadges } from "@/components/ui/Badge";
import { Absent, Empty, Panel, Row } from "@/components/ui/Panel";
import { featureLabel, featureValue, num, pct } from "@/lib/format";
import type { Detection, Factor, Scene } from "@/types/domain";

export function DetectionPanel({
  scene,
  detection,
  className = "",
}: {
  scene: Scene;
  detection: Detection | null;
  className?: string;
}) {
  if (!detection) {
    return (
      <Panel
        title="Detection"
        className={className}
        meta={<Badge tone="clear">0 detections</Badge>}
      >
        <div className="space-y-3">
          <div className="rounded-chart border border-clear/40 bg-clear/8 px-2.5 py-2">
            <div className="text-tiny font-semibold text-clear">Clean sea</div>
            <p className="mt-1 text-tiny leading-relaxed text-ink">
              No dark formations above threshold. Nothing to rewind, nothing to attribute.
            </p>
          </div>
          <Row label="Wind speed" value={`${num(scene.wind_speed_ms, 1)} m/s`} />
          <p className="text-micro leading-relaxed text-ink-faint">
            Wind sits inside the 3.0–12.0 m/s gate, so a slick here would have been
            detectable. A clean result on a detectable scene is the evidence that the
            detector is not inventing formations.
          </p>
          <ProvenanceBadges badges={scene.badges} />
        </div>
      </Panel>
    );
  }

  const oilLike = detection.class === "oil";

  return (
    <Panel
      title="Detection"
      className={className}
      meta={<Badge tone={detection.method === "rule_based" ? "caution" : "survey"}>
        {detection.detector}
      </Badge>}
    >
      <div className="space-y-3">
        {/* §2.2: the score, its class, its method and its note are one unit. */}
        <div className="rounded-chart border border-chart-line/70 bg-sea-mid/40 px-2.5 py-2">
          <div className="flex items-end justify-between gap-3">
            <div>
              <div className="text-micro text-ink-dim">P(oil)</div>
              <div
                className={`tnum text-figure font-semibold ${
                  oilLike ? "text-slick" : "text-ink-bright"
                }`}
              >
                {num(detection.p_oil, 3)}
              </div>
            </div>
            <div className="pb-1 text-right">
              <div className="text-micro text-ink-dim">Classified</div>
              <div className="text-readout font-semibold text-ink-bright">
                {detection.class}
              </div>
            </div>
          </div>

          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-chart-line/60">
            <div
              className={oilLike ? "h-full bg-slick" : "h-full bg-ink-faint"}
              style={{ width: `${Math.max(1, detection.p_oil * 100)}%` }}
            />
          </div>

          <MethodNote method={detection.method} note={detection.note} />

          <div className="mt-2 border-t border-chart-line/50 pt-1.5">
            <Row
              label="Base rate"
              value={num(detection.base_p_oil, 3)}
              hint="Where the score sits before any evidence is applied."
            />
            <Row
              label="Evidence observed"
              value={pct(detection.evidence_fraction, 0)}
              hint="Some §5.2 terms need pixels this build does not have."
            />
            <p className="mt-1 text-micro leading-relaxed text-ink-faint">
              The score is shrunk toward the base rate in proportion to the evidence
              actually available. Less evidence means a weaker claim, in both directions.
            </p>
          </div>
        </div>

        <WindGate detection={detection} />

        <div>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-micro text-ink-dim">What moved the score</span>
            <Badge>{detection.factor_basis.replace(/_/g, "-")}</Badge>
          </div>
          <div className="space-y-1.5">
            {detection.shap_factors.map((factor) => (
              <FactorRow key={factor.feature} factor={factor} />
            ))}
          </div>
        </div>

        <FeatureTable detection={detection} />

        <div className="border-t border-chart-line/50 pt-2">
          <p className="text-micro leading-relaxed text-ink-faint">{detection.provenance}</p>
        </div>
      </div>
    </Panel>
  );
}

/**
 * §5.2's wind gate. Below 3 m/s the sea surface itself mimics oil; above
 * 12 m/s slicks disperse below detectability. Either way the dark formation
 * stops being trustworthy evidence, and the console says so loudly.
 */
function WindGate({ detection }: { detection: Detection }) {
  const violated = detection.wind_gate_violated;
  const wind = detection.features.wind_speed_ms;
  const [low, high] = detection.wind_gate_range_ms;

  return (
    <div
      className={`rounded-chart border px-2.5 py-2 ${
        violated ? "border-caution bg-caution/12" : "border-clear/40 bg-clear/8"
      }`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span
          className={`text-tiny font-semibold ${violated ? "text-caution" : "text-clear"}`}
        >
          {violated ? "Wind gate violated" : "Wind gate satisfied"}
        </span>
        <span className="tnum text-tiny text-ink-bright">
          {wind === null ? "—" : `${num(wind, 1)} m/s`}
        </span>
      </div>

      {/* The gate as a band, with the reading placed inside or outside it. */}
      <div className="relative mt-2 h-1.5 rounded-full bg-chart-line/60">
        <div
          className="absolute inset-y-0 rounded-full bg-clear/35"
          style={{
            left: `${((low ?? 3) / 18) * 100}%`,
            width: `${(((high ?? 12) - (low ?? 3)) / 18) * 100}%`,
          }}
        />
        {wind !== null ? (
          <div
            className={`absolute -top-0.5 h-2.5 w-0.5 ${violated ? "bg-caution" : "bg-clear"}`}
            style={{ left: `${Math.min(99, (wind / 18) * 100)}%` }}
          />
        ) : null}
      </div>
      <div className="mt-0.5 flex justify-between text-micro text-ink-faint">
        <span className="tnum">0</span>
        <span className="tnum">
          gate {num(low ?? 3, 1)}–{num(high ?? 12, 1)} m/s
        </span>
        <span className="tnum">18</span>
      </div>

      <p className="mt-1.5 text-micro leading-relaxed text-ink">
        {detection.wind_gate_reason}
      </p>
    </div>
  );
}

function FactorRow({ factor }: { factor: Factor }) {
  const towardOil = factor.direction === "oil";
  const width = Math.min(100, Math.abs(factor.contribution) * 55);

  return (
    <div title={factor.rationale}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-micro text-ink">{featureLabel(factor.feature)}</span>
        <span className="tnum shrink-0 text-micro text-ink-dim">
          {featureValue(factor.feature, factor.value)}
        </span>
      </div>
      <div className="relative mt-0.5 h-1 bg-chart-line/50">
        <div className="absolute inset-y-0 left-1/2 w-px bg-chart-edge" />
        <div
          className={`absolute inset-y-0 ${towardOil ? "bg-slick" : "bg-ink-faint"}`}
          style={
            towardOil
              ? { left: "50%", width: `${width / 2}%` }
              : { right: "50%", width: `${width / 2}%` }
          }
        />
      </div>
      <p className="mt-0.5 text-micro leading-snug text-ink-faint">{factor.rationale}</p>
    </div>
  );
}

/** Every §5.2 feature, with a reason wherever a value is absent (§2.2). */
function FeatureTable({ detection }: { detection: Detection }) {
  const keys = Object.keys(detection.features) as (keyof Detection["features"])[];

  return (
    <div className="border-t border-chart-line/50 pt-2">
      <div className="mb-1 text-micro text-ink-dim">Measured features (§5.2)</div>
      {keys.map((key) => {
        const value = detection.features[key];
        const reason = detection.unavailable[key];
        return (
          <Row
            key={key}
            label={featureLabel(key)}
            hint={detection.feature_provenance[key]}
            value={
              value === null ? (
                <Absent reason={reason ?? "not measured in this build"} />
              ) : (
                featureValue(key, value)
              )
            }
          />
        );
      })}
    </div>
  );
}

export function DetectionEmpty() {
  return <Empty>No scene loaded.</Empty>;
}
