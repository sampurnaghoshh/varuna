/**
 * Mirrors backend/app/schemas/ws.py. Keep in sync (§8).
 *
 * The console depends on this shape exactly; a rename here breaks the demo
 * mid-run.
 */

export type Stage =
  | "SEGMENTING"
  | "DISCRIMINATING"
  | "REWINDING"
  | "FUSING"
  | "DONE"
  | "ERROR";

export interface PipelineEvent {
  stage: Stage;
  progress: number;
  message: string;
  payload: Record<string, unknown>;
}

/** §5.4 decision bands. UNATTRIBUTED is a first-class result, not an error. */
export type Verdict = "STRONG" | "MODERATE" | "UNATTRIBUTED";
