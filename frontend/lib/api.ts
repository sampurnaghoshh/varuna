/**
 * API client. Every §7 response carries {source, elapsed_ms}; the console
 * renders the source badge from it — a fixture-backed answer is never shown as
 * a live one (§2.2).
 *
 * The backend runs in Compose on :8000; this app runs natively on the host.
 */

export const API_BASE: string =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000/api/v1";

export const WS_BASE: string =
  process.env.NEXT_PUBLIC_WS_BASE ?? "ws://localhost:8000/api/v1/ws";

export type Source = "live" | "fixture";

export interface Envelope {
  source: Source;
  elapsed_ms: number;
}

export async function apiGet<T extends Envelope>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`GET ${path} failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function apiPost<T extends Envelope>(
  path: string,
  body?: unknown,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`POST ${path} failed: ${response.status}`);
  }
  return (await response.json()) as T;
}
