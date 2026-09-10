/**
 * Serves data/fixtures/ and data/geo/ to the console over HTTP.
 *
 * §15 builds the frontend against the fixture set, and §9 keeps that set
 * read-only. Reading it from disk here rather than copying it into public/
 * duplicates none of the 4.5 MB and — the reason that matters — makes
 * lib/data.ts speak HTTP exactly as it will to the live §7 API. Swapping to
 * the backend changes one file and this route stops being called.
 */

import { readFile } from "node:fs/promises";
import path from "node:path";

export const dynamic = "force-dynamic";

/** Only these subtrees of data/ are reachable. */
const ALLOWED_ROOTS = new Set(["fixtures", "geo"]);

const SEGMENT = /^[A-Za-z0-9._-]+$/;

/** next dev runs with cwd = frontend/, so data/ is one level up. */
const DATA_ROOT = path.join(process.cwd(), "..", "data");

export async function GET(
  _request: Request,
  { params }: { params: { path?: string[] } },
): Promise<Response> {
  const segments = params.path ?? [];
  const root = segments[0];

  const rejected =
    segments.length < 2 ||
    root === undefined ||
    !ALLOWED_ROOTS.has(root) ||
    !segments.every((segment) => SEGMENT.test(segment) && segment !== "..") ||
    !/\.(json|geojson)$/.test(segments[segments.length - 1]!);

  if (rejected) {
    return Response.json({ detail: "Not a readable fixture path." }, { status: 400 });
  }

  const filePath = path.join(DATA_ROOT, ...segments);
  if (!filePath.startsWith(path.join(DATA_ROOT) + path.sep)) {
    return Response.json({ detail: "Not a readable fixture path." }, { status: 400 });
  }

  try {
    const body = await readFile(filePath, "utf-8");
    return new Response(body, {
      status: 200,
      headers: { "content-type": "application/json; charset=utf-8" },
    });
  } catch {
    return Response.json(
      { detail: `No fixture at ${segments.join("/")}.` },
      { status: 404 },
    );
  }
}
