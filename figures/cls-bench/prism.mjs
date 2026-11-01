// Oriented prism helper (from the Anatomy kit figures on computerliteracy). Inlined into the page by prismScript().
import { readFileSync } from "node:fs";
import { iso, pathOf, cameraOf } from "../kit/iso-kit.mjs";
export function prismScript() { return readFileSync(new URL(import.meta.url), "utf8").replace(/^import .*$/gm, "").replace(/^export function prismScript[^\n]*\n/m, "").replace(/^export /gm, ""); }

export const clAdd = (a, b, s = 1) => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
const clSub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const clDot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const clCross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const clUnit = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
export function clViewOf(P) { const { sinA, cosA, sinE, cosE } = cameraOf(P); return [cosA * cosE, sinA * cosE, sinE]; }

function clHull(points) {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const turn = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of pts) { while (lower.length > 1 && turn(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop(); lower.push(p); }
  for (const p of pts.reverse()) { while (upper.length > 1 && turn(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop(); upper.push(p); }
  return pathOf([...lower.slice(0, -1), ...upper.slice(0, -1)], true);
}

// A convex planar ring (3D points) pushed along `dir`: the same {fill, outline, crease, top, shades} that kit.slabOf
// returns, so kit.solidSvg paints it, but for faces in any orientation (a lid on a hinge, a phone standing up).
export function clPrism(ring, dir, P) {
  const view = clViewOf(P);
  const { sinA, cosA } = cameraOf(P);
  const light = [-sinA, cosA, 0];
  const n = ring.length;
  const far = ring.map((p) => clAdd(p, dir));
  const ud = clUnit(dir);
  const c = ring.reduce((s, p) => clAdd(s, p, 1 / n), [0, 0, 0]);
  const nearVis = clDot(ud, view) < -1e-9;
  const farVis = clDot(ud, view) > 1e-9;
  const at = (p) => iso(p, P);
  const vis = [];
  const shades = [[], [], [], []];
  for (let i = 0; i < n; i++) {
    const a = ring[i], b = ring[(i + 1) % n];
    let nr = clUnit(clCross(clSub(b, a), ud));
    if (clDot(nr, clSub(clAdd(a, b, 1).map((v) => v / 2), c)) < 0) nr = nr.map((v) => -v);
    vis[i] = clDot(nr, view) > 1e-9;
    if (!vis[i]) continue;
    const shade = Math.min(3, Math.max(0, Math.floor(((clDot(nr, light) + 1) / 2) * 4)));
    shades[shade].push(pathOf([at(a), at(b), at(far[(i + 1) % n]), at(far[i])], true));
  }
  const runs = { out: [], crease: [] };
  const edge = (list, kind) => {
    // chain consecutive segments of one ring into polylines
    let run = null;
    for (const [i, j, from, keep] of list) {
      if (!keep) { run = null; continue; }
      if (run && run.last === i && run.kind === keep) { run.pts.push(at(from[j])); run.last = j; }
      else { run = { kind: keep, last: j, pts: [at(from[i]), at(from[j])] }; runs[keep].push(run); }
    }
  };
  for (const [capVis, from] of [[nearVis, ring], [farVis, far]]) {
    const list = [];
    for (let i = 0; i < n; i++) list.push([i, (i + 1) % n, from, vis[i] !== capVis ? "out" : vis[i] && capVis ? "crease" : null]);
    edge(list);
  }
  const verticals = [];
  for (let i = 0; i < n; i++) if (vis[(i + n - 1) % n] !== vis[i]) verticals.push(pathOf([at(ring[i]), at(far[i])]));
  const flat = (rs) => rs.map((r) => pathOf(r.pts)).join("");
  const cap = nearVis ? ring : farVis ? far : null;
  return {
    fill: clHull([...ring, ...far].map(at)),
    outline: flat(runs.out) + verticals.join(""),
    crease: flat(runs.crease),
    top: cap ? pathOf(cap.map(at), true) : "",
    shades: shades.map((s) => s.join("")),
    bevel: "",
    nearVis,
    farVis,
  };
}

