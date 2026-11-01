// Moving parts of the CLS bench, shared by the build (rest pose) and the live page (inlined, imports stripped).
import { readFileSync } from "node:fs";
import { iso, pathOf, slabOf, cylinder, solidSvg, lineSvg, dotsSvg, segment, ring, radialTicks, knurl, sideArc, lineOnTop } from "../kit/iso-kit.mjs";
import { clPrism } from "./prism.mjs";
export function sharedScript() { return readFileSync(new URL(import.meta.url), "utf8").replace(/^import .*$/gm, "").replace(/^export function sharedScript[^\n]*\n/m, "").replace(/^export /gm, ""); }

// ---- the real numbers (pdp-continual default core: z5 store + c3 router + awake remote replay) ----
export const CORE = { every: 38, batch: 32, gap: 300, margin: 0.3, digits: 10 };

// ---- world layout (units ~ px, x right/back, y toward viewer, z up) ----
export const Y0 = -7;                                   // the shuttle line: card-file slots, gantry, lap plate
export const TRAY = { x: -162, y: -24, w: 96, d: 34, r: 1.5 };
export const SLOT0 = -160, PITCH = 9.2;                 // ten slots, one per digit
export const CARD = { w: 1.4, d: 24, h: 16 };          // card thin in x, long in y
export const slotX = (i) => SLOT0 + PITCH * i + 4.6;    // the front card of slot i (centre x)
export const BAL = { x: 6, y: 4, pivot: 62, arm: 32, hang: 24, panR: 10 };
export const LAP = { x: 114, y: Y0, r: 28, z: 24, top: 28 };
export const RAIL_Z = 134, CARRY_Z = 100, PARK_X = -24;

// card standing upright, its centre-line at x = cx, bottom at z, height h
export function cardSvg(cx, z, h, P, opts = {}) {
  const plan = { x: cx - CARD.w / 2, y: Y0 - CARD.d / 2, w: CARD.w, d: CARD.d, r: 0.5 };
  return solidSvg(slabOf(plan, z, Math.max(0.3, h), P, 2, 0), { tone: "hi", crease: "lo", ...opts });
}

const rot = (x, z, t) => [x * Math.cos(t) - z * Math.sin(t), x * Math.sin(t) + z * Math.cos(t)];

// The c3 balance: theta > 0 tips the store pan (left) down. discs = weights on the store pan (its margin).
export function balanceSvg(theta, discs, P) {
  const { x: bx, y: by, pivot, arm, hang, panR } = BAL;
  const out = [];
  // needle (behind the beam, in front of the scale plate)
  const [nx, nz] = rot(0, 28, theta);
  out.push(`<path class="iso-line" data-tone="lit" data-free="end" style="stroke-width:1.1" d="${segment([bx, by - 2.5, pivot], [bx + nx, by - 2.5, pivot + nz], P)}"/>`);
  const ends = [-1, 1].map((s) => { const [ex, ez] = rot(s * arm, 0, theta); return [bx + ex, by, pivot + ez]; });
  const pan = (end, side) => {
    const [ex, ey, ez] = end;
    const top = ez - hang;
    const parts = [solidSvg(cylinder(ex, ey, panR, top - 2, 2, P, 40, 0.5), { tone: "mid" })];
    parts.push(lineSvg(ring(ex, ey, panR - 1.6, top, P, 40), { tone: "faint" }));
    if (side < 0) for (let k = 0; k < discs; k++) parts.push(solidSvg(cylinder(ex, ey, 4.5, top + k * 1.3, 1.3, P, 24), { tone: "hi", crease: "lo" }));
    // three stirrup wires from the hook to the rim
    const rim = [Math.PI * 0.25, Math.PI * 0.95, Math.PI * 1.6].map((a) => [ex + Math.cos(a) * (panR - 0.6), ey + Math.sin(a) * (panR - 0.6), top]);
    const hook = [ex, ey, ez - 3.4];
    parts.push(lineSvg(rim.map((p) => segment(hook, p, P)).join(""), { tone: "mid" }));
    parts.push(lineSvg(segment([ex, ey, ez - 1.4], hook, P), { tone: "mid" }));
    parts.push(dotsSvg([iso(hook, P), ...rim.map((p) => iso(p, P))], { size: 0.5, tone: "mid" }));
    return parts.join("");
  };
  out.push(pan(ends[0], -1));
  // beam: a bar in the x-z plane, 3 deep in y
  const bar = [[-arm - 2, -1.4], [arm + 2, -1.4], [arm + 2, 1.4], [-arm - 2, 1.4]].map(([x, z]) => { const [rx, rz] = rot(x, z, theta); return [bx + rx, by - 1.5, pivot + rz]; });
  out.push(solidSvg(clPrism(bar, [0, 3, 0], P), { tone: "mid", crease: "lo" }));
  // rider: the c3 threshold, a fixed weight on the cortex arm
  const rid = [[20, 1.4], [26, 1.4], [26, 5], [20, 5]].map(([x, z]) => { const [rx, rz] = rot(x, z, theta); return [bx + rx, by - 2.2, pivot + rz]; });
  out.push(solidSvg(clPrism(rid, [0, 4.4, 0], P), { tone: "hi", crease: "lo" }));
  out.push(dotsSvg([iso([bx, by + 1.5, pivot], P)], { size: 0.7, tone: "hi" }));
  out.push(pan(ends[1], 1));
  return out.join("");
}

// The lap plate's grooves (the cortex's weight space), turned by phase radians.
export function groovesSvg(phase, P) {
  const t = radialTicks(LAP.x, LAP.y, LAP.r - 2.5, 24, 4, LAP.top, [15.5, 15.5], P, phase);
  return lineSvg(t.minor, { tone: "lo" }) + lineSvg(t.major, { tone: "mid" });
}

// The shuttle: carriage on the rails at x = cx, hook block bottom at hookZ.
export function carriageSvg(cx, P) {
  const out = [];
  out.push(solidSvg(slabOf({ x: cx - 9, y: Y0 - 9, w: 18, d: 18, r: 2 }, RAIL_Z + 3, 5, P, 3, 0.5), { tone: "mid" }));
  out.push(dotsSvg([[-6, -6], [6, -6], [-6, 6], [6, 6]].map(([dx, dy]) => iso([cx + dx, Y0 + dy, RAIL_Z + 8], P)), { size: 0.45, tone: "lo" }));
  out.push(solidSvg(cylinder(cx, Y0, 4, RAIL_Z + 8, 4, P, 24), { tone: "mid" }));
  out.push(lineSvg(knurl(cx, Y0, 4, RAIL_Z + 8, RAIL_Z + 12, 22, P), { tone: "lo" }));
  out.push(lineSvg(ring(cx, Y0, 1.2, RAIL_Z + 12, P, 12), { tone: "mid" }));
  return out.join("");
}
export function hoistSvg(cx, hookZ, P) {
  const out = [];
  out.push(lineSvg(segment([cx, Y0, RAIL_Z], [cx, Y0, hookZ + 4], P), { tone: "mid" }));
  out.push(solidSvg(slabOf({ x: cx - 2.5, y: Y0 - 3, w: 5, d: 6, r: 1 }, hookZ, 4, P, 2, 0.3), { tone: "mid" }));
  out.push(dotsSvg([iso([cx, Y0, hookZ + 4], P)], { size: 0.55, tone: "hi" }));
  return out.join("");
}
