// Fig: the pdp-continual CLS dual as a bench instrument. node build.mjs [--light] [--out file]
import { writeFileSync } from "node:fs";
import * as k from "../kit/iso-kit.mjs";
import { clPrism, prismScript } from "./prism.mjs";
import * as S from "./shared.mjs";
import { sharedScript } from "./shared.mjs";

import { readFileSync } from "node:fs";
const kitScript = () => readFileSync(new URL("../kit/iso-kit.mjs", import.meta.url), "utf8").replace(/^export /gm, "");
const LIVE = readFileSync(new URL("./live.js", import.meta.url), "utf8");
const argv = process.argv.slice(2);
const theme = argv.includes("--light") ? "light" : "dark";
const OUT = argv.includes("--out") ? argv[argv.indexOf("--out") + 1] : new URL("../../docs/figures/cls-bench.html", import.meta.url).pathname;
const WIDTH = 640, HEIGHT = 420;
const { Y0, TRAY, SLOT0, PITCH, BAL, LAP, RAIL_Z } = S;
const PLATE = { x: -184, y: -44, w: 366, d: 100, r: 8 };
const POSTS = [-176, 168];

const P = k.fitProjection(
  [...k.boxCorners(PLATE, -17, 0), [POSTS[0], Y0, RAIL_Z + 12], [POSTS[1], Y0, RAIL_Z + 12], [PLATE.x, Y0, RAIL_Z + 12]],
  WIDTH, HEIGHT, { pad: 26, azimuth: 56, elevation: 30 });
const at = (p) => k.iso(p, P);
const Sv = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (pts, style) => k.dotsSvg(pts, style);
const byDepth = (pts) => [...pts].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
// a closed outline on a vertical face y = const (vents, slots): no free ends
const slotOnFace = (x0, x1, z0, z1, y) => k.pathOf([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]].map(at), true);

// ---------- base ----------
function base() {
  const out = [`<path class="iso-halo" d="${k.haloOf(PLATE, -17, 12, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(PLATE, 22))) {
    out.push(Sv(k.cylinder(x, y, 7, -17, 1.6, P, 28), { tone: "lo" }));
    out.push(Sv(k.cylinder(x, y, 5.4, -15.4, 3.4, P, 28), { tone: "mid" }));
  }
  out.push(Sv(k.slabOf(PLATE, -12, 12, P, 8, 1.6), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(PLATE, 5), 0, P), { tone: "lo" }));
  const screws = k.corners(PLATE, 10);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.2, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
  // the wake-tick ruler: one replay period, 38 waking arrivals
  const strip = RULER;
  out.push(Sv(k.slabOf(strip, 0, 0.6, P, 3), { tone: "lo", crease: "none" }));
  const minor = [], major = [];
  for (let i = 0; i <= 38; i++) {
    const x = RX0 + (i / 38) * (RX1 - RX0);
    const big = i % 19 === 0;
    (big ? major : minor).push(k.lineOnTop([x, strip.y], [x, strip.y + (big ? 3.8 : 2.2)], 0.6, P));
  }
  out.push(L(minor.join(""), { tone: "lo" }), L(major.join(""), { tone: "mid" }));
  out.push(D([at([strip.x + 2, strip.y + 3.5, 0.6]), at([strip.x + strip.w - 2, strip.y + 3.5, 0.6])], { size: 0.45 }));
  // label plate with the one engraved label
  const lab = { x: 122, y: 32, w: 48, d: 12, r: 1.2 };
  out.push(Sv(k.slabOf(lab, 0, 0.8, P, 4), { tone: "mid", crease: "none" }));
  out.push(L(k.planOutline(k.insetPlan(lab, 1.6), 0.8, P), { tone: "faint" }));
  out.push(k.faceTextSvg(k.topMatrix([lab.x + 4, lab.y + 5.6], 0.8, P, "x"), "z5 · c3 · awr", { size: 3.2, tone: "mid" }));
  out.push(L([0, 1].map((r) => k.lineOnTop([lab.x + 4, lab.y + 8 + r * 1.8], [lab.x + 4 + [30, 20][r], lab.y + 8 + r * 1.8], 0.8, P)).join(""), { tone: "lo", free: true }));
  out.push(D([at([lab.x + lab.w - 3, lab.y + 3, 0.8]), at([lab.x + lab.w - 3, lab.y + lab.d - 3, 0.8])], { size: 0.45 }));
  return out.join("");
}
const RULER = { x: -150, y: 42, w: 300, d: 7, r: 1 };
const RX0 = RULER.x + 4, RX1 = RULER.x + RULER.w - 4;

// ---------- gantry posts ----------
function post(x) {
  const out = [];
  out.push(Sv(k.slabOf({ x: x - 6, y: Y0 - 6, w: 12, d: 12, r: 1.5 }, 0, 4, P, 3, 0.5), { tone: "mid" }));
  out.push(D(k.corners({ x: x - 6, y: Y0 - 6, w: 12, d: 12 }, 2).map(([a, b]) => at([a, b, 4])), { size: 0.45, tone: "lo" }));
  out.push(Sv(k.cylinder(x, Y0, 3, 4, RAIL_Z - 4 - 4, P, 24), { tone: "mid", crease: "none" }));
  out.push(L([40, 80].map((z) => k.sideArc(x, Y0, 3, z, P, 12)).join(""), { tone: "faint" }));
  out.push(Sv(k.slabOf({ x: x - 5, y: Y0 - 10, w: 10, d: 20, r: 1.5 }, RAIL_Z - 4, 4, P, 3, 0.5), { tone: "mid" }));
  return out.join("");
}
const RAILS = [Y0 - 9, Y0 + 4];
function rail(y) {
  const plan = { x: POSTS[0] - 4, y, w: POSTS[1] - POSTS[0] + 8, d: 3, r: 0.8 };
  return Sv(k.slabOf(plan, RAIL_Z, 3, P, 2, 0.4), { tone: "mid" }) + D(POSTS.map((x) => at([x, y + 1.5, RAIL_Z + 3])), { size: 0.45, tone: "lo" });
}

// ---------- card file (the fast episodic store) ----------
function cardFile() {
  const out = [`<path class="iso-halo" d="${k.haloOf(TRAY, 0, 2, P)}" style="filter:blur(3px)"/>`];
  out.push(Sv(k.slabOf(TRAY, 0, 2, P, 3, 0.5), { tone: "mid" }));
  out.push(Sv(k.slabOf({ x: TRAY.x, y: TRAY.y, w: TRAY.w, d: 2, r: 0.5 }, 2, 14, P, 2, 0.4), { tone: "mid" }));
  out.push(Sv(k.slabOf({ x: TRAY.x, y: TRAY.y + 2, w: 2, d: TRAY.d - 2, r: 0.5 }, 2, 8, P, 2, 0.3), { tone: "mid" }));
  for (let i = 0; i < 10; i++) {
    const x0 = SLOT0 + PITCH * i;
    out.push(`<g data-slot="${i}">`);
    if (i) out.push(Sv(k.slabOf({ x: x0 - 0.4, y: TRAY.y + 2, w: 0.8, d: TRAY.d - 4, r: 0.3 }, 2, 6, P, 2), { tone: "lo", crease: "none" }));
    out.push(S.cardSvg(x0 + 2.4, 2, 16, P));
    out.push(`<g data-flag><g class="flag-lo">${Sv(k.slabOf({ x: x0 + 1.7, y: Y0 - 11, w: 1.4, d: 4, r: 0.4 }, 18, 1.6, P, 2), { tone: "mid", crease: "none" })}</g>`);
    out.push(`<g class="flag-hi">${Sv(k.slabOf({ x: x0 + 1.7, y: Y0 - 11, w: 1.4, d: 4, r: 0.4 }, 18, 6, P, 2), { tone: "hi", crease: "lo" })}</g></g>`);
    out.push(`<g data-front transform="">${S.cardSvg(x0 + 5.6, 2, 16, P)}</g>`);
    out.push(`</g>`);
  }
  out.push(Sv(k.slabOf({ x: TRAY.x + TRAY.w - 2, y: TRAY.y + 2, w: 2, d: TRAY.d - 2, r: 0.5 }, 2, 8, P, 2, 0.3), { tone: "mid" }));
  const front = { x: TRAY.x + 2, y: TRAY.y + TRAY.d - 2, w: TRAY.w - 4, d: 2, r: 0.5 };
  out.push(Sv(k.slabOf(front, 2, 8, P, 2, 0.3), { tone: "mid" }));
  // card-holder pulls on the front wall, one per slot (closed outlines on the face)
  const yF = TRAY.y + TRAY.d;
  out.push(L(Array.from({ length: 10 }, (_, i) => slotOnFace(SLOT0 + PITCH * i + 2.2, SLOT0 + PITCH * i + 7, 4.2, 6.6, yF)).join(""), { tone: "lo" }));
  return out.join("");
}

// ---------- balance (router c3) ----------
function balanceStatic() {
  const { x, y, pivot } = BAL;
  const out = [`<path class="iso-halo" d="${k.haloOf({ x: x - 28, y: y - 12, w: 56, d: 26, r: 4 }, 0, 2, P)}" style="filter:blur(3px)"/>`];
  const plinth = { x: x - 26, y: y - 11, w: 52, d: 24, r: 3 };
  out.push(Sv(k.slabOf(plinth, 0, 6, P, 4, 0.6), { tone: "mid", inner: L(k.sideSeam(plinth, 3, P, 4), { tone: "faint" }) }));
  out.push(D(k.corners(plinth, 3).map(([a, b]) => at([a, b, 6])), { size: 0.45, tone: "lo" }));
  // levelling screws
  for (const [a, b] of byDepth([[x - 20, y + 9], [x + 20, y + 9]])) {
    out.push(Sv(k.cylinder(a, b, 2.4, 6, 2.4, P, 18), { tone: "mid" }));
    out.push(L(k.knurl(a, b, 2.4, 6, 8.4, 14, P), { tone: "lo" }));
  }
  out.push(Sv(k.cylinder(x, y, 7, 6, 3, P, 32, 0.5), { tone: "mid" }));
  out.push(Sv(k.cylinder(x, y, 3, 9, pivot - 4 - 9, P, 24), { tone: "mid", crease: "none" }));
  out.push(L([24, 44].map((z) => k.sideArc(x, y, 3, z, P, 12)).join(""), { tone: "faint" }));
  // scale bracket and plate (behind the needle)
  out.push(Sv(k.slabOf({ x: x - 1, y: y - 6, w: 2, d: 1.6, r: 0.3 }, pivot - 4, 84 - pivot + 4, P, 2), { tone: "mid", crease: "none" }));
  const plate = { x: x - 11, y: y - 5, w: 22, d: 1.4, r: 0.4 };
  out.push(Sv(k.slabOf(plate, 81, 11, P, 2, 0.3), { tone: "hi", crease: "lo" }));
  const face = y - 5 + 1.4, top = 92;
  const minor = [], major = [], zero = [];
  for (let m = 0; m <= 0.6001; m += 0.05) {
    const th = THETA(m);
    const tx = x - Math.sin(th) * 28;
    const kind = Math.abs(m - 0.3) < 1e-6 ? zero : Math.abs(m) < 1e-6 || Math.abs(m - 0.6) < 1e-6 ? major : minor;
    kind.push(k.segment([tx, face, top], [tx, face, top - (kind === minor ? 2 : 3.6)], P));
  }
  out.push(L(minor.join(""), { tone: "lo" }), L(major.join(""), { tone: "mid" }), L(zero.join(""), { tone: "hi" }));
  // pivot block (knife-edge seat)
  out.push(Sv(k.slabOf({ x: x - 4, y: y - 4, w: 8, d: 8, r: 1 }, pivot - 4, 2.6, P, 2, 0.3), { tone: "mid" }));
  return out.join("");
}
const THETA = (m) => Math.max(-1, Math.min(1, (m - 0.3) / 0.3)) * (12 * Math.PI / 180);

function inputHolder() {
  const out = [];
  const blk = { x: BAL.x - 9, y: 22, w: 18, d: 7, r: 1.2 };
  out.push(Sv(k.slabOf(blk, 0, 4, P, 3, 0.4), { tone: "mid" }));
  out.push(L(k.planOutline({ x: BAL.x - 7, y: 24.8, w: 14, d: 1.4, r: 0.5 }, 4, P, 2), { tone: "lo" }));
  return out.join("");
}
const inputCard = () => Sv(k.slabOf({ x: BAL.x - 6, y: 24.9, w: 12, d: 1.2, r: 0.4 }, 4, 13, P, 2), { tone: "hi", crease: "lo" });

// ---------- lap (cortex, weight space) ----------
function lapStatic() {
  const out = [`<path class="iso-halo" d="${k.haloOf({ x: 78, y: -42, w: 72, d: 68, r: 4 }, 0, 2, P)}" style="filter:blur(3px)"/>`];
  const cab = { x: 80, y: -40, w: 68, d: 66, r: 3 };
  out.push(Sv(k.slabOf(k.insetPlan(cab, -1.5), 0, 2, P, 4, 0.4), { tone: "lo" }));
  out.push(Sv(k.slabOf(cab, 2, LAP.z - 2, P, 4, 0.8), { tone: "mid", inner: L(k.sideSeam(cab, 18, P, 4), { tone: "faint" }) }));
  const yF = cab.y + cab.d;
  out.push(L(Array.from({ length: 6 }, (_, i) => slotOnFace(88 + i * 6, 91.6 + i * 6, 6, 14.5, yF)).join(""), { tone: "lo" }));
  out.push(D([at([126, yF, 15]), at([142, yF, 15]), at([126, yF, 6]), at([142, yF, 6])], { size: 0.45, tone: "lo" }));
  out.push(L(slotOnFace(128.5, 139.5, 8, 12.5, yF), { tone: "lo" }));
  out.push(D(k.corners(cab, 3).map(([a, b]) => at([a, b, LAP.z])), { size: 0.45, tone: "lo" }));
  // table ring and the lap plate
  out.push(Sv(k.cylinder(LAP.x, LAP.y, LAP.r + 1.6, LAP.z, 1.2, P, 72), { tone: "lo" }));
  out.push(Sv(k.cylinder(LAP.x, LAP.y, LAP.r, LAP.z + 1.2, LAP.top - LAP.z - 1.2, P, 72, 0.6), { tone: "hi", crease: "lo", inner: L(k.sideArc(LAP.x, LAP.y, LAP.r, LAP.z + 2.4, P, 48), { tone: "faint" }) }));
  out.push(L(k.ring(LAP.x, LAP.y, LAP.r - 2.5, LAP.top, P, 72) + k.ring(LAP.x, LAP.y, LAP.r - 18, LAP.top, P, 48), { tone: "lo" }));
  return out.join("");
}
function lapFront() {
  const out = [];
  // conditioning ring, yoke post and arm (back left of the plate)
  const rc = [104, LAP.y - 19];
  out.push(Sv(k.cylinder(84.5, rc[1], 2.2, LAP.z, 12.5, P, 16), { tone: "mid" }));
  out.push(Sv(k.cylinder(rc[0], rc[1], 7, LAP.top, 5, P, 36, 0.4), { tone: "mid" }));
  out.push(L(k.ring(rc[0], rc[1], 5, LAP.top + 5, P, 28), { tone: "lo" }));
  out.push(Sv(k.slabOf({ x: 84.5, y: rc[1] - 1.5, w: rc[0] - 84.5, d: 3, r: 0.8 }, LAP.z + 12.5, 2, P, 2, 0.3), { tone: "mid" }));
  out.push(Sv(k.cylinder(rc[0], rc[1], 1.4, LAP.top + 5, LAP.z + 12.5 - LAP.top - 5, P, 12), { tone: "mid", crease: "none" }));
  out.push(D([at([84.5, rc[1], LAP.z + 14.5]), at([rc[0], rc[1], LAP.z + 14.5])], { size: 0.5, tone: "lo" }));
  // speed knob on the cabinet top, front right
  out.push(Sv(k.cylinder(141, 19, 4, LAP.z, 4, P, 28), { tone: "mid" }));
  out.push(L(k.knurl(141, 19, 4, LAP.z, LAP.z + 4, 22, P), { tone: "lo" }));
  out.push(L(k.lineOnTop([139, 19], [144, 17], LAP.z + 4, P), { tone: "hi", free: "end" }));
  return out.join("");
}

// ---------- assemble ----------
const T0 = S.CORE.every;
const svg = k.figureSvg({
  width: WIDTH, height: HEIGHT,
  label: "A bench instrument for a continual learner. Left, a card file holds one-shot memories, one slot per digit; a raised tab marks a digit unseen for 300 ticks. Centre, a two-pan balance weighs the store against the cortex on each image: the store answers only when its relative margin is at least 0.3, set by a fixed rider on the cortex arm. Right, a slowly turning lap plate is the cortex's weight space. Overhead, a hoist shuttle carries cards from absent digits to the lap every 38th waking arrival.",
  body: [
    base(),
    `<g data-part="ruler-pointer">${Sv(k.slabOf({ x: RX0 - 2.5, y: RULER.y - 3, w: 5, d: 10, r: 1 }, 0.6, 2, P, 2, 0.3), { tone: "mid" })}${L(k.lineOnTop([RX0, RULER.y - 3], [RX0, RULER.y + 7], 2.6, P), { tone: "hi" })}</g>`,
    post(POSTS[0]),
    cardFile(),
    balanceStatic(),
    `<g data-part="balance">${S.balanceSvg(0, 0, P)}</g>`,
    inputHolder(),
    `<g data-part="input">${inputCard()}</g>`,
    lapStatic(),
    `<g data-part="grooves">${S.groovesSvg(0, P)}</g>`,
    lapFront(),
    post(POSTS[1]),
    rail(RAILS[0]),
    `<g data-part="hoist">${S.hoistSvg(S.PARK_X, S.CARRY_Z + 16, P)}</g>`,
    `<g data-part="card-clip"><g data-part="card" style="opacity:0"></g></g>`,
    rail(RAILS[1]),
    `<g data-part="carriage">${S.carriageSvg(S.PARK_X, P)}</g>`,
  ],
});

// clip used while a card rises out of its slot: everything above the near top edges of the tray walls
const wallTop = 10;
const A = at([TRAY.x - 200, TRAY.y + TRAY.d, wallTop]), B = at([TRAY.x + TRAY.w, TRAY.y + TRAY.d, wallTop]), C = at([TRAY.x + TRAY.w, TRAY.y - 300, wallTop]);
const clipD = k.pathOf([A, B, C, [C[0], -4000], [A[0], -4000]], true);

const DATA = { P, RX0, RX1, clipD, THETA_MAX: 12 * Math.PI / 180 };
const live = `
const DATA = ${JSON.stringify(DATA)};
const P = DATA.P;
${LIVE}
`;

const body = k.plateHtml({
  fig: "Fig 1",
  title: "CLS dual: store, router, awake replay",
  hint: "Click to pause · → steps one tick",
  readout: "tick 1200 · digit 0 → cortex · replay in 38",
  keys: [
    { mark: "lit", label: "Shuttle card and needle: what moves when the dual acts" },
    { mark: "raised", label: "Card file: one-shot store, raised tab = digit absent ≥ 300 ticks" },
    { mark: "edge", label: "Lap plate: cortex weight space, the only part that changes connection strengths" },
  ],
  caption: "Every waking arrival is written into the card file and weighed on the balance: the store answers only when it disagrees with the cortex at relative margin ≥ 0.3 (c3); otherwise the cortex answers. Every 38th waking arrival the hoist carries one batch of 32 store memories, drawn only from digits absent ≥ 300 ticks, into the lap: awake remote replay, interleaved with answering, no nights. On fresh seeds it lifted returns +4.4 / +5.4 with lifetime accuracy +0.05 / +0.15 (two worlds). The stream here is illustrative; the rules and numbers are the system's.",
  body: `<div class="iso-stage" tabindex="0" role="img" aria-label="CLS dual bench: click to pause, right arrow steps one waking tick">${svg}</div>`,
});

const extraCss = `<style>
.iso [data-flag] .flag-hi{opacity:0;transition:opacity .4s}
.iso [data-flag][data-absent] .flag-hi{opacity:1}
.iso [data-flag][data-absent] .flag-lo{opacity:0}
.iso-stage{cursor:pointer;outline:none}
.embed .iso-page{width:min(760px,100% - 4px);margin:2px auto 8px}
html.embed body.iso[data-theme="light"]{--anatomy-page:#fbfbf8}
html.embed body.iso[data-theme="dark"]{--anatomy-page:#151514}
html.embed body.iso[data-theme="light"]{color-scheme:light}
@media (max-width:560px){.iso-page{width:100%;margin:0}.iso-plate .iso-svg{width:112%;margin:0 -6%}}
</style>`;

const html = k.pageHtml({ title: "CLS dual: store, router, awake replay · Learning, by Ez", theme, body: extraCss + body, script: kitScript() + prismScript() + sharedScript() + live, width: 760 });
writeFileSync(OUT, html);
console.log("wrote", OUT, (html.length / 1024).toFixed(0) + " KB");
