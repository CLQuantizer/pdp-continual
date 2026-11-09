// ---- live page: the model drives every moving part ----
(() => { const q = new URLSearchParams(location.search).get("theme"); if (q === "light" || q === "dark") document.body.dataset.theme = q; else if (!document.body.dataset.fixed) document.body.dataset.theme = matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"; })();
const stage = document.querySelector(".iso-stage");
const svg = stage.querySelector("svg");
const part = (n) => svg.querySelector(`[data-part="${n}"]`);
const readout = document.querySelector("[data-readout]");
const still = matchMedia("(prefers-reduced-motion: reduce)");
const el = { balance: part("balance"), grooves: part("grooves"), hoist: part("hoist"), card: part("card"), clip: part("card-clip"), carriage: part("carriage"), input: part("input"), pointer: part("ruler-pointer") };
const slots = [...svg.querySelectorAll("[data-slot]")];
// clip for a card rising out of the tray
const NS = "http://www.w3.org/2000/svg";
const defs = document.createElementNS(NS, "defs");
defs.innerHTML = `<clipPath id="cls-tray-clip"><path d="${DATA.clipD}"/></clipPath>`;
svg.prepend(defs);

// ---- the stream (illustrative) and the rules (real) ----
function mulberry(seed) { return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const rand = mulberry(49);
const SEASONS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 2, 5], [1, 4, 7], [0, 3, 8], [6, 9, 2]];
const SEASON_LEN = 450, BG = 0.02, P_DISAGREE = 0.12, TICK = 0.3;
const lastSeen = new Array(10).fill(-1e9);
let tick = 0, arrivals = 0, last = null;
function arrive() {
  const season = SEASONS[Math.floor(tick / SEASON_LEN) % SEASONS.length];
  const d = rand() < BG ? Math.floor(rand() * 10) : season[Math.floor(rand() * 3)];
  const disagree = rand() < P_DISAGREE;
  const m = disagree ? rand() * 0.6 : 0;
  lastSeen[d] = tick;
  tick++; arrivals++;
  return { d, disagree, m, store: disagree && m >= CORE.margin };
}
while (tick < 1200) arrive();
arrivals = 0;

// ---- shuttle timeline (sim seconds) ----
const PH = [["go", 1.0], ["lower", 0.7], ["lift", 1.0], ["travel", 1.6], ["drop", 0.9], ["press", 0.8], ["raise", 0.6], ["home", 1.0]];
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const lerp = (a, b, t) => a + (b - a) * t;
let shuttle = null; // { t, digit, sx, held }
const REST_HOOK = CARRY_Z + CARD.h;
function shuttlePose(s) {
  const pose = { cx: PARK_X, hook: REST_HOOK, card: null };
  if (!s) return pose;
  let t = s.t;
  for (const [name, len] of PH) {
    if (t > len) { t -= len; continue; }
    const u = ease(Math.min(1, t / len));
    const sx = s.sx;
    if (name === "go") { pose.cx = lerp(PARK_X, sx, u); }
    if (name === "lower") { pose.cx = sx; pose.hook = lerp(REST_HOOK, 2 + CARD.h, u); }
    if (name === "lift") { const z = lerp(2, CARRY_Z, u); pose.cx = sx; pose.card = { z, h: CARD.h, o: 1 }; pose.hook = z + CARD.h; }
    if (name === "travel") { pose.cx = lerp(sx, LAP.x, u); pose.card = { z: CARRY_Z, h: CARD.h, o: 1 }; pose.hook = CARRY_Z + CARD.h; }
    if (name === "drop") { const z = lerp(CARRY_Z, LAP.top, u); pose.cx = LAP.x; pose.card = { z, h: CARD.h, o: 1 }; pose.hook = z + CARD.h; }
    if (name === "press") { const h = lerp(CARD.h, 0.4, u); pose.cx = LAP.x; pose.card = { z: LAP.top, h, o: 1 - Math.max(0, (u - 0.7) / 0.3) }; pose.hook = LAP.top + h; }
    if (name === "raise") { pose.cx = LAP.x; pose.hook = lerp(LAP.top + 0.4, REST_HOOK, u); }
    if (name === "home") { pose.cx = lerp(LAP.x, PARK_X, u); }
    pose.phase = name;
    return pose;
  }
  return null; // finished
}
const SHUTTLE_LEN = PH.reduce((s, [, l]) => s + l, 0);

// ---- dynamic state ----
const LAP_STEP = 0.6 * Math.PI / 180;           // lap turn per item learned (1 per arrival, 32 per replay batch)
let phase = 0, phaseTarget = 0, pressed = false;
const beam = { x: 0, v: 0 }; let beamTarget = 0, discs = 0;
let inputZ = 0, pointerX = 0, pointerTarget = 0;
const pulse = new Array(10).fill(0);
let simAcc = 0;
let lastText = "";

function onArrival() {
  last = arrive();
  phaseTarget += LAP_STEP;
  beamTarget = last.disagree ? THETA_OF(last.m) : 0;
  discs = last.disagree ? Math.round(last.m / 0.15) : 0;
  inputZ = 3;
  pulse[last.d] = 1.6;
  pointerTarget = (arrivals % CORE.every) / CORE.every;
  if (arrivals % CORE.every === 0) {
    // uniform awake replay (awu): the batch is drawn uniformly from what the store holds, not from absent digits
    const held = lastSeen.map((t, d) => [d, tick - t]).filter(([, a]) => a < CORE.forget);
    if (held.length && !shuttle) {
      const [digit] = held[Math.floor(rand() * held.length)];
      shuttle = { t: 0, digit, sx: slotX(digit), held: held.length };
      pressed = false;
    }
  }
  slots.forEach((g, d) => { const f = g.querySelector("[data-flag]"); const on = tick - lastSeen[d] >= CORE.gap; if (on !== f.hasAttribute("data-absent")) f.toggleAttribute("data-absent", on); });
}
const THETA_OF = (m) => Math.max(-1, Math.min(1, (m - CORE.margin) / CORE.margin)) * DATA.THETA_MAX;

// advance simulated time (only while running, or by a step)
function advance(dt) {
  simAcc += dt;
  while (simAcc >= TICK) { simAcc -= TICK; onArrival(); }
  if (shuttle) {
    shuttle.t += dt;
    const pose = shuttlePose(shuttle);
    if (pose && pose.phase === "press" && !pressed) { pressed = true; phaseTarget += CORE.batch * LAP_STEP; }
    if (!pose) shuttle = null;
  }
}

// ---- drawing ----
let drawn = { beam: NaN, discs: -1, phase: NaN, cx: NaN, hook: NaN, card: "", input: NaN, pointer: NaN };
function draw() {
  if (Math.abs(beam.x - drawn.beam) > 2e-4 || discs !== drawn.discs) { el.balance.innerHTML = balanceSvg(beam.x, discs, P); drawn.beam = beam.x; drawn.discs = discs; }
  if (Math.abs(phase - drawn.phase) > 2e-4) { el.grooves.innerHTML = groovesSvg(phase, P); drawn.phase = phase; }
  const pose = shuttlePose(shuttle) || shuttlePose(null);
  if (pose.cx !== drawn.cx || pose.hook !== drawn.hook) {
    el.carriage.setAttribute("transform", translateAlong(P, [pose.cx - PARK_X, 0, 0]));
    el.hoist.innerHTML = hoistSvg(pose.cx, pose.hook, P);
    drawn.cx = pose.cx; drawn.hook = pose.hook;
  }
  const cardKey = pose.card ? `${pose.cx.toFixed(2)}|${pose.card.z.toFixed(2)}|${pose.card.h.toFixed(2)}|${pose.card.o.toFixed(2)}` : "";
  if (cardKey !== drawn.card) {
    if (pose.card) {
      el.card.innerHTML = cardSvg(pose.cx, pose.card.z, pose.card.h, P, { lit: true });
      el.card.style.opacity = String(pose.card.o);
      if (pose.card.z < 14 && pose.cx < TRAY.x + TRAY.w) el.clip.setAttribute("clip-path", "url(#cls-tray-clip)"); else el.clip.removeAttribute("clip-path");
    } else { el.card.innerHTML = ""; el.card.style.opacity = "0"; }
    drawn.card = cardKey;
  }
  if (inputZ !== drawn.input) { el.input.setAttribute("transform", translateAlong(P, [0, 0, inputZ])); drawn.input = inputZ; }
  if (pointerX !== drawn.pointer) { el.pointer.setAttribute("transform", translateAlong(P, [pointerX * (DATA.RX1 - DATA.RX0), 0, 0])); drawn.pointer = pointerX; }
  slots.forEach((g, d) => { const f = g.querySelector("[data-front]"); const v = pulse[d] > 0.01 ? translateAlong(P, [0, 0, pulse[d]]) : ""; if (f.getAttribute("transform") !== v) f.setAttribute("transform", v); });
  if (last) {
    const next = CORE.every - (arrivals % CORE.every);
    const who = last.store ? "store answers" : "cortex answers";
    const tail = shuttle ? `replaying ${CORE.batch}, uniform over ${shuttle.held} held` : `replay in ${next}`;
    const text = `tick ${tick} · ${last.d} → ${who} · ${tail}`;
    if (text !== lastText) { readout.textContent = text; lastText = text; }
  }
}

// ---- loop ----
let running = !still.matches, visible = false, frame = 0, prev = 0;
function settle(dt, calm) {
  if (calm) { beam.x = beamTarget; beam.v = 0; phase = phaseTarget; inputZ = 0; pointerX = pointerTarget; pulse.fill(0); return; }
  for (let i = 0; i < 4; i++) { const h = dt / 4; beam.v += ((beamTarget - beam.x) * 70 - beam.v * 9) * h; beam.x += beam.v * h; }
  if (Math.abs(beamTarget - beam.x) < 1e-5 && Math.abs(beam.v) < 1e-5) { beam.x = beamTarget; beam.v = 0; }
  phase += (phaseTarget - phase) * (1 - Math.exp(-dt / 1.1)); if (Math.abs(phaseTarget - phase) < 1e-5) phase = phaseTarget;
  inputZ *= Math.exp(-dt / 0.08); if (inputZ < 0.02) inputZ = 0;
  const wrap = pointerTarget < pointerX - 0.5;
  if (wrap) pointerX = pointerTarget; else { pointerX += (pointerTarget - pointerX) * (1 - Math.exp(-dt / 0.12)); if (Math.abs(pointerTarget - pointerX) < 1e-4) pointerX = pointerTarget; }
  for (let d = 0; d < 10; d++) { pulse[d] *= Math.exp(-dt / 0.15); if (pulse[d] < 0.01) pulse[d] = 0; }
}
function tickFrame(now) {
  const dt = prev ? Math.min((now - prev) / 1000, 1 / 20) : 1 / 60;
  prev = now;
  if (running) advance(dt);
  settle(dt, false);
  draw();
  frame = visible ? requestAnimationFrame(tickFrame) : 0;
}
const run = () => { if (!frame && visible) { prev = 0; frame = requestAnimationFrame(tickFrame); } };
new IntersectionObserver((es) => { visible = es[es.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "80px 0px" }).observe(stage);
const step = () => { advance(TICK); run(); };
stage.addEventListener("click", () => { running = !running; run(); });
stage.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight") { e.preventDefault(); running = false; step(); }
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); running = !running; run(); }
});
// still pose: a replay in flight, over the balance
function stillPose() {
  for (let i = 0; i < 4000 && !(shuttle && shuttlePose(shuttle)?.phase === "travel" && shuttle.t > 3.4); i++) advance(1 / 20);
  settle(0, true); draw();
}
if (still.matches || new URLSearchParams(location.search).has("still")) { running = false; stillPose(); } else { advance(0); draw(); }
window.__cls = { advance: (s) => { advance(s); settle(0, true); draw(); }, state: () => ({ tick, arrivals, shuttle, last }) };
// embedded in an iframe: report height to the parent page
if (window.parent !== window) { document.documentElement.classList.add("embed"); const post = () => window.parent.postMessage({ type: "cls-figure-height", height: Math.ceil(document.documentElement.scrollHeight) }, "*"); new ResizeObserver(post).observe(document.body); addEventListener("load", post); post(); }
