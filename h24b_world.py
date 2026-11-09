"""H24b generator and floors; run with .venv/bin/python h24b_world.py.

Serves compression (principle 2) and provides arbitrary recurring facts for the
priced history store (principle 1). No learner or hypothesis verdict here.
Only the requested RESULTS.md is written; raw measurements go to stdout.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import lzma
import math
from pathlib import Path
import string
import time
from types import FunctionType

import numpy as np
from h20_world import World, ALPHA, PROCS, WORLDS, kt_floors

F_ALPHA = ALPHA + "="
F_PROCS = PROCS + ["F"]
BUCKETS = ("0", "1", "2–4", "5+")
# Reuse the exact imported implementation with a private globals dictionary.
# Its original globals, alphabet and source file remain unchanged.
_kt_f = FunctionType(kt_floors.__code__,
                     dict(kt_floors.__globals__, ALPHA=F_ALPHA, A=len(F_ALPHA)))


class FactWorld(World):
    def __init__(self, seed):
        super().__init__(seed)
        self.fr = np.random.default_rng(np.random.SeedSequence([seed, 0xF24B]))
        self.facts = {}
        while len(self.facts) < 300:
            key = "".join(self.fr.choice(list(string.ascii_lowercase), 4))
            if key not in self.facts:
                self.facts[key] = "".join(self.fr.choice(list(string.digits), 4))
        self.keys = tuple(self.facts)
        self.buf["F"] = []

    def chunk(self, p):
        if p != "F":
            return super().chunk(p)
        return "".join(f"{self.keys[i]}={self.facts[self.keys[i]]};"
                       for i in self.fr.integers(300, size=16))


def stream_f(world, seed, n):
    """Same season algorithm as H20, with F as an eighth possible process."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    w, season = FactWorld(seed), WORLDS[world]
    syms, procs = [], []
    p = F_PROCS[int(w.r.integers(len(F_PROCS)))]
    while len(syms) < n:
        if p.startswith("E"):
            w.buf[p], w.eca[p] = [], None
        syms += w.take(p, season)
        procs += [p] * season
        p = F_PROCS[(F_PROCS.index(p) + 1 +
                     int(w.r.integers(len(F_PROCS) - 1))) % len(F_PROCS)]
    return "".join(syms[:n]), procs[:n]


def value_exposures(s, procs, mix):
    """Score observed value digits before learning; carry records across visits."""
    idx = np.flatnonzero(procs == "F")
    text = "".join(s[i] for i in idx)
    seen = {}
    stats = {b: dict(digits=0, records=0, bits=0.0, recall_bits=0.0)
             for b in BUCKETS}
    for start in range(0, len(text), 10):
        record = text[start:start + 10]
        if len(record) <= 5:
            continue
        key, value = record[:4], record[5:9]
        assert record[4] == "=" and value.isdigit()
        assert len(record) < 10 or record[9] == ";"
        prior = seen.get(key, 0)
        bucket = "0" if prior == 0 else "1" if prior == 1 else "2–4" if prior < 5 else "5+"
        stat = stats[bucket]
        stat["digits"] += len(value)
        stat["records"] += 1
        stat["bits"] += float(mix[idx[start + 5:start + 5 + len(value)]].sum())
        # Known record format, uniform unseen digits, exact recall after a full
        # previous value. This is a value-only reference, not a priced store.
        stat["recall_bits"] += len(value) * math.log2(10) if prior == 0 else 0
        seen[key] = prior + 1
    return stats


def lzma_bits(s):
    return 8 * len(lzma.compress(s.encode("ascii"),
                               preset=9 | lzma.PRESET_EXTREME)) / len(s) if s else None


def run(world, seed, n):
    begin = time.perf_counter()
    s, labels = stream_f(world, seed, n)
    assert len(s) == len(labels) == n and set(s) <= set(F_ALPHA)
    bits, mix, _ = _kt_f(s)
    del bits
    procs = np.asarray(labels)
    per_proc = {}
    for p in F_PROCS:
        mask = procs == p
        text = "".join(c for c, label in zip(s, labels) if label == p)
        per_proc[p] = dict(n=int(mask.sum()),
                           kt=float(mix[mask].mean()) if mask.any() else None,
                           lzma=lzma_bits(text))
    return dict(world=world, seed=seed, n=n, kt=float(mix.mean()),
                lzma=lzma_bits(s), per_proc=per_proc,
                exposures=value_exposures(s, procs, mix),
                seconds=time.perf_counter() - begin)


def smoke():
    original_alpha = kt_floors.__globals__["ALPHA"]
    for seed in (6, 7, 8):
        base, facts = World(seed), FactWorld(seed)
        assert len(facts.facts) == 300
        for key, value in facts.facts.items():
            assert len(key) == len(value) == 4
            assert all(c in string.ascii_lowercase for c in key)
            assert all(c in string.digits for c in value)
        for p in PROCS * 2:
            facts.chunk("F")
            assert base.chunk(p) == facts.chunk(p)
        for world in WORLDS:
            s, labels = stream_f(world, seed, 20000)
            assert (s, labels) == stream_f(world, seed, 20000)
            assert s[:12345] == stream_f(world, seed, 12345)[0]
            season = WORLDS[world]
            assert all(labels[i] != labels[i - 1] for i in range(season, len(s), season))
    # Force visits and interruptions, since a short season schedule can omit F.
    w = FactWorld(6)
    text = "".join("".join(w.take("F", size)) for size in (2048, 16384, 20000))
    for start in range(0, len(text) - 9, 10):
        rec = text[start:start + 10]
        assert rec == f"{rec[:4]}={w.facts[rec[:4]]};"
    toy = "abcd=1234;" * 8
    stats = value_exposures(toy, np.array(["F"] * len(toy)), np.ones(len(toy)))
    assert [stats[b]["digits"] for b in BUCKETS] == [4, 4, 12, 12]
    results = [run(w, s, 20000) for w in WORLDS for s in (6, 7, 8)]
    assert kt_floors.__globals__["ALPHA"] == original_alpha
    return results


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def report(results, elapsed, smoke_seconds):
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M UTC+8")
    lines = ["# H24b floors", "", now, "",
             "N = 524288 symbols per life; seeds 6, 7, 8; no fitted selection. "
             "KT is the prequential fixed-share mixture of orders 0–8 (α = 0.001). "
             "Alphabet: H20's 41 symbols plus `=` (42; uniform 5.3923 bits/symbol). "
             "lzma uses preset 9 with PRESET_EXTREME, including container overhead.", "",
             "F has 300 unique four-letter keys with fixed four-digit values per seed, "
             "16 uniformly sampled records per chunk. F uses an independent seeded RNG. "
             "Under an identical sequence of original-process calls, their symbols are unchanged "
             "(smoke-checked); whole H20 streams are not unchanged because eight-way seasons "
             "alter scheduling and subsequent shared-RNG consumption.", "",
             "## Whole-stream floors (bits/symbol)", "",
             "| World | Seed | KT mixture | lzma -9e | Seconds |",
             "|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f'| {r["world"]} | {r["seed"]} | {r["kt"]:.4f} | {r["lzma"]:.4f} | {r["seconds"]:.2f} |')
    for world in WORLDS:
        subset = [r for r in results if r["world"] == world]
        fields = [f'{np.mean([r[k] for r in subset]):.4f} ± {np.std([r[k] for r in subset]):.4f}' for k in ("kt", "lzma")]
        lines.append(f'| {world} | mean ± population std | {fields[0]} | {fields[1]} | — |')
    lines += ["", "## Per-process floors (bits/symbol)", "",
              "KT entries attribute whole-stream prequential bits by the true process. "
              "lzma entries independently compress that process's concatenated observations "
              "across visits; these are separate codes, not additive attributions of the "
              "whole-stream lzma code. — means the process was absent.", "",
              "| World | Seed | Process | Symbols | KT mixture | lzma -9e (isolated) |",
              "|---|---:|---|---:|---:|---:|"]
    for r in results:
        for p, v in r["per_proc"].items():
            lines.append(f'| {r["world"]} | {r["seed"]} | {p} | {v["n"]} | {fmt(v["kt"])} | {fmt(v["lzma"])} |')
    lines += ["", "## F value digits by prior key appearances", "",
              "Buckets count earlier observed records of the same key across all visits, "
              "before scoring each value. Means use only the four value positions, including "
              "observed digits of a final partial record. Exact recall is a value-only reference: "
              "uniform decimal digits on the first appearance, then probability 1 for the "
              "remembered digits. Its store is unpriced here; this is not two-part T. "
              "lzma does not expose per-digit code lengths.", "",
              "| World | Seed | Prior appearances | Value digits | KT bits/digit | Exact-recall bits/digit |",
              "|---|---:|---|---:|---:|---:|"]
    for r in results:
        for b, v in r["exposures"].items():
            count = v["digits"]
            lines.append(f'| {r["world"]} | {r["seed"]} | {b} | {count} | '
                         f'{fmt(v["bits"] / count if count else None)} | '
                         f'{fmt(v["recall_bits"] / count if count else None)} |')
    lines += ["", "Smoke: six lives at N = 20000 plus deterministic, format, RNG-isolation, "
              "visit-boundary and exposure-bucket assertions passed "
              f"({smoke_seconds:.2f} seconds). Full-run wall time: {elapsed:.2f} seconds.", "",
              "F's first value exposure costs the exact-recall reference 3.3219 bits/digit; every repeat costs 0.",
              "KT first-exposure value costs range from "
              f'{min(r["exposures"]["0"]["bits"] / r["exposures"]["0"]["digits"] for r in results):.4f} to '
              f'{max(r["exposures"]["0"]["bits"] / r["exposures"]["0"]["digits"] for r in results):.4f} bits/digit; '
              "at 5+ prior appearances they range from "
              f'{min(r["exposures"]["5+"]["bits"] / r["exposures"]["5+"]["digits"] for r in results):.4f} to '
              f'{max(r["exposures"]["5+"]["bits"] / r["exposures"]["5+"]["digits"] for r in results):.4f}.',
              f'lzma uses fewer whole-stream bits than KT in {sum(r["lzma"] < r["kt"] for r in results)} of {len(results)} lives.', ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="only run N=20000 validation; write no files")
    args = parser.parse_args()
    start = time.perf_counter()
    smoke_results = smoke()
    smoke_seconds = time.perf_counter() - start
    print(json.dumps(dict(smoke=smoke_results, seconds=smoke_seconds)), flush=True)
    if not args.smoke:
        start = time.perf_counter()
        results = []
        # Sequential: keep KT count tables and lzma -9e memory bounded.
        for world in WORLDS:
            for seed in (6, 7, 8):
                result = run(world, seed, 524288)
                results.append(result)
                print(json.dumps(result), flush=True)
        elapsed = time.perf_counter() - start
        output = report(results, elapsed, smoke_seconds)
        target = Path("results/h24b_floors/RESULTS.md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output)
        print(output, flush=True)
