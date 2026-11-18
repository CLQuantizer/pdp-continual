"""H26: replay keyed by content, not position. Numbers only. No verdict.

Copy of the H31 interleaved path (--sleep-block 1). Same ByteGRU, lr, seed,
score-before-step order and, on the uniform arm, the same replay sampler.
Each replay step is one Adam step on 8 windows of 128 bytes, initial state 0.
Only the choice of windows changes.

Arms
  uniform  8 uniform windows (H31 inter)
  cued8    8 content-cued windows
  mix44    4 content-cued + 4 uniform

A key is the L2-normalised projection k_c = P h_c. h_c is the live hidden
state at the start of chunk c, detached. P is 128 x 1024 with entries
N(0, 1/128) (variance 1/128), drawn from torch.Generator().manual_seed(2626).
Keys are written once. The cue is P h after the current chunk, L2-normalised.
Eligible chunks are c <= current chunk - 16. The window is the top 32 by
cosine, one drawn uniformly, or all eligible chunks when fewer than 32 are
eligible. No eligible chunk falls back to the uniform sampler.

Does not commit.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ALPHABET = 256
EMB = 64
HIDDEN = 1024
CHUNK = 128
REPLAY_BATCH = 32
CLIP = 1.0
SEED = 0
REPLAY_SEED = 30_000
E10M = 10_000_000
E3M = 3_000_000
E3M_SHA256 = '72953fccfeb78e55251024bd6ffbe269d1d8d730ce2e4fb028117a3613c45aac'
FULL = 100_000_000
SLICE = 1_000_000
PARAMS = 3_627_264
LOG2 = math.log(2)
OFFSETS = np.arange(CHUNK, dtype=np.int64)[:, None]
E10M_SHA256 = '5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97'
KEY_DIM = 128
TOPK = 32
RECENCY_GAP = 16
PROJ_SEED = 2626
ARMS = ('uniform', 'cued8', 'mix44')


def configure():
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha256_arr(raw):
    return hashlib.sha256(raw.tobytes()).hexdigest()


def load_enwik8(path):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != FULL:
        raise SystemExit(f'{path} has {raw.size} bytes, expected {FULL}')
    return raw


def arm_window_counts(arm, replay_batch):
    replay_batch = int(replay_batch)
    if arm == 'uniform':
        return 0, replay_batch
    if arm == 'cued8':
        if replay_batch != 8:
            raise SystemExit('cued8 requires replay batch 8')
        return 8, 0
    if arm == 'mix44':
        if replay_batch != 8:
            raise SystemExit('mix44 requires replay batch 8')
        return 4, 4
    raise SystemExit(f'unknown arm {arm}')


def make_projection(device):
    """Fixed 1024 -> 128 map. Variance 1/128. Its own generator, not the model RNG."""
    gen = torch.Generator()
    gen.manual_seed(PROJ_SEED)
    proj = torch.randn(KEY_DIM, HIDDEN, generator=gen)
    proj = proj * (1.0 / math.sqrt(KEY_DIM))
    return proj.to(device=device, dtype=torch.float32)


def uniform_expected_age(current_chunk, n_seen):
    """Expected (current chunk - start // 128) for one uniform window start."""
    high = int(n_seen) - CHUNK + 1
    if high <= 0:
        raise AssertionError(f'no uniform window at n_seen={n_seen}')
    max_chunk = (high - 1) // CHUNK
    total = 0
    if max_chunk > 0:
        total += CHUNK * (max_chunk - 1) * max_chunk // 2
    rem = high - max_chunk * CHUNK
    total += rem * max_chunk
    return float(current_chunk) - (total / high)


def n_eligible_at(current_chunk):
    return max(0, int(current_chunk) - RECENCY_GAP + 1)


def recent_threshold(n_eligible):
    """Lowest chunk index in the most recent 10% of eligible indices 0..n-1."""
    n_eligible = int(n_eligible)
    if n_eligible < 1:
        return None
    return int(math.floor(0.9 * n_eligible))


class ByteGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(ALPHABET, EMB)
        self.gru = nn.GRU(EMB, HIDDEN)
        self.readout = nn.Linear(HIDDEN, ALPHABET)

    def nll(self, prev, targets, state):
        # -1 is an absent predecessor, not a 257th symbol.
        emb = self.embedding(prev.clamp_min(0))
        emb = emb.masked_fill((prev < 0).unsqueeze(-1), 0)
        y, state = self.gru(emb, state)
        logits = self.readout(y)
        losses = F.cross_entropy(
            logits.reshape(-1, ALPHABET), targets.reshape(-1), reduction='none')
        return losses, state


def deposit(slice_sums, start, values):
    pos = 0
    n = len(values)
    while pos < n:
        idx = (start + pos) // SLICE
        room = SLICE - ((start + pos) % SLICE)
        take = min(room, n - pos)
        slice_sums[idx] += float(values[pos:pos + take].sum())
        pos += take


def pack_windows(raw, starts):
    """Same window layout as H31 sample_replay. starts are byte offsets."""
    starts = np.asarray(starts, dtype=np.int64)
    idx = starts[None, :] + OFFSETS
    targets = raw[idx].astype(np.int64)
    prev = np.empty(idx.shape, dtype=np.int64)
    at_zero = idx == 0
    if np.any(at_zero):
        prev[at_zero] = -1
        keep = ~at_zero
        prev[keep] = raw[idx[keep] - 1]
    else:
        prev[:] = raw[idx - 1]
    return prev, targets


def sample_replay(raw, n_seen, rng):
    """Uniform over starts that have a full 128-byte window. With replacement.

    One rng.integers draw of size REPLAY_BATCH, matching H31.
    """
    high = n_seen - CHUNK + 1
    starts = rng.integers(0, high, size=REPLAY_BATCH, dtype=np.int64)
    prev, targets = pack_windows(raw, starts)
    return prev, targets, starts


def _show_bytes(raw, start):
    start = int(start)
    if start < 0 or start >= len(raw):
        return "b''"
    n = min(40, len(raw) - start)
    blob = bytes(int(x) for x in raw[start:start + n])
    return repr(blob)


class StreamRun:
    def __init__(self, lr, replay, device, phase, sleep_block=1, arm='uniform', retrieval_log=5):
        self.lr = float(lr)
        self.replay = bool(replay)
        self.device = device
        self.phase = phase
        self.sleep_block = int(sleep_block)
        if self.sleep_block != 1:
            raise ValueError('H26 fixes sleep_block at 1')
        self.arm = arm
        self.n_cued_per_step, self.n_uniform_per_step = arm_window_counts(arm, REPLAY_BATCH)
        self.uses_index = self.n_cued_per_step > 0
        self.retrieval_log = int(retrieval_log)
        self.t0 = time.monotonic()
        torch.manual_seed(SEED)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(SEED)
        self.model = ByteGRU().to(device)
        self.model.train()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.rng = np.random.default_rng(REPLAY_SEED + SEED)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        if self.n_params != PARAMS:
            raise AssertionError(f'parameter count {self.n_params} != {PARAMS}')
        self.P = make_projection(device) if self.uses_index else None
        self.slice_sums = np.zeros(0, dtype=np.float64)
        self.total_bits = 0.0
        self.n_done = 0
        self.replay_steps = 0
        self.state = None
        self.first_chunk_bits = None
        self.second_chunk_bits = None
        self._chunk_index = 0
        self._log_mark = -1
        self._reported_slices = 0
        self.step_kinds = []
        self.keys = None
        self.n_keys = 0
        self.covered = None
        self.covered_cued = None
        self.age_sum_cued = 0.0
        self.n_cued = 0
        self.n_recent = 0
        self.age_sum_uniform = 0.0
        self.n_uniform = 0
        self.age_sum_uniform_eligible = 0.0
        self.n_uniform_eligible = 0
        self.n_recent_uniform_eligible = 0
        self.exp_age_sum_at_cued = 0.0
        self.n_exp_at_cued = 0
        self.exp_age_sum_eligible = 0.0
        self.n_exp_eligible = 0
        self.n_fallback_windows = 0
        self.retrieval_entries = []
        self._text_printed = False

    def _ensure(self, n_target):
        need = (n_target + SLICE - 1) // SLICE
        if len(self.slice_sums) < need:
            extra = np.zeros(need - len(self.slice_sums), dtype=np.float64)
            self.slice_sums = np.concatenate((self.slice_sums, extra))
        n_chunks = (n_target + CHUNK - 1) // CHUNK
        if self.covered is None:
            self.covered = np.zeros(n_chunks, dtype=np.bool_)
            self.covered_cued = np.zeros(n_chunks, dtype=np.bool_)
            if self.uses_index:
                self.keys = torch.zeros(n_chunks, KEY_DIM, dtype=torch.float32, device=self.device)

    def _step(self, losses):
        self.opt.zero_grad(set_to_none=True)
        losses.mean().backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
        self.opt.step()

    def _store_key(self, h_start, chunk_index):
        """Write k_c once from the hidden state at the start of chunk c."""
        with torch.no_grad():
            if h_start is None:
                hidden = torch.zeros(HIDDEN, dtype=torch.float32, device=self.device)
            else:
                hidden = h_start.detach().to(dtype=torch.float32, device=self.device).reshape(HIDDEN)
            key = self.P @ hidden
            key = key / torch.linalg.vector_norm(key).clamp_min(1e-12)
            self.keys[chunk_index].copy_(key)
        self.n_keys = int(chunk_index) + 1

    def _cue(self):
        with torch.no_grad():
            hidden = self.state.detach().to(dtype=torch.float32, device=self.device).reshape(HIDDEN)
            cue = self.P @ hidden
            cue = cue / torch.linalg.vector_norm(cue).clamp_min(1e-12)
            return cue

    def _uniform_starts(self, n_seen, n_windows):
        high = int(n_seen) - CHUNK + 1
        if high <= 0:
            raise AssertionError(f'uniform sampler has no window at n_seen={n_seen}')
        return self.rng.integers(0, high, size=int(n_windows), dtype=np.int64)

    def _cued_windows(self, chunk_index, n_windows):
        """Return starts, chunk indices, cosines, n_eligible. None starts means fall back."""
        n_eligible = n_eligible_at(chunk_index)
        if n_eligible < 1:
            return None
        if self.n_keys < chunk_index + 1:
            raise AssertionError(f'key {chunk_index} was not stored (n_keys={self.n_keys})')
        if n_eligible > self.n_keys:
            raise AssertionError(f'eligible {n_eligible} exceeds stored keys {self.n_keys}')
        cue = self._cue()
        scores = self.keys[:n_eligible] @ cue
        n_windows = int(n_windows)
        if n_eligible > TOPK:
            top_values, top_index = torch.topk(scores, TOPK, largest=True, sorted=True)
            top_index_np = top_index.detach().cpu().numpy()
            top_values_np = top_values.detach().float().cpu().numpy()
            slot = self.rng.integers(0, TOPK, size=n_windows, dtype=np.int64)
            chosen = top_index_np[slot]
            chosen_scores = top_values_np[slot]
            allowed = set(int(x) for x in top_index_np)
            if not set(int(x) for x in chosen).issubset(allowed):
                raise AssertionError('cued sample fell outside the top 32')
        else:
            scores_np = scores.detach().float().cpu().numpy()
            chosen = self.rng.integers(0, n_eligible, size=n_windows, dtype=np.int64)
            chosen_scores = scores_np[chosen]
        if int(chosen.max()) > chunk_index - RECENCY_GAP:
            raise AssertionError(
                f'cued chunk {int(chosen.max())} is after current {chunk_index} - {RECENCY_GAP}')
        starts = chosen.astype(np.int64) * CHUNK
        return starts, chosen.astype(np.int64), chosen_scores.astype(np.float64), n_eligible

    def _account(self, starts, modes, chunk_index, n_seen, n_eligible, scores=None):
        starts = np.asarray(starts, dtype=np.int64)
        n = int(starts.shape[0])
        exp_age = uniform_expected_age(chunk_index, n_seen)
        eligible_time = n_eligible_at(chunk_index) >= 1
        if eligible_time:
            self.exp_age_sum_eligible += exp_age * n
            self.n_exp_eligible += n
        retrieved = starts // CHUNK
        threshold = None if not n_eligible else recent_threshold(n_eligible)
        cued_positions = []
        for i in range(n):
            chunk = int(retrieved[i])
            age = int(chunk_index) - chunk
            if chunk < 0 or chunk >= len(self.covered):
                raise AssertionError(f'retrieved chunk {chunk} outside the store')
            self.covered[chunk] = True
            mode = modes[i]
            if mode == 'cued':
                if age < RECENCY_GAP:
                    raise AssertionError(f'cued age {age} at chunk {chunk_index}')
                self.covered_cued[chunk] = True
                self.age_sum_cued += age
                self.n_cued += 1
                cued_positions.append(i)
                if threshold is not None and chunk >= threshold:
                    self.n_recent += 1
            elif mode == 'uniform':
                self.age_sum_uniform += age
                self.n_uniform += 1
                if eligible_time:
                    self.age_sum_uniform_eligible += age
                    self.n_uniform_eligible += 1
                    if threshold is not None and chunk < n_eligible and chunk >= threshold:
                        self.n_recent_uniform_eligible += 1
            elif mode == 'fallback':
                self.n_fallback_windows += 1
                self.age_sum_uniform += age
                self.n_uniform += 1
            else:
                raise AssertionError(f'unknown retrieval mode {mode}')
        n_cued_here = len(cued_positions)
        if n_cued_here:
            self.exp_age_sum_at_cued += exp_age * n_cued_here
            self.n_exp_at_cued += n_cued_here
            self._maybe_log(
                chunk_index, n_eligible, retrieved, scores, cued_positions)

    def _maybe_log(self, chunk_index, n_eligible, retrieved, scores, cued_positions):
        if len(self.retrieval_entries) >= self.retrieval_log:
            return
        if scores is None:
            raise AssertionError('cued retrieval has no cosine scores')
        chunks = [int(retrieved[i]) for i in cued_positions]
        cosines = [float(scores[i]) for i in cued_positions]
        if any(c > int(chunk_index) - RECENCY_GAP for c in chunks):
            raise AssertionError(f'retrieval log left the eligible range at chunk {chunk_index}')
        top_i = max(range(len(cosines)), key=lambda i: (cosines[i], -chunks[i]))
        entry = {
            'cue_chunk': int(chunk_index),
            'n_eligible': int(n_eligible),
            'retrieved_chunks': chunks,
            'cosines': cosines,
            'top_chunk': chunks[top_i],
            'top_cosine': cosines[top_i],
            'cue_byte': int(chunk_index) * CHUNK,
            'top_byte': chunks[top_i] * CHUNK,
        }
        self.retrieval_entries.append(entry)

    def _print_retrieval(self, raw):
        for i, entry in enumerate(self.retrieval_entries, start=1):
            if entry.get('printed'):
                continue
            chunks = entry['retrieved_chunks']
            cosines = [round(float(x), 6) for x in entry['cosines']]
            print(
                f'H26 retrieval i={i} cue_chunk={entry["cue_chunk"]} '
                f'n_eligible={entry["n_eligible"]} retrieved={chunks} cosines={cosines}',
                flush=True)
            if not self._text_printed:
                cue_txt = _show_bytes(raw, entry['cue_byte'])
                top_txt = _show_bytes(raw, entry['top_byte'])
                print(
                    f'H26 retrieval text i={i} cue_chunk={entry["cue_chunk"]} '
                    f'cue_byte={entry["cue_byte"]} cue={cue_txt} '
                    f'top_chunk={entry["top_chunk"]} top_byte={entry["top_byte"]} top={top_txt}',
                    flush=True)
                entry['cue_text'] = cue_txt
                entry['top_text'] = top_txt
                self._text_printed = True
            entry['printed'] = True

    def _replay_batch(self, raw, n_seen, chunk_index):
        if self.arm == 'uniform':
            # H31 sleep-block 1: one uniform draw of REPLAY_BATCH, then one step.
            r_prev, r_tgt, starts = sample_replay(raw, n_seen, self.rng)
            r_prev_t = torch.tensor(r_prev, dtype=torch.long, device=self.device)
            r_tgt_t = torch.tensor(r_tgt, dtype=torch.long, device=self.device)
            r_losses, _ = self.model.nll(r_prev_t, r_tgt_t, None)
            self._step(r_losses)
            del r_losses
            self.replay_steps += 1
            self.step_kinds.append('R')
            self._account(
                starts, ['uniform'] * int(starts.shape[0]), chunk_index, n_seen,
                n_eligible_at(chunk_index))
            return
        parts = []
        modes = []
        score_parts = []
        if self.n_uniform_per_step:
            uniform = self._uniform_starts(n_seen, self.n_uniform_per_step)
            parts.append(uniform)
            modes.extend(['uniform'] * int(uniform.shape[0]))
            score_parts.append(np.full(int(uniform.shape[0]), np.nan, dtype=np.float64))
        cued = self._cued_windows(chunk_index, self.n_cued_per_step)
        if cued is None:
            fallback = self._uniform_starts(n_seen, self.n_cued_per_step)
            parts.append(fallback)
            modes.extend(['fallback'] * int(fallback.shape[0]))
            score_parts.append(np.full(int(fallback.shape[0]), np.nan, dtype=np.float64))
            n_eligible = 0
        else:
            starts, _chosen, chosen_scores, n_eligible = cued
            if int(starts.max()) + CHUNK > int(n_seen):
                raise AssertionError('cued window runs past the bytes seen so far')
            parts.append(starts)
            modes.extend(['cued'] * int(starts.shape[0]))
            score_parts.append(np.asarray(chosen_scores, dtype=np.float64))
        starts = np.concatenate(parts)
        if int(starts.shape[0]) != REPLAY_BATCH:
            raise AssertionError(f'replay windows {starts.shape[0]} != {REPLAY_BATCH}')
        scores = np.concatenate(score_parts)
        r_prev, r_tgt = pack_windows(raw, starts)
        r_prev_t = torch.tensor(r_prev, dtype=torch.long, device=self.device)
        r_tgt_t = torch.tensor(r_tgt, dtype=torch.long, device=self.device)
        r_losses, _ = self.model.nll(r_prev_t, r_tgt_t, None)
        self._step(r_losses)
        del r_losses
        self.replay_steps += 1
        self.step_kinds.append('R')
        self._account(starts, modes, chunk_index, n_seen, n_eligible, scores)
        if raw is not None and self.retrieval_entries:
            self._print_retrieval(raw)

    def _log(self, end, n_target):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        running = self.total_bits / end
        done = end // SLICE
        slice_bpb = ''
        if done > 0 and end % SLICE == 0:
            slice_bpb = f' slice_bpb={self.slice_sums[done - 1] / SLICE:.6f}'
        elif end == n_target and end % SLICE != 0:
            idx = (end - 1) // SLICE
            nbytes = end - idx * SLICE
            slice_bpb = f' slice_bpb={self.slice_sums[idx] / nbytes:.6f}'
        print(
            f'H26 phase={self.phase} n={end}/{n_target} bpb={running:.6f}'
            f'{slice_bpb} replay_steps={self.replay_steps} '
            f'wall_s={time.monotonic() - self.t0:.1f}',
            flush=True)

    def _report_slices(self, end, n_target):
        while self._reported_slices < end // SLICE:
            i = self._reported_slices
            print(
                f'H26 phase={self.phase} slice={i} '
                f'slice_bpb={self.slice_sums[i] / SLICE:.6f} '
                f'n={end}/{n_target} bpb={self.total_bits / end:.6f}',
                flush=True)
            self._reported_slices += 1

    def advance(self, raw, n_target):
        if n_target > len(raw):
            raise AssertionError(f'n_target {n_target} exceeds file {len(raw)}')
        if n_target < self.n_done:
            raise AssertionError('advance would rewind the stream')
        self._ensure(n_target)
        while self.n_done < n_target:
            start = self.n_done
            end = min(start + CHUNK, n_target)
            tlen = end - start
            if start == 0:
                prev_np = np.empty(tlen, dtype=np.int64)
                prev_np[0] = -1
                if tlen > 1:
                    prev_np[1:] = raw[:tlen - 1]
            else:
                prev_np = raw[start - 1:end - 1].astype(np.int64, copy=False)
            tgt_np = raw[start:end].astype(np.int64, copy=False)
            # Reference only. The uniform arm does not touch the index.
            h_start = self.state if self.uses_index else None
            prev_t = torch.tensor(prev_np, dtype=torch.long, device=self.device).unsqueeze(1)
            tgt_t = torch.tensor(tgt_np, dtype=torch.long, device=self.device).unsqueeze(1)
            # Score this chunk under the weights and state from before the update.
            losses, state = self.model.nll(prev_t, tgt_t, self.state)
            bit_vals = (losses.detach().double() / LOG2).cpu().numpy()
            if not np.isfinite(bit_vals).all():
                raise FloatingPointError(f'{self.phase}: nonfinite bits at {start}')
            if self._chunk_index == 0:
                self.first_chunk_bits = bit_vals.copy()
            elif self._chunk_index == 1:
                self.second_chunk_bits = bit_vals.copy()
            deposit(self.slice_sums, start, bit_vals)
            chunk_bits = float(bit_vals.sum())
            self.total_bits += chunk_bits
            self.step_kinds.append('L')
            self.state = state.detach()
            self._step(losses)
            del losses
            if self.uses_index:
                self._store_key(h_start, self._chunk_index)
            if self.replay and end >= CHUNK:
                self._replay_batch(raw, end, self._chunk_index)
            self._chunk_index += 1
            self.n_done = end
            mark = end // 200_000
            if mark > self._log_mark or end == n_target:
                self._log_mark = mark
                self._report_slices(end, n_target)
                self._log(end, n_target)
        gap = abs(self.total_bits - float(self.slice_sums.sum()))
        if gap > 1e-4:
            raise AssertionError(f'slice sums drifted by {gap}')
        return self.result()

    def _retrieval_summary(self):
        n_stored = int(self._chunk_index)
        if self.covered is None:
            n_covered = 0
            n_covered_cued = 0
        else:
            n_covered = int(self.covered[:n_stored].sum())
            n_covered_cued = int(self.covered_cued[:n_stored].sum())

        def mean(total, count):
            if count == 0:
                return None
            return float(total) / float(count)

        return {
            'n_stored': n_stored,
            'n_covered': n_covered,
            'coverage': None if n_stored == 0 else n_covered / n_stored,
            'n_covered_cued': n_covered_cued,
            'coverage_cued_windows': None if n_stored == 0 else n_covered_cued / n_stored,
            'n_cued_windows': int(self.n_cued),
            'age_sum_cued': float(self.age_sum_cued),
            'mean_age_cued': mean(self.age_sum_cued, self.n_cued),
            'n_recent': int(self.n_recent),
            'fraction_recent_10': mean(self.n_recent, self.n_cued),
            'n_uniform_windows': int(self.n_uniform),
            'age_sum_uniform': float(self.age_sum_uniform),
            'mean_age_uniform': mean(self.age_sum_uniform, self.n_uniform),
            'n_uniform_eligible': int(self.n_uniform_eligible),
            'age_sum_uniform_eligible': float(self.age_sum_uniform_eligible),
            'mean_age_uniform_eligible': mean(
                self.age_sum_uniform_eligible, self.n_uniform_eligible),
            'n_recent_uniform_eligible': int(self.n_recent_uniform_eligible),
            'fraction_recent_10_uniform_eligible': mean(
                self.n_recent_uniform_eligible, self.n_uniform_eligible),
            'n_exp_at_cued': int(self.n_exp_at_cued),
            'exp_age_sum_at_cued': float(self.exp_age_sum_at_cued),
            'mean_expected_uniform_age_at_cued': mean(
                self.exp_age_sum_at_cued, self.n_exp_at_cued),
            'n_exp_eligible': int(self.n_exp_eligible),
            'exp_age_sum_eligible': float(self.exp_age_sum_eligible),
            'mean_expected_uniform_age_eligible': mean(
                self.exp_age_sum_eligible, self.n_exp_eligible),
            'n_fallback_windows': int(self.n_fallback_windows),
            'recent_rule': 'chunk >= floor(0.9 * n_eligible), eligible indices 0 .. n_eligible-1',
            'age_rule': 'current chunk - (window start // 128)',
            'retrieval_log': [
                {k: v for k, v in entry.items() if k != 'printed'}
                for entry in self.retrieval_entries
            ],
        }

    def result(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        n = int(self.n_done)
        slices = []
        for i in range((n + SLICE - 1) // SLICE):
            start = i * SLICE
            end = min(start + SLICE, n)
            bits = float(self.slice_sums[i])
            slices.append({
                'index': i,
                'start': start,
                'end': end,
                'bytes': end - start,
                'bits': bits,
                'bits_per_byte': bits / (end - start),
            })
        total = float(self.total_bits)
        replay_tokens = int(self.replay_steps * REPLAY_BATCH * CHUNK)
        out = {
            'phase': self.phase,
            'arm': self.arm,
            'lr': self.lr,
            'replay': self.replay,
            'seed': SEED,
            'replay_seed': REPLAY_SEED + SEED,
            'n': n,
            'bits': total,
            'bits_per_byte': total / n,
            'slices': slices,
            'replay_steps': int(self.replay_steps),
            'replay_tokens': replay_tokens,
            'traffic': replay_tokens / n,
            'sleep_block': 1,
            'replay_batch': int(REPLAY_BATCH),
            'n_cued_per_step': int(self.n_cued_per_step),
            'n_uniform_per_step': int(self.n_uniform_per_step),
            'wall_seconds': time.monotonic() - self.t0,
            'parameters': int(self.n_params),
            'device': str(self.device),
            'first_chunk_bits_per_byte': (
                None if self.first_chunk_bits is None else float(self.first_chunk_bits.mean())),
            'retrieval': self._retrieval_summary(),
        }
        if n <= 2048:
            out['step_kinds'] = ''.join(self.step_kinds)
        return out


def _reference_first_chunk(raw):
    torch.manual_seed(SEED)
    model = ByteGRU()
    prev = torch.empty(CHUNK, 1, dtype=torch.long)
    prev[0, 0] = -1
    prev[1:, 0] = torch.from_numpy(raw[:CHUNK - 1].astype(np.int64))
    tgt = torch.from_numpy(raw[:CHUNK].astype(np.int64))[:, None]
    with torch.no_grad():
        losses, _ = model.nll(prev, tgt, None)
    return (losses.double() / LOG2).numpy()


def sanity_checks():
    """Prequential order, carried state, and replay-between-chunks. CPU, 512 bytes."""
    sums = np.zeros(3, dtype=np.float64)
    deposit(sums, 999_900, np.ones(200, dtype=np.float64))
    if not (sums[0] == 100 and sums[1] == 100 and sums[2] == 0):
        raise AssertionError(f'deposit split failed: {sums}')
    probe = ByteGRU()
    n_params = sum(p.numel() for p in probe.parameters())
    if n_params != PARAMS:
        raise AssertionError(f'parameter count {n_params} != {PARAMS}')
    del probe
    raw = (np.arange(512) % 256).astype(np.uint8)
    ref = _reference_first_chunk(raw)
    cpu = torch.device('cpu')
    global REPLAY_BATCH
    saved = REPLAY_BATCH
    REPLAY_BATCH = 8
    try:
        replay = StreamRun(1e-3, True, cpu, 'sanity-replay', arm='uniform', retrieval_log=0)
        replay.advance(raw, 256)
        noreplay = StreamRun(1e-3, False, cpu, 'sanity-noreplay', arm='uniform', retrieval_log=0)
        noreplay.advance(raw, 256)
        frozen_r = StreamRun(0.0, True, cpu, 'sanity-frozen-replay', arm='uniform', retrieval_log=0)
        frozen_r.advance(raw, 256)
        frozen_n = StreamRun(0.0, False, cpu, 'sanity-frozen-noreplay', arm='uniform', retrieval_log=0)
        frozen_n.advance(raw, 256)
    finally:
        REPLAY_BATCH = saved
    if replay.replay_steps != 2 or noreplay.replay_steps != 0:
        raise AssertionError(
            f'replay steps replay={replay.replay_steps} noreplay={noreplay.replay_steps}')
    if not np.allclose(replay.first_chunk_bits, ref, atol=1e-5):
        raise AssertionError('first chunk was not scored under the initial weights')
    if not np.allclose(replay.first_chunk_bits, noreplay.first_chunk_bits, atol=1e-5):
        raise AssertionError('replay changed the chunk that is scored before it runs')
    if abs(float(replay.second_chunk_bits.mean() - noreplay.second_chunk_bits.mean())) < 1e-6:
        raise AssertionError('replay did not change the next live chunk')
    if not np.allclose(frozen_r.first_chunk_bits, frozen_n.first_chunk_bits, atol=1e-5):
        raise AssertionError('lr 0 first chunks diverged')
    if not np.allclose(frozen_r.second_chunk_bits, frozen_n.second_chunk_bits, atol=1e-5):
        raise AssertionError('replay corrupted the carried live state')
    print(
        f'H26 sanity ok params={n_params} '
        f'first_bpb={float(ref.mean()):.4f} '
        f'replay_chunk1={float(replay.second_chunk_bits.mean()):.4f} '
        f'noreplay_chunk1={float(noreplay.second_chunk_bits.mean()):.4f}',
        flush=True)


def run_arm(raw, n_bytes, lr, replay_batch, arm, device, retrieval_log=5):
    """One prequential pass. Sleep block is 1. Batch 0 would be no replay; H26 does not use it."""
    global REPLAY_BATCH
    replay_batch = int(replay_batch)
    n_bytes = int(n_bytes)
    if replay_batch < 1:
        raise SystemExit('replay-batch must be >= 1')
    if arm not in ARMS:
        raise SystemExit(f'arm must be one of {ARMS}')
    if n_bytes <= 0 or n_bytes > len(raw):
        raise SystemExit(f'n-bytes {n_bytes} is outside 1..{len(raw)}')
    REPLAY_BATCH = replay_batch
    run = StreamRun(
        float(lr), True, device, arm, sleep_block=1, arm=arm, retrieval_log=retrieval_log)
    run.advance(raw, n_bytes)
    out = run.result()
    finite = bool(np.isfinite(out['bits'])) and all(
        np.isfinite(sl['bits']) for sl in out['slices'])
    out['finite'] = finite
    out['replay_batch'] = replay_batch
    if not finite:
        raise FloatingPointError(f'{arm}: nonfinite bits')
    if n_bytes >= CHUNK:
        expect_steps = (n_bytes + CHUNK - 1) // CHUNK
    else:
        expect_steps = 0
    if out['replay_steps'] != expect_steps:
        raise AssertionError(
            f'{arm}: replay_steps {out["replay_steps"]} != {expect_steps}')
    return out


def _pack(raw, digest, e3m_digest, arm, lr, replay_batch, n_bytes, row, session, gpu_name, kernel_id):
    return {
        'arm': arm,
        'kernel_id': kernel_id,
        'lr': float(lr),
        'replay_batch': int(replay_batch),
        'sleep_block': 1,
        'n_bytes': int(n_bytes),
        'gpu_name': gpu_name,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'sha256_file': sha256_arr(raw),
        'sha256_e10m': digest,
        'sha256_e3m': e3m_digest,
        'file_bytes': int(raw.size),
        'seed': SEED,
        'replay_seed': REPLAY_SEED + SEED,
        'projection_seed': PROJ_SEED,
        'parameters': int(row['parameters']),
        'session_wall_seconds': session,
        'e3m': row,
    }


def kernel_main(out, lr, replay_batch, arm, n_bytes, kernel_id):
    configure()
    t0 = time.monotonic()
    global REPLAY_BATCH
    REPLAY_BATCH = 32
    sanity_checks()
    path = find_kaggle_enwik8()
    print(f'H26 data {path} bytes={path.stat().st_size}', flush=True)
    raw = load_enwik8(path)
    digest = sha256_arr(raw[:E10M])
    e3m_digest = sha256_arr(raw[:E3M])
    file_digest = sha256_arr(raw)
    print(
        f'H26 sha256_file={file_digest} sha256_e10m={digest} sha256_e3m={e3m_digest}',
        flush=True)
    if digest != E10M_SHA256:
        raise SystemExit(f'E10M sha256 {digest} != {E10M_SHA256}')
    if e3m_digest != E3M_SHA256:
        raise SystemExit(f'E3M sha256 {e3m_digest} != {E3M_SHA256}')
    if int(n_bytes) != E3M:
        raise SystemExit(f'scored n-bytes is {n_bytes}, expected {E3M}')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for the scored run')
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(
        f'H26 arm={arm} device cuda name={gpu_name} torch={torch.__version__} '
        f'lr={float(lr):g} replay_batch={int(replay_batch)} sleep_block=1 '
        f'n={int(n_bytes)} seed={SEED}',
        flush=True)
    row = run_arm(raw, n_bytes, lr, replay_batch, arm, device, retrieval_log=5)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    session = time.monotonic() - t0
    payload = _pack(
        raw, digest, e3m_digest, arm, lr, replay_batch, n_bytes, row,
        session, gpu_name, kernel_id)
    dump(Path(out) / f'{arm}.json', payload)
    retrieval = row['retrieval']
    print('KERNEL_DONE ' + json.dumps({
        'kernel_id': kernel_id,
        'arm': arm,
        'lr': float(lr),
        'replay_batch': int(replay_batch),
        'sleep_block': 1,
        'n': row['n'],
        'bits': row['bits'],
        'bits_per_byte': row['bits_per_byte'],
        'replay_steps': row['replay_steps'],
        'traffic': row['traffic'],
        'wall_seconds': row['wall_seconds'],
        'session_wall_seconds': session,
        'gpu_name': gpu_name,
        'sha256_e3m': e3m_digest,
        'finite': row['finite'],
        'mean_age_cued': retrieval['mean_age_cued'],
        'fraction_recent_10': retrieval['fraction_recent_10'],
        'coverage': retrieval['coverage'],
        'mean_age_uniform_eligible': retrieval['mean_age_uniform_eligible'],
        'mean_expected_uniform_age_at_cued': retrieval['mean_expected_uniform_age_at_cued'],
    }), flush=True)
    return payload


def find_kaggle_enwik8():
    hits = [
        p for p in Path('/kaggle/input').rglob('*')
        if p.is_file() and p.name == 'enwik8' and p.stat().st_size == FULL
    ]
    if len(hits) != 1:
        raise SystemExit(f'expected one enwik8 file of {FULL} bytes, found {hits}')
    return hits[0]


def _fmt(value, digits=6):
    if value is None:
        return 'na'
    return f'{value:.{digits}f}'


def log_elapsed_seconds(path):
    path = Path(path)
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())
    times = [float(row['time']) for row in rows if isinstance(row, dict) and 'time' in row]
    if not times:
        return None
    return max(times)


def _load_arm(out_dir, arm):
    path = Path(out_dir) / 'raw' / arm / f'{arm}.json'
    if not path.is_file():
        raise SystemExit(f'missing {path}')
    payload = json.loads(path.read_text())
    if payload.get('arm') != arm:
        raise SystemExit(f'{path} arm is {payload.get("arm")}')
    if payload.get('sha256_e3m') != E3M_SHA256:
        raise SystemExit(f'{arm} E3M sha256 {payload.get("sha256_e3m")}')
    if payload.get('sha256_e10m') != E10M_SHA256:
        raise SystemExit(f'{arm} E10M sha256 {payload.get("sha256_e10m")}')
    if int(payload.get('sleep_block', 0)) != 1:
        raise SystemExit(f'{arm} sleep_block is not 1')
    if int(payload.get('replay_batch', 0)) != 8:
        raise SystemExit(f'{arm} replay_batch is not 8')
    if abs(float(payload['lr']) - 3e-4) > 1e-15:
        raise SystemExit(f'{arm} lr {payload["lr"]}')
    if int(payload['seed']) != 0:
        raise SystemExit(f'{arm} seed {payload["seed"]}')
    row = payload['e3m']
    if row['n'] != E3M or not row['finite']:
        raise SystemExit(f'{arm} is not a finite E3M run')
    if len(row['slices']) != 3:
        raise SystemExit(f'{arm} has {len(row["slices"])} slices')
    for i, sl in enumerate(row['slices']):
        if sl['index'] != i or sl['bytes'] != SLICE or sl['start'] != i * SLICE:
            raise SystemExit(f'{arm} slice {i} is not a 1 MB block')
    if int(row['replay_steps']) != (E3M + CHUNK - 1) // CHUNK:
        raise SystemExit(f'{arm} replay_steps {row["replay_steps"]}')
    return payload


def write_report(out_dir):
    out_dir = Path(out_dir)
    kernels = json.loads((out_dir / 'kernels.json').read_text())
    smoke = None
    smoke_path = out_dir / 'smoke.json'
    if smoke_path.is_file():
        smoke = json.loads(smoke_path.read_text())
    arms = []
    for name in ARMS:
        payload = _load_arm(out_dir, name)
        row = payload['e3m']
        meta = kernels[name]
        elapsed = meta.get('log_elapsed_seconds')
        if elapsed is None:
            elapsed = payload['session_wall_seconds']
            elapsed_source = 'script_wall'
        else:
            elapsed_source = meta.get('elapsed_source') or 'log_clock'
        retrieval = row['retrieval']
        arms.append({
            'arm': name,
            'bits': row['bits'],
            'bits_per_byte': row['bits_per_byte'],
            'slices': row['slices'],
            'replay_steps': row['replay_steps'],
            'traffic': row['traffic'],
            'wall_seconds': row['wall_seconds'],
            'gpu_name': payload['gpu_name'],
            'kernel_id': meta.get('id', payload['kernel_id']),
            'kernel_version': meta.get('version'),
            'elapsed_seconds': elapsed,
            'elapsed_source': elapsed_source,
            'session_wall_seconds': payload['session_wall_seconds'],
            'log_elapsed_seconds': meta.get('log_elapsed_seconds'),
            'status': meta.get('status'),
            'torch_version': payload['torch_version'],
            'parameters': payload['parameters'],
            'seed': payload['seed'],
            'replay_seed': payload['replay_seed'],
            'lr': payload['lr'],
            'replay_batch': int(payload['replay_batch']),
            'retrieval': retrieval,
        })
    by_name = {row['arm']: row for row in arms}
    counts = {row['arm']: row['replay_steps'] for row in arms}
    if len(set(counts.values())) != 1:
        raise SystemExit(f'replay steps differ: {counts}')
    base = by_name['uniform']['bits_per_byte']
    gaps = {
        'cued8_minus_uniform': by_name['cued8']['bits_per_byte'] - base,
        'mix44_minus_uniform': by_name['mix44']['bits_per_byte'] - base,
    }
    gpu_hours = [row['elapsed_seconds'] / 3600.0 for row in arms]
    lines = [
        '# H26 content-keyed replay on enwik8 E3M',
        '',
        f'E3M bytes: {E3M}. sha256: `{E3M_SHA256}`.',
        'E10M sha256, checked and not scored: '
        f'`{E10M_SHA256}`.',
        'GRU bits/byte = sum of live −log2 p / 3000000. Replay is not part of the code length.',
        '',
        'Fixed settings: lr 0.0003, seed 0, replay seed 30000, embedding 64, hidden 1024,',
        'chunk 128, clip 1.0, Adam, sleep block 1, replay batch 8.',
        'Projection P is 128 x 1024, entries N(0, 1/128) with variance 1/128,',
        f'generator seed {PROJ_SEED}. Keys and the cue are L2-normalised. Eligible chunks',
        'satisfy c <= current chunk - 16. A cued window is uniform on the top 32 eligible',
        'chunks, or on all eligible chunks when fewer than 32 are eligible.',
        '',
        '## E3M',
        '',
        '| Arm | Bits/byte | Replay steps | Wall seconds | GPU |',
        '|---|---:|---:|---:|---|',
    ]
    for row in arms:
        lines.append(
            f"| {row['arm']} | {_fmt(row['bits_per_byte'])} | {row['replay_steps']} | "
            f"{row['wall_seconds']:.3f} | {row['gpu_name']} |")
    lines += ['', '## 1 MB slices', '']
    for row in arms:
        lines += [
            f"### {row['arm']}",
            '',
            '| Slice | Start | End | Bits | Bits/byte |',
            '|---:|---:|---:|---:|---:|',
        ]
        for sl in row['slices']:
            lines.append(
                f"| {sl['index']} | {sl['start']} | {sl['end']} | {_fmt(sl['bits'], 3)} | "
                f"{_fmt(sl['bits_per_byte'])} |")
        lines.append('')
    lines += [
        '## Gaps',
        '',
        'Pre-registered differences. Not scored in this file.',
        '',
        f"cued8 − uniform = {_fmt(gaps['cued8_minus_uniform'])} "
        f"({_fmt(by_name['cued8']['bits_per_byte'])} − {_fmt(base)}).",
        f"mix44 − uniform = {_fmt(gaps['mix44_minus_uniform'])} "
        f"({_fmt(by_name['mix44']['bits_per_byte'])} − {_fmt(base)}).",
        '',
        'Bar 1, not scored here: mix44 − uniform ≤ 0.01 bits/byte.',
        'Bar 2, not scored here: cued8 − uniform > 0.01 bits/byte.',
        '',
        '## Age and coverage',
        '',
        'Age = current chunk − (window start // 128), in chunks. A cued window starts',
        'at the retrieved chunk. Mean cued age is over content-index windows only.',
        'Uniform expected age is the expected age of one uniform start, averaged over',
        'the same windows (one expectation per cued window, at that step).',
        'Uniform realized age is the uniform arm, on steps with current chunk >= 16.',
        'The recent 10% is chunk >= floor(0.9 * n_eligible), with eligible indices',
        '0 .. n_eligible−1 and n_eligible = current chunk − 15.',
        'Coverage is the fraction of scored chunks whose byte range contains at least',
        'one replay window start. Stored chunks are the chunks scored in the run.',
        '',
        '| Arm | Cued windows | Mean cued age | Uniform expected age at those windows | '
        'Fraction of cued windows in recent 10% | Chunks retrieved | Stored chunks | Coverage |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in arms:
        ret = row['retrieval']
        lines.append(
            f"| {row['arm']} | {ret['n_cued_windows']} | {_fmt(ret['mean_age_cued'], 4)} | "
            f"{_fmt(ret['mean_expected_uniform_age_at_cued'], 4)} | "
            f"{_fmt(ret['fraction_recent_10'])} | {ret['n_covered']} | {ret['n_stored']} | "
            f"{_fmt(ret['coverage'])} |")
    uniform_ret = by_name['uniform']['retrieval']
    lines += [
        '',
        f"Uniform arm realized mean age on steps with current chunk >= 16: "
        f"{_fmt(uniform_ret['mean_age_uniform_eligible'], 4)} "
        f"over {uniform_ret['n_uniform_eligible']} windows.",
        f"Uniform arm expected mean age on those windows: "
        f"{_fmt(uniform_ret['mean_expected_uniform_age_eligible'], 4)}.",
        f"Uniform arm fraction of those windows in the recent 10% of the eligible log: "
        f"{_fmt(uniform_ret['fraction_recent_10_uniform_eligible'])}.",
        f"Uniform arm coverage: {_fmt(uniform_ret['coverage'])} "
        f"({uniform_ret['n_covered']} / {uniform_ret['n_stored']}).",
        '',
        '## Kernels',
        '',
        'Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that',
        'clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.',
        'Script wall is time.monotonic from kernel_main entry through the result write. Queue time',
        'is not included. Total GPU hours is the sum across the three kernels.',
        '',
        '| Kernel | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours |',
        '|---|---:|---|---:|---|---:|---:|',
    ]
    for row in arms:
        lines.append(
            f"| {row['kernel_id']} | {row['kernel_version']} | {row['status']} | "
            f"{row['elapsed_seconds']:.3f} | {row['elapsed_source']} | "
            f"{row['session_wall_seconds']:.3f} | {row['elapsed_seconds'] / 3600.0:.4f} |")
    lines += [
        '',
        f"Total GPU hours: {_fmt(sum(gpu_hours), 4)}.",
        '',
    ]
    note = kernels.get('note')
    if note:
        lines += [note, '']
    lines += [
        '## Protocol',
        '',
        '- One online pass per arm. Prequential: −log2 p of each byte is recorded before that chunk is trained.',
        '- Sleep block 1: score the chunk, one Adam step, then one replay step. That is the H31 inter path.',
        '- Each replay step trains 8 windows of 128 bytes from a zero hidden state. The live state is kept.',
        '- uniform draws one rng.integers of size 8, the H31 sampler, seed 30000.',
        '- cued8 draws 8 windows from the content index. mix44 draws 4 uniform starts, then 4 cued windows.',
        '- Before 16 chunks are eligible, a cued window uses the uniform sampler (one integers draw of that width).',
        '- Keys are written once, from the hidden state at the start of the chunk, and are not refreshed.',
        '- The cue is the hidden state after the current chunk. Cosine is the dot product of L2-normalised vectors.',
        '- E3M is bytes [0, 3000000). 1 MB slices are 1000000 bytes.',
        '- Traffic = replay steps × 8 × 128 / bytes.',
        '- Code is h26_cued.py. Private dataset ezzzzzx/pdp-code-h26. Data dataset ezzzzzx/enwik8.',
        '',
    ]
    if smoke is not None:
        eq = smoke['equivalence']
        lines += [
            '## Smoke',
            '',
            'CPU. Not a score.',
            '',
            f"Bytes: {eq['n_bytes']}.",
            f"h31 --sleep-block 1 bits: {eq['h31_bits']:.12f}.",
            f"h26 uniform bits: {eq['h26_uniform_bits']:.12f}.",
            f"Absolute difference: {eq['abs_diff']:.3e}.",
            f"Within 1e-6: {eq['within_1e-6']}.",
            '',
        ]
        lines.append('Finite bits on 8192 bytes:')
        lines.append('')
        for name in ARMS:
            item = smoke['finite'][name]
            lines.append(
                f"- {name}: bits={item['bits']:.12f} finite={item['finite']} "
                f"replay_steps={item['replay_steps']}.")
        lines.append('')
        ret = smoke['retrieval']
        lines += [
            f"Retrieval check: arm {ret['arm']}, bytes {ret['n_bytes']}.",
            f"Logged cued steps: {len(ret['steps'])}.",
            f"Every logged index <= cue chunk - 16: {ret['all_within_gap']}.",
            '',
        ]
        for line in ret['lines']:
            lines.append(line)
        lines.append('')
    text = '\n'.join(lines)
    if not text.endswith('\n'):
        text += '\n'
    (out_dir / 'RESULTS.md').write_text(text)
    merged = {
        'e3m_bytes': E3M,
        'sha256_e3m': E3M_SHA256,
        'sha256_e10m': E10M_SHA256,
        'arms': {
            row['arm']: {
                'lr': row['lr'],
                'replay_batch': row['replay_batch'],
                'sleep_block': 1,
                'bits': row['bits'],
                'bits_per_byte': row['bits_per_byte'],
                'slices': [
                    {
                        'index': sl['index'],
                        'start': sl['start'],
                        'end': sl['end'],
                        'bits': sl['bits'],
                        'bits_per_byte': sl['bits_per_byte'],
                    }
                    for sl in row['slices']
                ],
                'replay_steps': row['replay_steps'],
                'traffic': row['traffic'],
                'wall_seconds': row['wall_seconds'],
                'gpu_name': row['gpu_name'],
                'kernel_id': row['kernel_id'],
                'kernel_version': row['kernel_version'],
                'elapsed_seconds': row['elapsed_seconds'],
                'elapsed_source': row['elapsed_source'],
                'session_wall_seconds': row['session_wall_seconds'],
                'log_elapsed_seconds': row['log_elapsed_seconds'],
                'kernel_status': row['status'],
                'torch_version': row['torch_version'],
                'parameters': row['parameters'],
                'seed': row['seed'],
                'replay_seed': row['replay_seed'],
                'retrieval': row['retrieval'],
            }
            for row in arms
        },
        'gaps': gaps,
        'bars': {
            'mix44_minus_uniform_le': 0.01,
            'cued8_minus_uniform_gt': 0.01,
            'scored_by': 'orchestrator',
        },
        'gpu_hours_total': sum(gpu_hours),
        'replay_steps_matched': counts['uniform'],
        'smoke': smoke,
    }
    dump(out_dir / 'results.json', merged)
    print(f'H26 wrote {out_dir / "RESULTS.md"} and results.json', flush=True)


def self_check():
    """Indexer arithmetic only. Does not seed the model RNG."""
    if n_eligible_at(15) != 0 or n_eligible_at(16) != 1 or n_eligible_at(100) != 85:
        raise AssertionError('eligibility boundary')
    if recent_threshold(100) != 90 or recent_threshold(1) != 0 or recent_threshold(10) != 9:
        raise AssertionError('recent 10% boundary')
    # n_seen = 128, only start 0, age at chunk 0 is 0.
    if abs(uniform_expected_age(0, 128) - 0.0) > 1e-12:
        raise AssertionError('expected age at one window')
    # n_seen = 256, current chunk 1. Starts 0..128. 128 in chunk 0, one in chunk 1.
    age = uniform_expected_age(1, 256)
    if abs(age - (1 - 1 / 129)) > 1e-12:
        raise AssertionError(f'expected age {age}')
    raw = np.arange(400, dtype=np.uint8)
    rng = np.random.default_rng(30000)
    starts = rng.integers(0, 200, size=8, dtype=np.int64)
    prev, targets = pack_windows(raw, starts)
    if prev.shape != (128, 8) or targets.shape != (128, 8):
        raise AssertionError('window shape')
    if int(targets[0, 0]) != int(raw[int(starts[0])]):
        raise AssertionError('window target')
    if int(starts[0]) == 0:
        if int(prev[0, 0]) != -1:
            raise AssertionError('missing predecessor')
    else:
        if int(prev[0, 0]) != int(raw[int(starts[0]) - 1]):
            raise AssertionError('predecessor byte')
    gen = torch.Generator()
    gen.manual_seed(PROJ_SEED)
    left = torch.randn(KEY_DIM, HIDDEN, generator=gen)
    gen2 = torch.Generator()
    gen2.manual_seed(PROJ_SEED)
    right = torch.randn(KEY_DIM, HIDDEN, generator=gen2)
    if not torch.equal(left, right):
        raise AssertionError('projection seed is not stable')
    scaled = left / math.sqrt(KEY_DIM)
    # Cosine is invariant to a positive rescaling of P.
    hidden = torch.randn(HIDDEN)
    a = scaled @ hidden
    b = (left * 3.5) @ hidden
    a = a / torch.linalg.vector_norm(a)
    b = b / torch.linalg.vector_norm(b)
    if float((a - b).abs().max()) > 1e-5:
        raise AssertionError('normalised projection depends on scale')
    print('H26 self-check ok', flush=True)


def main():
    global SEED
    ap = argparse.ArgumentParser()
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--self-check', action='store_true')
    ap.add_argument('--lr', type=float)
    ap.add_argument('--replay-batch', type=int)
    ap.add_argument('--arm')
    ap.add_argument('--n-bytes', type=int)
    ap.add_argument('--out')
    ap.add_argument('--data', default='data/enwik8/enwik8')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sleep-block', type=int, default=1)
    ap.add_argument('--retrieval-log', type=int, default=5)
    args = ap.parse_args()
    if args.self_check:
        self_check()
        if args.arm is None and not args.assemble:
            return
    if int(args.sleep_block) != 1:
        raise SystemExit('H26 fixes --sleep-block at 1')
    SEED = int(args.seed)
    if args.assemble:
        write_report(args.out or 'results/h26')
        return
    missing = [
        flag for flag, value in (
            ('--lr', args.lr),
            ('--replay-batch', args.replay_batch),
            ('--arm', args.arm),
            ('--n-bytes', args.n_bytes),
            ('--out', args.out),
        ) if value is None
    ]
    if missing:
        raise SystemExit('missing ' + ' '.join(missing))
    configure()
    raw = load_enwik8(args.data)
    digest = sha256_arr(raw[:E10M])
    e3m_digest = sha256_arr(raw[:E3M])
    if digest != E10M_SHA256:
        raise SystemExit(f'E10M sha256 {digest} != {E10M_SHA256}')
    if e3m_digest != E3M_SHA256:
        raise SystemExit(f'E3M sha256 {e3m_digest} != {E3M_SHA256}')
    print(
        f'H26 file bytes={raw.size} sha256_e3m={e3m_digest} sha256_e10m={digest} '
        f'arm={args.arm} lr={float(args.lr):g} replay_batch={int(args.replay_batch)} '
        f'sleep_block=1 n={int(args.n_bytes)} seed={SEED} '
        f'replay_seed={REPLAY_SEED + SEED}',
        flush=True)
    t0 = time.monotonic()
    row = run_arm(
        raw, args.n_bytes, args.lr, args.replay_batch, args.arm, torch.device('cpu'),
        retrieval_log=int(args.retrieval_log))
    session = time.monotonic() - t0
    payload = _pack(
        raw, digest, e3m_digest, args.arm, args.lr, args.replay_batch,
        args.n_bytes, row, session, 'cpu', None)
    dump(Path(args.out) / f'{args.arm}.json', payload)
    kinds = row.get('step_kinds', '')
    retrieval = row['retrieval']
    print(
        f'H26 arm={args.arm} bits={row["bits"]:.12f} bpb={row["bits_per_byte"]:.12f} '
        f'finite={row["finite"]} replay_steps={row["replay_steps"]} '
        f'sleep_block=1 traffic={row["traffic"]:.6f} '
        f'wall_s={row["wall_seconds"]:.3f} '
        f'mean_age_cued={retrieval["mean_age_cued"]} '
        f'coverage={retrieval["coverage"]} step_kinds={kinds}',
        flush=True)


if __name__ == '__main__':
    main()
