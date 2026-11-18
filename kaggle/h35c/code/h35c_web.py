"""H35c: byte-level transformers and data-selection arms on raw C4.

Fixed in advance, not tuned: context 512, AdamW lr 6e-4, 200-step warmup,
cosine decay to 10% of the peak at the last step, weight decay 0.1, batch 32,
grad clip 1.0, dropout 0. Snapshot for prog_snap is the initial model before
step 200 and then the weights from the previous 200-step boundary, refreshed
every 200 steps. The reference model is the small net trained 1,000 steps on
the first 5_000_000 bytes of enwik8, seed 0, then frozen.

Selection peeks at a document use only that document's first 128 bytes (or the
whole document if it is shorter). Evaluation bytes are never passed into
selection. Each selected document is cut into non-overlapping 512-byte
sequences and consumed once, in selection order. A round draws 96 unread
documents and keeps the highest-scoring ones until the sequence queue can fill
the next batch of 32. Leftover sequences from the last document stay in the
queue and are trained on. They are not dropped and not repeated.
rand_plus may walk its shuffled pool a second time if one pass does not fill
its step count. Sequences taken on that second pass are counted as re-read.
Every other arm still stops when its document list runs out.
"""
import contextlib
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:
    FlopCounterMode = None

LN2 = math.log(2)
CTX = 512
BATCH = 32
WARMUP = 200
LR = 6e-4
WD = 0.1
CLIP = 1.0
K_CAND = 96
PEEK = 128
SNAP_EVERY = 200
REF_STEPS = 1000
S_DEFAULT = 3000
NEED_SEQS = S_DEFAULT * BATCH
CLS_MIN = 2.0
JACCARD = 0.8
N_PERM = 128
NGRAM = 5
STOP = ('the', 'be', 'to', 'of', 'and', 'that', 'have', 'with')
SIZES = {
    'small': dict(layers=4, d_model=256, heads=4),
    'medium': dict(layers=8, d_model=512, heads=8),
}
ARMS = (
    'rand', 'heur', 'dedup', 'ppl', 'cls', 'pipeline',
    'high_loss', 'rho', 'prog_snap', 'rand_plus',
)
OFFLINE = ('rand', 'heur', 'dedup', 'ppl', 'cls', 'pipeline', 'rand_plus')
ONLINE = ('high_loss', 'rho', 'prog_snap')
ROOT = Path('/root/pdp')
DATA = ROOT / 'data' / 'h35c'

_BULLET = ('●', '•', '∙', '·', '○', '◦', '*', '-')
_JS = re.compile(r'\bjavascript\b', re.I)
_SENT = re.compile(r'[.!?]+')


class SelectionGuard:
    """Eval must not run while a selection peek is in progress."""

    def __init__(self):
        self.in_selection = False
        self.eval_during_selection = 0


GUARD = SelectionGuard()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def seed_all(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def lr_at(step, total):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    span = max(1, (total - 1) - WARMUP)
    progress = min(1.0, max(0.0, (step - WARMUP) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR * 0.1 + LR * 0.9 * cosine


def amp_ctx(device):
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return contextlib.nullcontext()


class CausalSelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        if d % heads != 0:
            raise ValueError(f'{heads} heads do not divide d_model {d}')
        self.heads = heads
        self.head_dim = d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        return x + self.down(F.gelu(self.up(h)))


class ByteLM(nn.Module):
    def __init__(self, layers, d_model, heads):
        super().__init__()
        self.layers = layers
        self.d_model = d_model
        self.heads = heads
        self.tok = nn.Embedding(256, d_model)
        self.pos = nn.Embedding(CTX, d_model)
        self.bos = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.bos)
        self.blocks = nn.ModuleList(Block(d_model, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d_model)
        self._init_gpt()

    def _init_gpt(self):
        """GPT-2 initialisation. Fixed, not swept: N(0, 0.02), residual
        projections scaled by 1/sqrt(2*layers) so the untrained loss stays
        near 8 bits/byte instead of a peaked tied-embedding blow-up."""
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.bos, mean=0.0, std=0.02)
        scale = 0.02 / math.sqrt(2 * self.layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, mean=0.0, std=scale)
            nn.init.normal_(block.down.weight, mean=0.0, std=scale)

    def forward(self, tokens):
        # tokens: (B, T) int64, T <= 512. Position t predicts tokens[:, t]
        # from BOS and tokens[:, :t]. Right-padding does not affect earlier
        # positions under the causal mask.
        b, t = tokens.shape
        if t > CTX:
            raise ValueError(f'context {t} exceeds {CTX}')
        tok = self.tok(tokens)
        bos = self.bos.view(1, 1, -1).expand(b, 1, -1)
        x = torch.cat((bos, tok[:, :-1, :]), dim=1)
        x = x + self.pos(torch.arange(t, device=tokens.device))
        for block in self.blocks:
            x = block(x)
        return F.linear(self.ln(x), self.tok.weight)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def make_model(size, seed, device):
    seed_all(seed)
    spec = SIZES[size]
    model = ByteLM(**spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    return model, opt, scaler


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def frozen_copy(model, device):
    spec = dict(layers=model.layers, d_model=model.d_model, heads=model.heads)
    other = ByteLM(**spec).to(device)
    other.load_state_dict(model.state_dict())
    other.eval()
    for p in other.parameters():
        p.requires_grad_(False)
    return other


def analytical_forward_flops(model, batch, seq):
    d = model.d_model
    layers = model.layers
    per_token = layers * (24 * d * d + 4 * seq * d) + 2 * d * 256
    return int(batch * seq * per_token)


def measure_flops(size, device):
    """Forward and one train-step FLOPs. Restores the torch RNG."""
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    model, opt, scaler = make_model(size, seed=12345, device=device)
    tokens = torch.randint(0, 256, (BATCH, CTX), device=device)
    analytical = analytical_forward_flops(model, BATCH, CTX)
    measured_fwd = measured_step = None
    if FlopCounterMode is not None:
        with FlopCounterMode(display=False) as cm:
            with amp_ctx(device):
                model(tokens)
        measured_fwd = int(cm.get_total_flops())
        opt.zero_grad(set_to_none=True)
        with FlopCounterMode(display=False) as cm:
            with amp_ctx(device):
                loss = F.cross_entropy(model(tokens).view(-1, 256), tokens.view(-1))
            loss.backward()
        measured_step = int(cm.get_total_flops())
    del model, opt, scaler, tokens
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    torch.random.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return {
        'analytical_forward_batch32': analytical,
        'measured_forward_batch32': measured_fwd,
        'measured_train_step_batch32': measured_step,
    }


def flops_scale(measured):
    """Map a T=512 batch-32 forward onto the analytical formula."""
    analytical = measured['analytical_forward_batch32']
    got = measured['measured_forward_batch32']
    if got is None or got <= 0:
        return 1.0, False
    return got / analytical, True


def forward_flops(scale, size, batch, seq):
    spec = SIZES[size]
    # Temporary arithmetic only; same formula as analytical_forward_flops.
    d = spec['d_model']
    layers = spec['layers']
    per_token = layers * (24 * d * d + 4 * seq * d) + 2 * d * 256
    return scale * batch * seq * per_token


def train_step_flops(measured, scale, size):
    got = measured['measured_train_step_batch32']
    if got is not None and got > 0:
        return float(got)
    return 3.0 * forward_flops(scale, size, BATCH, CTX)


class Pool:
    def __init__(self, root):
        root = Path(root)
        self.root = root
        self.offsets = np.load(root / 'pool_offsets.npy')
        self.bin = np.memmap(root / 'pool.bin', dtype=np.uint8, mode='r')
        self.n = int(len(self.offsets) - 1)
        self.lengths = np.diff(self.offsets).astype(np.int32)
        self.nseq = (self.lengths // CTX).astype(np.int32)

    def text(self, i):
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return bytes(self.bin[a:b]).decode('utf-8', errors='replace')

    def preview(self, i, n=200):
        a = int(self.offsets[i])
        b = min(int(self.offsets[i + 1]), a + n)
        return bytes(self.bin[a:b])

    def sequences_of(self, i):
        n = int(self.nseq[i])
        if n <= 0:
            return []
        a = int(self.offsets[i])
        return [(i, k) for k in range(n)]

    def peek_batch(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        lengths = np.minimum(self.lengths[ids], PEEK).astype(np.int64)
        width = int(lengths.max()) if len(ids) else 1
        batch = np.zeros((len(ids), width), dtype=np.int64)
        for row, doc in enumerate(ids):
            n = int(lengths[row])
            a = int(self.offsets[doc])
            batch[row, :n] = np.asarray(self.bin[a:a + n], dtype=np.int64)
        return torch.from_numpy(batch), torch.from_numpy(lengths)

    def gather(self, rows):
        batch = np.empty((len(rows), CTX), dtype=np.int64)
        for i, (doc, seq) in enumerate(rows):
            a = int(self.offsets[int(doc)]) + int(seq) * CTX
            batch[i] = np.asarray(self.bin[a:a + CTX], dtype=np.int64)
        return torch.from_numpy(batch)


def load_byte_docs(path_bin, path_off=None):
    if path_off is None:
        raw = Path(path_bin).read_bytes()
        return [raw]
    offsets = np.load(path_off)
    blob = np.memmap(path_bin, dtype=np.uint8, mode='r')
    docs = []
    for i in range(len(offsets) - 1):
        a, b = int(offsets[i]), int(offsets[i + 1])
        docs.append(bytes(blob[a:b]))
    return docs


def chunk_doc(raw):
    """Non-overlapping pieces, each (array uint8, length). Tail included."""
    out = []
    n = len(raw)
    full = n // CTX
    for i in range(full):
        out.append(raw[i * CTX:(i + 1) * CTX])
    tail = n % CTX
    if tail:
        out.append(raw[-tail:])
    return out


def ends_terminal(line):
    s = line.rstrip()
    if not s:
        return False
    if s[-1] in '"\'’”»':
        s = s[:-1].rstrip()
    return bool(s) and s[-1] in '.!?'


def is_bullet(line):
    t = line.lstrip()
    return bool(t) and t[0] in _BULLET


def heur_keep(text):
    """C4 line rule plus the Gopher-style document rules named in the spec."""
    low = text.lower()
    if 'lorem ipsum' in low or _JS.search(low):
        return False
    lines = text.splitlines() or [text]
    if lines and sum(is_bullet(ln) for ln in lines) / len(lines) > 0.30:
        return False
    kept = [ln for ln in lines if ends_terminal(ln)]
    body = '\n'.join(kept)
    if len(_SENT.findall(body)) < 3:
        return False
    words = body.split()
    if len(words) < 50:
        return False
    mean = sum(len(w) for w in words) / len(words)
    if mean < 3.0 or mean > 10.0:
        return False
    n_words = len(words)
    ellipsis = body.count('...') + body.count('…')
    if body.count('#') / n_words > 0.1 or ellipsis / n_words > 0.1:
        return False
    tokens = {w.strip('.,;:!?\"\'()[]{}').lower() for w in words}
    if sum(1 for sw in STOP if sw in tokens) < 2:
        return False
    return True


def _grams(text):
    words = text.split()
    if len(words) < NGRAM:
        return []
    return [' '.join(words[i:i + NGRAM]).encode('utf-8') for i in range(len(words) - NGRAM + 1)]


def build_dedup_cache(pool):
    """Normalised sha256 and datasketch MinHash hashvalues for every document.

    Hashvalues are a uint32 array, not live MinHash objects, so 60,000
    documents stay inside the VPS memory limit. Permutations are the
    datasketch default seed, shared, scheme affine32.
    """
    from datasketch import MinHash

    t0 = time.perf_counter()
    base = MinHash(num_perm=N_PERM)
    perms = base.permutations
    scheme = base.scheme
    digests = [None] * pool.n
    hv = np.zeros((pool.n, N_PERM), dtype=np.uint32)
    has = np.zeros(pool.n, dtype=bool)
    for doc_id in range(pool.n):
        text = ' '.join(pool.text(doc_id).lower().split())
        digests[doc_id] = hashlib.sha256(text.encode('utf-8')).digest()
        grams = _grams(text)
        if grams:
            mh = MinHash(num_perm=N_PERM, permutations=perms, scheme=scheme)
            mh.update_batch(grams)
            hv[doc_id] = mh.hashvalues
            has[doc_id] = True
        if doc_id and doc_id % 10000 == 0:
            print(f'minhash {doc_id}', flush=True)
    wall = time.perf_counter() - t0
    return {'digests': digests, 'hv': hv, 'has': has, 'scheme': scheme, 'wall_s': wall}


def dedup_keep_ids(cache, ids):
    """Exact normalised dedup, then MinHash LSH. Jaccard >= 0.8, first copy wins."""
    from datasketch import MinHash, MinHashLSH

    t0 = time.perf_counter()
    scheme = cache['scheme']
    seen = set()
    exact = []
    for doc_id in ids:
        doc_id = int(doc_id)
        digest = cache['digests'][doc_id]
        if digest in seen:
            continue
        seen.add(digest)
        exact.append(doc_id)

    def wrap(doc_id):
        return MinHash(num_perm=N_PERM, hashvalues=cache['hv'][doc_id], scheme=scheme)

    lsh = MinHashLSH(threshold=JACCARD, num_perm=N_PERM)
    kept = []
    for doc_id in exact:
        if cache['has'][doc_id]:
            mh = wrap(doc_id)
            hit = False
            for other in lsh.query(mh):
                if mh.jaccard(wrap(int(other))) >= JACCARD:
                    hit = True
                    break
            if hit:
                continue
            lsh.insert(str(doc_id), mh)
        kept.append(doc_id)
    return kept, time.perf_counter() - t0


def cls_keep_ids(ids, scores):
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) == 0:
        return ids
    sc = np.asarray(scores, dtype=np.float64)[ids]
    passed = ids[sc >= CLS_MIN]
    if len(passed) * 2 >= len(ids):
        return passed
    order = np.argsort(-sc, kind='stable')
    return ids[order[:len(ids) // 2]]


def ppl_keep_ids(scores):
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind='stable')
    return order[:len(order) // 2]


def plan_sequences(pool, doc_ids, n_seq, max_epochs=1):
    """Fill `n_seq` sequences. A second epoch re-reads from the start of the order.

    Returns plan, documents used in the first pass, whether the plan is full,
    how many sequences came from epoch 2 or later, and how many epochs were used.
    """
    plan = np.empty((n_seq, 2), dtype=np.int32)
    filled = 0
    used = []
    reread = 0
    epochs_used = 0
    for epoch in range(max_epochs):
        epochs_used = epoch + 1
        for doc_id in doc_ids:
            doc_id = int(doc_id)
            if epoch == 0:
                used.append(doc_id)
            n = int(pool.nseq[doc_id])
            if n <= 0:
                continue
            take = min(n, n_seq - filled)
            plan[filled:filled + take, 0] = doc_id
            plan[filled:filled + take, 1] = np.arange(take, dtype=np.int32)
            if epoch > 0:
                reread += take
            filled += take
            if filled >= n_seq:
                return plan, used, True, reread, epochs_used
    return plan[:filled], used, False, reread, epochs_used


def sequence_supply(pool, doc_ids):
    return int(sum(int(pool.nseq[int(i)]) for i in doc_ids))


@torch.no_grad()
def batch_bpb(model, tokens, lengths, device):
    if GUARD.in_selection is False and getattr(batch_bpb, 'protect', False):
        pass
    tokens = tokens.to(device, non_blocking=True)
    lengths = lengths.to(device, non_blocking=True)
    with amp_ctx(device):
        logits = model(tokens)
    nll = F.cross_entropy(logits.float().view(-1, 256), tokens.view(-1), reduction='none')
    nll = nll.view(tokens.shape[0], tokens.shape[1])
    ar = torch.arange(tokens.shape[1], device=device)
    mask = ar.unsqueeze(0) < lengths.unsqueeze(1)
    bits = (nll / LN2 * mask).sum(dim=1) / lengths.clamp(min=1).double()
    return bits.detach().cpu().numpy()


def score_ids(model, pool, ids, device, chunk=64):
    GUARD.in_selection = True
    try:
        model.eval()
        out = np.empty(len(ids), dtype=np.float64)
        for start in range(0, len(ids), chunk):
            part = ids[start:start + chunk]
            tokens, lengths = pool.peek_batch(part)
            out[start:start + len(part)] = batch_bpb(model, tokens, lengths, device)
        return out
    finally:
        GUARD.in_selection = False


def train_step(model, opt, scaler, batch, step, total, device):
    model.train()
    lr = lr_at(step, total)
    for group in opt.param_groups:
        group['lr'] = lr
    batch = batch.to(device, non_blocking=True)
    opt.zero_grad(set_to_none=True)
    with amp_ctx(device):
        loss = F.cross_entropy(model(batch).view(-1, 256), batch.view(-1))
    if not torch.isfinite(loss):
        return float('nan'), lr
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
    return float(loss.detach().float()) / LN2, lr


@torch.no_grad()
def corpus_bpb(model, docs, device, chunk=32):
    if GUARD.in_selection:
        GUARD.eval_during_selection += 1
        raise RuntimeError('evaluation ran during selection')
    model.eval()
    pieces = []
    for doc in docs:
        for piece in chunk_doc(doc):
            pieces.append(piece)
    if not pieces:
        raise RuntimeError('empty eval set')
    total_bits = 0.0
    total_bytes = 0
    for start in range(0, len(pieces), chunk):
        part = pieces[start:start + chunk]
        lengths = [len(p) for p in part]
        width = max(lengths)
        tokens = np.zeros((len(part), width), dtype=np.int64)
        for i, piece in enumerate(part):
            tokens[i, :len(piece)] = np.frombuffer(piece, dtype=np.uint8).astype(np.int64)
        bpb = batch_bpb(
            model, torch.from_numpy(tokens), torch.tensor(lengths, dtype=torch.long), device)
        for i, length in enumerate(lengths):
            total_bits += float(bpb[i]) * length
            total_bytes += length
    return total_bits / total_bytes, total_bytes


def draw_ids(rng, unread, k):
    avail = np.flatnonzero(unread)
    if len(avail) == 0:
        return avail
    k = min(k, len(avail))
    pick = rng.choice(len(avail), size=k, replace=False)
    return avail[pick]


def run_online(model, opt, scaler, pool, arm, steps, seed, device, ref=None,
               capture_step=None, log_every=100):
    rng = np.random.default_rng(int(seed))
    unread = np.ones(pool.n, dtype=bool)
    queue = []
    used = []
    losses = []
    reject = np.zeros(pool.n, dtype=np.int32)
    n_draws = 0
    n_scored = 0
    scored_tokens = 0
    snap_tokens = 0
    ref_tokens = 0
    capture = None
    snap = None
    held = None
    done = 0
    if arm == 'prog_snap':
        snap = frozen_copy(model, device)
        held = clone_state(model)
    t_select = 0.0
    t0 = time.perf_counter()
    exhausted = False
    for step in range(steps):
        if arm == 'prog_snap' and step > 0 and step % SNAP_EVERY == 0:
            snap.load_state_dict(held)
        if arm == 'prog_snap' and step % SNAP_EVERY == 0:
            held = clone_state(model)
        while len(queue) < BATCH:
            ts = time.perf_counter()
            cands = draw_ids(rng, unread, K_CAND)
            if len(cands) == 0:
                exhausted = True
                t_select += time.perf_counter() - ts
                break
            n_draws += 1
            n_scored += int(len(cands))
            cur = score_ids(model, pool, cands, device)
            scored_tokens += int(np.minimum(pool.lengths[cands], PEEK).sum())
            extra = None
            if arm == 'prog_snap':
                extra = score_ids(snap, pool, cands, device)
                snap_tokens += int(np.minimum(pool.lengths[cands], PEEK).sum())
                scores = extra - cur
            elif arm == 'rho':
                extra = score_ids(ref, pool, cands, device)
                ref_tokens += int(np.minimum(pool.lengths[cands], PEEK).sum())
                scores = cur - extra
            else:
                scores = cur
            if capture_step is not None and step == capture_step and capture is None:
                capture = {
                    'step': step,
                    'ids': cands.astype(int).tolist(),
                    'scores': [float(s) for s in scores],
                    'cur_bpb': [float(s) for s in cur],
                }
            order = np.argsort(-scores, kind='stable')
            taken = set()
            for j in order:
                doc_id = int(cands[j])
                unread[doc_id] = False
                used.append(doc_id)
                taken.add(doc_id)
                queue.extend(pool.sequences_of(doc_id))
                if len(queue) >= BATCH:
                    break
            for doc_id in cands:
                if int(doc_id) not in taken:
                    reject[int(doc_id)] += 1
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_select += time.perf_counter() - ts
            if exhausted:
                break
        if len(queue) < BATCH:
            exhausted = True
            break
        rows = queue[:BATCH]
        del queue[:BATCH]
        bpb, _lr = train_step(model, opt, scaler, pool.gather(rows), step, steps, device)
        done = step + 1
        if step % log_every == 0 or step == steps - 1:
            losses.append([step, bpb])
            print(f'{arm} step {step} bpb {bpb:.4f}', flush=True)
        if not math.isfinite(bpb):
            break
    if device.type == 'cuda':
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {
        'arm': arm,
        'steps_completed': done,
        'exhausted': exhausted,
        'losses': losses,
        'used': used,
        'reject': reject,
        'n_draws': n_draws,
        'n_scored': n_scored,
        'scored_tokens': int(scored_tokens),
        'snap_tokens': int(snap_tokens),
        'ref_tokens': int(ref_tokens),
        'wall_s': wall,
        'wall_select_s': t_select,
        'capture': capture,
        'queue_left': len(queue),
    }


def run_offline(model, opt, scaler, pool, doc_ids, steps, device, log_every=100, tag='rand', max_epochs=1):
    plan, used, ok, reread, epochs_used = plan_sequences(pool, doc_ids, steps * BATCH, max_epochs=max_epochs)
    if not ok:
        raise RuntimeError(f'{tag} has {len(plan)} sequences, needs {steps * BATCH}')
    losses = []
    t0 = time.perf_counter()
    done = 0
    for step in range(steps):
        rows = plan[step * BATCH:(step + 1) * BATCH]
        bpb, _lr = train_step(model, opt, scaler, pool.gather(rows), step, steps, device)
        done = step + 1
        if step % log_every == 0 or step == steps - 1:
            losses.append([step, bpb])
            print(f'{tag} step {step} bpb {bpb:.4f}', flush=True)
        if not math.isfinite(bpb):
            break
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return {
        'arm': tag,
        'steps_completed': done,
        'exhausted': False,
        'losses': losses,
        'used': used,
        'wall_s': time.perf_counter() - t0,
        'wall_select_s': 0.0,
        'n_draws': 0,
        'n_scored': 0,
        'scored_tokens': 0,
        'snap_tokens': 0,
        'ref_tokens': 0,
        'sequences_reread': int(reread),
        'epochs': int(epochs_used),
    }


def shuffled(ids, seed):
    rng = np.random.default_rng(int(seed))
    ids = np.asarray(list(ids), dtype=np.int64)
    return ids[rng.permutation(len(ids))]


def arm_documents(arm, seed, masks, scores):
    if arm in ('rand', 'rand_plus', 'high_loss', 'rho', 'prog_snap'):
        return shuffled(np.arange(masks['n']), seed)
    if arm == 'heur':
        return shuffled(masks['heur'], seed)
    if arm == 'dedup':
        return shuffled(masks['dedup'], seed)
    if arm == 'ppl':
        return shuffled(ppl_keep_ids(scores['ppl']), seed)
    if arm == 'cls':
        return shuffled(cls_keep_ids(np.arange(masks['n']), scores['cls']), seed)
    if arm == 'pipeline':
        return shuffled(cls_keep_ids(masks['pipe_before_cls'], scores['cls']), seed)
    raise KeyError(arm)


def load_masks(root):
    root = Path(root)
    z = np.load(root / 'filters_cpu.npz')
    return {
        'n': int(z['n']),
        'heur': z['heur'],
        'dedup': z['dedup'],
        'pipe_before_cls': z['pipe_before_cls'],
        'minhash_wall_s': float(z['minhash_wall_s']),
        'heur_wall_s': float(z['heur_wall_s']),
    }


def save_cpu_filters(pool, out_dir):
    out_dir = Path(out_dir)
    t0 = time.perf_counter()
    heur = []
    for i in range(pool.n):
        if heur_keep(pool.text(i)):
            heur.append(i)
        if i and i % 10000 == 0:
            print(f'heur {i}', flush=True)
    heur_wall = time.perf_counter() - t0
    heur = np.asarray(heur, dtype=np.int32)
    print(f'heur keep {len(heur)} / {pool.n} in {heur_wall:.1f}s', flush=True)
    cache = build_dedup_cache(pool)
    print(f'minhash cache in {cache["wall_s"]:.1f}s', flush=True)
    dedup_ids, wall_all = dedup_keep_ids(cache, range(pool.n))
    print(f'dedup keep {len(dedup_ids)} in {wall_all:.1f}s', flush=True)
    pipe_ids, wall_pipe = dedup_keep_ids(cache, heur)
    print(f'pipe before cls {len(pipe_ids)} in {wall_pipe:.1f}s', flush=True)
    wall_all = cache['wall_s'] + wall_all
    dedup_ids = np.asarray(dedup_ids, dtype=np.int32)
    pipe_ids = np.asarray(pipe_ids, dtype=np.int32)
    path = out_dir / 'filters_cpu.npz'
    np.savez(
        path, n=np.int32(pool.n), heur=heur, dedup=dedup_ids, pipe_before_cls=pipe_ids,
        minhash_wall_s=np.float64(wall_all + wall_pipe), heur_wall_s=np.float64(heur_wall))
    supply = {
        'pool_sequences': int(pool.nseq.sum()),
        'docs_with_sequence': int((pool.nseq > 0).sum()),
        'heur_docs': int(len(heur)),
        'heur_sequences': sequence_supply(pool, heur),
        'dedup_docs': int(len(dedup_ids)),
        'dedup_sequences': sequence_supply(pool, dedup_ids),
        'pipe_before_cls_docs': int(len(pipe_ids)),
        'pipe_before_cls_sequences': sequence_supply(pool, pipe_ids),
        'heur_wall_s': heur_wall,
        'dedup_wall_s': wall_all,
        'pipe_dedup_wall_s': wall_pipe,
        'need_sequences': NEED_SEQS,
    }
    dump(out_dir / 'filters_cpu.json', supply)
    print(json.dumps(supply), flush=True)
    return supply


def ref_batch(raw, step):
    n = len(raw) // CTX
    start = (step * BATCH) % n
    idx = (np.arange(BATCH) + start) % n
    batch = np.empty((BATCH, CTX), dtype=np.int64)
    base = np.frombuffer(raw[:n * CTX], dtype=np.uint8)
    for row, j in enumerate(idx):
        batch[row] = base[j * CTX:(j + 1) * CTX].astype(np.int64)
    return torch.from_numpy(batch)


def train_reference(device, raw, steps=REF_STEPS):
    model, opt, scaler = make_model('small', seed=0, device=device)
    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        bpb, _lr = train_step(model, opt, scaler, ref_batch(raw, step), step, steps, device)
        if step % 100 == 0 or step == steps - 1:
            losses.append([step, bpb])
            print(f'ref step {step} bpb {bpb:.4f}', flush=True)
        if not math.isfinite(bpb):
            break
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return model, {'losses': losses, 'wall_s': time.perf_counter() - t0, 'steps': steps, 'seed': 0}


@torch.no_grad()
def score_documents_bpb(model, pool, device, doc_batch=32, piece_batch=128):
    """Bits/byte of every pool document. Chunks do not cross documents."""
    model.eval()
    scores = np.empty(pool.n, dtype=np.float64)
    n_tokens = 0
    for start in range(0, pool.n, doc_batch):
        ids = list(range(start, min(pool.n, start + doc_batch)))
        pieces = []
        owners = []
        for doc_id in ids:
            a = int(pool.offsets[doc_id])
            b = int(pool.offsets[doc_id + 1])
            raw = bytes(pool.bin[a:b])
            for piece in chunk_doc(raw):
                pieces.append(piece)
                owners.append(doc_id - start)
        bits = np.zeros(len(ids), dtype=np.float64)
        nbytes = np.zeros(len(ids), dtype=np.float64)
        for p0 in range(0, len(pieces), piece_batch):
            part = pieces[p0:p0 + piece_batch]
            own = owners[p0:p0 + piece_batch]
            lengths = [len(p) for p in part]
            width = max(lengths)
            tokens = np.zeros((len(part), width), dtype=np.int64)
            for i, piece in enumerate(part):
                tokens[i, :len(piece)] = np.frombuffer(piece, dtype=np.uint8).astype(np.int64)
            bpb = batch_bpb(
                model, torch.from_numpy(tokens), torch.tensor(lengths), device)
            for i, owner in enumerate(own):
                bits[owner] += float(bpb[i]) * lengths[i]
                nbytes[owner] += lengths[i]
            n_tokens += int(sum(lengths))
        scores[start:start + len(ids)] = bits / np.maximum(nbytes, 1.0)
        if start and start % 5120 == 0:
            print(f'ppl scored {start}', flush=True)
    return scores, n_tokens


def _as_score_vector(logits, n):
    got = logits.detach().float().cpu().numpy()
    if got.ndim == 0:
        got = np.asarray([float(got)], dtype=np.float64)
    got = np.asarray(got, dtype=np.float64).reshape(-1)
    if len(got) != n:
        raise RuntimeError(f'classifier returned {len(got)} scores for {n} documents')
    return got


def score_classifier(pool, model_dir, device, batch=16):
    """FineWeb-Edu regression logit. fp32: the checkpoint is float32, and the
    keep rule is a threshold on that logit."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True)
    model.to(device).eval()
    scores = np.empty(pool.n, dtype=np.float64)
    per_doc_flops = None
    if FlopCounterMode is not None:
        try:
            sample = tok(
                [pool.text(0)], return_tensors='pt', padding=True,
                truncation=True, max_length=512)
            sample = {k: v.to(device) for k, v in sample.items()}
            with FlopCounterMode(display=False) as cm:
                with torch.no_grad():
                    model(**sample)
            per_doc_flops = float(cm.get_total_flops())
        except Exception as exc:
            print(f'cls flop count failed: {type(exc).__name__}', flush=True)
            per_doc_flops = None
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    t0 = time.perf_counter()
    for start in range(0, pool.n, batch):
        texts = [pool.text(i) for i in range(start, min(pool.n, start + batch))]
        enc = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits.squeeze(-1)
        scores[start:start + len(texts)] = _as_score_vector(logits, len(texts))
        if not np.isfinite(scores[start:start + len(texts)]).all():
            raise RuntimeError(f'non-finite classifier scores at {start}')
        if start and start % 5000 == 0:
            print(f'cls scored {start}', flush=True)
    wall = time.perf_counter() - t0
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return scores, wall, per_doc_flops


def finish_arm(model, device, c4_docs, wiki_docs, row):
    t0 = time.perf_counter()
    c4, c4_bytes = corpus_bpb(model, c4_docs, device)
    wiki, wiki_bytes = corpus_bpb(model, wiki_docs, device)
    row['c4_bpb'] = c4
    row['wiki_bpb'] = wiki
    row['c4_bytes'] = c4_bytes
    row['wiki_bytes'] = wiki_bytes
    row['wall_eval_s'] = time.perf_counter() - t0
    row['eval_during_selection'] = GUARD.eval_during_selection
    print(f"{row['arm']} c4 {c4:.4f} wiki {wiki:.4f}", flush=True)
    return row


def slim_row(row, size, seed, steps):
    used = row.get('used') or []
    reject = row.get('reject')
    out = {
        'arm': row['arm'],
        'size': size,
        'seed': int(seed),
        'steps_requested': int(steps),
        'steps_completed': int(row.get('steps_completed') or 0),
        'exhausted': bool(row.get('exhausted')),
        'c4_bpb': row.get('c4_bpb'),
        'wiki_bpb': row.get('wiki_bpb'),
        'c4_bytes': row.get('c4_bytes'),
        'wiki_bytes': row.get('wiki_bytes'),
        'wall_s': row.get('wall_s'),
        'wall_select_s': row.get('wall_select_s'),
        'wall_eval_s': row.get('wall_eval_s'),
        'n_draws': row.get('n_draws'),
        'n_scored': row.get('n_scored'),
        'scored_tokens': row.get('scored_tokens'),
        'snap_tokens': row.get('snap_tokens'),
        'ref_tokens': row.get('ref_tokens'),
        'n_selected': len(used),
        'selected_ids': [int(x) for x in used],
        'losses': row.get('losses'),
        'eval_during_selection': row.get('eval_during_selection', GUARD.eval_during_selection),
        'queue_left': row.get('queue_left', 0),
        'sequences_reread': int(row.get('sequences_reread') or 0),
        'epochs': int(row.get('epochs') or (2 if row['arm'] == 'rand_plus' else 1)),
        'peek_bytes': PEEK if row['arm'] in ONLINE else None,
        'k_cand': K_CAND if row['arm'] in ONLINE else None,
    }
    if reject is not None:
        top = np.argsort(-reject, kind='stable')[:10]
        out['rejected_top'] = [
            {'id': int(i), 'count': int(reject[i])} for i in top if reject[i] > 0
        ]
    if row.get('capture'):
        out['capture'] = row['capture']
    return out


def load_eval(root):
    root = Path(root)
    c4 = load_byte_docs(root / 'c4_val.bin', root / 'c4_val_offsets.npy')
    wiki = [Path(root / 'wikitext103_test.bin').read_bytes()]
    return c4, wiki


def run_one(arm, size, seed, steps, pool, masks, scores, ref, device, c4, wiki, delta=0):
    total = steps + (delta if arm == 'rand_plus' else 0)
    model, opt, scaler = make_model(size, seed, device)
    if arm in ONLINE:
        row = run_online(model, opt, scaler, pool, arm, total, seed, device, ref=ref)
    else:
        docs = arm_documents(arm, seed, masks, scores)
        epochs = 2 if arm == 'rand_plus' else 1
        row = run_offline(
            model, opt, scaler, pool, docs, total, device, tag=arm, max_epochs=epochs)
    finish_arm(model, device, c4, wiki, row)
    del model, opt, scaler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return slim_row(row, size, seed, total)


def cpu_checks(pool):
    """Fast local checks that do not replace the GPU smoke."""
    device = torch.device('cpu')
    lines = []
    for name, spec in SIZES.items():
        model = ByteLM(**spec)
        lines.append(f'params {name} {model.n_params()}')
    model = ByteLM(**SIZES['small'])
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (2, 64), generator=g)
    y = x.clone()
    y[0, 40:] = (y[0, 40:] + 17) % 256
    with torch.no_grad():
        a = model(x)
        b = model(y)
    # Position t may depend on tokens[:t] only, so token 40 changes logits from 40 on.
    early = (a[0, :40] - b[0, :40]).abs().max().item()
    late = (a[0, 40:] - b[0, 40:]).abs().max().item()
    lines.append(f'causal_early_max_abs {early:.3e} causal_late_max_abs {late:.3e}')
    # Padding past the real bytes must not change the real positions.
    short = x[:, :48]
    pad = torch.zeros(2, 64, dtype=torch.long)
    pad[:, :48] = short
    with torch.no_grad():
        c = model(short)
        d = model(pad)
    pad_gap = (c - d[:, :48]).abs().max().item()
    lines.append(f'pad_gap {pad_gap:.3e}')
    good = (
        'The cat sat on the mat and looked at the birds in the tall green tree. '
        'It was happy to be outside with the people who live on that street. '
        'Birds sang in the tree while the children read a book and talked. '
        'People have time to read and to think about the day with care.'
    )
    bad_short = 'hi'
    js = 'click here javascript now. Yes. No.'
    lines.append(f'heur_good {int(heur_keep(good))} heur_short {int(heur_keep(bad_short))} heur_js {int(heur_keep(js))}')
    lrs = [lr_at(0, 3000), lr_at(199, 3000), lr_at(200, 3000), lr_at(2999, 3000)]
    lines.append('lr ' + ' '.join(f'{v:.6e}' for v in lrs))
    # One real step on a repeated batch must be finite.
    model, opt, scaler = make_model('small', 0, device)
    batch = torch.randint(0, 256, (2, CTX))
    bpb, lr = train_step(model, opt, scaler, batch, 0, 3000, device)
    lines.append(f'one_step_bpb {bpb:.4f} lr {lr:.6e} finite {int(math.isfinite(bpb))}')
    lines.append(f'guard {GUARD.eval_during_selection}')
    lines.append(f'pool_docs {pool.n} pool_bytes {int(pool.lengths.sum())} seqs {int(pool.nseq.sum())}')
    text = '\n'.join(lines)
    print(text, flush=True)
    return {'ok': early == 0.0 and pad_gap == 0.0 and math.isfinite(bpb) and heur_keep(good) and not heur_keep(js),
            'text': text, 'causal_early': early, 'pad_gap': pad_gap}


def time_steps(size, device, kind, n_steps, pool):
    """Wall seconds per step for the GPU-hour plan. Uses random pool bytes."""
    model, opt, scaler = make_model(size, 0, device)
    snap = frozen_copy(model, device) if kind == 'prog_snap' else None
    ref = frozen_copy(model, device) if kind == 'rho' and size == 'small' else None
    if kind == 'rho' and size != 'small':
        ref, _, _ = make_model('small', 1, device)
        ref.eval()
    rng = np.random.default_rng(0)

    def one(step):
        if kind in ONLINE:
            cands = rng.choice(pool.n, size=K_CAND, replace=False)
            score_ids(model, pool, cands, device)
            if kind == 'prog_snap':
                score_ids(snap, pool, cands, device)
            elif kind == 'rho':
                score_ids(ref, pool, cands, device)
        rows = []
        while len(rows) < BATCH:
            doc = int(rng.integers(0, pool.n))
            if pool.nseq[doc] > 0:
                rows.append((doc, 0))
        train_step(model, opt, scaler, pool.gather(rows), step, 3000, device)

    # The first steps compile CUDA kernels. Time only the steps after that.
    for step in range(4):
        one(step)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(n_steps):
        one(100 + step)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    del model, opt, scaler, snap, ref
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return wall / n_steps


if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'checks'
    pool = Pool(DATA)
    if cmd == 'checks':
        result = cpu_checks(pool)
        dump(ROOT / 'results' / 'h35c' / 'cpu_checks.json', result)
        if not result['ok']:
            raise SystemExit('cpu checks failed')
    elif cmd == 'filters':
        save_cpu_filters(pool, DATA)
    else:
        raise SystemExit(f'unknown {cmd}')
