"""A fine-tune's training in PyTorch, op by op as program.py lays it out: what finetune.train_segment runs, and what a
worker re-runs in a dispute to show (or to check) any state between two ops.

Training and a dispute's re-run are one computation: the same ops, kernels and inputs on the same device give the same
bits (TF32 off, deterministic kernels: compute.TorchBackend), so the states a worker posts in a dispute are the ones its
training went through, and they chain to the checkpoints it committed to. Every op makes a new tensor (nothing is
changed in place), so a state is just a dict of tensors and a snapshot of it is a shallow copy.
"""
from __future__ import annotations

import contextlib
import threading

import torch
import torch.nn.functional as F

from .program import Node, Op

_F32 = torch.float32


def _rot(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Executor:
    """Runs a Program's ops on states of `rt` (finetune._Runtime: the base model on one device). `tamper`, for tests,
    is a worker that cheats in one op: tamper(where, op) gives a factor its output is multiplied by, or None."""

    def __init__(self, rt, spec, program, tamper=None):
        self.model, self.device, self.spec, self.prog, self.tamper = rt.model, rt.device, spec, program, tamper
        self.a = {k: a for k, (_, a, _, _) in enumerate(rt.lora[spec])}
        self.where: tuple = ()
        self._seq: dict[int, tuple] = {}
        self._head: tuple = (None, None)  # the head as float32 while a sequence's head and loss run

    def w(self, name: str) -> torch.Tensor:
        return self.model.w[name]

    def w32(self, name: str) -> torch.Tensor:
        """A projection's frozen weights as float32, the exact cast its matmul runs on."""
        return self.w(name).to(_F32)

    def seq(self, i: int):
        """One training sequence: (ids, its target positions, their targets) on the device."""
        if i not in self._seq:
            d = self.prog.data
            n = int(d.train_len[i])
            ids = torch.from_numpy(d.train_ids[i, :n].astype("int64")).to(self.device)
            pos = torch.from_numpy((d.train_mask[i, 1:n] != 0).nonzero()[0].astype("int64")).to(self.device)
            self._seq = {i: (ids, pos, ids[pos + 1])}
        return self._seq[i]

    def head32(self, name: str) -> torch.Tensor:
        """The head's weights cast to float32 once for all of a sequence's head ops (each slices its block): the cast
        is exact, so it's the same numbers as casting each block, and it's what llm.py's _HeadLoss holds too."""
        if self._head[0] != name:
            self._head = (name, self.w(name).to(_F32))
        return self._head[1]

    def tables(self, t: int):
        return self.model._tables(t)

    # --- running ---

    def run(self, op: Op, st: dict) -> None:
        out = RUN[op.kind](self, op, st)
        if out is not None:
            if self.tamper is not None:
                f = self.tamper(self.where, op)
                if f is not None:
                    out = out * f
            st[op.out] = out.contiguous()
        for d in op.drops:
            del st[d]

    def run_atoms(self, node: Node, lo: int, hi: int, st: dict, where: tuple = ()) -> None:
        for i in range(lo, hi):
            a = node.atoms[i]
            if isinstance(a, Op):
                self.where = where
                self.run(a, st)
            else:
                self.run_atoms(a, 0, len(a.atoms), st, where + (a.label,))
                if a.label == "head and loss":
                    self._head = (None, None)


# --- the ops, as referee.py checks them -------------------------------------------------------------------------------

def _x(st, op, i=0):
    return st[op.ins[i]]


def _acc(st, op, i, y):
    return st[op.ins[i]] + y if len(op.ins) > i else y


def r_embed(ex, op, st):
    ids, _, _ = ex.seq(op.attrs["seq"])
    return ex.w(op.attrs["weight"])[ids].to(_F32)


def _norm_rsqrt(xs, eps):
    return torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + eps)


def r_rmsnorm(ex, op, st):
    x = _x(st, op)
    xs = x.view(x.shape[0], -1, op.attrs["group"])
    return (ex.w(op.attrs["weight"]) * (xs * _norm_rsqrt(xs, op.attrs["eps"]))).view(x.shape)


def r_linear(ex, op, st):
    y = F.linear(_x(st, op), ex.w32(op.attrs["weight"]))
    bias = op.attrs.get("bias")
    return y + ex.w(bias) if bias else y


def r_lora_down(ex, op, st):
    return _x(st, op) @ ex.a[op.attrs["module"]].T


def r_lora_up(ex, op, st):
    base, xa, b = (st[n] for n in op.ins)
    return base + (xa @ b.T) * op.attrs["scaling"]


def r_rope(ex, op, st):
    x = _x(st, op)
    cos, sin = ex.tables(x.shape[0])
    xs = x.view(x.shape[0], -1, op.attrs["hd"])
    return (xs * cos[:, None] + _rot(xs) * sin[:, None]).view(x.shape)


def _dims(op):
    a = op.attrs
    return a["H"], a["KV"], a["hd"], a["H"] // a["KV"]


def _q4(x, KV, G, hd):  # [T, H·hd] -> [KV, G, T, hd]
    return x.view(x.shape[0], KV, G, hd).permute(1, 2, 0, 3)


def _kv4(x, KV, hd):  # [T, KV·hd] -> [KV, 1, T, hd]
    return x.view(x.shape[0], KV, hd).permute(1, 0, 2).unsqueeze(1)


def r_attn_scores(ex, op, st):
    q, k = (st[n] for n in op.ins)
    H, KV, hd, G = _dims(op)
    T = q.shape[0]
    mask = torch.full((T, T), float("-inf"), device=q.device, dtype=_F32).triu(1)
    return ((_q4(q, KV, G, hd) @ _kv4(k, KV, hd).transpose(-1, -2)) * op.attrs["c"] + mask).reshape(H * T, T)


def r_softmax(ex, op, st):
    return torch.softmax(_x(st, op), dim=-1)


def r_attn_mix(ex, op, st):
    p, v = (st[n] for n in op.ins)
    H, KV, hd, G = _dims(op)
    T = v.shape[0]
    return (p.view(KV, G, T, T) @ _kv4(v, KV, hd)).permute(2, 0, 1, 3).reshape(T, H * hd)


def r_add(ex, op, st):
    return st[op.ins[0]] + st[op.ins[1]]


def r_silu_mul(ex, op, st):
    return F.silu(st[op.ins[0]]) * st[op.ins[1]]


def _head(ex, op, st):
    _, pos, tgt = ex.seq(op.attrs["seq"])
    w = ex.head32(op.attrs["weight"])[op.attrs["b0"]:op.attrs["b1"]]
    return F.linear(st[op.ins[0]][pos], w), w, tgt


def r_head_stats(ex, op, st):
    lo, _, _ = _head(ex, op, st)
    return _stats(lo, op, st)


def _stats(lo, op, st):
    """A head block's running max and sum for the log-softmax, from its logits."""
    mb = lo.max(-1).values
    if len(op.ins) > 1:
        prev = st[op.ins[1]]
        m = torch.maximum(prev[:, 0], mb)
        s = prev[:, 1] * torch.exp(prev[:, 0] - m) + torch.exp(lo - m[:, None]).sum(-1)
    else:
        m, s = mb, torch.exp(lo - mb[:, None]).sum(-1)
    return torch.stack((m, s), dim=-1)


def r_head_lse(ex, op, st):
    ms = _x(st, op)
    return ms[:, :1] + torch.log(ms[:, 1:2])


def r_head_grad(ex, op, st):
    lo, w, tgt = _head(ex, op, st)
    return _grad(lo, w, tgt, op, st)


def _grad(lo, w, tgt, op, st):
    """A head block's share of the gradient into the hidden states, from its logits: softmax minus one-hot.
    Every row subtracts at one column: 1 where its target falls in this block, and 0 elsewhere, which leaves the number
    as it was (x - 0.0 is x, to the bit). So no row is picked out with .nonzero(), which made the CPU wait for the GPU
    38 times a sequence."""
    q = torch.exp(lo - st[op.ins[1]])
    b0, b1 = op.attrs["b0"], op.attrs["b1"]
    hit = ((tgt >= b0) & (tgt < b1)).to(q.dtype)
    col = (tgt - b0).clamp(0, b1 - b0 - 1)
    rows = torch.arange(len(tgt), device=q.device)
    q[rows, col] = q[rows, col] - hit
    return _acc(st, op, 2, (q * op.attrs["ic"]) @ w)


def r_scatter(ex, op, st):
    ids, pos, _ = ex.seq(op.attrs["seq"])
    out = torch.zeros(len(ids), ex.prog.arch.hidden, device=ids.device, dtype=_F32)
    if op.ins:
        out[pos] = st[op.ins[0]]
    return out


def r_rmsnorm_bwd(ex, op, st):
    x, dy = st[op.ins[0]], st[op.ins[1]]
    g = op.attrs["group"]
    xs, dys = x.view(x.shape[0], -1, g), dy.view(x.shape[0], -1, g)
    r = _norm_rsqrt(xs, op.attrs["eps"])
    dyw = dys * ex.w(op.attrs["weight"])
    dm = (dyw * xs).mean(-1, keepdim=True)
    return _acc(st, op, 2, (r * (dyw - xs * ((r * r) * dm))).view(x.shape))


def r_linear_bwd(ex, op, st):
    return _acc(st, op, 1, _x(st, op) @ ex.w32(op.attrs["weight"]))


def r_scale(ex, op, st):
    return _x(st, op) * op.attrs["c"]


def r_lora_grad(ex, op, st):
    return _acc(st, op, 2, st[op.ins[0]].T @ st[op.ins[1]])


def r_lora_bwd_mid(ex, op, st):
    return st[op.ins[0]] @ st[op.ins[1]]


def r_lora_bwd_x(ex, op, st):
    return st[op.ins[1]] + st[op.ins[0]] @ ex.a[op.attrs["module"]]


def r_silu_bwd_g(ex, op, st):
    g, u, dm = (st[n] for n in op.ins)
    sig = torch.sigmoid(g)
    return (dm * u) * (sig * (1 + g * (1 - sig)))


def r_silu_bwd_u(ex, op, st):
    return st[op.ins[1]] * F.silu(st[op.ins[0]])


def r_attn_dp(ex, op, st):
    do, v = (st[n] for n in op.ins)
    H, KV, hd, G = _dims(op)
    T = v.shape[0]
    return (_q4(do, KV, G, hd) @ _kv4(v, KV, hd).transpose(-1, -2)).reshape(H * T, T)


def r_attn_dv(ex, op, st):
    p, do = (st[n] for n in op.ins)
    H, KV, hd, G = _dims(op)
    T = do.shape[0]
    return (p.view(KV, G, T, T).transpose(-1, -2) @ _q4(do, KV, G, hd)).sum(1).permute(1, 0, 2).reshape(T, KV * hd)


def r_softmax_bwd(ex, op, st):
    p, dp = (st[n] for n in op.ins)
    return p * (dp - (dp * p).sum(-1, keepdim=True))


def r_attn_dq(ex, op, st):
    ds, k = (st[n] for n in op.ins)
    H, KV, hd, G = _dims(op)
    T = k.shape[0]
    return ((ds * op.attrs["c"]).view(KV, G, T, T) @ _kv4(k, KV, hd)).permute(2, 0, 1, 3).reshape(T, H * hd)


def r_attn_dk(ex, op, st):
    ds, q = (st[n] for n in op.ins)
    H, KV, hd, G = _dims(op)
    T = q.shape[0]
    dss = (ds * op.attrs["c"]).view(KV, G, T, T)
    return (dss.transpose(-1, -2) @ _q4(q, KV, G, hd)).sum(1).permute(1, 0, 2).reshape(T, KV * hd)


def r_rope_bwd(ex, op, st):
    dy = _x(st, op)
    cos, sin = ex.tables(dy.shape[0])
    hd = op.attrs["hd"]
    ys = dy.view(dy.shape[0], -1, hd)
    a, b = ys * cos[:, None], ys * sin[:, None]
    return (a + torch.cat((b[..., hd // 2:], -b[..., :hd // 2]), dim=-1)).view(dy.shape)


def r_adam_m(ex, op, st):
    m, g = (st[n] for n in op.ins)
    return m * op.attrs["b1"] + g * op.attrs["c1m"]


def r_adam_v(ex, op, st):
    v, g = (st[n] for n in op.ins)
    return v * op.attrs["b2"] + (g * g) * op.attrs["c2m"]


def r_adam_b(ex, op, st):
    b, m, v = (st[n] for n in op.ins)
    a = op.attrs
    return b - ((m / a["c1"]) / ((v / a["c2"]).sqrt() + a["eps"])) * a["lr"]


def r_free(ex, op, st):
    return None


RUN = {name[2:]: fn for name, fn in globals().items() if name.startswith("r_")}


# --- a fine-tune's executor, with the work Executor repeats done once ----------------------------------------------

ROOM_SHARE = 0.5  # of the device memory spare when a FastExecutor starts, what its reuse may hold
CPU_ROOM = 1 << 30  # on the CPU, which has no free-memory query that costs nothing


def spare_bytes(device) -> int:
    """Device memory a FastExecutor could hold: CUDA's free memory plus what PyTorch has cached but isn't using; a
    fixed GiB on the CPU; none elsewhere."""
    dev = torch.device(device)
    if dev.type == "cuda":
        free, _ = torch.cuda.mem_get_info(dev)
        return int(free + torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev))
    return CPU_ROOM if dev.type == "cpu" else 0


@contextlib.contextmanager
def no_fill():
    """PyTorch's deterministic mode (compute.TorchBackend) fills every new block of memory with NaN, about 9,000 extra
    kernels a fine-tune step. A fine-tune's ops write each tensor whole before anything reads it, so the fill changes no
    number. It's off only while a fine-tune runs, and put back after: DeepSeek's code makes weights with torch.empty
    and then loads them, and relies on the fill."""
    det = getattr(torch.utils, "deterministic", None)
    if det is None or not hasattr(det, "fill_uninitialized_memory"):
        yield
        return
    old = det.fill_uninitialized_memory
    det.fill_uninitialized_memory = False
    try:
        yield
    finally:
        det.fill_uninitialized_memory = old


def _layer_part(label: str):
    """("forward" or "backward", layer) for a micro-batch's layer node (program._Micro), else None."""
    parts = label.split()
    if len(parts) == 3 and parts[0] == "layer" and parts[2] in ("forward", "backward") and parts[1].isdigit():
        return parts[2], int(parts[1])
    return None


def _nbytes(t) -> int:
    return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0


class FastExecutor(Executor):
    """A fine-tune's Executor that does once what Executor does again: what finetune.train_segment and a dispute's
    re-runs (Party) use. It gives the same bits as Executor, op for op (tests/test_tracer.py compares every tensor
    after several steps), so workers with and without it agree, and the states it reaches in a dispute chain to the
    checkpoints either way. Measured on Qwen3-0.6B (the everyday chats, batch 8, RTX 4070 SUPER, 2026-11-10): 1.90 s a
    step against Executor's 2.71 s, 1.42x, the same bits, at 4.9 GiB peak against 1.9.

    What it keeps, while spare device memory allows (`room` bytes, ROOM_SHARE of what's spare when it starts):
    - each projection's frozen weights cast to float32, once, not on every use (the cast is exact);
    - a layer's forward outputs, for the recompute that opens the layer's backward. The same op on the very same input
      tensors gives the same output (nothing is changed in place), so each is looked up instead of run again. Only
      when the forward ran in this executor and every input is the tensor the forward read: a re-run that starts
      inside a layer, as a dispute's can, recomputes;
    - each head block's logits, from its head_stats to its head_grad in the same sequence.
    With `tamper` (tests' cheating workers) it keeps none of them, so a cheat lands where Executor's would."""

    def __init__(self, rt, spec, program, tamper=None, room=None):
        super().__init__(rt, spec, program, tamper)
        if tamper is not None:
            room = 0
        self.room = int(spare_bytes(self.device) * ROOM_SHARE) if room is None else room
        self.used = 0
        self._w32: dict[str, torch.Tensor] = {}
        self._fwd: dict[tuple, tuple[list, int]] = {}  # (where, layer) -> ([(op, input tensors, output)], bytes)
        self._recording: list | None = None
        self._replay: list | None = None  # [record, index of the next op it has]
        self._logits: dict[tuple, tuple] = {}  # (sequence, block start) -> (logits, the hidden states, bytes)
        self._depth = 0

    def _take(self, n: int) -> bool:
        if self.used + n > self.room:
            return False
        self.used += n
        return True

    def w32(self, name: str) -> torch.Tensor:
        t = self._w32.get(name)
        if t is None:
            t = self.w(name).to(_F32)
            if self._take(_nbytes(t)):
                self._w32[name] = t
        return t

    # --- running ---

    def run_atoms(self, node: Node, lo: int, hi: int, st: dict, where: tuple = ()) -> None:
        if self._depth:
            return self._atoms(node, lo, hi, st, where)
        self._depth = 1
        try:
            with no_fill():
                self._atoms(node, lo, hi, st, where)
        finally:
            self._depth = 0
            for _, n in self._fwd.values():  # a forward whose backward this run didn't reach
                self.used -= n
            self._fwd.clear()
            self._drop_logits()

    def _atoms(self, node: Node, lo: int, hi: int, st: dict, where: tuple) -> None:
        for i in range(lo, hi):
            a = node.atoms[i]
            if isinstance(a, Op):
                self.where = where
                self.run(a, st)
                continue
            sub = where + (a.label,)
            part = _layer_part(a.label)
            if part and part[0] == "forward" and self.room and self._recording is None and self._replay is None:
                rec = self._recording = []
                try:
                    self._atoms(a, 0, len(a.atoms), st, sub)
                finally:
                    self._recording = None
                if rec:
                    rec.pop()  # the forward's closing add, which the backward's recompute leaves out
                n = sum(_nbytes(out) for _, _, out in rec)
                if rec and self._take(n):
                    self._fwd[(where, part[1])] = (rec, n)
            elif part and part[0] == "backward" and (where, part[1]) in self._fwd and self._replay is None:
                rec, n = self._fwd.pop((where, part[1]))
                self._replay = [rec, 0]
                try:
                    self._atoms(a, 0, len(a.atoms), st, sub)
                finally:
                    self._replay = None
                    self.used -= n
            else:
                self._atoms(a, 0, len(a.atoms), st, sub)
            if a.label == "head and loss":
                self._head = (None, None)
                self._drop_logits()

    def run(self, op: Op, st: dict) -> None:
        if self._replay is not None and self._replayed(op, st):
            return
        if self.tamper is None and op.kind == "head_stats":
            return self._head_stats(op, st)
        if self.tamper is None and op.kind == "head_grad" and self._head_grad(op, st):
            return
        ins = tuple(st[n] for n in op.ins) if self._recording is not None else None
        super().run(op, st)
        if ins is not None:
            self._recording.append((op, ins, st[op.out] if op.out else None))

    def _replayed(self, op: Op, st: dict) -> bool:
        """Puts the forward's output for `op` in place of running it, if `op` is the recompute's next op on the very
        inputs the forward read; otherwise the recompute is over (or took another way) and the rest runs."""
        rec, j = self._replay
        if j < len(rec):
            was, ins, out = rec[j]
            if (was.kind, was.out, was.ins) == (op.kind, op.out, op.ins) and all(
                    st.get(n) is t for n, t in zip(op.ins, ins)):
                self._replay[1] = j + 1
                if op.out:
                    st[op.out] = out
                for d in op.drops:
                    del st[d]
                return True
        self._replay = None
        return False

    def _head_stats(self, op: Op, st: dict) -> None:
        lo, _, _ = _head(self, op, st)
        n = _nbytes(lo)
        if self.room and self._take(n):
            self._logits[(op.attrs["seq"], op.attrs["b0"])] = (lo, st[op.ins[0]], n)
        st[op.out] = _stats(lo, op, st).contiguous()
        for d in op.drops:
            del st[d]

    def _head_grad(self, op: Op, st: dict) -> bool:
        key = (op.attrs["seq"], op.attrs["b0"])
        kept = self._logits.get(key)
        if kept is None or kept[1] is not st[op.ins[0]]:
            return False
        lo, _, n = self._logits.pop(key)
        self.used -= n
        _, _, tgt = self.seq(op.attrs["seq"])
        w = self.head32(op.attrs["weight"])[op.attrs["b0"]:op.attrs["b1"]]
        st[op.out] = _grad(lo, w, tgt, op, st).contiguous()
        for d in op.drops:
            del st[d]
        return True

    def _drop_logits(self) -> None:
        for _, _, n in self._logits.values():
            self.used -= n
        self._logits.clear()


# --- disputes: a worker's side (dispute.py has the rules) -------------------------------------------------------------

HARD = 1e-2  # a child whose result is this far (relative) from a recompute is taken at once; else the furthest
CANDIDATES = 32  # tiles of a picked op the challenger checks with the referee before naming one (or withdrawing)


def node_at(prog, root, path) -> tuple[Node, tuple]:
    """The node a game is at, and the labels training runs its ops under (tracer.Executor's `where`)."""
    n = prog.segment(int(root[1])) if root[0] == "segment" else prog.unit()
    labels = []
    for i in path:
        n = n.atoms[i]
        labels.append(n.label)
    skip = 0 if root[0] == "segment" else 1  # a unit's first level is segments, which training doesn't label
    return n, tuple(labels[skip:])


class TorchWeights:
    """The base model's weights as a worker has them loaded, for running the referee itself (dispute.Outside)."""

    _NAMES = {torch.bfloat16: ("BF16", torch.int16), torch.float16: ("F16", torch.int16), torch.float32: ("F32", torch.int32)}

    def __init__(self, model):
        self.model = model

    def shape(self, name):
        if name not in self.model.w:
            from .dispute import WeightsUnavailable
            raise WeightsUnavailable(f"the base model has no {name}")
        t = self.model.w[name]
        return self._NAMES[t.dtype][0], tuple(t.shape)

    def read(self, name, r0, r1) -> bytes:
        t = self.model.w[name]
        t = t[r0:r1] if t.ndim > 1 else t
        return t.contiguous().view(self._NAMES[t.dtype][1]).cpu().numpy().tobytes()


class _AsNumpy(dict):
    """A state's tensors as float32 NumPy arrays, converted when first read."""

    def __init__(self, st: dict):
        super().__init__()
        self.st = st

    def __missing__(self, name):
        a = self.st[name].detach().cpu().numpy()
        self[name] = a
        return a

    def shapes(self) -> dict:
        return {n: tuple(t.shape) for n, t in self.st.items()}


class Party:
    """One worker's side of its disputes: states it has made or fetched (by root), and how to run them.
    `remote` has put(bytes) -> blob id and get(blob id) -> bytes (the node's blob store), and may have put_many and
    get_many, which move a state's tensors several at a time (client._Remote): one at a time, a state of Qwen3-0.6B's
    336 tensors took two round trips to a distant node each, and a defender missed its move (round 7280)."""

    def __init__(self, rt, spec, data, remote, tamper=None):
        from . import merkle

        self.rt, self.spec, self.data, self.remote, self.tamper, self.merkle = rt, spec, data, remote, tamper, merkle
        self.states: dict[str, tuple[dict, bytes]] = {}
        self.tensors: dict[tuple, torch.Tensor] = {}  # by (blob id, rows, cols): a reshape has the same bytes
        self.hashes: dict[int, tuple] = {}  # id(tensor) -> (tensor, root, blob): every op makes a new tensor
        self.on_node: set[str] = set()  # blobs the node holds: fetched from it, or sent to it
        self._sending: dict[str, threading.Event] = {}  # blobs on their way to the node, set once they're there
        self._sending_lock = threading.Lock()

    def executor(self, prog, honest: bool) -> Executor:
        return FastExecutor(self.rt, self.spec, prog, None if honest else self.tamper)

    def gpu(self):
        """The device's lock (finetune._Runtime.busy) around a re-run, so a dispute's and a check's never hold two
        steps' activations at once; nothing to wait for on the CPU, or for a runtime without one."""
        import contextlib
        return getattr(self.rt, "busy", None) or contextlib.nullcontext()

    def source(self):
        """The base model's weights as this worker holds them, for running the referee itself (dispute.Outside)."""
        return TorchWeights(self.rt.model)

    def adopt(self, theta) -> str:
        """One of this worker's own checkpoints as a state, loaded without fetching anything: a game's ends are
        checkpoints its defender committed to, and so are a unit's segment ends. Returns its root."""
        from .dispute import state_manifest
        from .program import split_theta

        arrays = split_theta(self.spec, theta)
        m, _ = state_manifest(arrays)
        root = self.merkle.state_root(m)
        if root not in self.states:
            st = {}
            for name, (troot, blob, rows, cols) in self.merkle.parse_manifest(m).items():
                k = (blob, rows, cols)
                if k not in self.tensors:  # (a copy: a checkpoint may be a read-only memory map)
                    self.tensors[k] = torch.tensor(arrays[name], device=self.rt.device)
                    self.hashes[id(self.tensors[k])] = (self.tensors[k], troot, blob)
                st[name] = self.tensors[k]
            self.states[root] = (st, m)
        return root

    def load(self, root: str) -> tuple[dict, bytes]:
        if root not in self.states:
            m = self.remote.get(root)
            if self.merkle.state_root(m) != root:
                raise ValueError("the node sent another state than the one asked for")
            entries = self.merkle.parse_manifest(m)
            want = list(dict.fromkeys(b for _, b, rows, cols in entries.values() if (b, rows, cols) not in self.tensors))
            get_many = getattr(self.remote, "get_many", None)
            got = dict(zip(want, get_many(want) if get_many else [self.remote.get(b) for b in want]))
            st = {}
            for name, (troot, blob, rows, cols) in entries.items():
                k = (blob, rows, cols)
                if k not in self.tensors:
                    a = torch.frombuffer(bytearray(got[blob]), dtype=_F32).reshape(rows, cols)
                    self.tensors[k] = a.to(self.rt.device)
                    self.hashes[id(self.tensors[k])] = (self.tensors[k], troot, blob)
                st[name] = self.tensors[k]
            self.on_node.update([root, *(b for _, b, _, _ in entries.values())])  # (all of them: it served this state)
            self.states[root] = (st, m)
        st, m = self.states[root]
        return dict(st), m

    def publish(self, st: dict, ready=()) -> str:
        return self.publish_all([st], ready)[0]

    def publish_all(self, states: list[dict], ready=()) -> list[str]:
        """Put states on the node, every tensor of theirs it doesn't hold yet (several at a time, with the remote's
        put_many) and their manifests, and return their roots. `ready`: states the node holds whole already
        (dispute.Game.ready), which send nothing."""
        roots, send = [], {}
        for st in states:
            entries, new = {}, {}
            for name, t in st.items():
                hit = self.hashes.get(id(t))
                if hit is None or hit[0] is not t:
                    a = t.detach().cpu().numpy()
                    hit = (t, self.merkle.root_of(a), self.merkle.blob_of(a))
                    self.hashes[id(t)] = hit
                entries[name] = (hit[1], hit[2], t.shape[0], t.shape[1])
                self.tensors.setdefault((hit[2], t.shape[0], t.shape[1]), t)
                if hit[2] not in self.on_node:
                    new[hit[2]] = t
            m = self.merkle.manifest_bytes(entries)
            root = self.merkle.state_root(m)
            if root in ready:
                self.on_node.update([root, *new])
            else:
                send |= new
                send[root] = m
            roots.append(root)
            self.states[root] = (dict(st), m)
        # A worker plays its games at once (client.WorkerClient._move), and two may name the same tensor: one sends it,
        # the other waits for that, as the node takes a blob from one upload of it at a time.
        with self._sending_lock:
            theirs = [self._sending[k] for k in send if k in self._sending]
            send = {k: v for k, v in send.items() if k not in self._sending}
            for k in send:
                self._sending[k] = threading.Event()
        try:
            raws = [v if isinstance(v, bytes) else v.detach().cpu().numpy().astype("<f4").tobytes() for v in send.values()]
            put_many = getattr(self.remote, "put_many", None)
            if put_many is not None:
                put_many(raws)
            else:
                for raw in raws:
                    self.remote.put(raw)
            self.on_node.update(send)
        finally:
            with self._sending_lock:
                for k in send:
                    self._sending.pop(k).set()
        for sent in theirs:
            sent.wait()
        return roots

    def forget(self) -> None:
        self.states.clear()
        self.tensors.clear()
        self.hashes.clear()
        self.on_node.clear()

    # --- the defender ---

    def split(self, prog, view: dict, checkpoints=None) -> list[str]:
        """The state roots at the boundaries of the current range's children, as its training went. At a unit's top
        level the boundaries are segment ends, so with `checkpoints` (this worker's own for the unit, by index) they're
        those, not re-run, and the node already holds the ones the unit's reveal opened (view["ready"])."""
        from .program import groups

        ready, bounds = set(view.get("ready") or ()), groups(view["lo"], view["hi"])[:-1]
        if checkpoints is not None and view["root"] == ["unit"] and not view["path"]:
            states = [self.states[self.adopt(checkpoints[g1])][0] for _, g1 in bounds]
        else:
            node, where = node_at(prog, view["root"], view["path"])
            st, _ = self.load(view["a"])
            ex, states = self.executor(prog, honest=False), []
            for g0, g1 in bounds:
                with self.gpu():
                    ex.run_atoms(node, g0, g1, st, where)
                states.append(dict(st))
        return [view["a"]] + self.publish_all(states, ready) + [view["b"]]

    def evidence(self, prog, view: dict) -> bytes:
        from .dispute import Outside, make_evidence

        node, _ = node_at(prog, view["root"], view["path"])
        op = node.atoms[view["lo"]]
        (before, mb), (after, ma) = self.load(view["a"]), self.load(view["b"])
        out = Outside(self.spec, self.data, self.source(), keep=False)
        b, a = _AsNumpy(before), _AsNumpy(after)
        evidence, _, _ = make_evidence(op, view["tile"], b, a, (mb, ma), out, shapes=b.shapes())
        return evidence

    # --- the challenger ---

    def pick(self, prog, view: dict) -> dict:
        """The first child that doesn't come out of a recompute from the defender's state before it (else the one
        furthest off), and for an op, its worst tile. A replica that finds nothing wrong with the other's unit
        concedes, which costs it nothing before its first pick."""
        from .program import groups

        node, where = node_at(prog, view["root"], view["path"])
        ex, roots = self.executor(prog, honest=True), view["roots"]
        best = (-1.0, 0)
        for c, (g0, g1) in enumerate(groups(view["lo"], view["hi"])):
            mine, _ = self.load(roots[c])
            theirs, _ = self.load(roots[c + 1])
            with self.gpu():
                ex.run_atoms(node, g0, g1, mine, where)
            dev = deviation(mine, theirs)
            best = max(best, (dev, -c))
            if dev >= HARD:
                break
        if view.get("why") == "replicas" and not view.get("played") and best[0] < HARD:
            return {"concede": True}
        child = -best[1]
        g0, g1 = groups(view["lo"], view["hi"])[child]
        atom = node.atoms[g0] if g1 - g0 == 1 else None
        move = {"pick": child}
        if isinstance(atom, Op) and atom.out:
            tile = self.worst_tile(prog, atom, roots[child], roots[child + 1])
            if tile is None:  # nothing the referee would refuse: play on and lose, or withdraw
                return {"concede": True}
            move["tile"] = tile
        return move

    def worst_tile(self, prog, op: Op, a: str, b: str) -> int | None:
        """The tile of the defender's output of `op` the referee would refuse, among the CANDIDATES furthest from a
        recompute, or None: then the challenger can only withdraw (dispute.play's concede), not win."""
        from . import merkle, referee
        from .dispute import Inputs, Outside

        before, _ = self.load(a)
        after, _ = self.load(b)
        mine = dict(before)
        self.executor(prog, honest=True).run(op, mine)
        if op.out not in after:
            return 0
        n = merkle.n_leaves(*op.shape)
        x, y = mine[op.out].reshape(-1), after[op.out].reshape(-1)
        pad = n * merkle.LEAF - x.numel()
        diff = torch.nn.functional.pad(_diff(x, y), (0, pad)).view(n, -1).amax(1)
        size = torch.nn.functional.pad(_size(y), (0, pad)).view(n, -1).amax(1)
        off = torch.nan_to_num(diff / (size + 1e-30), nan=float("inf"))
        order = torch.argsort(off, descending=True)[:CANDIDATES].tolist()
        out = Outside(self.spec, self.data, self.source(), keep=False)
        full = _AsNumpy(before)
        theirs = after[op.out].detach().cpu().numpy().reshape(-1)
        for tile in order:
            said = theirs[tile * merkle.LEAF:(tile + 1) * merkle.LEAF]
            try:
                ok, _ = referee.judge(op, Inputs(full.shapes(), out, full=full), tile, said)
            except referee.Unresolved:  # the referee can't say: not a tile to win on
                continue
            if not ok:
                return tile
        return None


def _diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a − b|, 0 where they're equal (the attention mask's −∞ included), ∞ where either is NaN."""
    d = torch.where(a == b, torch.zeros_like(a), (a - b).abs())
    return torch.nan_to_num(d, nan=float("inf"))


def _size(b: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(b), b.abs(), torch.zeros_like(b))


def deviation(mine: dict, theirs: dict) -> float:
    """How far a state is from a recompute of it: the largest relative difference of any tensor, inf if they don't
    hold the same tensors."""
    if set(mine) != set(theirs):
        return float("inf")
    worst = 0.0
    for name, a in mine.items():
        b = theirs[name]
        if a is b:
            continue
        if a.shape != b.shape:
            return float("inf")
        d = float((_diff(a, b).max() / (_size(b).max() + 1e-30)).item()) if a.numel() else 0.0
        if d != d:
            return float("inf")
        worst = max(worst, d)
    return worst
