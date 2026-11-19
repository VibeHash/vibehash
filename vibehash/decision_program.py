"""What one unit of a decision model's fine-tune computes, op by op (decision.py): program.py's tree and rules, for
Laya's model. decision_tracer.py runs it in PyTorch.

A unit is segments, a segment Adam steps, a step one item of its batch at a time (adding that item's gradients into
G) and then the Adam update, as program.py's are. An item is:
    the embedding (the token embeddings and their LayerNorm);
    each encoder layer forward, keeping its input;
    the encoder's final LayerNorm and the question type's embedding added to every position;
    each head layer forward, keeping its input;
    "head and loss": the rows at the option markers through the scorer to one number an option, the gradient of the
    cross-entropy against the item's target probabilities (softmax minus target, over its options), and back through
    the scorer to the head's last output;
    each head layer backward, then the final LayerNorm's, then each encoder layer backward, each recomputing its forward
    from its input first (activation checkpointing, as program.py's layers do).
Every op reads named 2-D tensors of the state and writes one, and nothing is changed in place, so a state between any
two ops is well defined and any op can be checked on its own.

The arithmetic is float32 with frozen weights cast exactly, attention written out (bidirectional, with the sliding
layers' window as a mask of -inf), as program.py's. The ops' kinds that program.py doesn't have (LayerNorm, GELU and
its gated form, ReLU, the option rows, the loss) are decision_tracer.py's; the referee has no entries for them yet,
which is why a decision model's fine-tune runs only on a chain that takes them (ledger.DECISION_GATE).
"""
from __future__ import annotations

import functools
import math
import types

import numpy as np

from .decision import ENCODER_TARGETS, HEAD_TARGETS, DecisionArch, arch_of
from .finetune import OPTIMIZERS
from .program import Node, Op, Program, f32, tree_size

EMBED = "encoder.embeddings.tok_embeddings.weight"


class DecisionProgram(Program):
    """The tree of one unit of a decision model's fine-tune whose seed is `seed` (Program's, with Laya's items)."""

    def __init__(self, spec, data, lr: float, batch_size: int, seed: int, segments: int = 1, per_segment: int = 1):
        self.spec, self.data, self.seed = spec, data, seed
        self.lr, self.batch_size, self.segments, self.per_segment = lr, batch_size, segments, per_segment
        self.arch: DecisionArch = data.arch()
        self.mods = spec.modules()
        self.mod = {name: k for k, (name, _, _) in enumerate(self.mods)}
        b1, b2, eps = (f32(x) for x in OPTIMIZERS[self.spec.optimizer])
        self.adam = {"b1": b1, "b2": b2, "eps": eps, "c1m": f32(1 - b1), "c2m": f32(1 - b2), "lr": f32(lr)}

    def step(self, t: int) -> Node:
        def build():
            idx = self.batch(t)
            return [self.micro(t, j, int(i), len(idx)) for j, i in enumerate(idx)] + [self.adam_step(t)]
        return Node(f"step {t}", build)

    def micro(self, t: int, j: int, seq: int, count: int) -> Node:
        def build():
            return _Item(self, seq, count, first=j == 0).atoms()
        return Node(f"step {t} item {j} (#{seq})", build)

    def forward(self, i: int, val: bool = False) -> Node:
        """One item's forward alone, to its options' numbers (s.z): what scoring runs. Its layers aren't labelled as a
        unit's are, so an executor keeps nothing of them for a backward that never comes."""
        return Node(f"item {i}", _Item(self, i, 1, first=True, val=val, forward_only=True).atoms())


def _shape(a: DecisionArch, module: str) -> tuple[int, int]:
    """[out, in] of any projection, adapted or not."""
    d, f = a.hidden, a.ffn
    for end, shape in ((".attn.Wqkv", (3 * d, d)), (".attn.Wo", (d, d)), (".mlp.Wi", (2 * f, d)), (".mlp.Wo", (d, f)),
                       (".in_proj", (3 * d, d)), (".out_proj", (d, d)), (".linear1", (4 * d, d)),
                       (".linear2", (d, 4 * d)), ("scorer.1", (d, d))):
        if module.endswith(end):
            return shape
    raise KeyError(module)


class _Item:
    """Builds one item's forward and backward, keeping track of every tensor's shape as it goes."""

    def __init__(self, p: DecisionProgram, i: int, count: int, first: bool, val: bool = False,
                 forward_only: bool = False):
        self.p, self.a, self.spec, self.i, self.val = p, p.arch, p.spec, i, val
        ids, markers, _, qtype = p.data.item(i, val)
        self.T, self.K, self.qtype = len(ids), len(markers), qtype
        self.ic = f32(1.0 / count)
        self.shapes = p.theta_shapes(grads=not first)
        self.ops: list[Op] = []
        self.s32 = f32(self.spec.scaling)
        self.fwd_only = forward_only
        targets = set(self.spec.targets)
        # What the backward must reach: an encoder layer's adapters need the gradient through the whole head; a head
        # layer's, through the scorer. Nothing below the lowest adapted part is computed.
        self.enc_grads = not forward_only and bool(targets & set(ENCODER_TARGETS))
        self.head_grads = not forward_only and (self.enc_grads or bool(targets & set(HEAD_TARGETS)))

    def emit(self, kind, out, ins, attrs, shape, drops=()):
        self.ops.append(Op(kind, out, tuple(ins), attrs, tuple(drops), tuple(shape), tuple(self.shapes[i] for i in ins)))
        if out:
            self.shapes[out] = tuple(shape)
        for d in drops:
            del self.shapes[d]

    def take(self, label: str) -> Node:
        ops, self.ops = self.ops, []
        return Node(label, ops)

    def atoms(self) -> list:
        a, T, d = self.a, self.T, self.a.hidden
        L, HL = a.layers, a.head_layers
        fwd = "" if self.fwd_only else " forward"
        self.emit("d_embed", "h.e", (), {"item": self.i, "val": self.val, "weight": EMBED}, (T, d))
        self.emit("ln", "h.0", ("h.e",), {"weight": "encoder.embeddings.norm.weight", "bias": None, "eps": a.norm_eps},
                  (T, d), ("h.e",))
        out = [self.take("embedding")]
        for i in range(L):
            self.layer(i, recompute=False)
            out.append(self.take(f"layer {i}{fwd}"))
        self.emit("ln", "e.x", (f"h.{L}",), {"weight": "encoder.final_norm.weight", "bias": None, "eps": a.norm_eps},
                  (T, d), () if self.enc_grads else (f"h.{L}",))
        self.emit("d_type", "e.0", ("e.x",), {"weight": "type_emb.weight", "row": self.qtype}, (T, d), ("e.x",))
        out.append(self.take("encoder out"))
        for j in range(HL):
            self.head_layer(j, recompute=False)
            out.append(self.take(f"head {j}{fwd}"))
        self.scorer()
        out.append(self.take("scorer" if self.fwd_only else "head and loss"))
        if self.fwd_only:
            return out
        if self.head_grads:
            for j in reversed(range(HL)):
                self.head_backward(j)
                out.append(self.take(f"head {j} backward"))
        if self.enc_grads:
            self.emit("ln_bwd", "g.h", (f"h.{L}", "g.e"), {"weight": "encoder.final_norm.weight", "eps": a.norm_eps},
                      (T, d), ("g.e", f"h.{L}"))
            out.append(self.take("encoder out backward"))
            for i in reversed(range(L)):
                self.layer_backward(i)
                out.append(self.take(f"layer {i} backward"))
        left = tuple(n for n in self.shapes if n[:2] in ("g.", "h.", "e."))
        if left:
            self.emit("free", "", (), {}, (0, 0), left)
            out.append(self.take("end"))
        return out

    # --- pieces ---

    def proj(self, module: str, x: str, out: str, rows: int):
        """x·Wᵀ (+ the bias) (+ the adapter's (x·Aᵀ)·Bᵀ·alpha/rank) of `module` into `out`. The adapter's x·Aᵀ is kept,
        as `out` + "a", for the backward."""
        o, _ = _shape(self.a, module)
        attrs = {"weight": self.spec.weight_of(module), "bias": self.spec.bias_of(module)}
        k = self.p.mod.get(module)
        if k is None:
            self.emit("linear", out, (x,), attrs, (rows, o))
            return
        self.emit("linear", out + "0", (x,), attrs, (rows, o))
        self.emit("lora_down", out + "a", (x,), {"module": k}, (rows, self.spec.rank))
        self.emit("lora_up", out, (out + "0", out + "a", f"B.{module}"), {"scaling": self.s32}, (rows, o), (out + "0",))

    def proj_bwd(self, module: str, dy: str, out: str, fwd: str, rows: int, drops=(), dx: bool = True):
        """dx of `module`'s projection from dy into `out`, and its adapter's gradient into G. `fwd` is the forward's
        output, whose x·Aᵀ the gradient reads. `drops` go with the last op; with `dx` False only the gradient is made."""
        o, inn = _shape(self.a, module)
        weight = self.spec.weight_of(module)
        k = self.p.mod.get(module)
        if k is None:
            if dx:
                self.emit("linear_bwd", out, (dy,), {"weight": weight}, (rows, inn), tuple(drops))
            elif drops:
                self.emit("free", "", (), {}, (0, 0), tuple(drops))
            return
        r, g = self.spec.rank, f"G.{module}"
        if dx:
            self.emit("linear_bwd", out + "0", (dy,), {"weight": weight}, (rows, inn))
        self.emit("scale", out + "s", (dy,), {"c": self.s32}, (rows, o))
        self.emit("lora_grad", g, (out + "s", fwd + "a") + ((g,) if g in self.shapes else ()), {}, (o, r))
        if not dx:
            self.emit("free", "", (), {}, (0, 0), (out + "s",) + tuple(drops))
            return
        self.emit("lora_bwd_mid", out + "m", (out + "s", f"B.{module}"), {}, (rows, r), (out + "s",))
        self.emit("lora_bwd_x", out, (out + "m", out + "0"), {"module": k}, (rows, inn),
                  (out + "m", out + "0") + tuple(drops))

    def attention(self, qkv: str, H: int, hd: int, theta: float, window: int):
        """Bidirectional attention over the item from a fused query-key-value output: q, k and v cut from it, the
        rotary positions (unless `theta` is 0), the scores with the window's mask (none if `window` is 0), softmax, mix."""
        T, d = self.T, H * hd
        self.emit("cols", "a.q", (qkv,), {"c0": 0, "c1": d}, (T, d))
        self.emit("cols", "a.k", (qkv,), {"c0": d, "c1": 2 * d}, (T, d))
        self.emit("cols", "a.v", (qkv,), {"c0": 2 * d, "c1": 3 * d}, (T, d), (qkv,))
        q, k = "a.q", "a.k"
        if theta:
            rope = {"hd": hd, "theta": theta}
            self.emit("rope2", "a.qr", (q,), rope, (T, d), (q,))
            self.emit("rope2", "a.kr", (k,), rope, (T, d), (k,))
            q, k = "a.qr", "a.kr"
        self.qk = (q, k)
        att = {"H": H, "KV": H, "hd": hd}
        self.emit("attn_bi", "a.s", (q, k), {**att, "c": f32(1.0 / math.sqrt(hd)), "window": window}, (H * T, T))
        self.emit("softmax", "a.sm", ("a.s",), {}, (H * T, T), ("a.s",))
        self.emit("attn_mix", "a.o", ("a.sm", "a.v"), att, (T, d))

    def attention_backward(self, H: int, hd: int, theta: float):
        """From the gradient at the mix's output (b.do) to the fused query-key-value gradient (b.dqkv)."""
        T, d = self.T, H * hd
        q, k = self.qk
        att = {"H": H, "KV": H, "hd": hd}
        self.emit("attn_dp", "b.dp", ("b.do", "a.v"), att, (H * T, T))
        self.emit("attn_dv", "b.dv", ("a.sm", "b.do"), att, (T, d), ("b.do",))
        self.emit("softmax_bwd", "b.ds", ("a.sm", "b.dp"), {}, (H * T, T), ("b.dp",))
        c = {**att, "c": f32(1.0 / math.sqrt(hd))}
        self.emit("attn_dq", "b.dq", ("b.ds", k), c, (T, d))
        self.emit("attn_dk", "b.dk", ("b.ds", q), c, (T, d), ("b.ds",))
        dq, dk = "b.dq", "b.dk"
        if theta:
            rope = {"hd": hd, "theta": theta}
            self.emit("rope2_bwd", "b.dq1", (dq,), rope, (T, d), (dq,))
            self.emit("rope2_bwd", "b.dk1", (dk,), rope, (T, d), (dk,))
            dq, dk = "b.dq1", "b.dk1"
        self.emit("cat", "b.dqkv", (dq, dk, "b.dv"), {}, (T, 3 * d), (dq, dk, "b.dv"))

    def _close(self, top: str, nxt: str, keep_input: bool):
        """A layer's last op: its output, dropping its activations (and its input, when no backward will read it)."""
        live = tuple(n for n in self.shapes if n.startswith("a."))
        self.emit("add", nxt, ("a.r", "a.d"), {}, (self.T, self.a.hidden), live + (() if keep_input else (top,)))

    # --- the encoder ---

    def layer(self, i: int, recompute: bool):
        a, T, d = self.a, self.T, self.a.hidden
        p = f"encoder.layers.{i}."
        x = f"h.{i}"
        if i:  # layer 0 has no attention norm (ModernBERT's attn_norm is Identity there)
            self.emit("ln", "a.x", (x,), {"weight": p + "attn_norm.weight", "bias": None, "eps": a.norm_eps}, (T, d))
            x = "a.x"
        self.proj(p + "attn.Wqkv", x, "a.qkv", T)
        self.attention("a.qkv", a.heads, a.head_dim, a.theta(i), 0 if a.is_global(i) else a.window)
        self.proj(p + "attn.Wo", "a.o", "a.y", T)
        self.emit("add", "a.r", (f"h.{i}", "a.y"), {}, (T, d), ("a.y",))
        self.emit("ln", "a.x2", ("a.r",), {"weight": p + "mlp_norm.weight", "bias": None, "eps": a.norm_eps}, (T, d))
        self.proj(p + "mlp.Wi", "a.x2", "a.wi", T)
        self.emit("geglu", "a.m", ("a.wi",), {"f": a.ffn}, (T, a.ffn))
        self.proj(p + "mlp.Wo", "a.m", "a.d", T)
        if not recompute:
            self._close(f"h.{i}", f"h.{i + 1}", self.enc_grads)

    def layer_backward(self, i: int):
        a, T, d = self.a, self.T, self.a.hidden
        p = f"encoder.layers.{i}."
        self.layer(i, recompute=True)
        self.proj_bwd(p + "mlp.Wo", "g.h", "b.dm", "a.d", T)
        self.emit("geglu_bwd", "b.dwi", ("a.wi", "b.dm"), {"f": a.ffn}, (T, 2 * a.ffn), ("b.dm",))
        self.proj_bwd(p + "mlp.Wi", "b.dwi", "b.dx2", "a.wi", T, ("b.dwi",))
        self.emit("ln_bwd", "b.dr", ("a.r", "b.dx2", "g.h"), {"weight": p + "mlp_norm.weight", "eps": a.norm_eps},
                  (T, d), ("b.dx2",))
        self.proj_bwd(p + "attn.Wo", "b.dr", "b.do", "a.y", T)
        self.attention_backward(a.heads, a.head_dim, a.theta(i))
        last = i == 0  # nothing below layer 0 is trained: its input's gradient isn't needed
        self.proj_bwd(p + "attn.Wqkv", "b.dqkv", "b.dx", "a.qkv", T, ("b.dqkv",), dx=not last)
        if not last:
            self.emit("ln_bwd", "g.h", (f"h.{i}", "b.dx", "b.dr"),
                      {"weight": p + "attn_norm.weight", "eps": a.norm_eps}, (T, d), ("b.dx", "b.dr"))
        live = tuple(n for n in self.shapes if n[:2] in ("a.", "b.")) + (f"h.{i}",) + (("g.h",) if last else ())
        self.emit("free", "", (), {}, (0, 0), live)

    # --- the decision head ---

    def head_layer(self, j: int, recompute: bool):
        """nn.TransformerEncoderLayer with norm_first: x + attention(norm1(x)), then + linear2(relu(linear1(norm2(·))))."""
        a, T, d = self.a, self.T, self.a.hidden
        p = f"head.layers.{j}."
        ln = {"eps": a.head_eps}
        self.emit("ln", "a.x", (f"e.{j}",), {"weight": p + "norm1.weight", "bias": p + "norm1.bias", **ln}, (T, d))
        self.proj(p + "self_attn.in_proj", "a.x", "a.qkv", T)
        self.attention("a.qkv", a.head_heads, d // a.head_heads, 0.0, 0)
        self.proj(p + "self_attn.out_proj", "a.o", "a.y", T)
        self.emit("add", "a.r", (f"e.{j}", "a.y"), {}, (T, d), ("a.y",))
        self.emit("ln", "a.x2", ("a.r",), {"weight": p + "norm2.weight", "bias": p + "norm2.bias", **ln}, (T, d))
        self.proj(p + "linear1", "a.x2", "a.u", T)
        self.emit("relu", "a.m", ("a.u",), {}, (T, 4 * d))
        self.proj(p + "linear2", "a.m", "a.d", T)
        if not recompute:
            self._close(f"e.{j}", f"e.{j + 1}", self.head_grads)

    def head_backward(self, j: int):
        a, T, d = self.a, self.T, self.a.hidden
        p = f"head.layers.{j}."
        self.head_layer(j, recompute=True)
        self.proj_bwd(p + "linear2", "g.e", "b.dm", "a.d", T)
        self.emit("relu_bwd", "b.du", ("a.u", "b.dm"), {}, (T, 4 * d), ("b.dm",))
        self.proj_bwd(p + "linear1", "b.du", "b.dx2", "a.u", T, ("b.du",))
        self.emit("ln_bwd", "b.dr", ("a.r", "b.dx2", "g.e"), {"weight": p + "norm2.weight", "eps": a.head_eps},
                  (T, d), ("b.dx2",))
        self.proj_bwd(p + "self_attn.out_proj", "b.dr", "b.do", "a.y", T)
        self.attention_backward(a.head_heads, d // a.head_heads, 0.0)
        last = j == 0 and not self.enc_grads  # the encoder's adapters need the gradient on through the head's input
        self.proj_bwd(p + "self_attn.in_proj", "b.dqkv", "b.dx", "a.qkv", T, ("b.dqkv",), dx=not last)
        if not last:
            self.emit("ln_bwd", "g.e", (f"e.{j}", "b.dx", "b.dr"), {"weight": p + "norm1.weight", "eps": a.head_eps},
                      (T, d), ("b.dx", "b.dr"))
        live = tuple(n for n in self.shapes if n[:2] in ("a.", "b.")) + (f"e.{j}",) + (("g.e",) if last else ())
        self.emit("free", "", (), {}, (0, 0), live)

    def scorer(self):
        """The rows at the option markers through LayerNorm, Linear, GELU and Linear to one number each (s.z); then,
        training, the loss's gradient and back to the head's output (g.e, zero but at the markers)."""
        a, K, d = self.a, self.K, self.a.hidden
        top = f"e.{a.head_layers}"
        self.emit("d_gather", "s.m", (top,), {"item": self.i, "val": self.val}, (K, d), (top,))
        self.emit("ln", "s.x", ("s.m",), {"weight": "scorer.0.weight", "bias": "scorer.0.bias", "eps": a.head_eps},
                  (K, d))
        self.proj("scorer.1", "s.x", "s.u", K)
        self.emit("gelu", "s.g", ("s.u",), {}, (K, d))
        self.emit("linear", "s.z", ("s.g",), {"weight": "scorer.3.weight", "bias": "scorer.3.bias"}, (K, 1))
        if self.fwd_only:
            return
        self.emit("d_loss", "s.dz", ("s.z",), {"item": self.i, "ic": self.ic}, (K, 1), ("s.z",))
        self.emit("linear_bwd", "s.dg", ("s.dz",), {"weight": "scorer.3.weight"}, (K, d), ("s.dz",))
        self.emit("gelu_bwd", "s.du", ("s.u", "s.dg"), {}, (K, d), ("s.dg", "s.g"))
        self.proj_bwd("scorer.1", "s.du", "s.dx", "s.u", K, ("s.du",), dx=self.head_grads)
        if self.head_grads:
            self.emit("ln_bwd", "s.dm", ("s.m", "s.dx"), {"weight": "scorer.0.weight", "eps": a.head_eps}, (K, d),
                      ("s.dx",))
            self.emit("d_scatter", "g.e", ("s.dm",), {"item": self.i}, (self.T, d), ("s.dm",))
        live = tuple(n for n in self.shapes if n.startswith("s."))
        if live:
            self.emit("free", "", (), {}, (0, 0), live)


@functools.lru_cache(maxsize=64)
def dispute_size(spec, batch_size: int, segments: int, per_segment: int) -> tuple[int, int]:
    """program.dispute_size for a decision model: the tree of a stand-in item of seq_len tokens with two options (how
    many ops an item has, and their shapes, depend on its length and its options alone; an option adds a row to the
    scorer's tensors, which are small beside the layers')."""
    T = spec.seq_len
    arch = arch_of(spec.model)

    def item(i, val=False):
        return np.zeros(T, np.int64), np.array([1, 2]), np.array([0.5, 0.5], np.float32), 0
    data = types.SimpleNamespace(arch=lambda: arch, item=item, n_train=1)
    p = DecisionProgram(spec, data, 1e-4, 1, 0)
    micros = [_Item(p, 0, 1, first).atoms() for first in (True, False)]
    return tree_size(p, micros, spec, batch_size, segments, per_segment)
