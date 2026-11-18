"""A decision model's fine-tune in PyTorch, op by op as decision_program.py lays it out (tracer.py's executor, with
Laya's ops added), and the model's frozen weights on a device.

The ops are written out as float32 arithmetic, not called as fused library functions whose algorithm differs by
device or version: LayerNorm as mean, centre, mean square, rsqrt and scale; GELU as 0.5·x·(1 + erf(x/√2)) (exact
GELU, which ModernBERT's "gelu" and nn.GELU() both are); the rotary tables computed in float64 on the CPU for each
base, then rounded. The same ops on the same inputs on the same device give the same bits, so a replay of a segment
is the segment.
"""
from __future__ import annotations

import math

import torch

from . import tracer
from .decision import DecisionArch
from .llm import STORED

_F32 = torch.float32
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


class Weights:
    """A decision model's frozen weights on one device, as stored (f16, bf16 or f32), and its rotary tables."""

    def __init__(self, arch: DecisionArch, weights: dict[str, torch.Tensor], device="cpu"):
        missing = [n for n in arch.tensor_names() if n not in weights]
        if missing:
            raise ValueError(f"the checkpoint has no {missing[0]}")
        self.arch, self.device = arch, torch.device(device)
        self.w = {n: t.to(self.device) for n, t in weights.items()}
        self._tables: dict[tuple[int, float], tuple[torch.Tensor, torch.Tensor]] = {}

    def tables(self, length: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
        """cos and sin, [length, head_dim], in the rotate-half layout, for one rotary base: float64 on the CPU, then
        float32 on the device (as llm.rope_table's)."""
        key = (length, theta)
        if key not in self._tables:
            d = self.arch.head_dim
            inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            angles = torch.outer(torch.arange(length, dtype=torch.float64), inv)
            angles = torch.cat((angles, angles), dim=-1)
            self._tables[key] = (angles.cos().to(_F32).to(self.device), angles.sin().to(_F32).to(self.device))
        return self._tables[key]


def shapes(arch: DecisionArch) -> dict[str, tuple[int, ...]]:
    """Every tensor's shape, as the configs make it."""
    d, f, v = arch.hidden, arch.ffn, arch.vocab
    out = {"encoder.embeddings.tok_embeddings.weight": (v, d), "encoder.embeddings.norm.weight": (d,),
           "encoder.final_norm.weight": (d,), "type_emb.weight": (3, d)}
    for i in range(arch.layers):
        p = f"encoder.layers.{i}."
        out |= {p + "attn_norm.weight": (d,), p + "attn.Wqkv.weight": (3 * d, d), p + "attn.Wo.weight": (d, d),
                p + "mlp_norm.weight": (d,), p + "mlp.Wi.weight": (2 * f, d), p + "mlp.Wo.weight": (d, f)}
    for j in range(arch.head_layers):
        p = f"head.layers.{j}."
        out |= {p + "self_attn.in_proj_weight": (3 * d, d), p + "self_attn.in_proj_bias": (3 * d,),
                p + "self_attn.out_proj.weight": (d, d), p + "self_attn.out_proj.bias": (d,),
                p + "linear1.weight": (4 * d, d), p + "linear1.bias": (4 * d,), p + "linear2.weight": (d, 4 * d),
                p + "linear2.bias": (d,)}
        out |= {p + f"norm{k}.{w}": (d,) for k in (1, 2) for w in ("weight", "bias")}
    out |= {"scorer.0.weight": (d,), "scorer.0.bias": (d,), "scorer.1.weight": (d, d), "scorer.1.bias": (d,),
            "scorer.3.weight": (1, d), "scorer.3.bias": (1,)}
    return out


def load(arch: DecisionArch, read, device="cpu") -> Weights:
    """Weights from `read(name) -> (safetensors dtype, shape, raw bytes)`, as hub.load_tensor gives them, checked
    against the configs' shapes (finetune.runtime's loader, for a decision model)."""
    expect, weights = shapes(arch), {}
    for name in arch.tensor_names():
        stored, shape, raw = read(name)
        if stored not in STORED:
            raise ValueError(f"{name} is stored as {stored}; only BF16, F16 and F32 weights are supported")
        if tuple(shape) != expect[name]:
            raise ValueError(f"{name} has shape {tuple(shape)}, but the configs make it {expect[name]}")
        weights[name] = torch.frombuffer(bytearray(raw), dtype=STORED[stored]).reshape(shape)
    return Weights(arch, weights, device)


class DecisionExecutor(tracer.FastExecutor):
    """tracer.FastExecutor for a decision model's program: its items, its rotary tables, and its ops (OPS, then
    tracer's)."""

    ops = None  # set below, once the ops are defined

    def __init__(self, rt, spec, program, tamper=None, room=None):
        super().__init__(rt, spec, program, tamper, room)
        self._items: dict[tuple[int, bool], tuple] = {}

    def item(self, i: int, val: bool = False):
        """One item on the device: (ids, marker positions, target probabilities, question type)."""
        key = (i, bool(val))
        if key not in self._items:
            ids, markers, target, qtype = self.prog.data.item(i, val)
            self._items = {key: (torch.from_numpy(ids).to(self.device), torch.from_numpy(markers).to(self.device),
                                 torch.from_numpy(target).to(self.device), qtype)}
        return self._items[key]

    def rope(self, length: int, theta: float):
        return self.model.tables(length, theta)


# --- the ops ---------------------------------------------------------------------------------------------------------

def _x(st, op, i=0):
    return st[op.ins[i]]


def _acc(st, op, i, y):
    return st[op.ins[i]] + y if len(op.ins) > i else y


def r_d_embed(ex, op, st):
    ids = ex.item(op.attrs["item"], op.attrs["val"])[0]
    return ex.w(op.attrs["weight"])[ids].to(_F32)


def _centre(x):
    return x - x.mean(-1, keepdim=True)


def _rstd(xc, eps):
    return torch.rsqrt((xc * xc).mean(-1, keepdim=True) + eps)


def r_ln(ex, op, st):
    """LayerNorm over the last axis: (x − mean) · rsqrt(mean square of that + eps) · weight (+ bias)."""
    xc = _centre(_x(st, op))
    y = (xc * _rstd(xc, op.attrs["eps"])) * ex.w(op.attrs["weight"])
    bias = op.attrs.get("bias")
    return y + ex.w(bias) if bias else y


def r_ln_bwd(ex, op, st):
    """dx of LayerNorm from dy (+ the third input, if any): r · (dy·w − mean(dy·w) − xc · r² · mean(dy·w · xc))."""
    x, dy = st[op.ins[0]], st[op.ins[1]]
    xc = _centre(x)
    r = _rstd(xc, op.attrs["eps"])
    dyw = dy * ex.w(op.attrs["weight"])
    dx = r * ((dyw - dyw.mean(-1, keepdim=True)) - xc * ((r * r) * (dyw * xc).mean(-1, keepdim=True)))
    return _acc(st, op, 2, dx)


def r_cols(ex, op, st):
    return _x(st, op)[:, op.attrs["c0"]:op.attrs["c1"]]


def r_cat(ex, op, st):
    return torch.cat([st[n] for n in op.ins], dim=1)


def r_rope2(ex, op, st):
    x = _x(st, op)
    cos, sin = ex.rope(x.shape[0], op.attrs["theta"])
    xs = x.view(x.shape[0], -1, op.attrs["hd"])
    return (xs * cos[:, None] + tracer._rot(xs) * sin[:, None]).view(x.shape)


def r_rope2_bwd(ex, op, st):
    dy = _x(st, op)
    cos, sin = ex.rope(dy.shape[0], op.attrs["theta"])
    hd = op.attrs["hd"]
    ys = dy.view(dy.shape[0], -1, hd)
    a, b = ys * cos[:, None], ys * sin[:, None]
    return (a + torch.cat((b[..., hd // 2:], -b[..., :hd // 2]), dim=-1)).view(dy.shape)


def window_mask(T: int, window: int, device) -> torch.Tensor:
    """0 where |i − j| <= window, −inf elsewhere: ModernBERT's sliding layers (masking_utils'
    sliding_window_bidirectional_overlay)."""
    pos = torch.arange(T, device=device)
    far = (pos[:, None] - pos[None, :]).abs() > window
    return torch.zeros((T, T), device=device, dtype=_F32).masked_fill(far, float("-inf"))


def r_attn_bi(ex, op, st):
    """Bidirectional scores, q·kᵀ·c, with the window's mask added when the layer has one."""
    q, k = (st[n] for n in op.ins)
    H, KV, hd, G = tracer._dims(op)
    T = q.shape[0]
    s = (tracer._q4(q, KV, G, hd) @ tracer._kv4(k, KV, hd).transpose(-1, -2)) * op.attrs["c"]
    if op.attrs["window"]:
        s = s + window_mask(T, op.attrs["window"], q.device)
    return s.reshape(H * T, T)


def _gelu(x):
    return (0.5 * x) * (1 + torch.erf(x * _INV_SQRT2))


def _gelu_grad(x):
    """d gelu / dx: 0.5 · (1 + erf(x/√2)) + x · φ(x)."""
    return 0.5 * (1 + torch.erf(x * _INV_SQRT2)) + x * (torch.exp(-0.5 * (x * x)) * _INV_SQRT_2PI)


def r_gelu(ex, op, st):
    return _gelu(_x(st, op))


def r_gelu_bwd(ex, op, st):
    u, dy = st[op.ins[0]], st[op.ins[1]]
    return dy * _gelu_grad(u)


def r_geglu(ex, op, st):
    """ModernBERT's MLP gate: gelu of the first half of Wi's output times the second half."""
    wi, f = _x(st, op), op.attrs["f"]
    return _gelu(wi[:, :f]) * wi[:, f:]


def r_geglu_bwd(ex, op, st):
    wi, dm = st[op.ins[0]], st[op.ins[1]]
    f = op.attrs["f"]
    a, g = wi[:, :f], wi[:, f:]
    return torch.cat(((dm * g) * _gelu_grad(a), dm * _gelu(a)), dim=1)


def r_relu(ex, op, st):
    return torch.relu(_x(st, op))


def r_relu_bwd(ex, op, st):
    u, dy = st[op.ins[0]], st[op.ins[1]]
    return dy * (u > 0).to(dy.dtype)


def r_d_type(ex, op, st):
    """The question type's embedding added to every position (Laya's h + type_emb(qtype))."""
    return _x(st, op) + ex.w(op.attrs["weight"])[op.attrs["row"]].to(_F32)


def r_d_gather(ex, op, st):
    markers = ex.item(op.attrs["item"], op.attrs["val"])[1]
    return _x(st, op)[markers]


def r_d_scatter(ex, op, st):
    _, markers, _, _ = ex.item(op.attrs["item"])
    dm = _x(st, op)
    out = torch.zeros(op.shape, device=dm.device, dtype=_F32)
    out[markers] = dm
    return out


def r_d_loss(ex, op, st):
    """The gradient of the item's cross-entropy against its target, times 1/items in the step: softmax − target."""
    z = _x(st, op)[:, 0]
    target = ex.item(op.attrs["item"])[2]
    return ((torch.softmax(z, dim=0) - target) * op.attrs["ic"])[:, None]


OPS = {name[2:]: fn for name, fn in globals().items() if name.startswith("r_")}
DecisionExecutor.ops = {**tracer.RUN, **OPS}
