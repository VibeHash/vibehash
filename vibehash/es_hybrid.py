"""Evo v2's engine for Qwen3.5's hybrid layers (Qwen3.5-0.8B, 2B, 4B, 9B and Qwen3.8-27B share the design): three of
every four layers are Gated DeltaNet linear attention, a fixed-size memory per head (conv state + recurrent state)
instead of a growing KV cache, and every fourth is gated full attention.

The rule is es_fast's (the shared-prompt rule, rank-1 twists, a checker teacher-forcing through the engine that wrote
the answers), on a decoder written for this architecture:

* **The shared reading** of every prompt but its last token is transformers' own forward (untwisted; Gated DeltaNet's
  chunked algorithm), done by worker and checker over the same prompts in one batch, so both get the same states.
* **Everything twisted runs step by step**: each answer token is one decode step, Gated DeltaNet's recurrent rule, for
  the worker answering and for the checker teacher-forcing alike. (A checker reading a whole answer in one forward
  would use the chunked algorithm, which rounds differently from the steps that wrote it.)
* **A pass is always `rows_per_pass(config)` rows**, the same for every worker and checker of a model, padded with idle
  rows, because cuBLAS rounds a row by how many rows share the product. A row's numbers then depend on nothing but the
  row, wherever it sits in the pass. The recurrent state is big (Qwen3.5-4B: 24 layers x 32 heads x 128 x 128 float32,
  48 MB a row), so only one pass of rows is ever resident: rows that finish are replaced by waiting ones at the next
  step (continuous batching), each at its own step and cache slot.
* Each pass is one CUDA graph on CUDA (the buffers are fixed); elsewhere (the CPU tests' tiny model) it runs eagerly.

Twisted linears (TENSORS): full attention's q/k/v/o, linear attention's in_proj_qkv, in_proj_z and out_proj, and the
MLP's gate/up/down; member (j, s) adds s * sigma / sqrt(rank) * (x B_j) A_j^T to each one's output.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from . import es

log = logging.getLogger("vibehash.worker")

# Every projection but linear attention's in_proj_a and in_proj_b (hidden -> one number a value head: each head's
# memory decay and write strength, through softplus/exp and a sigmoid). Those are 0.1% of the projections' weights,
# and a rank-1 twist of their 16-48 outputs moves every head's memory horizon at once rather than what it stores.
TENSORS = ("model.layers.*_proj.weight", "model.layers.*.in_proj_qkv.weight", "model.layers.*.in_proj_z.weight")

# Rows a pass, fixed by the model alone (a worker and its checkers must agree; another value rounds differently). By
# hidden size: 0.8B, 2B, 4B, 9B, 27B. A step's cost is mostly passes over the recurrent state (float32, the model's
# own mamba_ssm_dtype: 18 MB a row in 0.8B, 48 MB in 4B), so rows-steps a second level off early and more rows mostly
# cost memory and make each check (one member's rows in a pass of padding) dearer. Measured on the RTX 4070 SUPER
# (12 GB): 0.8B 2,750 rows-steps/s at 64 (3,200 at 128, twice the memory); 4B 520 at 16 (31 ms a step; 600 at 24,
# 650 at 32), but a round of 64 pairs peaks at 10.9 GiB reserved at 16 and spills into system memory past ~11 (at 32
# it ran 3x slower). 2B, 9B and 27B are provisional, unmeasured.
ROWS_BY_HIDDEN = {1024: 64, 2048: 32, 2560: 16, 4096: 8, 5120: 8}
ROWS_SMALL = 8  # anything else (the tests' tiny models)
_ENGINE: dict = {}


def is_hybrid(model) -> bool:
    return "linear_attention" in (getattr(model.config, "layer_types", None) or ())


def rows_per_pass(cfg) -> int:
    return ROWS_BY_HIDDEN.get(int(cfg.hidden_size), ROWS_SMALL)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rope(x, cos, sin):
    """Qwen3.5's partial rotary embedding: the first cos.shape[-1] numbers of each head rotate, the rest pass."""
    rd = cos.shape[-1]
    xr = x[..., :rd]
    return torch.cat([xr * cos + _rotate_half(xr) * sin, x[..., rd:]], -1)


def _l2norm(x, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


# Gated DeltaNet's state update S' = d S + k delta^T (the decay, then the token's write) as one pass over the float32
# state instead of two (S.mul_(d), then S.baddbmm_(k^T, delta), cuBLAS's K=1 product): each number is
# fma(k_i, delta_j, S_ij * d), rounded as those two passes round it. Measured on the RTX 4070 SUPER (2026-11-08,
# vibe-08b-2's shapes): of 335M numbers none came out different (a multiply then an add differs in 4% of them), and the
# 18 layers' updates take 6.2 ms a step instead of 12.2, in a 25 ms step whose passes over the state were 13.7 ms. A
# device uses it only once it has given the two passes' bits at the engine's shape (onepass_for), so a machine's numbers
# are what they were whatever its cuBLAS does: a speed change only, for workers and checkers alike.
ONEPASS = True  # False: the two passes always (what tests compare it with)
_ONEPASS: dict = {}  # (device index, B, Dk, Dv) -> the kernel, or None where the two passes stay
_THREADS = 256
_ONEPASS_SRC = r"""
extern "C" __global__ void decay_write(float* S, const float* d, const float* k, const float* dl, int dkdv, int dv) {
    const int b = blockIdx.y;
    const int r = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // four numbers of one row of S (dv % 4 == 0)
    if (r >= dkdv) return;
    const int i = r / dv, j = r - i * dv;
    float4* p = reinterpret_cast<float4*>(S + (size_t)b * dkdv + r);
    float4 s = *p;
    const float db = d[b], ki = k[(size_t)b * (dkdv / dv) + i];
    const float4 e = *reinterpret_cast<const float4*>(dl + (size_t)b * dv + j);
    s.x = __fmaf_rn(ki, e.x, __fmul_rn(s.x, db));
    s.y = __fmaf_rn(ki, e.y, __fmul_rn(s.y, db));
    s.z = __fmaf_rn(ki, e.z, __fmul_rn(s.z, db));
    s.w = __fmaf_rn(ki, e.w, __fmul_rn(s.w, db));
    *p = s;
}
"""


def _build_onepass(index: int):
    """decay_write compiled for CUDA device `index` by the NVRTC that PyTorch's CUDA builds ship, and loaded. (Not
    torch.cuda._compile_kernel: it adds a CUDA toolkit's include folders, and stops without CUDA_HOME, which a miner's
    computer has no reason to have; this kernel includes nothing.)"""
    import ctypes
    from torch.cuda import _utils as cu

    if torch.version.hip:
        raise RuntimeError("written for NVIDIA's compiler")
    nvrtc = cu._get_gpu_rtc_library()
    nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p  # const char* nvrtcGetErrorString(nvrtcResult)

    def ok(r):
        if r != 0:
            raise RuntimeError(f"NVRTC: {(nvrtc.nvrtcGetErrorString(r) or b'error').decode()}")
    p = torch.cuda.get_device_properties(index)
    opts = [f"--gpu-architecture=sm_{p.major}{p.minor}".encode(), b"-ftz=false"]
    prog = ctypes.c_void_p()
    ok(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), _ONEPASS_SRC.encode(), b"decay_write.cu", 0, None, None))
    try:
        if nvrtc.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts)) != 0:
            n = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
            text = ctypes.create_string_buffer(n.value)
            nvrtc.nvrtcGetProgramLog(prog, text)
            raise RuntimeError(f"NVRTC: {text.value.decode(errors='replace').strip()}")
        n = ctypes.c_size_t()
        ok(nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(n)))
        cubin = ctypes.create_string_buffer(n.value)
        ok(nvrtc.nvrtcGetCUBIN(prog, cubin))
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    with torch.cuda.device(index):
        return cu._cuda_load_module(cubin.raw, ["decay_write"])["decay_write"]


def _launch(kern, S, d, k, delta) -> None:
    B, Dk, Dv = S.shape
    with torch.cuda.device(S.device):  # the kernel's launch goes to the current device's current stream (a graph's)
        kern(grid=(-(-(Dk * Dv // 4) // _THREADS), B, 1), block=(_THREADS, 1, 1), args=[S, d, k, delta, Dk * Dv, Dv])


def decay_write(S, d, k, delta, kern=None) -> None:
    """S = d S + k^T delta, in place (S [B, Dk, Dv], d [B, 1, 1], k [B, 1, Dk], delta [B, 1, Dv], all float32): one
    pass with `kern` (onepass_for), else the two."""
    if (kern is not None and d.is_contiguous() and k.is_contiguous() and delta.is_contiguous() and S.is_contiguous()
            and S.data_ptr() % 16 == 0 and delta.data_ptr() % 16 == 0):
        _launch(kern, S, d, k, delta)
        return
    S.mul_(d)
    S.baddbmm_(k.transpose(1, 2), delta)


def onepass_for(dev, B: int, Dk: int, Dv: int):
    """The one-pass update's kernel for float32 states [B, Dk, Dv] on CUDA device `dev`, if it builds there (PyTorch's
    own NVRTC, _build_onepass) and gives the two passes' bits on test states of that shape, a quarter
    of them faded down to float32's subnormals as a long-decayed memory is; None otherwise, and the engine keeps the
    two passes. Worked out once a process for each shape."""
    if not ONEPASS:
        return None
    dev = torch.device(dev)
    key = (dev.index if dev.index is not None else torch.cuda.current_device(), B, Dk, Dv)
    if key in _ONEPASS:
        return _ONEPASS[key]
    kern = None
    if Dv % 4 == 0 and 0 < B <= 65535:
        try:
            _pin_precision()  # (TF32 would round baddbmm_'s inputs: the engine has it off, and so does this check)
            kern = _build_onepass(key[0])
            with torch.cuda.device(key[0]):
                g =torch.Generator(device=dev).manual_seed(7919 * B + 31 * Dk + Dv)
                S = torch.randn(B, Dk, Dv, generator=g, device=dev) * 0.3
                S[: B // 4] *= 1e-38
                d = torch.rand(B, 1, 1, generator=g, device=dev) * 0.999 + 0.001
                k = F.normalize(torch.randn(B, 1, Dk, generator=g, device=dev), dim=-1)
                delta = torch.randn(B, 1, Dv, generator=g, device=dev) * 0.1
                want, got = S.clone(), S.clone()
                want.mul_(d)
                want.baddbmm_(k.transpose(1, 2), delta)
                _launch(kern, got, d, k, delta)
                off = int((want.view(torch.int32) != got.view(torch.int32)).sum())
                if off:
                    log.info("the one-pass state update rounds %d of %d numbers differently from the two passes on "
                             "%s: the engine keeps the two", off, want.numel(), torch.cuda.get_device_name(key[0]))
                    kern = None
                del S, want, got
        except Exception as e:  # noqa: BLE001 - an older PyTorch, no NVRTC, a GPU it doesn't build for: two passes
            log.info("the one-pass state update isn't available on cuda:%d (%s): the engine keeps the two passes",
                     key[0], e)
            kern = None
    _ONEPASS[key] = kern
    return kern


def _shared(rt, rows):
    """The shared reading (transformers' forward, untwisted) of every prompt but its last token. Returns (ids, mask,
    pos, states): states[l] is (conv inputs [prompts, C, K-1], recurrent state [prompts, Hv, Dk, Dv]) for a linear
    attention layer, (keys, values) [prompts, KVH, W-1, D] for a full attention one."""
    dev = next(rt.model.parameters()).device
    ids, mask = es._left_pad(rows, rt.pad, dev)
    W = ids.shape[1]
    if W < 2:
        raise ValueError("a prompt of one token has nothing to share")
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = rt.model(input_ids=ids[:, :W - 1], attention_mask=mask[:, :W - 1], position_ids=pos[:, :W - 1],
                   use_cache=True, logits_to_keep=1)
    K = rt.model.config.linear_conv_kernel_dim
    states = []
    for kind, layer in zip(rt.model.config.layer_types, out.past_key_values.layers):
        if kind == "linear_attention":
            states.append((layer.conv_states[0][..., -(K - 1):], layer.recurrent_states[0]))
        else:
            states.append((layer.keys, layer.values))
    return ids, mask, pos, states


class _Hybrid:
    """Buffers (and CUDA graphs) for decoding R rows through one resident pass of n rows, a KV cache of L slots for
    the full attention layers, answers of up to T tokens. Rows are member-major (member k's answer to prompt q is row
    k * nq + q); row R is a spare that idle slots write to. loop: a job's loop_stop (period, run), or () for none: an
    answering row also stops at its first looping token (es2.loop_end), counted on the device a period at a time."""

    def __init__(self, rt, linears, n, R, PR, L, T, loop=()):
        model, cfg = rt.model, rt.model.config
        p = next(model.parameters())
        dev, dt = p.device, p.dtype
        self.rt, self.n, self.R, self.L, self.T, self.dev, self.dt = rt, n, R, L, T, dev, dt
        self.loop = tuple(loop)
        if self.loop:  # count[row, p - 1]: the tokens in a row each equal to the one p before (es2.loop_end)
            self.lags = torch.arange(1, self.loop[0] + 1, device=dev)
            self.count = torch.zeros(n, self.loop[0], dtype=torch.long, device=dev)
        self.kinds = list(cfg.layer_types)
        self.layers = list(model.model.layers)
        self.embed, self.norm, self.rot = model.model.embed_tokens, model.model.norm, model.model.rotary_emb
        self.lm_w = model.lm_head.weight
        self.H, self.KVH, self.D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.G = self.H // self.KVH
        self.Hk, self.Hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.Dk, self.Dv, self.K = cfg.linear_key_head_dim, cfg.linear_value_head_dim, cfg.linear_conv_kernel_dim
        self.C = 2 * self.Hk * self.Dk + self.Hv * self.Dv
        self.names = {id(m): nm for nm, m in linears}
        self.tw = {id(m): (torch.zeros(m.in_features, PR, dtype=dt, device=dev),
                           torch.zeros(PR, m.out_features, dtype=dt, device=dev)) for nm, m in linears}
        z = lambda *s, d=dt: torch.zeros(*s, dtype=d, device=dev)  # noqa: E731
        self.conv, self.rec, self.kc, self.vc = {}, {}, {}, {}
        for l, kind in enumerate(self.kinds):
            if kind == "linear_attention":
                self.conv[l] = z(n, self.C, self.K - 1)
                self.rec[l] = z(n, self.Hv, self.Dk, self.Dv, d=torch.float32)
            else:
                self.kc[l] = z(n, self.KVH, L, self.D)
                self.vc[l] = z(n, self.KVH, L, self.D)
        self.bias = torch.full((n, L), float("-inf"), dtype=dt, device=dev)
        self.zero = z(())
        self.tok = z(n, d=torch.long)
        self.cap = torch.full((n,), T, dtype=torch.long, device=dev)  # each row's most tokens (Spec2.task_tokens)
        self.pos = z(n, d=torch.long)
        self.col = z(n, d=torch.long)  # each row's step: the answer token it reads next is number col
        self.slot = z(n, d=torch.long)  # the KV cache slot it writes: W - 1 + col
        self.w1 = z((), d=torch.long)  # W - 1, where every row's answer starts in the cache
        self.rowid = torch.full((n,), R, dtype=torch.long, device=dev)
        self.done = torch.ones(n, dtype=torch.bool, device=dev)
        self.sel = z(n, PR)
        self.ar = torch.arange(n, device=dev)
        self.out = z(R + 1, T, d=torch.long)
        self.ans = z(R + 1, T, d=torch.long)  # teacher forcing: each row's committed answer, its length, each token's
        self.alen = z(R + 1, d=torch.long)  # gap below the step's highest logit and its log-probability
        self.gap = z(R + 1, T, d=torch.float32)
        self.lp = z(R + 1, T, d=torch.float32)
        self.eos = torch.tensor(rt.eos, device=dev)
        self.graphs = {}
        self.onepass = None  # the state's decay and write in one pass (onepass_for), where it's this machine's numbers
        if dev.type == "cuda":
            self.pool = torch.cuda.graph_pool_handle()
            self.stream = torch.cuda.Stream()
            if any(k == "linear_attention" for k in self.kinds):
                self.onepass = onepass_for(dev, n * self.Hv, self.Dk, self.Dv)

    def load(self, fac):
        by_name = {nm: i for i, nm in self.names.items()}
        for nm, (A, B) in fac.items():
            Bt, Aa = self.tw[by_name[nm]]
            Bt.copy_(B.permute(1, 0, 2).reshape(Bt.shape))
            Aa.copy_(A.permute(0, 2, 1).reshape(Aa.shape))

    def _lin(self, mod, x):
        y = F.linear(x, mod.weight, mod.bias)
        t = self.tw.get(id(mod))
        if t is None:
            return y
        Bt, Aa = t
        return y.addmm_(torch.mm(x, Bt) * self.sel, Aa)  # the row's twist: one nonzero factor column, times coef

    def _linear_attn(self, l, la, x):
        """Gated DeltaNet's recurrent step for every row (transformers' torch_recurrent_gated_delta_rule's arithmetic,
        the state's products as batched GEMMs)."""
        n, Hk, Hv, Dk, Dv = self.n, self.Hk, self.Hv, self.Dk, self.Dv
        qkv = self._lin(la.in_proj_qkv, x)
        zg = self._lin(la.in_proj_z, x)
        b = F.linear(x, la.in_proj_b.weight)
        a = F.linear(x, la.in_proj_a.weight)
        cs = self.conv[l]
        win = torch.cat([cs, qkv[:, :, None]], -1)  # the causal conv's window: the last K inputs
        cs.copy_(win[:, :, 1:])
        c = F.silu((win.float() * la.conv1d.weight[:, 0, :].float()).sum(-1).to(x.dtype))
        q, k, v = torch.split(c, [Hk * Dk, Hk * Dk, Hv * Dv], -1)
        q, k = q.reshape(n, Hk, Dk).float(), k.reshape(n, Hk, Dk).float()
        if Hv // Hk > 1:
            q, k = q.repeat_interleave(Hv // Hk, dim=1), k.repeat_interleave(Hv // Hk, dim=1)
        B = n * Hv
        q = (_l2norm(q) / (Dk ** 0.5)).reshape(B, 1, Dk)
        k = _l2norm(k).reshape(B, 1, Dk)
        beta = b.sigmoid().float().reshape(B, 1, 1)
        d = (-la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)).exp().reshape(B, 1, 1)  # the decay
        # S' = d S + k delta^T, delta = (v - d k^T S) beta, out = q^T S' = d q^T S + (q.k) delta: the state is read
        # once for both products, then decayed and updated in place: 3 passes over it a step with the one-pass
        # update (decay_write), 5 without (6 before either).
        Sv = self.rec[l].view(B, Dk, Dv)
        kq = torch.bmm(torch.cat([k, q], 1), Sv) * d  # [B, 2, Dv]: d k^T S, d q^T S
        delta = (v.float().reshape(B, 1, Dv) - kq[:, :1]) * beta
        o = (kq[:, 1:] + torch.bmm(q, k.transpose(1, 2)) * delta).reshape(B, Dv).to(x.dtype)
        decay_write(Sv, d, k, delta, self.onepass)
        o = la.norm(o, zg.reshape(n * Hv, Dv)).reshape(n, Hv * Dv)
        return self._lin(la.out_proj, o)

    def _full_attn(self, l, at, x, cos, sin):
        n, H, KVH, D, G = self.n, self.H, self.KVH, self.D, self.G
        qg = self._lin(at.q_proj, x).view(n, H, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(n, H * D)
        q = _rope(at.q_norm(q), cos, sin)
        k = _rope(at.k_norm(self._lin(at.k_proj, x).view(n, KVH, D)), cos, sin)
        v = self._lin(at.v_proj, x).view(n, KVH, D)
        kc, vc = self.kc[l], self.vc[l]
        kc[self.ar, :, self.slot] = k
        vc[self.ar, :, self.slot] = v
        o = F.scaled_dot_product_attention(q.reshape(n, KVH, G, D), kc, vc, attn_mask=self.bias[:, None, None, :],
                                           scale=at.scaling)
        return self._lin(at.o_proj, o.reshape(n, H * D) * torch.sigmoid(gate))

    def _pass(self):
        """One decode step's logits for every resident row, their states moved on."""
        self.bias.index_put_((self.ar, self.slot), self.zero)
        h = self.embed(self.tok)
        cos, sin = self.rot(h, self.pos[:, None])
        for l, (kind, layer) in enumerate(zip(self.kinds, self.layers)):
            x = layer.input_layernorm(h)
            if kind == "linear_attention":
                h = h + self._linear_attn(l, layer.linear_attn, x)
            else:
                h = h + self._full_attn(l, layer.self_attn, x, cos, sin)
            x = layer.post_attention_layernorm(h)
            mlp = layer.mlp
            h = h + self._lin(mlp.down_proj, mlp.act_fn(self._lin(mlp.gate_proj, x)) * self._lin(mlp.up_proj, x))
        return F.linear(self.norm(h), self.lm_w)

    def _step(self, forced: bool):
        logits = self._pass()
        rows, col = self.rowid, self.col.clamp(max=self.T - 1)
        if forced:
            nxt = self.ans[rows, col]
            got = logits.gather(1, nxt[:, None])[:, 0].float()
            self.gap.index_put_((rows, col), logits.max(-1).values.float() - got)
            self.lp.index_put_((rows, col), got - torch.logsumexp(logits.float(), -1))
            fin = self.col + 1 >= self.alen[rows]
        else:
            nxt = logits.argmax(-1)
            self.out.index_put_((rows, col), nxt)
            fin = (nxt[:, None] == self.eos).any(-1) | (self.col + 1 >= self.cap)
            if self.loop:  # the token p back is in the row's own answer so far: out[row, col - p], for col >= p
                lag = self.col[:, None] - self.lags[None, :]
                hit = (lag >= 0) & (self.out[rows[:, None], lag.clamp(min=0)] == nxt[:, None])
                self.count.copy_((self.count + 1) * hit)
                fin = fin | (self.count >= self.loop[1]).any(-1)
        act = (~self.done).long()
        self.tok.copy_(nxt)
        self.pos.add_(act)
        self.col.add_(act)
        torch.clamp(self.col + self.w1, max=self.L - 1, out=self.slot)
        self.done.logical_or_(fin)
        self.rowid.masked_fill_(self.done, self.R)

    def run(self, forced: bool):
        if self.dev.type != "cuda":
            return self._step(forced)
        g = self.graphs.get(forced)
        if g is not None:
            g.replay()
            return
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            self._step(forced)
            g = torch.cuda.CUDAGraph()
            g.capture_begin(self.pool)
            try:
                self._step(forced)
            finally:
                g.capture_end()
        cur.wait_stream(self.stream)
        self.graphs[forced] = g

    def decode(self, rt, rows, members, fac, sigma, rank, forced: bool, order, caps=None):
        """Every row in `order` (row ids) through the resident pass, waiting rows taking slots as rows finish. caps:
        each prompt's most answer tokens (answering stops a row there), T for all when None."""
        dev, dt, n = self.dev, self.dt, self.n
        nq = len(rows)
        capq = torch.tensor([min(int(c), self.T) for c in caps] if caps is not None else [self.T] * nq,
                            dtype=torch.long, device=dev)
        self.load({nm: (A.to(dt), B.to(dt)) for nm, (A, B) in fac.items()})
        r = next(iter(fac.values()))[0].shape[2]
        M = len(members)
        selm = torch.zeros(M, self.sel.shape[1], dtype=dt, device=dev)
        coef = torch.tensor([s * sigma / math.sqrt(rank) for _, s in members], device=dev, dtype=dt)
        for i in range(r):
            selm[torch.arange(M, device=dev), torch.tensor([j * r + i for j, _ in members], device=dev)] = coef
        ids, mask, pos, states = _shared(rt, rows)
        W = ids.shape[1]
        for l, kind in enumerate(self.kinds):  # a row's cache past its prompt starts as zeros, whoever had the slot
            if kind == "full_attention":
                states[l] = tuple(F.pad(s, (0, 0, 0, self.L - (W - 1))) for s in states[l])
        pbias = torch.full((nq, self.L), float("-inf"), dtype=dt, device=dev)
        pbias[:, :W - 1].masked_fill_(mask[:, :W - 1].bool(), 0.0)
        last_tok, last_pos = ids[:, W - 1], pos[:, W - 1]
        self.w1.fill_(W - 1)
        self.done.fill_(True)
        self.rowid.fill_(self.R)
        self.bias.fill_(float("-inf"))
        order = torch.as_tensor(order, dtype=torch.long, device=dev)
        at, total = 0, order.numel()
        while True:
            free = torch.nonzero(self.done).squeeze(1)
            if at < total and free.numel():
                take = min(free.numel(), total - at)
                slots, rid = free[:take], order[at:at + take]
                at += take
                q, k = rid % nq, rid // nq
                for l, kind in enumerate(self.kinds):
                    a, b = states[l]
                    if kind == "linear_attention":
                        self.conv[l][slots] = a[q]
                        self.rec[l][slots] = b[q]
                    else:
                        self.kc[l][slots] = a[q]
                        self.vc[l][slots] = b[q]
                self.bias[slots] = pbias[q]
                self.tok[slots] = last_tok[q]
                self.cap[slots] = capq[q]
                self.pos[slots] = last_pos[q]
                self.col[slots] = 0
                if self.loop:
                    self.count[slots] = 0
                self.slot[slots] = W - 1
                self.sel[slots] = selm[k]
                self.rowid[slots] = rid
                self.done[slots] = False
            elif free.numel() == n:
                break
            self.run(forced)


def _engine(rt, linears, fac, R, rows, T, loop=()):
    """The engine for R rows of these prompts' answers up to T tokens, made once and kept while the shape repeats."""
    W = max(len(r) for r in rows)
    L = -(-(W + T - 1) // 32) * 32
    A0 = next(iter(fac.values()))[0]
    PR = A0.shape[0] * A0.shape[2]
    n = rows_per_pass(rt.model.config)
    key = (id(rt.model), n, R, PR, L, T, tuple(nm for nm, _ in linears), tuple(loop))
    eng = _ENGINE.get(key)
    if eng is None:
        _ENGINE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        eng = _ENGINE[key] = _Hybrid(rt, linears, n, R, PR, L, T, loop)
    return eng


def longest_first(lens) -> list[int]:
    """The rows with something to read (length > 0), longest first (ties in row order): the order teacher forcing
    takes them into the pass. A row's numbers don't depend on when it starts or what shares its pass (the module's
    docstring), so the order only decides how long the pass runs: one long row started last runs on alone at the
    full pass's cost. Measured on vibe-08b-2's replay (2026-11-07, RTX 4070 SUPER; 64 members x 16 targets of 2 to
    483 tokens, 846 steps' work for 64 slots): 28.6 s in row order, 22.2 s longest first."""
    return sorted((r for r, n in enumerate(lens) if n > 0), key=lambda r: -lens[r])


def _pin_precision():
    # The recurrent state's products are float32 GEMMs: TF32 would round them otherwise, and differently from a
    # checker that left it off (on Intel GPUs too, through oneDNN). compute.pin_precision: only writes the settings.
    from .compute import pin_precision
    pin_precision()


def answer_order(nq: int, M: int, expect=None) -> list[int]:
    """The order answering takes rows (member k's answer to prompt q is row k * nq + q) into the pass: member by member
    without `expect`; with it (each prompt's expected answer length), the prompts expected longest first, every
    member's row of a prompt before the next prompt's. Like longest_first, it decides only how long the pass runs."""
    if expect is None:
        return list(range(M * nq))
    qs = sorted(range(nq), key=lambda q: -float(expect[q]))
    return [k * nq + q for q in qs for k in range(M)]


def answer(rt, linears, fac, members, rows, T, sigma, rank, loop=(), caps=None, expect=None):
    """Every member's greedy answers to every prompt (es_fast.answer_members on a hybrid model). loop: the job's
    loop_stop, () for none: a looping row stops there (its tokens past that are another row's, left in the buffer).
    caps: each prompt's most tokens (Spec2.task_tokens), T for all when None. expect: each prompt's expected answer
    length, to take the long ones first (answer_order); the answers are the same either way."""
    from .es2 import cut_loop

    _pin_precision()
    nq, M = len(rows), len(members)
    R = M * nq
    eng = _engine(rt, linears, fac, R, rows, T, loop)
    eng.decode(rt, rows, members, fac, sigma, rank, False, answer_order(nq, M, expect), caps)
    toks = eng.out[:R].tolist()
    answers = []
    for i, row in enumerate(toks):
        if caps is not None:
            row = row[:caps[i % nq]]  # a row stopped at its cap: what's past it is stale
        row = cut_loop(row, loop)  # first: a row that stopped looping has stale tokens past its end
        cut = next((i + 1 for i, t in enumerate(row) if t in rt.eos), len(row))
        answers.append(row[:cut])
    return [answers[k * nq:(k + 1) * nq] for k in range(M)]


def forced(rt, linears, fac, members, rows, answers, T, sigma, rank):
    """Teacher forcing of every member's committed answers (answers[member][prompt]) through the decode steps that
    wrote them: (worst gap below the step's highest logit, mean log-probability) of each, two [members, prompts]
    float tensors (0 for an empty answer)."""
    _pin_precision()
    nq, M = len(rows), len(members)
    R = M * nq
    ans, alen = torch.zeros(R + 1, T, dtype=torch.long), torch.zeros(R + 1, dtype=torch.long)
    for k, member in enumerate(answers):
        for q, a in enumerate(member):
            if len(a) > T:
                raise ValueError(f"an answer of {len(a)} tokens, past the {T} answering stops at")
            ans[k * nq + q, :len(a)] = torch.tensor(a, dtype=torch.long)
            alen[k * nq + q] = len(a)
    eng = _engine(rt, linears, fac, R, rows, T)
    eng.ans.copy_(ans)
    eng.alen.copy_(alen)
    eng.gap.zero_()
    eng.lp.zero_()
    eng.decode(rt, rows, members, fac, sigma, rank, True, longest_first(alen[:R].tolist()))
    cols = torch.arange(T, device=eng.dev)[None, :] < eng.alen[:R, None]
    gaps = torch.where(cols, eng.gap[:R], 0.0).max(-1).values.view(M, nq)
    lps = torch.where(cols, eng.lp[:R], 0.0).sum(-1) / eng.alen[:R].clamp(min=1)
    return gaps, lps.view(M, nq)
