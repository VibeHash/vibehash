"""Evo v2 on an NPU (Intel AI Boost first): Qwen3.5's twisted decode step compiled by OpenVINO, for measuring how far
an NPU's answers sit from a checker's greedy choice (docs/CROSS-HARDWARE.md, "NPUs", step 2). Not a worker backend:
nothing here takes network work.

NPUs run fixed shapes through their vendor's compiler, in fp16 (Intel's has no float32 at all), so the step is
es_hybrid's (_Hybrid._pass) written as one pure function, every state an input and an output:

* inputs: each resident row's embedded token, its rotary cos/sin, its attention bias and the cache slot it writes
  (one-hot), its twist (sel: the member's coef in its pair's column) and, teacher forcing, the token it must read;
  every layer's state (Gated DeltaNet's conv window and recurrent state, full attention's keys and values); and every
  twisted linear's factors (Bt [in, PR], Aa [PR, out]), so one compiled model serves every round;
* outputs: the greedy token, the highest logit, the forced token's logit, the log-sum-exp, and every state moved on.

The host loop is es_hybrid's (_Hybrid.decode and _step): a pass of `n` rows, rows replaced as they finish. The shared
reading of every prompt but its last token is transformers' own forward on the CPU in float32 (es_hybrid._shared),
untwisted, as a worker on any device does; only the twisted steps run on the NPU. So what's measured is the NPU's
decode, the part a worker's answers depend on the twist through.

The base model's weights are constants of the compiled model (an NPU takes them in fp16); its weights must be the
round's own (a job's line applied) before `compile`: nothing here reloads them.

    eng = es_npu.Engine(rt, rt.linears, PR, L, n=16, device="NPU")
    answers = es_npu.answer(eng, rt, fac, members, rows, T, sigma, rank)
    gaps, lps = es_npu.forced(eng, rt, fac, members, rows, answers, T, sigma, rank)

`rt` is es_client.runtime2(spec, "cpu"): the CPU float32 model, read once for the constants.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import es_hybrid

NEG = -30000.0  # a masked cache slot's bias: below anything softmax keeps, and inside fp16 (no -inf arithmetic)
# Gated DeltaNet's read-out q^T S is computed QSCALE * sqrt(Dk) times too large (q unit-length times QSCALE rather
# than over sqrt(Dk)), and the gated norm after it, which a scale doesn't move, takes the factor back out of its
# epsilon. Some heads' outputs are ~1e-5, fp16 subnormals, which an NPU holds to a few bits, and the norm then blows
# those bits up to full size: on Qwen3.5-0.8B (Intel AI Boost) the layer's output was 3e-2 off float32 that way, 20x
# what an Intel GPU's fp16 does. Scaled, they sit in fp16's normal range (the largest, ~0.35, becomes ~250).
QSCALE = 64.0


def _scaled(x):
    """x over its largest magnitude (per last-dim row; 1 for a row of zeros) and that magnitude: a norm's input scaled
    before it's squared, so the squares of small values don't fall out of fp16's range."""
    m = x.abs().amax(-1, keepdim=True)
    m = torch.where(m > 0, m, torch.ones_like(m))
    return x / m, m


def _rnorm(ss, m, eps: float, scale: float = 1.0):
    """1 / sqrt(ss + t^2), t = sqrt(eps) * scale / m: a norm's factor for its scaled input u = x / m (ss: u's sum or
    mean of squares) when x is `scale` times the norm's own input, never squaring a number out of fp16's range.
    t^2 is the norm's epsilon in u's units: m * m underflows to 0 for m ~1e-4, and t^2 overflows for a small m and a
    big scale, so where t > 1 it's w / sqrt(ss * w^2 + 1), w = 1 / t, the same number. (torch.where picks one; the
    other side's inf or nan is never used.)"""
    w = m / (math.sqrt(eps) * scale)
    return torch.where(w < 1.0, w * torch.rsqrt(ss * (w * w) + 1.0), torch.rsqrt(ss + 1.0 / (w * w)))


def _l2norm(x, eps: float = 1e-6):
    """es_hybrid._l2norm, scale-safe in fp16 (_scaled): x / sqrt(sum x^2 + eps)."""
    u, m = _scaled(x)
    return u * _rnorm((u * u).sum(dim=-1, keepdim=True), m, eps)


def _gated_norm(norm, x, z, scale: float = 1.0):
    """Qwen3_5RMSNormGated (weight * x / rms(x) * silu(z)) of x / scale, given x, scale-safe in fp16 (_scaled)."""
    u, m = _scaled(x)
    return norm.weight * (u * _rnorm((u * u).mean(-1, keepdim=True), m, norm.variance_epsilon, scale)) * F.silu(z)


def _rope(x, cos, sin):
    rd = cos.shape[-1]
    xr = x[..., :rd]
    return torch.cat([xr * cos + es_hybrid._rotate_half(xr) * sin, x[..., rd:]], -1)


class _Step(torch.nn.Module):
    """_Hybrid._pass as a pure function of its inputs (the module docstring), for OpenVINO to trace."""

    def __init__(self, model, linears, n: int, L: int):
        super().__init__()
        cfg = model.config
        self.model, self.n, self.L = model, n, L
        self.kinds = list(cfg.layer_types)
        self.H, self.KVH, self.D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.G = self.H // self.KVH
        self.Hk, self.Hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.Dk, self.Dv, self.K = cfg.linear_key_head_dim, cfg.linear_value_head_dim, cfg.linear_conv_kernel_dim
        self.order = [id(m) for _, m in linears]  # the factors' input order: rt.linears' (sorted by name)
        self.nlin = len(linears)
        self.nla = sum(k == "linear_attention" for k in self.kinds)
        self.nfa = len(self.kinds) - self.nla

    def _lin(self, mod, x, tw, sel):
        y = F.linear(x, mod.weight, mod.bias)
        t = tw.get(id(mod))
        if t is None:
            return y
        Bt, Aa = t
        return y + torch.matmul(torch.matmul(x, Bt) * sel, Aa)

    def _linear_attn(self, la, x, cs, S, tw, sel):
        n, Hk, Hv, Dk, Dv = self.n, self.Hk, self.Hv, self.Dk, self.Dv
        qkv = self._lin(la.in_proj_qkv, x, tw, sel)
        zg = self._lin(la.in_proj_z, x, tw, sel)
        b = F.linear(x, la.in_proj_b.weight)
        a = F.linear(x, la.in_proj_a.weight)
        win = torch.cat([cs, qkv[:, :, None]], -1)
        c = F.silu((win * la.conv1d.weight[:, 0, :]).sum(-1))
        q, k, v = torch.split(c, [Hk * Dk, Hk * Dk, Hv * Dv], -1)
        q, k = q.reshape(n, Hk, Dk), k.reshape(n, Hk, Dk)
        if Hv // Hk > 1:
            g = Hv // Hk  # repeat_interleave, as an expand (the NPU compiler can't take its lowering)
            q = q[:, :, None, :].expand(n, Hk, g, Dk).reshape(n, Hv, Dk)
            k = k[:, :, None, :].expand(n, Hk, g, Dk).reshape(n, Hv, Dk)
        B = n * Hv
        q = (_l2norm(q) * QSCALE).reshape(B, 1, Dk)  # QSCALE * sqrt(Dk) times the model's q: the read-out's scale
        k = _l2norm(k).reshape(B, 1, Dk)
        beta = torch.sigmoid(b).reshape(B, 1, 1)
        d = torch.exp(-torch.exp(la.A_log) * F.softplus(a + la.dt_bias)).reshape(B, 1, 1)
        Sv = S.reshape(B, Dk, Dv)
        kq = torch.matmul(torch.cat([k, q], 1), Sv) * d
        delta = (v.reshape(B, 1, Dv) - kq[:, :1]) * beta
        o = (kq[:, 1:] + torch.matmul(q, k.transpose(1, 2)) * delta).reshape(B, Dv)
        S2 = Sv * d + torch.matmul(k.transpose(1, 2), delta)
        o = _gated_norm(la.norm, o, zg.reshape(B, Dv), QSCALE * Dk ** 0.5).reshape(n, Hv * Dv)
        return self._lin(la.out_proj, o, tw, sel), win[:, :, 1:], S2.reshape(n, Hv, Dk, Dv)

    def _full_attn(self, at, x, cos, sin, kc, vc, bias, oh, tw, sel):
        n, H, KVH, D, G = self.n, self.H, self.KVH, self.D, self.G
        qg = self._lin(at.q_proj, x, tw, sel).reshape(n, H, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(n, H * D)
        q = _rope(at.q_norm(q), cos, sin)
        k = _rope(at.k_norm(self._lin(at.k_proj, x, tw, sel).reshape(n, KVH, D)), cos, sin)
        v = self._lin(at.v_proj, x, tw, sel).reshape(n, KVH, D)
        w = oh[:, None, :, None]  # the row's slot: [n, 1, L, 1]
        kc2 = kc * (1.0 - w) + k[:, :, None, :] * w
        vc2 = vc * (1.0 - w) + v[:, :, None, :] * w
        s = torch.matmul(q.reshape(n, KVH, G, D), kc2.transpose(2, 3)) * at.scaling + bias[:, None, None, :]
        o = torch.matmul(torch.softmax(s, -1), vc2)
        return self._lin(at.o_proj, o.reshape(n, H * D) * torch.sigmoid(gate), tw, sel), kc2, vc2

    def forward(self, h, cos, sin, bias, oh, sel, want, *rest):
        nla, nfa, nlin = self.nla, self.nfa, self.nlin
        conv, rec = rest[:nla], rest[nla:2 * nla]
        kc, vc = rest[2 * nla:2 * nla + nfa], rest[2 * nla + nfa:2 * nla + 2 * nfa]
        fac = rest[2 * nla + 2 * nfa:]
        tw = {m: (fac[2 * i], fac[2 * i + 1]) for i, m in enumerate(self.order)}
        new_conv, new_rec, new_k, new_v = [], [], [], []
        ia = ifa = 0
        for kind, layer in zip(self.kinds, self.model.model.layers):
            x = layer.input_layernorm(h)
            if kind == "linear_attention":
                y, c2, s2 = self._linear_attn(layer.linear_attn, x, conv[ia], rec[ia], tw, sel)
                new_conv.append(c2)
                new_rec.append(s2)
                ia += 1
            else:
                y, k2, v2 = self._full_attn(layer.self_attn, x, cos, sin, kc[ifa], vc[ifa], bias, oh, tw, sel)
                new_k.append(k2)
                new_v.append(v2)
                ifa += 1
            h = h + y
            x = layer.post_attention_layernorm(h)
            mlp = layer.mlp
            h = h + self._lin(mlp.down_proj, mlp.act_fn(self._lin(mlp.gate_proj, x, tw, sel))
                              * self._lin(mlp.up_proj, x, tw, sel), tw, sel)
        logits = F.linear(self.model.model.norm(h), self.model.lm_head.weight)
        top = logits.max(-1)
        got = logits.gather(1, want[:, None])[:, 0]
        lse = torch.logsumexp(logits, -1)
        return (top.indices, top.values, got, lse, *new_conv, *new_rec, *new_k, *new_v)


class Engine:
    """One compiled step for n resident rows, L cache slots and PR factor columns (pairs x rank), on an OpenVINO
    device ("NPU", "GPU", "CPU"). precision: the device's inference precision ("f16" on an NPU, which has nothing
    else; "f32" on the CPU makes this the CPU's own float32 engine, the test's reference)."""

    def __init__(self, rt, linears, PR: int, L: int, n: int = 16, device: str = "NPU", precision: str | None = None,
                 cache_dir: str | Path | None = None, options: dict | None = None, log=print):
        import openvino as ov

        model = rt.model
        p = next(model.parameters())
        if p.device.type != "cpu" or p.dtype != torch.float32:
            raise ValueError("es_npu reads the base from the CPU float32 model (es_client.runtime2(spec, 'cpu'))")
        if not es_hybrid.is_hybrid(model):
            raise ValueError("es_npu runs Qwen3.5-style (Gated DeltaNet) models only")
        cfg = model.config
        self.rt, self.n, self.L, self.PR, self.device = rt, n, L, PR, device
        self.linears = list(linears)
        self.step = _Step(model, self.linears, n, L)
        st = self.step
        self.C = 2 * st.Hk * st.Dk + st.Hv * st.Dv
        self.kinds = st.kinds
        self.embed = model.model.embed_tokens.weight.detach().numpy()
        self.rot = model.model.rotary_emb
        rd = self.rot(torch.zeros(1, 1, cfg.hidden_size), torch.zeros(1, 1, dtype=torch.long))[0].shape[-1]
        self.rd = rd
        ex = self._example()
        t0 = time.perf_counter()
        with torch.no_grad():
            ovm = ov.convert_model(st, example_input=ex)
        ovm.reshape({p: ov.PartialShape(list(x.shape)) for p, x in zip(ovm.inputs, ex)})  # static: the NPU's only kind
        log(f"es_npu: traced the step ({n} rows, {L} slots, {PR} factor columns) in {time.perf_counter() - t0:.0f} s")
        core = ov.Core()
        if cache_dir:
            core.set_property({"CACHE_DIR": str(cache_dir)})
        conf = dict(options or {})  # the device's own settings (EXECUTION_MODE_HINT, ...)
        if precision:
            conf["INFERENCE_PRECISION_HINT"] = precision
        t0 = time.perf_counter()
        self.compiled = core.compile_model(ovm, device, conf)
        self.req = self.compiled.create_infer_request()
        try:
            prec = self.compiled.get_property("INFERENCE_PRECISION_HINT")
        except RuntimeError:  # an NPU model loaded from CACHE_DIR doesn't answer it; the device does
            prec = core.get_property(device, "INFERENCE_PRECISION_HINT")
        self.precision = prec.get_type_name()  # "f16", "f32"
        log(f"es_npu: compiled for {device} ({self.precision}) in {time.perf_counter() - t0:.0f} s")
        del ovm
        self.nin = len(ex)

    def _example(self):
        st, n, L, PR = self.step, self.n, self.L, self.PR
        f = torch.float32
        ex = [torch.zeros(n, self.rt.model.config.hidden_size), torch.ones(n, 1, self.rd), torch.zeros(n, 1, self.rd),
              torch.zeros(n, L), torch.zeros(n, L), torch.zeros(n, PR), torch.zeros(n, dtype=torch.long)]
        ex += [torch.zeros(n, self.C, st.K - 1, dtype=f) for _ in range(st.nla)]
        ex += [torch.zeros(n, st.Hv, st.Dk, st.Dv, dtype=f) for _ in range(st.nla)]
        ex += [torch.zeros(n, st.KVH, L, st.D, dtype=f) for _ in range(2 * st.nfa)]
        for _, m in self.linears:
            ex += [torch.zeros(m.in_features, PR), torch.zeros(PR, m.out_features)]
        return tuple(ex)

    def load(self, fac, members_cols: int):
        """A round's factors ({name: (A [P, out, r], B [P, in, r])}) into the step's factor inputs, as _Hybrid.load."""
        if members_cols != self.PR:
            raise ValueError(f"the engine was compiled for {self.PR} factor columns, the round has {members_cols}")
        at = 7 + 2 * self.step.nla + 2 * self.step.nfa
        for i, (nm, m) in enumerate(self.linears):
            A, B = fac[nm]
            Bt = B.permute(1, 0, 2).reshape(m.in_features, self.PR).float().numpy()
            Aa = A.permute(0, 2, 1).reshape(self.PR, m.out_features).float().numpy()
            self.req.set_input_tensor(at + 2 * i, _tensor(Bt))
            self.req.set_input_tensor(at + 2 * i + 1, _tensor(Aa))

    def decode(self, rows, members, fac, sigma, rank, forced: bool, order, T: int, ans=None, alen=None):
        """_Hybrid.decode and _step on the host, each step on the device. Returns (out, gap, lp): [R + 1, T] arrays,
        row R the idle rows' spare."""
        rt, n, L, st = self.rt, self.n, self.L, self.step
        nq, M = len(rows), len(members)
        R = M * nq
        r = next(iter(fac.values()))[0].shape[2]
        P = next(iter(fac.values()))[0].shape[0]
        self.load(fac, P * r)
        selm = np.zeros((M, self.PR), np.float32)
        for k, (j, s) in enumerate(members):
            for i in range(r):
                selm[k, j * r + i] = s * sigma / math.sqrt(rank)
        with torch.no_grad():
            ids, mask, pos, states = es_hybrid._shared(rt, rows)
        W = ids.shape[1]
        if W - 1 + T > L:
            raise ValueError(f"{W - 1} prompt slots and {T} answer tokens don't fit {L} cache slots")
        la = [(a.float().numpy(), b.float().numpy()) for kind, (a, b) in zip(self.kinds, states)
              if kind == "linear_attention"]
        fa = [tuple(F.pad(s, (0, 0, 0, L - (W - 1))).float().numpy() for s in ab)
              for kind, ab in zip(self.kinds, states) if kind != "linear_attention"]
        pbias = np.full((nq, L), NEG, np.float32)
        pbias[:, :W - 1][mask[:, :W - 1].bool().numpy()] = 0.0
        last_tok, last_pos = ids[:, W - 1].numpy(), pos[:, W - 1].numpy()
        conv = [np.zeros((n, self.C, st.K - 1), np.float32) for _ in range(st.nla)]
        rec = [np.zeros((n, st.Hv, st.Dk, st.Dv), np.float32) for _ in range(st.nla)]
        kc = [np.zeros((n, st.KVH, L, st.D), np.float32) for _ in range(st.nfa)]
        vc = [np.zeros((n, st.KVH, L, st.D), np.float32) for _ in range(st.nfa)]
        bias = np.full((n, L), NEG, np.float32)
        tok, cap = np.zeros(n, np.int64), np.full(n, T, np.int64)
        p_, col, slot = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64)
        rowid, done = np.full(n, R, np.int64), np.ones(n, bool)
        sel = np.zeros((n, self.PR), np.float32)
        out = np.zeros((R + 1, T), np.int64)
        gap = np.zeros((R + 1, T), np.float32)
        lp = np.zeros((R + 1, T), np.float32)
        if forced:
            ans_, alen_ = np.asarray(ans), np.asarray(alen)
        eos = np.array(rt.eos)
        order = list(order)
        at = 0
        nla, nfa = st.nla, st.nfa
        steps = 0
        while True:
            free = np.nonzero(done)[0]
            if at < len(order) and free.size:
                take = min(free.size, len(order) - at)
                slots, rid = free[:take], np.array(order[at:at + take])
                at += take
                q, k = rid % nq, rid // nq
                for i in range(nla):
                    conv[i][slots], rec[i][slots] = la[i][0][q], la[i][1][q]
                for i in range(nfa):
                    kc[i][slots], vc[i][slots] = fa[i][0][q], fa[i][1][q]
                bias[slots] = pbias[q]
                tok[slots], p_[slots], col[slots], slot[slots] = last_tok[q], last_pos[q], 0, W - 1
                sel[slots], rowid[slots], done[slots] = selm[k], rid, False
            elif free.size == n:
                break
            # one step (_Hybrid._pass's start, then the device, then _step's bookkeeping)
            bias[np.arange(n), slot] = 0.0
            oh = np.zeros((n, L), np.float32)
            oh[np.arange(n), slot] = 1.0
            with torch.no_grad():
                h = torch.from_numpy(self.embed[tok])
                cos, sin = self.rot(h, torch.from_numpy(p_)[:, None])
            c0 = col.clip(max=T - 1)
            want = ans_[rowid, c0] if forced else np.zeros(n, np.int64)
            ins = [self.embed[tok], cos.reshape(n, 1, -1).numpy(), sin.reshape(n, 1, -1).numpy(), bias, oh, sel, want,
                   *conv, *rec, *kc, *vc]
            for i, x in enumerate(ins):
                self.req.set_input_tensor(i, _tensor(np.ascontiguousarray(x)))
            self.req.infer()
            o = [self.req.get_output_tensor(i).data for i in range(4 + 2 * nla + 2 * nfa)]
            arg, top, got, lse = (np.asarray(x) for x in o[:4])
            states = o[4:]
            for i in range(nla):
                np.copyto(conv[i], states[i])
                np.copyto(rec[i], states[nla + i])
            for i in range(nfa):
                np.copyto(kc[i], states[2 * nla + i])
                np.copyto(vc[i], states[2 * nla + nfa + i])
            steps += 1
            if forced:
                nxt = want
                gap[rowid, c0] = top.astype(np.float32) - got.astype(np.float32)
                lp[rowid, c0] = got.astype(np.float32) - lse.astype(np.float32)
                fin = col + 1 >= alen_[rowid]
            else:
                nxt = arg.astype(np.int64)
                out[rowid, c0] = nxt
                fin = np.isin(nxt, eos) | (col + 1 >= cap)
            act = (~done).astype(np.int64)
            tok = nxt.copy()
            p_ += act
            col += act
            slot = np.minimum(col + (W - 1), L - 1)
            done |= fin
            rowid[done] = R
        self.steps = steps
        return out, gap, lp


def _tensor(x):
    import openvino as ov
    return ov.Tensor(x)


def cache_slots(rows, T: int) -> int:
    """es_hybrid._engine's L: the prompts' slots and T answer tokens, rounded up to 32."""
    W = max(len(r) for r in rows)
    return -(-(W + T - 1) // 32) * 32


def answer(eng: Engine, rt, fac, members, rows, T, sigma, rank):
    """Every member's greedy answers to every prompt on the device (es_hybrid.answer without loop_stop or caps)."""
    nq, M = len(rows), len(members)
    out, _, _ = eng.decode(rows, members, fac, sigma, rank, False, range(M * nq), T)
    answers = []
    for row in out[:M * nq].tolist():
        cut = next((i + 1 for i, t in enumerate(row) if t in rt.eos), len(row))
        answers.append(row[:cut])
    return [answers[k * nq:(k + 1) * nq] for k in range(M)]


def forced(eng: Engine, rt, fac, members, rows, answers, T, sigma, rank):
    """Teacher forcing on the device (es_hybrid.forced): (worst gap, mean log-probability), [members, prompts]."""
    nq, M = len(rows), len(members)
    R = M * nq
    ans, alen = np.zeros((R + 1, T), np.int64), np.zeros(R + 1, np.int64)
    for k, member in enumerate(answers):
        for q, a in enumerate(member):
            if len(a) > T:
                raise ValueError(f"an answer of {len(a)} tokens, past the {T} answering stops at")
            ans[k * nq + q, :len(a)] = a
            alen[k * nq + q] = len(a)
    _, gap, lp = eng.decode(rows, members, fac, sigma, rank, True, es_hybrid.longest_first(alen[:R].tolist()), T,
                            ans, alen)
    cols = np.arange(T)[None, :] < alen[:R, None]
    gaps = np.where(cols, gap[:R], 0.0).max(-1).reshape(M, nq)
    lps = (np.where(cols, lp[:R], 0.0).sum(-1) / np.maximum(alen[:R], 1)).reshape(M, nq)
    return torch.from_numpy(gaps.astype(np.float32)), torch.from_numpy(lps.astype(np.float32))
