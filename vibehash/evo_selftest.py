"""Can this device check its own Evo answers? (docs/GPUS.md, risk 13; docs/CROSS-HARDWARE.md, F4)

The GPU self-test (compute.self_test) trains a float32 MLP: a card can pass it and still get Evo wrong, in its bf16
matmuls (a P100 report), its graph capture (ROCm), or kernels its PyTorch build doesn't have (the RX 6700/6600). An
Evo checker teacher-forces answers through the engine that wrote them, so on one device a worker's own answers
checked by itself give gaps of exactly 0, and a member claiming its twin's answers gives more. `check(device)` runs
that on a tiny random Qwen3.5 (three Gated DeltaNet layers and one full attention, the 0.8B's kinds of layer), with
the network's own code: es_client.runtime2, es_fast.answer_members (CUDA graphs on CUDA and ROCm) and gaps_members.

A device whose own answers don't come back at 0 would have its honest work rejected by every checker like it, and
reject theirs, so the worker doesn't take Evo work on it (es_client.Es2Work.why_not) and `vibehash selftest` says so.
It takes a few seconds, once a device and process.
"""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
_done: dict[str, dict] = {}
_lock = threading.Lock()
RETRY_S = 600.0  # how long a self-check that errored stands before it's tried again


def tiny_qwen35(folder: Path, seed: int = 0) -> Path:
    """A random four-layer Qwen3.5 text model (three Gated DeltaNet layers, one gated full attention; twice as many
    value heads as key heads, as the 4B has) with a byte-level tokenizer and a chat template, saved as Hugging Face
    saves models."""
    import json

    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens(specials)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = TEMPLATE
    folder.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(folder)
    ids = {s: tok.convert_tokens_to_ids(s) for s in specials}
    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(
        vocab_size=len(vocab) + len(specials), hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, max_position_embeddings=2048, tie_word_embeddings=True,
        eos_token_id=ids["<|im_end|>"], pad_token_id=ids["<|endoftext|>"])
    model = Qwen3_5ForCausalLM(config)
    with torch.no_grad():  # norms away from their start, so the model isn't flat
        for name, p in model.named_parameters():
            if "norm" in name:
                p.add_(torch.randn_like(p) * 0.3)
    model.save_pretrained(folder)
    (folder / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": [ids["<|im_end|>"]], "pad_token_id": ids["<|endoftext|>"]}))
    return folder


def check(device: str) -> dict:
    """{"ok", "worst": the worst gap of the device's own answers checked by itself (0 when ok), "swap": the smallest
    worst gap of a member claiming its twin's answers where they differ (None when the twins answered alike), "seconds"}.
    Raises what the device raises (a missing kernel, a failed graph capture)."""
    from . import es2_rounds, es_fast, es_local
    from .es_client import prompt_rows, round_factors, runtime2

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        found = es_local.find(str(tiny_qwen35(Path(d) / "tiny35", seed=3)))
        es_local.register(found)
        spec = es_local.spec2_for(found, "arithmetic", pairs=4, pairs_per_unit=2, problems=5, max_tokens=10,
                                  sigma=0.05, check=2, margin=1e-3, lp_tol=0.02)
        rt = runtime2(spec, device)
        seed = 777
        rows = prompt_rows(rt, es2_rounds.items(spec, seed))
        fac = round_factors(rt, spec, seed)
        members = [(j, s) for j in range(spec.pairs) for s in (1, -1)]
        got = es_fast.answer_members(rt, rt.linears, fac, members, rows, spec.max_tokens, spec.sigma)
        own = es_fast.gaps_members(rt, rt.linears, fac, members, rows, got, spec.sigma, max_tokens=spec.max_tokens)
        worst = max(float(g.max()) for g in own if g.numel())
        swapped = [k for k in range(len(members)) if got[k] != got[k ^ 1]]
        swap = None
        if swapped:
            swaps = es_fast.gaps_members(rt, rt.linears, fac, [members[k] for k in swapped], rows,
                                         [got[k ^ 1] for k in swapped], spec.sigma, max_tokens=spec.max_tokens)
            swap = min(float(g.max()) for g in swaps)
    return {"ok": worst == 0.0 and (swap is None or swap > 0.0), "worst": worst, "swap": swap,
            "seconds": time.perf_counter() - t0}


def cached(device: str) -> dict:
    """check(device), once a device and process; an error is {"ok": False, "error": "..."}. A result is kept for
    good, an error only for RETRY_S: it may be passing (the card briefly full while another program loads)."""
    with _lock:
        got = _done.get(device)
        if got is None or ("error" in got and time.monotonic() - got["at"] >= RETRY_S):
            try:
                got = check(device)
            except Exception as e:  # noqa: BLE001 - whatever stops it, the device can't mine Evo for now
                got = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "at": time.monotonic()}
            _done[device] = got
        return got


def problem(device: str) -> str | None:
    """Why `device` shouldn't take Evo work, or None (cached)."""
    got = cached(device)
    return None if got["ok"] else problem_of(device, got)


def problem_of(device: str, got: dict) -> str:
    """What a failed check(device) means, in words."""
    if "error" in got:
        return f"Evo's self-check on {device} failed ({got['error']})"
    if got["worst"] != 0.0:
        return (f"Evo's self-check on {device}: its own answers, checked by itself, came back {got['worst']:.3g} below "
                "the greedy choice instead of exactly 0, so checkers like it would reject its honest work. Update the "
                "graphics driver or PyTorch (`vibehash install-gpu`)")
    return f"Evo's self-check on {device}: a member claiming its twin's answers wasn't caught"
