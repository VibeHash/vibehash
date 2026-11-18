"""Fine-tuning System 1 decision models, like Laya: a bidirectional encoder that reads a question, its options and the
case in one pass, and a small decision head that gives each option a probability. A fine-tune of one is a fine-tune
job (finetune.py) like any other: units of Adam steps on LoRA adapters with A frozen, checkpoints of [B | m | v],
segments that checkers replay, results averaged into the next round's start. What differs is the model, the data and
the loss, all behind the spec (DecisionSpec) and the dataset (DecisionDataset), so the protocol code is the same.

**The model** is Laya's (convaiinnovations/laya, Apache-2.0): a ModernBERT encoder (pre-norm layers, LayerNorm
without biases, a fused query-key-value projection, rotary positions with one base for the global layers and another
for the sliding-window ones, every `every`-th layer global and the rest seeing `window` tokens either way, a gated GELU
MLP), then the decision head: the question type's embedding added to every position, `head_layers` standard
pre-norm transformer layers (PyTorch's nn.TransformerEncoderLayer: LayerNorm with biases, biased projections, ReLU),
and the scorer (LayerNorm, Linear, GELU, Linear to one number), read at each option's [MASK] marker. The options'
numbers, softmaxed over the question's own options, are the model's probabilities.

**The input** is Laya's layout, built by the poster with the model's own tokenizer (`prepare`), so the fine-tuned
model reads at inference exactly what it trained on:
    [CLS] <type> question: <instructions> [SEP] [MASK] option 0 [MASK] option 1 ... [SEP] <the case> [SEP]
Workers get token ids, the markers' positions and the target probabilities, never text.

**The loss** is cross-entropy against the target distribution over the options (Laya's soft cross-entropy, its "ce"
loss): a strictly proper scoring rule, so it rewards honest probabilities, which is what makes Laya's confidence worth
reading. Laya's default "rlcd" loss adds a policy-gradient term over noisy logits; its noise is a random draw each
step, which a replayed unit would have to draw identically, so it isn't here yet.

**What's trained**: LoRA on every linear layer by default, encoder and head (TARGETS). Laya's own recipe trains every
weight; the network trains adapters, whose average is exact (finetune.py), and which travel as megabytes, not the
model's 850 MB. The head's biases and norms stay as the base has them.

The model's PyTorch side, the ops a unit runs and their tracer, are decision_program.py and decision_tracer.py. A
fine-tune of a decision model is new to the chain, so only a chain whose genesis takes them (ledger.DECISION_GATE: a
local testnet, a test) can hold one, until the referee checks their ops (DESIGN.md §8, "Decision models").
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .finetune import (_BASE, _SHA, MAX_SEQ_LEN, MAX_VAL, OPTIMIZERS, SEQ_STEP, DataError, LoraSpec, TokenDataset,
                       _ARRAYS)

FAMILY = "laya"
# What LoRA can adapt, in their order: the encoder's four projections in each layer, the head's four in each of its
# layers, and the scorer's hidden layer. Every one by default, as near as adapters get to Laya's own full fine-tune.
ENCODER_TARGETS = ("qkv", "o", "wi", "wo")
HEAD_TARGETS = ("head_qkv", "head_o", "head_ff1", "head_ff2")
TARGETS = ENCODER_TARGETS + HEAD_TARGETS + ("scorer",)
DEFAULT_TARGETS = TARGETS
DIMS = ("family", "vocab", "hidden", "layers", "heads", "ffn", "window", "every", "head_layers")
QTYPES = {"choice": 0, "score": 1, "noul": 2}  # Laya's question types, as its type embedding numbers them
MAX_OPTIONS = 255  # Laya feeds its action head k / 255
MAX_HEAD_LAYERS = 8
MAX_BASE_PARAMS = 2_000_000_000  # laya-multilingual (mmBERT) and Laya (ModernBERT-large) are 0.3-0.4B
OPTION_TOKENS = 48  # Laya's cap on one option's text, in tokens (common.build_head)
DEFAULT_INSTRUCTIONS = "Classify the input into the correct category."  # Laya's, for a CSV of labelled rows
ENCODER_CONFIG = "encoder/config.json"
AGENT_FILE = "rl_agent_config.json"
TOKENIZER_FILE, TOKENIZER_CONFIG = "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"
# The small files a Laya checkpoint's manifest pins (hub.build_manifest's `extra`): the encoder's shape, the head's
# settings and the tokenizer, which every worker and the poster must read the same.
EXTRA_FILES = ("config.json", ENCODER_CONFIG, AGENT_FILE, TOKENIZER_FILE, TOKENIZER_CONFIG,
               "tokenizer/special_tokens_map.json")
UNIT_STEPS, SEGMENT_STEPS = 16, 4  # as finetune's: 4 segments a unit, 3 of them challenged
BATCH = 16  # items a step; Laya's own recipe steps on 64 (8 x 8 accumulated)
LR = 2e-4  # Adam on B; adapters take bigger steps than Laya's full fine-tune (2.5e-5 encoder, 1e-4 head)
EVAL_BATCH = 1  # held-back items scored one at a time: each is its own length
TOLERANCE = 1e-4  # finetune.TOLERANCE: a replay's distance from a claim, of the claim's size, for each of B, m and v


def is_decision_spec(d) -> bool:
    """Whether a parsed fine-tune spec is a decision model's: its model's family is FAMILY. Nodes from before decision
    models refuse it (finetune.LoraSpec.problems: an unknown family), so no spec they took changes meaning."""
    return isinstance(d, dict) and isinstance(d.get("model"), dict) and d["model"].get("family") == FAMILY


# --- the model's shape, without PyTorch -----------------------------------------------------------------------------

def _int(config: dict, key: str, default=None, lo: int = 1, hi: int = 1 << 20) -> int:
    v = config.get(key, default)
    if type(v) is not int or not lo <= v <= hi:
        raise ValueError(f"{key} must be a whole number from {lo} to {hi}, not {v!r}")
    return v


def _float(config: dict, key: str, default=None) -> float:
    v = config.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise ValueError(f"{key} must be a positive number, not {v!r}")
    return float(v)


@dataclass(frozen=True)
class DecisionArch:
    """What a Laya checkpoint's encoder/config.json and rl_agent_config.json say about its shape and arithmetic. Refuses
    what decision_program doesn't compute, rather than train a different model under the checkpoint's name."""
    vocab: int
    hidden: int
    layers: int
    heads: int
    ffn: int
    window: int  # a sliding layer sees tokens at most this far either way (ModernBERT's local_attention // 2)
    every: int  # every `every`-th layer, from layer 0, is global
    head_layers: int
    norm_eps: float = 1e-5
    rope_global: float = 160000.0
    rope_local: float = 10000.0
    max_positions: int = 8192
    max_len: int = 512  # Laya's budgets: a whole sequence, and the question half of it (common.build_sequence)
    head_max_len: int = 192
    head_eps: float = 1e-5  # nn.TransformerEncoderLayer's and nn.LayerNorm's default

    @classmethod
    def from_configs(cls, encoder: dict, agent: dict) -> "DecisionArch":
        if not isinstance(encoder, dict) or not isinstance(agent, dict):
            raise ValueError("the encoder's config.json and rl_agent_config.json must be JSON objects")
        if encoder.get("model_type") != "modernbert":
            raise ValueError(f"the encoder's model_type is {encoder.get('model_type')!r}; decision models here have a "
                             "ModernBERT encoder")
        problems = [why for bad, why in [
            (encoder.get("hidden_activation", "gelu") != "gelu", f"activation {encoder.get('hidden_activation')!r}"),
            (bool(encoder.get("attention_bias")), "attention biases"),
            (bool(encoder.get("mlp_bias")), "MLP biases"),
            (bool(encoder.get("norm_bias")), "norm biases"),
        ] if bad]
        if problems:
            raise ValueError(f"not supported yet: {', '.join(problems)}")
        hidden, heads, layers = _int(encoder, "hidden_size"), _int(encoder, "num_attention_heads"), \
            _int(encoder, "num_hidden_layers", hi=256)
        if hidden % heads or (hidden // heads) % 2:
            raise ValueError("hidden_size must be a multiple of the heads, and each head's width even")
        every = _int(encoder, "global_attn_every_n_layers", 1, hi=layers)
        types = encoder.get("layer_types")
        if types is not None:
            want = ["full_attention" if i % every == 0 else "sliding_attention" for i in range(layers)]
            if list(types) != want:
                raise ValueError("layer_types isn't every global_attn_every_n_layers-th layer global, the rest sliding")
        rope = encoder.get("rope_parameters")
        if isinstance(rope, dict) and ("full_attention" in rope or "sliding_attention" in rope):
            kinds = {rope.get(t, {}).get("rope_type", "default") for t in ("full_attention", "sliding_attention")}
            g, s = rope.get("full_attention", {}), rope.get("sliding_attention", {})
        else:
            kinds, g, s = {"default"}, {"rope_theta": encoder.get("global_rope_theta", 160000.0)}, \
                {"rope_theta": encoder.get("local_rope_theta", 10000.0)}
        if kinds != {"default"}:
            raise ValueError(f"rotary scaling {sorted(kinds - {'default'})[0]!r} isn't supported yet")
        head_layers = agent.get("head_layers", 2)
        if type(head_layers) is not int or not 0 <= head_layers <= MAX_HEAD_LAYERS:
            raise ValueError(f"head_layers must be 0-{MAX_HEAD_LAYERS}, not {head_layers!r}")
        local = _int(encoder, "local_attention", 128, hi=1 << 20)
        out = cls(vocab=_int(encoder, "vocab_size"), hidden=hidden, layers=layers, heads=heads,
                  ffn=_int(encoder, "intermediate_size"), window=local // 2, every=every, head_layers=head_layers,
                  norm_eps=_float(encoder, "norm_eps", 1e-5), rope_global=_float(g, "rope_theta"),
                  rope_local=_float(s, "rope_theta"), max_positions=_int(encoder, "max_position_embeddings", 8192),
                  max_len=_int(agent, "max_len", 512, lo=16), head_max_len=_int(agent, "head_max_len", 192, lo=16))
        if out.hidden % out.head_heads:
            raise ValueError("the head's width must be a multiple of its heads")
        return out

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def head_heads(self) -> int:
        """The decision head's attention heads: Laya's max(1, d // 64)."""
        return max(1, self.hidden // 64)

    def is_global(self, i: int) -> bool:
        return i % self.every == 0

    def theta(self, i: int) -> float:
        return self.rope_global if self.is_global(i) else self.rope_local

    def dims(self) -> dict:
        return {"family": FAMILY, "vocab": self.vocab, "hidden": self.hidden, "layers": self.layers,
                "heads": self.heads, "ffn": self.ffn, "window": self.window, "every": self.every,
                "head_layers": self.head_layers}

    def tensor_names(self) -> list[str]:
        """Every tensor training reads, in the checkpoint's names (the action head and temperatures aren't read)."""
        names = ["encoder.embeddings.tok_embeddings.weight", "encoder.embeddings.norm.weight"]
        for i in range(self.layers):
            p = f"encoder.layers.{i}."
            names += ([p + "attn_norm.weight"] if i else []) + [p + "attn.Wqkv.weight", p + "attn.Wo.weight",
                                                                 p + "mlp_norm.weight", p + "mlp.Wi.weight",
                                                                 p + "mlp.Wo.weight"]
        names += ["encoder.final_norm.weight", "type_emb.weight"]
        for j in range(self.head_layers):
            p = f"head.layers.{j}."
            names += [p + n for n in ("self_attn.in_proj_weight", "self_attn.in_proj_bias", "self_attn.out_proj.weight",
                                      "self_attn.out_proj.bias", "linear1.weight", "linear1.bias", "linear2.weight",
                                      "linear2.bias", "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias")]
        names += [f"scorer.{k}.{w}" for k in (0, 1, 3) for w in ("weight", "bias")]
        return names

    def check(self, seq_len: int) -> None:
        if seq_len > self.max_positions:
            raise ValueError(f"sequences of {seq_len} tokens are longer than the encoder's {self.max_positions}")


def arch_of(dims: dict) -> DecisionArch:
    """An arch with a spec's dims (no config: the rest at Laya's values), for sizing work without the model's files."""
    return DecisionArch(**{k: v for k, v in dims.items() if k != "family"})


# --- the spec -----------------------------------------------------------------------------------------------------

_SHAPES = {"qkv": ("3d", "d"), "o": ("d", "d"), "wi": ("2f", "d"), "wo": ("d", "f"), "head_qkv": ("3d", "d"),
           "head_o": ("d", "d"), "head_ff1": ("4d", "d"), "head_ff2": ("d", "4d"), "scorer": ("d", "d")}
_MODULES = {"qkv": "encoder.layers.{i}.attn.Wqkv", "o": "encoder.layers.{i}.attn.Wo", "wi": "encoder.layers.{i}.mlp.Wi",
            "wo": "encoder.layers.{i}.mlp.Wo", "head_qkv": "head.layers.{i}.self_attn.in_proj",
            "head_o": "head.layers.{i}.self_attn.out_proj", "head_ff1": "head.layers.{i}.linear1",
            "head_ff2": "head.layers.{i}.linear2", "scorer": "scorer.1"}


@dataclass(frozen=True)
class DecisionSpec(LoraSpec):
    """A decision model's fine-tune: LoraSpec's fields and protocol, with Laya's modules, shapes and sizes."""
    targets: tuple[str, ...] = DEFAULT_TARGETS
    seq_len: int = 512
    # Laya's (max_len, head_max_len) the items were laid out at: what the downloaded checkpoint's rl_agent_config.json
    # says, so inference lays a question out as training did. Workers never need it; it is pinned so everyone who
    # downloads the result gets the same.
    budgets: tuple[int, int] = (512, 192)

    kind = "decision"

    def __post_init__(self):
        if isinstance(self.dims, dict):
            object.__setattr__(self, "dims", tuple(sorted(self.dims.items())))
        if isinstance(self.budgets, list):
            object.__setattr__(self, "budgets", tuple(self.budgets))
        if not isinstance(self.targets, str):  # TARGETS order, so equal specs compare (and serialize) equal
            known = tuple(t for t in TARGETS if t in set(self.targets))
            object.__setattr__(self, "targets", known + tuple(sorted(set(self.targets) - set(TARGETS), key=str)))

    @property
    def arch(self) -> DecisionArch:
        return arch_of(self.model)

    def problems(self) -> list[str]:
        m, out = self.model, []
        if not isinstance(self.base, str) or not _BASE.match(self.base):
            out.append("base must be hf:org/name@<40-hex commit>")
        if not isinstance(self.manifest, str) or not _SHA.match(self.manifest):
            out.append("manifest must be a sha256")
        ok = set(m) == set(DIMS) and m.get("family") == FAMILY and all(
            type(m.get(k)) is int and 1 <= m[k] <= 1 << 20 for k in DIMS[1:-1]) \
            and type(m.get("head_layers")) is int and 0 <= m["head_layers"] <= MAX_HEAD_LAYERS
        if not ok:
            out.append(f"model must give {', '.join(DIMS)} (family {FAMILY}, head_layers 0-{MAX_HEAD_LAYERS})")
        elif m["hidden"] % m["heads"] or (m["hidden"] // m["heads"]) % 2 or m["hidden"] % max(1, m["hidden"] // 64) \
                or m["every"] > m["layers"]:
            out.append("the hidden width must be a multiple of the heads with an even head width, and every <= layers")
        elif self.base_params > MAX_BASE_PARAMS:
            out.append(f"decision models are at most {MAX_BASE_PARAMS / 1e9:g}B parameters for now; "
                       f"this one is {self.base_params / 1e9:.1f}B")
        if type(self.rank) is not int or not 1 <= self.rank <= 256:
            out.append("LoRA rank must be 1-256")
        if not 0 < self.alpha <= 4096:
            out.append("LoRA alpha must be in (0, 4096]")
        if not self.targets or not set(self.targets) <= set(TARGETS):
            out.append(f"LoRA targets must be some of {', '.join(TARGETS)}")
        elif ok and m["head_layers"] == 0 and set(self.targets) & set(HEAD_TARGETS):
            out.append("a model without head layers has no head projections to adapt")
        if type(self.seq_len) is not int or not 16 <= self.seq_len <= MAX_SEQ_LEN:
            out.append(f"seq_len must be 16-{MAX_SEQ_LEN}")
        b = self.budgets
        if not (isinstance(b, tuple) and len(b) == 2 and all(type(x) is int for x in b)
                and 16 <= b[1] <= b[0] <= MAX_SEQ_LEN and self.seq_len <= b[0]):
            out.append(f"budgets must be [max_len, head_max_len], 16 <= head_max_len <= max_len <= {MAX_SEQ_LEN}, "
                       "and hold the items")
        if self.optimizer not in OPTIMIZERS:
            out.append(f"optimizer must be one of {', '.join(OPTIMIZERS)}")
        if self.precision != "float32":
            out.append("fine-tunes train in float32")
        return out

    def to_json(self) -> dict:
        return {**super().to_json(), "budgets": list(self.budgets)}

    @classmethod
    def from_json(cls, d: dict, seed: int) -> "DecisionSpec":
        spec = super().from_json(d, seed)
        budgets = d["budgets"]
        if not isinstance(budgets, list):
            raise TypeError("budgets must be a list")
        object.__setattr__(spec, "budgets", tuple(budgets))
        return spec

    def shape(self, target: str) -> tuple[int, int]:
        """A projection's weight shape, [out, in]."""
        m = self.model
        size = {"d": m["hidden"], "2d": 2 * m["hidden"], "3d": 3 * m["hidden"], "4d": 4 * m["hidden"],
                "f": m["ffn"], "2f": 2 * m["ffn"]}
        return tuple(size[s] for s in _SHAPES[target])

    def modules(self) -> list[tuple[str, int, int]]:
        """Every adapted projection, (module name, out, in): the encoder's layer by layer, then the head's, then the
        scorer's, targets in TARGETS order. The order of the B blocks in the vector."""
        m, out = self.model, []
        for i in range(m["layers"]):
            out += [(_MODULES[t].format(i=i), *self.shape(t)) for t in ENCODER_TARGETS if t in self.targets]
        for j in range(m["head_layers"]):
            out += [(_MODULES[t].format(i=j), *self.shape(t)) for t in HEAD_TARGETS if t in self.targets]
        if "scorer" in self.targets:
            out.append((_MODULES["scorer"], *self.shape("scorer")))
        return out

    @staticmethod
    def weight_of(module: str) -> str:
        """A module's weight tensor in the checkpoint: nn.MultiheadAttention keeps its fused input projection as a bare
        parameter, in_proj_weight."""
        return module + "_weight" if module.endswith(".in_proj") else module + ".weight"

    @staticmethod
    def bias_of(module: str) -> str | None:
        """Its bias, or None: the encoder's projections have none, the head's and the scorer's do."""
        if module.startswith("encoder."):
            return None
        return module + "_bias" if module.endswith(".in_proj") else module + ".bias"

    @property
    def base_params(self) -> int:
        m = self.model
        d, f, L, H = m["hidden"], m["ffn"], m["layers"], m["head_layers"]
        encoder = m["vocab"] * d + d + L * (4 * d * d + 3 * d * f + 2 * d) - d + d
        head = H * (12 * d * d + 13 * d) + 3 * d + (d * d + 4 * d + 1)  # layers, type embedding, scorer
        return encoder + head

    def flops_per_step(self, batch_size: int) -> int:
        """Nominal FLOPs a step: every item counted at the full seq_len. Per token, as finetune's: the frozen matmuls
        forward, again when each layer is recomputed, and back to the inputs (6 a weight); attention's scores and mixing
        likewise; the adapters (6 a weight of A and B). The scorer reads only the markers: left out."""
        m, T = self.model, self.seq_len
        d, f = m["hidden"], m["ffn"]
        enc = m["layers"] * (6 * (4 * d * d + 3 * d * f) + 3 * 4 * T * d)
        head = m["head_layers"] * (6 * 12 * d * d + 3 * 4 * T * d)
        adapters = sum(out + inp for _, out, inp in self.modules()) * self.rank
        return (enc + head + 6 * adapters) * T * batch_size

    def peft_config(self) -> dict:
        """No PEFT format for these: the result is the merged checkpoint (export_merged). What describes the adapters."""
        alpha = int(self.alpha) if float(self.alpha).is_integer() else self.alpha
        return {"r": self.rank, "lora_alpha": alpha, "target_modules": list(self.targets)}


# --- the data -----------------------------------------------------------------------------------------------------

_DARRAYS = _ARRAYS + (("agent", np.uint8), ("train_target", np.float32), ("train_type", np.uint8),
                      ("val_target", np.float32), ("val_type", np.uint8))


@dataclass(frozen=True)
class DecisionDataset(TokenDataset):
    """Items in Laya's layout: token ids [n, length] (zero-padded on the right) and how long each is, as a
    TokenDataset's; `mask` marks each option's [MASK] marker, in option order (a row's options are its 1s); `target`
    [n, most options] is each item's target probability for each option (zeros past its own); `type` its question type
    (QTYPES). Plus the encoder's config.json (`config`) and rl_agent_config.json (`agent`) as bytes, which the job's
    manifest pins."""
    agent: bytes = b""
    train_target: np.ndarray | None = None
    train_type: np.ndarray | None = None
    val_target: np.ndarray | None = None
    val_type: np.ndarray | None = None

    kind = "decisions"

    def item(self, i: int, val: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """(ids, marker positions, target probabilities, question type) of one item."""
        ids, lens, mask, target, typ = (self.val_ids, self.val_len, self.val_mask, self.val_target, self.val_type) \
            if val else (self.train_ids, self.train_len, self.train_mask, self.train_target, self.train_type)
        n = int(lens[i])
        markers = np.flatnonzero(mask[i, :n])
        return ids[i, :n].astype(np.int64), markers.astype(np.int64), target[i, :len(markers)].astype(np.float32), \
            int(typ[i])

    def arch(self) -> DecisionArch:
        if "arch" not in self._parsed:
            self.check_shape()
        return self._parsed["arch"]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: getattr(self, name) for name, _ in _DARRAYS if name not in ("manifest", "config", "agent")}
        for name in ("manifest", "config", "agent"):
            arrays[name] = np.frombuffer(getattr(self, name), dtype=np.uint8)
        with open(path, "wb") as f:
            np.savez_compressed(f, **arrays)
        return path

    @classmethod
    def from_npz(cls, path, max_bytes: int | None = None) -> "DecisionDataset":
        """As TokenDataset.from_npz: nothing unpickled, and a file whose arrays would take more than `max_bytes` (default
        64 times its size, or 64 MB if that's more) refused before it's loaded."""
        import os

        limit = max(2**26, 64 * os.path.getsize(path)) if max_bytes is None else max_bytes
        need = 0
        try:
            with zipfile.ZipFile(path) as z:
                members = set(z.namelist())
                for name, target in _DARRAYS:
                    with z.open(name if name in members else f"{name}.npy") as m:
                        version = np.lib.format.read_magic(m)
                        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else \
                            np.lib.format.read_array_header_2_0
                        shape, _, dtype = read(m)
                    if dtype != np.dtype(target):
                        raise ValueError(f"{name} must be {np.dtype(target).name}, not {dtype}")
                    need += math.prod(shape) * dtype.itemsize
        except (zipfile.BadZipFile, KeyError) as e:
            raise ValueError(f"not a decision dataset: {e}") from e
        if need > limit:
            raise ValueError(f"the dataset would take {need / 2**20:,.0f} MB of memory once loaded, over the "
                             f"{limit / 2**20:,.0f} MB allowed for a file its size")
        with np.load(path, allow_pickle=False) as f:
            arrays = {name: np.ascontiguousarray(f[name]) for name, _ in _DARRAYS}
        d = cls(*(arrays[name] for name, _ in _ARRAYS[:6]), manifest=arrays["manifest"].tobytes(),
                config=arrays["config"].tobytes(), agent=arrays["agent"].tobytes(),
                train_target=arrays["train_target"], train_type=arrays["train_type"],
                val_target=arrays["val_target"], val_type=arrays["val_type"])
        d.check_shape()
        return d

    def check_shape(self) -> None:
        """Consistent on its own: lengths within the rows, ids not negative, 2 to MAX_OPTIONS markers a row, on real
        tokens and never at position 0, targets that are probabilities over a row's own options, known question
        types, and configs that describe a model this code trains."""
        from .hub import Manifest

        for ids, lens, mask, target, typ, what in (
                (self.train_ids, self.train_len, self.train_mask, self.train_target, self.train_type, "training"),
                (self.val_ids, self.val_len, self.val_mask, self.val_target, self.val_type, "validation")):
            if target is None or typ is None or ids.ndim != 2 or mask.shape != ids.shape or ids.shape[0] == 0 \
                    or lens.shape != (ids.shape[0],) or typ.shape != lens.shape or target.ndim != 2 \
                    or target.shape[0] != ids.shape[0]:
                raise ValueError(f"the {what} items' arrays don't match")
            if lens.min() < 2 or lens.max() > ids.shape[1] or ids.min() < 0 or mask.max() > 1:
                raise ValueError(f"the {what} items have impossible lengths or ids")
            beyond = np.arange(ids.shape[1])[None, :] >= lens[:, None]
            if mask[:, 0].any() or (mask.astype(bool) & beyond).any() or (ids[beyond] != 0).any():
                raise ValueError(f"the {what} items mark options where there are no tokens")
            k = mask.sum(axis=1).astype(np.int64)
            if k.min() < 2 or k.max() > min(MAX_OPTIONS, target.shape[1]):
                raise ValueError(f"every {what} item needs 2 to {MAX_OPTIONS} options, with a target for each")
            past = np.arange(target.shape[1])[None, :] >= k[:, None]
            if not np.all(np.isfinite(target)) or target.min() < 0 or (target[past] != 0).any() \
                    or np.abs(target.sum(axis=1) - 1).max() > 1e-3:
                raise ValueError(f"a {what} item's targets aren't probabilities over its own options")
            if typ.max() >= len(QTYPES):
                raise ValueError(f"a {what} item has an unknown question type")
        try:
            self._parsed["manifest"] = Manifest.from_bytes(self.manifest)
            self._parsed["config"] = json.loads(self.config)
            self._parsed["agent"] = json.loads(self.agent)
            self._parsed["arch"] = DecisionArch.from_configs(self._parsed["config"], self._parsed["agent"])
        except (ValueError, UnicodeDecodeError, RecursionError) as e:
            raise ValueError(f"the dataset's manifest, encoder config or rl_agent_config.json is malformed: {e}") \
                from None

    def check(self, spec) -> None:
        """The dataset suits this job: the manifest it pins, for the job's base, with these configs, of the shape the
        spec says; items fit the job's length; every id is in the vocabulary."""
        if not isinstance(spec, DecisionSpec):
            raise ValueError("a decision dataset is for a decision model's fine-tune")
        if "manifest" not in self._parsed:
            self.check_shape()
        manifest = self._parsed["manifest"]
        if hashlib.sha256(self.manifest).hexdigest() != spec.manifest:
            raise ValueError("the dataset's model manifest isn't the one the job pins")
        if (manifest.repo, manifest.commit) != (spec.repo, spec.commit):
            raise ValueError("the dataset's manifest is for another model")
        for path, raw in ((ENCODER_CONFIG, self.config), (AGENT_FILE, self.agent)):
            pinned = manifest.files.get(path)
            if pinned is None or hashlib.sha256(raw).hexdigest() != pinned["sha256"]:
                raise ValueError(f"the dataset's {path} isn't the one the manifest pins")
        dims = self._parsed["arch"].dims()
        if dims != spec.model:
            raise ValueError(f"the job's model shape doesn't match its configs ({dims})")
        if max(self.train_ids.shape[1], self.val_ids.shape[1]) > spec.seq_len:
            raise ValueError(f"the dataset's items are longer than the job's {spec.seq_len} tokens")
        if max(int(self.train_ids.max()), int(self.val_ids.max())) >= spec.model["vocab"]:
            raise ValueError("the dataset has token ids outside the model's vocabulary")


def is_decision_npz(names) -> bool:
    """Whether a dataset file's arrays are a decision dataset's (tasks.load_dataset tells kinds apart by them)."""
    return {"train_target", "agent"} <= set(names)


# --- Laya's layout, from text ---------------------------------------------------------------------------------------

class LayaTokenizer:
    """A Laya checkpoint's tokenizer (tokenizer/tokenizer.json, through the `tokenizers` package) and the sequence layout
    Laya builds (laya.common.build_head and build_sequence), from the files its manifest pins."""

    def __init__(self, tokenizer_json: bytes, tokenizer_config: dict):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise DataError("fine-tuning needs the `tokenizers` package: pip install \"vibehash[finetune]\"") from None
        self.tok = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
        names = {k: tokenizer_config.get(k) for k in ("cls_token", "sep_token", "mask_token")}
        names = {k: v.get("content") if isinstance(v, dict) else v for k, v in names.items()}
        ids = {k: self.tok.token_to_id(v) if isinstance(v, str) else None for k, v in names.items()}
        if None in ids.values():
            raise DataError("the model's tokenizer has no [CLS], [SEP] or [MASK] token, which Laya's layout needs")
        self.mask_token = names["mask_token"]
        self.cls, self.sep, self.mask = ids["cls_token"], ids["sep_token"], ids["mask_token"]

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def head(self, qtype: str, instructions: str, options: list[str], head_max_len: int):
        """(ids, marker positions, distinct options) of the question half: build_head's arithmetic."""
        head_ids = self.encode(f"{qtype} question: {instructions.replace(self.mask_token, ' ')}")
        opt_ids = [[self.mask] + self.encode(" " + o.replace(self.mask_token, " "))[:OPTION_TOKENS] for o in options]
        budget = head_max_len - sum(len(o) for o in opt_ids)
        if budget < 16:
            per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            budget = head_max_len - sum(len(o) for o in opt_ids)
        ids = [self.cls] + head_ids[:max(8, budget)] + [self.sep]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids += o
        ids.append(self.sep)
        return ids, markers, len({tuple(o) for o in opt_ids})

    def sequence(self, head: list[int], markers: list[int], text: str, max_len: int) -> tuple[list[int], list[int]]:
        """The whole sequence, the case after the question: build_sequence's, with the case cut to the room left."""
        room = max(0, max_len - len(head) - 1)
        state = self.encode(text.replace(self.mask_token, " "))[:max_len][:room]
        ids = (head + state + [self.sep])[:max_len]
        return ids, [m for m in markers if m < max_len]


def options_of(labels, descriptions: dict | None = None) -> list[str]:
    """Option texts for a choice question, in label order: Laya's render_options ("label: description", or the label
    alone when it has no description)."""
    descriptions = descriptions or {}
    out = []
    for k in labels:
        v = descriptions.get(k)
        if v is None or v == "":
            out.append(str(k))
        else:
            out.append(f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(', ', ': '))}")
    return out


def read_rows(path: str | Path, text_column: str = "text", label_column: str = "label") -> list[tuple[str, str]]:
    """(text, label) rows from a CSV file with a header, or JSON lines of {"text": ..., "label": ...}. Labels are text;
    a row with no text or no label is an error that says where."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig")
    rows = []
    if path.suffix.lower() == ".csv":
        reader = csv.DictReader(io.StringIO(raw))
        if not reader.fieldnames:
            raise DataError(f"{path.name} is empty")
        for col in (text_column, label_column):
            if col not in reader.fieldnames:
                raise DataError(f"{path.name} has no `{col}` column (it has {', '.join(reader.fieldnames)})")
        for n, r in enumerate(reader, 2):
            rows.append((r.get(text_column) or "", (r.get(label_column) or "").strip(), f"line {n}"))
    else:
        for n, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError as e:
                raise DataError(f"line {n} of {path.name} isn't JSON: {e}") from None
            if not isinstance(r, dict):
                raise DataError(f"line {n} of {path.name} isn't a {{\"text\": ..., \"label\": ...}} object")
            label = r.get(label_column)
            rows.append((r.get(text_column) if isinstance(r.get(text_column), str) else "",
                         str(label).strip() if isinstance(label, (str, int)) and not isinstance(label, bool) else "",
                         f"line {n}"))
    out = []
    for text, label, at in rows:
        if not text.strip():
            raise DataError(f"{at.capitalize()} of {path.name} has no `{text_column}`")
        if not label:
            raise DataError(f"{at.capitalize()} of {path.name} has no `{label_column}`")
        out.append((text, label))
    if not out:
        raise DataError(f"{path.name} has no rows")
    return out


@dataclass
class Prepared:
    data: DecisionDataset
    rows: int  # in the file
    labels: list[str]  # the options, in order
    cut: int  # cases cut short to fit max_len
    seq_len: int
    head_tokens: int  # the question half's length
    max_len: int  # Laya's budgets the items were laid out at
    head_max_len: int


def prepare(path: str | Path, tokenizer: LayaTokenizer, manifest: bytes, config: bytes, agent: bytes, *,
            instructions: str = DEFAULT_INSTRUCTIONS, descriptions: dict | None = None, labels: list | None = None,
            text_column: str = "text", label_column: str = "label", seed: bytes = b"", max_len: int | None = None,
            head_max_len: int | None = None) -> Prepared:
    """Labelled rows as one choice question over every label (in order of first appearance, unless `labels` gives the
    order), in Laya's layout: hold back a tenth (at most MAX_VAL items) to measure progress, and size the items to the
    longest (a multiple of SEQ_STEP).

    The budgets are the checkpoint's own (rl_agent_config.json's max_len and head_max_len) unless given. When the
    options don't fit the question budget whole, Laya cuts every option to the same few tokens, and many labels can
    come out the same (Banking77's 77 at Laya's 192: four pairs). Then, unless `head_max_len` was given, the budget is
    raised to the next multiple of SEQ_STEP that holds them whole, as laya-banking77 did (192 to 512)."""
    arch = DecisionArch.from_configs(json.loads(config), json.loads(agent))
    max_len = max_len or arch.max_len
    if not 32 <= max_len <= min(MAX_SEQ_LEN, arch.max_positions):
        raise DataError(f"max_len must be 32 to {min(MAX_SEQ_LEN, arch.max_positions)} tokens")
    rows = read_rows(path, text_column, label_column)
    seen = list(dict.fromkeys(label for _, label in rows))
    labels = [str(x) for x in labels] if labels else seen
    unknown = sorted(set(seen) - set(labels))
    if unknown:
        raise DataError(f"{Path(path).name} has labels the label list doesn't: {', '.join(unknown[:5])}")
    if not 2 <= len(labels) <= MAX_OPTIONS:
        raise DataError(f"a decision needs 2 to {MAX_OPTIONS} labels; {Path(path).name} has {len(labels)}")
    options = options_of(labels, descriptions)
    if head_max_len is None:
        head_max_len = min(arch.head_max_len, max_len)
        whole = sum(1 + len(tokenizer.encode(" " + o.replace(tokenizer.mask_token, " "))[:OPTION_TOKENS]) for o in options)
        if head_max_len - whole < 16:  # build_head would cut every option short
            head_max_len = min(max_len, -(-(whole + 16) // SEQ_STEP) * SEQ_STEP)
    if not 16 <= head_max_len <= max_len:
        raise DataError(f"the question budget must be 16 to {max_len} tokens")
    head, markers, distinct = tokenizer.head("choice", instructions, options, head_max_len)
    if distinct < len(labels):
        raise DataError(f"{len(labels) - distinct} of the {len(labels)} options come out the same once Laya cuts them to "
                        f"fit a {head_max_len}-token question budget: give the labels shorter or more distinct names, "
                        "or a longer --max-len")
    if markers[-1] >= max_len - 1:
        raise DataError(f"the question and its {len(labels)} options take {len(head)} tokens, which leaves no room for "
                        f"the text in {max_len}: give a longer --max-len")
    index = {label: i for i, label in enumerate(labels)}
    items, cut = [], 0
    for text, label in rows:
        ids, ms = tokenizer.sequence(head, markers, text, max_len)
        cut += len(tokenizer.encode(text)) > max_len - len(head) - 1
        items.append((ids, ms, index[label]))
    order = np.random.default_rng(int.from_bytes(hashlib.sha256(seed or Path(path).read_bytes()).digest()[:8], "big")
                                  ).permutation(len(items))
    if len(items) < 2:
        raise DataError(f"{Path(path).name} has {len(items)} row; it takes at least 2 (one is held back)")
    n_val = max(1, min(len(items) // 10, MAX_VAL))
    seq_len = min(max_len, -(-max(len(ids) for ids, _, _ in items) // SEQ_STEP) * SEQ_STEP)
    k = len(labels)

    def pack(which):
        ids = np.zeros((len(which), seq_len), dtype=np.int32)
        mask = np.zeros((len(which), seq_len), dtype=np.uint8)
        lens = np.zeros(len(which), dtype=np.int32)
        target = np.zeros((len(which), k), dtype=np.float32)
        for row, j in enumerate(which):
            s, ms, y = items[j]
            ids[row, :len(s)], lens[row] = s, len(s)
            mask[row, ms] = 1
            target[row, y] = 1.0
        return ids, lens, mask, target, np.zeros(len(which), dtype=np.uint8)

    val, train = pack(order[:n_val]), pack(order[n_val:])
    data = DecisionDataset(*train[:3], *val[:3], manifest=manifest, config=config, agent=agent,
                           train_target=train[3], train_type=train[4], val_target=val[3], val_type=val[4])
    data.check_shape()
    return Prepared(data, len(rows), labels, cut, seq_len, len(head), max_len, head_max_len)


# --- running it (PyTorch, through finetune's runtime) ---------------------------------------------------------------

def memory_mb(spec: DecisionSpec, batch_size: int, weight_bytes: int) -> int:
    """GPU memory a worker needs: the weights as stored, each layer's input kept for the backward (one item at a time),
    the adapters' state four times over, the CUDA context, and the largest layer's moment (its recomputed activations,
    the attention scores, and its biggest matrix cast to float32)."""
    m, T = spec.model, spec.seq_len
    d, f = m["hidden"], m["ffn"]
    kept = (m["layers"] + m["head_layers"] + 1) * T * d * 4
    layer = (3 * max(m["heads"], max(1, d // 64)) * T * T + 12 * T * max(2 * f, 4 * d) + 4 * d * max(2 * f, 4 * d)) * 4
    total = weight_bytes + kept + 4 * spec.n_params * 4 + layer + 700 * 2**20
    return int(math.ceil(total / 2**20 / 256) * 256)


def evaluate(spec: DecisionSpec, data: DecisionDataset, theta: np.ndarray, backend=None) -> tuple[float, float]:
    """(mean log loss, share of items whose most likely option is the target's) over the held-back items, with the
    adapters in `theta`. For people following the job, not part of any check."""
    probs = probabilities(spec, data, theta, backend)
    loss = right = 0.0
    for i, p in enumerate(probs):
        _, _, t, _ = data.item(i, val=True)
        loss -= float(np.sum(t * np.log(np.maximum(p, 1e-12))))
        right += float(np.argmax(p) == np.argmax(t))
    return loss / len(probs), right / len(probs)


def probabilities(spec: DecisionSpec, data: DecisionDataset, theta: np.ndarray, backend=None) -> list[np.ndarray]:
    """Each held-back item's probabilities over its options (temperature 1), computed by the ops training runs."""
    import torch

    from . import finetune
    from .decision_program import DecisionProgram
    from .decision_tracer import DecisionExecutor

    rt = finetune.runtime(spec, data, finetune.device_of(backend))
    prog = DecisionProgram(spec, data, 1e-4, 1, 0)
    out = []
    with rt.busy, torch.no_grad():
        from .compute import pin_precision
        pin_precision()
        st0 = {k: torch.tensor(v, device=rt.device) for k, v in _b_only(spec, theta).items()}
        ex = DecisionExecutor(rt, spec, prog)
        for i in range(len(data.val_len)):
            st = dict(st0)
            node = prog.forward(i, val=True)
            ex.run_atoms(node, 0, len(node.atoms), st, (node.label,))
            z = st["s.z"].reshape(-1).double()
            out.append(torch.softmax(z, dim=0).cpu().numpy())
    return out


def _b_only(spec, theta) -> dict:
    from .program import split_theta
    return {k: v for k, v in split_theta(spec, theta).items() if k.startswith("B.")}


def calibration(probs: list[np.ndarray], targets: list[np.ndarray], bins: int = 15) -> dict:
    """How the probabilities measure up on items with known answers: accuracy, mean log loss, Brier score, and the
    expected calibration error over `bins` equal-width bins of the top probability (Laya's ece_score)."""
    conf = np.array([float(p.max()) for p in probs])
    right = np.array([float(np.argmax(p) == np.argmax(t)) for p, t in zip(probs, targets)])
    loss = float(np.mean([-np.sum(t * np.log(np.maximum(p, 1e-12))) for p, t in zip(probs, targets)]))
    brier = float(np.mean([np.sum((p - t) ** 2) for p, t in zip(probs, targets)]))
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (conf > lo) & (conf <= hi)
        if inside.any():
            ece += inside.mean() * abs(conf[inside].mean() - right[inside].mean())
    return {"items": len(probs), "accuracy": float(right.mean()), "log_loss": loss, "brier": brier,
            "ece": float(ece), "mean_confidence": float(conf.mean())}


# --- the result: a Laya checkpoint with the adapters merged in -----------------------------------------------------

def export_merged(spec: DecisionSpec, theta: np.ndarray, manifest, out: str | Path, about: dict | None = None, *,
                  float32: bool = False, read_file=None, progress=None, temperature: float = 1.0) -> dict:
    """The base checkpoint with the adapters folded into its weights (W + alpha/rank·B·A), in Laya's own layout, so
    `laya.Agent(folder)` loads it as it loads the base. NumPy only (finetune.export_merged does the merging).

    The base's fitted temperatures (rl_agent_config.json's `temperature` and `temperature_by_options`, and the
    checkpoint's `temperature` tensor) were fitted to the base's logits, not the fine-tune's, so they're set to
    `temperature` (1: the probabilities the job trained and scored)."""
    from . import finetune

    def edit(path: str, data: bytes) -> bytes:
        if path != AGENT_FILE:
            return data
        cfg = json.loads(data)
        cfg["max_len"], cfg["head_max_len"] = spec.budgets  # the layout the job trained on
        cfg["temperature"] = [float(temperature)] * len(QTYPES)
        cfg["temperature_by_options"] = {}
        cfg["fine_tuned"] = True
        cfg.setdefault("training", {})["vibehash"] = {k: v for k, v in (about or {}).items() if v is not None}
        return (json.dumps(cfg, indent=2) + "\n").encode()

    def tensor(name: str, values: np.ndarray) -> np.ndarray:
        return np.full_like(values, temperature) if name == "temperature" else values
    return finetune.export_merged(spec, theta, manifest, out, about, float32=float32, read_file=read_file,
                                  progress=progress, edit_file=edit, edit_tensor=tensor,
                                  readme=lambda info, folder: _readme(spec, info, folder))


def _readme(spec: DecisionSpec, info: dict, folder: str) -> str:
    about = {k: v for k, v in info.items() if k not in ("base", "manifest", "rank", "alpha", "targets", "dtype",
                                                         "rounding", "trained", "projections changed", "job_id")}
    lines = [f"# {spec.repo}, fine-tuned on Vibe Hash", "",
             f"A Laya decision model: the base checkpoint with job `{info.get('job_id', '?')}`'s LoRA adapters merged "
             f"into its weights (W + {spec.alpha:g}/{spec.rank}·B·A on {info['projections changed']} projections), "
             "in the base's own layout.", "",
             f"- Base model: [{spec.repo}](https://huggingface.co/{spec.repo}) at commit `{spec.commit}`; its licence "
             "applies to this model too",
             f"- Adapters: LoRA rank {spec.rank}, alpha {spec.alpha:g}, on {', '.join(spec.targets)}",
             "- Temperatures: reset to 1, the probabilities the job trained and was scored on (the base's were fitted "
             "to the base)"]
    lines += [f"- {k.replace('_', ' ').capitalize()}: {v}" for k, v in about.items() if v is not None]
    lines += ["", "```python", "import laya", "", f'agent = laya.Agent("{folder}")', "```", ""]
    return "\n".join(lines)


# --- the poster's side ------------------------------------------------------------------------------------------------

def is_decision_repo(listing) -> bool:
    """Whether a Hugging Face repository holds a Laya checkpoint: rl_agent_config.json and the encoder's config beside
    model.safetensors."""
    return {AGENT_FILE, ENCODER_CONFIG, "model.safetensors"} <= set(listing)


@dataclass
class Plan:
    """A decision fine-tune worked out but not posted."""
    job: object  # tasks.TrainingJob
    prepared: Prepared
    weight_bytes: int
    rounds_per_epoch: int = 0
    workers: int | None = None
    licence: str | None = None

    @property
    def rounds(self) -> int:
        return self.job.budget // self.job.fee_per_round if self.job.fee_per_round else 0

    def describe(self, data_name: str) -> str:
        job, spec, p = self.job, self.job.spec, self.prepared
        m = spec.model

        def vibe(syn):
            return f"{syn / 1e8:,.8g} VIBE"
        epochs = f" ({self.rounds / self.rounds_per_epoch:.1f} epochs)" if self.rounds_per_epoch and self.rounds else ""
        seen = (f" · {self.workers} worker{'s' * (self.workers != 1)} took part in the last rounds"
                if self.workers is not None else "")
        shown = ", ".join(p.labels[:6]) + (f" and {len(p.labels) - 6} more" if len(p.labels) > 6 else "")
        return "\n".join([
            f"Model     {spec.repo} @ {spec.commit[:12]} (pinned)",
            f"          a Laya decision model: ModernBERT encoder, {m['layers']} layers of width {m['hidden']}, and a "
            f"{m['head_layers']}-layer decision head · {self.weight_bytes / 1e9:.1f} GB of weights · licence "
            f"{self.licence or 'not stated'}",
            f"Data      {data_name}: {p.rows:,} rows, one choice over {len(p.labels)} labels ({shown})",
            f"          {len(p.data.train_len):,} to train, {len(p.data.val_len):,} held back · the question takes "
            f"{p.head_tokens} tokens (budget {p.head_max_len} of {p.max_len}), items up to {p.seq_len}"
            + (f" · {p.cut} texts cut short" if p.cut else ""),
            f"Training  LoRA rank {spec.rank} (alpha {spec.alpha:g}) on {', '.join(spec.targets)}: "
            f"{spec.n_adapter:,} trained weights, A frozen; cross-entropy over the options",
            f"          Adam at {job.lr:g}, {job.batch_size} items a step, {job.steps_per_unit} steps a unit, "
            f"{job.units_per_round} units a round: one epoch is about {self.rounds_per_epoch:,} rounds",
            f"Workers   a GPU with at least {job.memory_mb / 1024:.1f} GB, and PyTorch; each fetches the weights from "
            f"Hugging Face once{seen}",
            f"Cost      {vibe(job.fee_per_round)} a round · a budget of {vibe(job.budget)} buys {self.rounds:,} "
            f"rounds{epochs}, paid only for work that passes its checks",
        ])


def plan(data_path: str | Path, ref: str, *, hub=None, job_id: str = "decision", budget: int = 0, fee: int = 0,
         rank: int = 16, targets=DEFAULT_TARGETS, lr: float | None = None, units_per_round: int = 8,
         instructions: str = DEFAULT_INSTRUCTIONS, descriptions: dict | None = None, label_column: str = "label",
         text_column: str = "text", max_len: int | None = None, ask=None, progress=None,
         licence: str | None = None) -> Plan:
    """Work out a fine-tune of the Laya checkpoint `ref` (org/name[@revision]) on labelled rows: pin the commit, check the
    model is one this code trains, get its manifest (cached, or built by streaming the weights once, if `ask` agrees),
    lay the rows out with its own tokenizer, and size the job. The data source's hash is a placeholder until the dataset
    is saved (the caller does that)."""
    from . import hub as hubmod
    from .tasks import TrainingJob

    hub = hub or hubmod.Hub()
    repo, commit = hub.resolve(ref)
    listing = hub.list_files(repo, commit)
    if not is_decision_repo(listing):
        raise DataError(f"{repo} isn't a Laya checkpoint (it has no {AGENT_FILE} and {ENCODER_CONFIG})")
    encoder, agent = hub.get_file(repo, commit, ENCODER_CONFIG), hub.get_file(repo, commit, AGENT_FILE)
    try:
        arch = DecisionArch.from_configs(json.loads(encoder), json.loads(agent))
    except ValueError as e:
        raise DataError(f"{repo} can't be fine-tuned here yet: {e}") from None
    mbytes = model_manifest(repo, commit, hub, listing, ask=ask, progress=progress)
    manifest = hubmod.Manifest.from_bytes(mbytes)
    for path, raw in ((ENCODER_CONFIG, encoder), (AGENT_FILE, agent)):
        if hashlib.sha256(raw).hexdigest() != manifest.files.get(path, {}).get("sha256"):
            raise DataError(f"{path} isn't the one its manifest pins; try again")
    files = {p: hubmod.fetch_file(manifest, p, hub=hub) for p in (TOKENIZER_FILE, TOKENIZER_CONFIG)
             if p in manifest.files}
    if TOKENIZER_FILE not in files:
        raise DataError(f"{repo} has no {TOKENIZER_FILE}, which laying the rows out needs")
    tokenizer = LayaTokenizer(files[TOKENIZER_FILE], json.loads(files.get(TOKENIZER_CONFIG, b"{}")))
    prepared = prepare(data_path, tokenizer, mbytes, encoder, agent, instructions=instructions,
                       descriptions=descriptions, labels=list(descriptions) if descriptions else None,
                       text_column=text_column, label_column=label_column, max_len=max_len)
    spec = DecisionSpec(base=f"hf:{repo}@{commit}", manifest=hashlib.sha256(mbytes).hexdigest(), dims=arch.dims(),
                        rank=rank, alpha=2.0 * rank, targets=tuple(targets), seq_len=prepared.seq_len, seed=11,
                        budgets=(prepared.max_len, prepared.head_max_len))
    weight_bytes = manifest.bytes_for(n for n in arch.tensor_names() if n in manifest.tensors)
    job = TrainingJob(job_id, "", budget=budget, fee_per_round=fee, spec=spec,
                      data_source={"kind": "tokens", "sha256": "0" * 64}, init_seed=11, lr=lr or LR,
                      batch_size=BATCH, steps_per_unit=UNIT_STEPS, checkpoint_every=SEGMENT_STEPS,
                      units_per_round=units_per_round, memory_mb=memory_mb(spec, BATCH, weight_bytes),
                      tolerance=TOLERANCE, max_update_ratio=1.0)
    per_round = units_per_round * UNIT_STEPS * BATCH
    return Plan(job, prepared, weight_bytes, max(1, math.ceil(len(prepared.data.train_len) / per_round)),
                licence=licence)


def model_manifest(repo: str, commit: str, hub, listing, ask=None, progress=None) -> bytes:
    """finetune.model_manifest for a Laya checkpoint: the same cache, with the small files Laya needs pinned too."""
    from . import finetune
    from . import hub as hubmod

    path = finetune.manifest_path(repo, commit)
    if path.exists():
        try:
            m = hubmod.Manifest.from_bytes(path.read_bytes())
            if (m.repo, m.commit) == (repo, commit) and {ENCODER_CONFIG, AGENT_FILE} <= set(m.files):
                return m.to_bytes()
        except hubmod.IntegrityError:
            pass
    total = listing[hubmod.SINGLE_FILE].size
    if ask is not None and not ask(total):
        raise DataError("stopped: a model's weights are checked once before a job can pin them")
    m = hubmod.build_manifest(repo, commit, hub=hub, listing=listing, extra=EXTRA_FILES, progress=progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(m.to_bytes())
    tmp.replace(path)
    return m.to_bytes()


def read_descriptions(path: str | Path) -> dict:
    """--labels: a JSON object of {label: description}, in option order."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        raise DataError(f"can't read the labels file {path}: {e}") from None
    if not isinstance(d, dict) or not d or not all(isinstance(k, str) for k in d):
        raise DataError(f"{path} must be a JSON object of {{\"label\": \"description\", ...}}")
    return d
