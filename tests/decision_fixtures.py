"""A tiny Laya-shaped checkpoint for decision-model tests, with no network: a ModernBERT encoder and a decision head
of random weights in f16 (as Laya stores them), laid out as the Hugging Face repository convaiinnovations/laya is
(config.json, encoder/config.json, rl_agent_config.json, model.safetensors, tokenizer/), with a word-level tokenizer,
its hub.Manifest, and the tensors put straight into the tensor cache under $VIBEHASH_HOME."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from vibehash import decision, hub

ENCODER = {"architectures": ["ModernBertForMaskedLM"], "model_type": "modernbert", "vocab_size": 64, "hidden_size": 32,
           "num_hidden_layers": 4, "num_attention_heads": 4, "intermediate_size": 24, "global_attn_every_n_layers": 3,
           "local_attention": 4, "max_position_embeddings": 256, "norm_eps": 1e-5, "norm_bias": False,
           "attention_bias": False, "mlp_bias": False, "hidden_activation": "gelu", "pad_token_id": 3,
           "cls_token_id": 1, "sep_token_id": 2, "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": True,
           "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
           "rope_parameters": {"full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                               "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}}
AGENT = {"encoder": "answerdotai/ModernBERT-large", "head_layers": 2, "max_len": 48, "head_max_len": 24,
         "temperature": [1.5, 1.2, 1.9], "temperature_by_options": {"choice:2": 1.9}, "act_costs": {"escalate": 0.5}}
WORDS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "choice", "question:", "classify", "the", "input", "into",
         "correct", "category.", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
         "red", "green", "blue", "x", "y", "z", "cat", "dog", "bird", "fish"]


def tokenizer_json() -> bytes:
    """A word-level tokenizers.Tokenizer: words split on whitespace, specials as added tokens."""
    vocab = {w: i for i, w in enumerate(WORDS)}
    specials = [{"id": vocab[t], "content": t, "single_word": False, "lstrip": False, "rstrip": False,
                 "normalized": False, "special": True} for t in ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]")]
    tok = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": specials, "normalizer": None,
           "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": None, "decoder": None,
           "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}}
    return json.dumps(tok).encode()


TOKENIZER_CONFIG = {"cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]", "pad_token": "[PAD]",
                    "unk_token": "[UNK]", "tokenizer_class": "PreTrainedTokenizerFast"}


def f16(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype="<f2").tobytes()


def weights(encoder: dict, agent: dict, seed: int = 0) -> dict[str, tuple[str, list[int], bytes]]:
    """Random weights shaped as the configs make them, at a pretrained model's scale: matrices N(0, 1/in), norms near
    1, biases small; f16, keyed by the checkpoint's names. The action head and temperatures too, which training doesn't
    read but a checkpoint has."""
    from vibehash.decision_tracer import shapes

    arch = decision.DecisionArch.from_configs(encoder, agent)
    rng = np.random.default_rng(seed)
    out = {}
    for name, shape in shapes(arch).items():
        if name == "encoder.layers.0.attn_norm.weight":
            continue
        if name.endswith("norm.weight") or ".norm" in name and name.endswith("weight") or name == "scorer.0.weight":
            a = rng.normal(1.0, 0.1, shape)
        elif name.endswith("bias"):
            a = rng.normal(0.0, 0.1, shape)
        else:
            a = rng.normal(0.0, 1 / np.sqrt(shape[-1]), shape)
        out[name] = ("F16", list(shape), f16(a))
    d = arch.hidden
    out["act_head.0.weight"] = ("F16", [8, d + 4], f16(rng.normal(0, 0.1, (8, d + 4))))
    out["act_head.0.bias"] = ("F16", [8], f16(np.zeros(8)))
    out["act_head.2.weight"] = ("F16", [2, 8], f16(rng.normal(0, 0.1, (2, 8))))
    out["act_head.2.bias"] = ("F16", [2], f16(np.zeros(2)))
    out["temperature"] = ("F32", [3], np.asarray(agent["temperature"], dtype="<f4").tobytes())
    return out


def write_safetensors(path: Path, tensors: dict) -> None:
    header, offset, blobs = {}, 0, []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    with open(path, "wb") as f:
        f.write(len(head).to_bytes(8, "little") + head + b"".join(blobs))


def listing(folder: Path) -> dict[str, hub.FileInfo]:
    """What the Hub's tree API would say about every file under the folder."""
    out = {}
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        name, data = p.relative_to(folder).as_posix(), p.read_bytes()
        if p.suffix == ".safetensors":
            out[name] = hub.FileInfo(name, len(data), "0" * 40, hashlib.sha256(data).hexdigest())
        else:
            out[name] = hub.FileInfo(name, len(data), hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(), None)
    return out


class TinyLaya:
    """A tiny Laya checkpoint, its manifest, and its tensors in the cache under `home` (`use` points $VIBEHASH_HOME at
    it), so finetune.runtime loads it without asking the Hub."""

    def __init__(self, home: Path, seed: int = 0, head_layers: int = 2, **encoder):
        self.home = Path(home)
        self.encoder, self.agent = {**ENCODER, **encoder}, {**AGENT, "head_layers": head_layers}
        self.folder = self.home / "src" / f"laya-{seed}-{head_layers}"
        self.tensors = weights(self.encoder, self.agent, seed)
        (self.folder / "encoder").mkdir(parents=True, exist_ok=True)
        (self.folder / "tokenizer").mkdir(parents=True, exist_ok=True)
        (self.folder / "config.json").write_text(json.dumps({"model_type": "laya", "architectures": ["LayaTypedDecisions"]}))
        (self.folder / "encoder" / "config.json").write_text(json.dumps(self.encoder, indent=2))
        (self.folder / "rl_agent_config.json").write_text(json.dumps(self.agent, indent=2))
        (self.folder / "tokenizer" / "tokenizer.json").write_bytes(tokenizer_json())
        (self.folder / "tokenizer" / "tokenizer_config.json").write_text(json.dumps(TOKENIZER_CONFIG))
        write_safetensors(self.folder / "model.safetensors", self.tensors)
        self.commit = hashlib.sha1((self.folder / "model.safetensors").read_bytes()).hexdigest()
        self.repo = "tiny/laya"
        self.manifest = hub.build_manifest(self.repo, self.commit, local_dir=self.folder, listing=listing(self.folder),
                                           extra=decision.EXTRA_FILES)
        self.manifest_bytes = self.manifest.to_bytes()
        self.encoder_bytes = (self.folder / "encoder" / "config.json").read_bytes()
        self.agent_bytes = (self.folder / "rl_agent_config.json").read_bytes()
        root = self.home / "cache" / "hub" / self.commit
        root.mkdir(parents=True, exist_ok=True)
        for name, (_, _, raw) in self.tensors.items():
            (root / self.manifest.tensors[name].sha256).write_bytes(raw)

    def use(self, test) -> "TinyLaya":
        from vibehash import finetune

        old = os.environ.get("VIBEHASH_HOME")
        os.environ["VIBEHASH_HOME"] = str(self.home)
        test.addCleanup(lambda: os.environ.pop("VIBEHASH_HOME") if old is None else os.environ.update(VIBEHASH_HOME=old))
        finetune.unload()
        test.addCleanup(finetune.unload)
        return self

    @property
    def arch(self) -> decision.DecisionArch:
        return decision.DecisionArch.from_configs(self.encoder, self.agent)

    def spec(self, seq_len: int = 48, **kw) -> decision.DecisionSpec:
        return decision.DecisionSpec(base=f"hf:{self.repo}@{self.commit}",
                                     manifest=hashlib.sha256(self.manifest_bytes).hexdigest(), dims=self.arch.dims(),
                                     seq_len=seq_len, **{"budgets": (self.arch.max_len, self.arch.head_max_len), **kw})

    def tokenizer(self) -> decision.LayaTokenizer:
        return decision.LayaTokenizer(tokenizer_json(), TOKENIZER_CONFIG)

    def rows(self, n: int, seed: int = 1) -> list[tuple[str, str]]:
        """Learnable rows: a few random letters and one colour word; the label is the colour."""
        rng = np.random.default_rng(seed)
        letters, colours = WORDS[13:30], ["red", "green", "blue"]
        out = []
        for _ in range(n):
            c = colours[int(rng.integers(3))]
            words = [letters[int(k)] for k in rng.integers(0, len(letters), int(rng.integers(2, 8)))]
            words.insert(int(rng.integers(len(words) + 1)), c)
            out.append((" ".join(words), c))
        return out

    def dataset(self, n: int = 60, seed: int = 1, path: Path | None = None) -> decision.DecisionDataset:
        path = path or self.home / f"rows-{n}-{seed}.csv"
        lines = ["text,label"] + [f"{t},{label}" for t, label in self.rows(n, seed)]
        path.write_text("\n".join(lines) + "\n")
        p = decision.prepare(path, self.tokenizer(), self.manifest_bytes, self.encoder_bytes, self.agent_bytes)
        return p.data

    def torch_reference(self, extra=None):
        """Laya's DecisionModel built from transformers' ModernBertModel and PyTorch's own layers (laya.common's
        construction), in float32, with this checkpoint's weights (or `extra`, {name: float32 array}, in their place).
        Returns forward(ids, markers, qtype) -> the options' numbers."""
        import torch
        import torch.nn as nn
        from transformers import ModernBertConfig, ModernBertModel

        cfg = ModernBertConfig(**{k: v for k, v in self.encoder.items() if k not in ("architectures", "model_type")})
        cfg._attn_implementation = "eager"
        enc = ModernBertModel(cfg).eval()
        d = self.encoder["hidden_size"]
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, 0.0, batch_first=True, norm_first=True)
        head = nn.TransformerEncoder(layer, self.agent["head_layers"], enable_nested_tensor=False).eval() \
            if self.agent["head_layers"] else None
        type_emb = nn.Embedding(3, d)
        scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)).eval()
        values = {}
        for name, (dtype, shape, raw) in self.tensors.items():
            a = np.frombuffer(raw, dtype="<f2" if dtype == "F16" else "<f4").astype(np.float32).reshape(shape)
            values[name] = (extra or {}).get(name, a)
        with torch.no_grad():
            for prefix, module in (("encoder.", enc), ("head.", head), ("scorer.", scorer)):
                if module is None:
                    continue
                own = dict(module.state_dict())
                for k in own:
                    if prefix + k in values:
                        own[k] = torch.from_numpy(np.ascontiguousarray(values[prefix + k]))
                module.load_state_dict(own)
            type_emb.weight.copy_(torch.from_numpy(values["type_emb.weight"]))

        def forward(ids, markers, qtype):
            with torch.no_grad():
                ids_t = torch.as_tensor(ids, dtype=torch.long)[None]
                h = enc(input_ids=ids_t, attention_mask=torch.ones_like(ids_t)).last_hidden_state
                h = h + type_emb(torch.tensor([qtype]))[:, None, :]
                if head is not None:
                    h = head(h)
                m = h[0, torch.as_tensor(markers, dtype=torch.long)]
                return scorer(m)[:, 0].double().numpy()
        return forward
