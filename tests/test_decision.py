"""Fine-tuning System 1 decision models, like Laya (decision.py, decision_program.py, decision_tracer.py).

Everything runs offline on a tiny random Laya-shaped checkpoint (tests/decision_fixtures.py) with CPU PyTorch. The
model is checked against transformers' ModernBertModel and PyTorch's own nn.TransformerEncoderLayer, built the way
laya.common.DecisionModel builds them, when transformers is installed.
"""
import copy
import dataclasses
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vibehash import decision, finetune
from vibehash.blobs import file_sha256
from vibehash.crypto import KeyPair
from vibehash.ledger import Chain, InvalidBlock, Tx
from vibehash.tasks import InvalidJob, TrainingJob, load_dataset

try:
    import torch
except ImportError:
    torch = None

HAVE_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None
HAVE_HF = torch is not None and importlib.util.find_spec("transformers") is not None
needs_torch = unittest.skipIf(torch is None or not HAVE_TOKENIZERS, "needs PyTorch and tokenizers")
needs_hf = unittest.skipIf(not HAVE_HF or not HAVE_TOKENIZERS, "needs transformers and tokenizers")
VIBE = 10**8

if torch is not None and HAVE_TOKENIZERS:
    from tests.decision_fixtures import WORDS, TinyLaya

# Laya's own encoder/config.json and rl_agent_config.json (convaiinnovations/laya at 85cc74dd, Apache-2.0), the parts
# that describe the model.
LAYA_ENCODER = {
    "architectures": ["ModernBertForMaskedLM"], "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 50281,
    "classifier_activation": "gelu", "classifier_bias": False, "classifier_pooling": "mean", "cls_token_id": 50281,
    "eos_token_id": 50282, "global_attn_every_n_layers": 3, "hidden_activation": "gelu", "hidden_size": 1024,
    "intermediate_size": 2624, "layer_norm_eps": 1e-05,
    "layer_types": ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(28)],
    "local_attention": 128, "max_position_embeddings": 8192, "mlp_bias": False, "model_type": "modernbert",
    "norm_bias": False, "norm_eps": 1e-05, "num_attention_heads": 16, "num_hidden_layers": 28, "pad_token_id": 50283,
    "rope_parameters": {"full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
    "sep_token_id": 50282, "tie_word_embeddings": True, "vocab_size": 50368}
LAYA_AGENT = {"encoder": "answerdotai/ModernBERT-large", "head_layers": 2, "max_len": 512, "head_max_len": 192,
              "max_prefixes": 6, "act_costs": {"escalate": 0.5}}


def tmpdir(test) -> Path:
    d = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    test.addCleanup(d.cleanup)
    return Path(d.name)


def laya_spec(**kw) -> decision.DecisionSpec:
    """A spec shaped like Laya, without its files."""
    dims = decision.DecisionArch.from_configs(LAYA_ENCODER, LAYA_AGENT).dims()
    return decision.DecisionSpec(base="hf:convaiinnovations/laya@" + "8" * 40, manifest="a" * 64, dims=dims, **kw)


def job_of(spec=None, **kw) -> TrainingJob:
    args = dict(budget=10 * VIBE, fee_per_round=VIBE, spec=spec or laya_spec(),
                data_source={"kind": "tokens", "sha256": "b" * 64}, lr=2e-4, batch_size=16, steps_per_unit=16,
                checkpoint_every=4, tolerance=1e-4, max_update_ratio=1.0)
    return TrainingJob("laya-ft", "", **{**args, **kw})


class SpecTests(unittest.TestCase):
    def test_lays_own_configs_describe_it(self):
        a = decision.DecisionArch.from_configs(LAYA_ENCODER, LAYA_AGENT)
        self.assertEqual((a.layers, a.hidden, a.heads, a.head_dim, a.ffn, a.window, a.every), (28, 1024, 16, 64, 2624,
                                                                                                64, 3))
        self.assertEqual((a.head_layers, a.head_heads, a.rope_global, a.rope_local), (2, 16, 160000.0, 10000.0))
        self.assertTrue(a.is_global(0) and a.is_global(27) and not a.is_global(1))
        s = laya_spec()
        self.assertEqual(s.base_params, 421_029_889, "Laya-421M: its model.safetensors, the action head left out")
        self.assertEqual(len(s.modules()), 28 * 4 + 2 * 4 + 1)
        self.assertEqual(s.n_adapter, 16 * (28 * (3072 + 1024 + 5248 + 1024) + 2 * (3072 + 1024 + 4096 + 1024) + 1024))
        self.assertLess(s.n_params * 4, 64 * 2**20, "a checkpoint of B, m and v is tens of MB")

    def test_unsupported_encoders_are_refused(self):
        for bad in ({"hidden_activation": "silu"}, {"attention_bias": True}, {"norm_bias": True},
                    {"model_type": "bert"}, {"layer_types": ["full_attention"] * 28},
                    {"rope_parameters": {"full_attention": {"rope_type": "yarn"}}}):
            with self.assertRaises(ValueError, msg=bad):
                decision.DecisionArch.from_configs({**LAYA_ENCODER, **bad}, LAYA_AGENT)
        with self.assertRaises(ValueError):
            decision.DecisionArch.from_configs(LAYA_ENCODER, {**LAYA_AGENT, "head_layers": 99})

    def test_round_trip_through_a_job_spec(self):
        job = job_of(laya_spec(rank=8, targets=("scorer", "qkv"), budgets=(512, 384)))
        again = TrainingJob.from_spec_json(job.spec_json(), "", job.budget)
        self.assertEqual(again.spec.budgets, (512, 384))
        self.assertIsInstance(again.spec, decision.DecisionSpec)
        self.assertEqual(again.spec, job.spec)
        self.assertEqual(again.spec.targets, ("qkv", "scorer"), "targets keep TARGETS' order")
        self.assertTrue(again.finetune and again.decision)
        self.assertEqual(again.spec_json(), job.spec_json())

    def test_a_node_from_before_refuses_the_spec(self):
        """Read as a language model's fine-tune (as a node without decision.py reads it), the spec is refused."""
        d = json.loads(job_of().spec_json())
        self.assertTrue(finetune.LoraSpec.from_json(d, seed=11).problems())

    def test_bad_specs_are_refused(self):
        for kw in ({"targets": ("q",)}, {"rank": 0}, {"seq_len": 4096}, {"targets": ()}, {"budgets": (512, 600)},
                   {"budgets": (256, 192)}):
            with self.assertRaises(InvalidJob, msg=kw):
                job_of(laya_spec(**kw))
        d = json.loads(job_of().spec_json())
        del d["budgets"]
        with self.assertRaises(InvalidJob):
            TrainingJob.from_spec_json(json.dumps(d), "", VIBE)
        no_head = dict(laya_spec().model, head_layers=0)
        with self.assertRaises(InvalidJob):
            job_of(decision.DecisionSpec(base=laya_spec().base, manifest="a" * 64, dims=no_head, targets=("head_o",)))

    def test_its_disputes_fit_a_home_connection(self):
        """A job is refused when a dispute over one of its units couldn't be played from home (tasks); Laya's units, at
        the settings `train` picks, can be."""
        from vibehash.program import dispute_size

        moves, split = dispute_size(laya_spec(), decision.BATCH, 4, 4)
        self.assertGreater(moves, 10)
        self.assertLess(split, 512 * 2**20)
        self.assertEqual(job_of().phase_lengths()["moves"], moves)


class GateTests(unittest.TestCase):
    """A decision model's fine-tune is a prototype: only a chain whose genesis takes them holds one."""

    def test_refused_without_the_gate_and_taken_with_it(self):
        from vibehash.ledger import VERSION, _apply_user_tx

        req = KeyPair.generate(b"req")
        job = job_of(budget=5 * VIBE)
        tx = Tx.signed(req, "post_job", "system", job.budget, 0, job.spec_json(), chain_id="t")
        for gate in (False, True):
            s = copy.deepcopy(Chain({req.address: 100 * VIBE}, chain_id="t", decision=gate).state)
            self.assertEqual(s.decision, gate)
            if gate:
                _apply_user_tx(s, tx, 1, "t", VERSION)
                self.assertIn("laya-ft", s.jobs)
            else:
                with self.assertRaisesRegex(InvalidBlock, "decision"):
                    _apply_user_tx(s, tx, 1, "t", VERSION)

    def test_the_genesis_says_so(self):
        g = {"chain_id": "t", "allocations": {}, "decision": True}
        self.assertTrue(Chain.from_genesis(g).state.decision)
        self.assertFalse(Chain.from_genesis({"chain_id": "t", "allocations": {}}).state.decision)
        self.assertNotEqual(Chain.from_genesis(g).genesis_hash,
                            Chain.from_genesis({"chain_id": "t", "allocations": {}}).genesis_hash)


@needs_torch
class DataTests(unittest.TestCase):
    def setUp(self):
        self.tiny = TinyLaya(tmpdir(self)).use(self)

    def test_lays_out_items_as_laya_does(self):
        tok = self.tiny.tokenizer()
        head, markers, distinct = tok.head("choice", "classify the input", ["red", "green: a b", "blue"], 24)
        w = {x: i for i, x in enumerate(WORDS)}
        unknown = w["[UNK]"]  # "green:" isn't a word of the tiny vocabulary
        self.assertEqual(head, [w["[CLS]"], w["choice"], w["question:"], w["classify"], w["the"], w["input"],
                                w["[SEP]"], w["[MASK]"], w["red"], w["[MASK]"], unknown, w["a"], w["b"], w["[MASK]"],
                                w["blue"], w["[SEP]"]])
        self.assertEqual(markers, [7, 9, 13])
        self.assertEqual(distinct, 3)
        ids, ms = tok.sequence(head, markers, "x y z", 48)
        self.assertEqual(ids, head + [w["x"], w["y"], w["z"], w["[SEP]"]])
        ids, ms = tok.sequence(head, markers, " ".join(["x"] * 100), 20)
        self.assertEqual(ids, head + [w["x"]] * 3 + [w["[SEP]"]], "the case is cut to the room left before the [SEP]")
        self.assertEqual(ms, markers)
        ids, ms = tok.sequence(head, markers, "x", 12)
        self.assertEqual((ids, ms), (head[:12], [7, 9]), "a sequence cut inside the options loses their markers")

    def test_options_that_dont_fit_get_a_bigger_question_budget(self):
        """Laya's 192 tokens cut Banking77's 77 options until four pairs are the same; prepare raises the budget to hold
        them whole, as laya-banking77 did, and the job pins it for the checkpoint it downloads."""
        labels = [f"{a} {b}" for a in ("red", "green", "blue", "cat", "dog") for b in ("a b c", "a b d", "x y z")]
        path = self.tiny.home / "many.csv"
        rows = ["text,label"] + [f"x {lab} y,{lab}" for lab in labels * 2]
        path.write_text("\n".join(rows) + "\n")
        p = decision.prepare(path, self.tiny.tokenizer(), self.tiny.manifest_bytes, self.tiny.encoder_bytes,
                             self.tiny.agent_bytes, max_len=128)
        self.assertGreater(p.head_max_len, self.tiny.arch.head_max_len)
        self.assertEqual(p.head_max_len % 64, 0)
        self.assertEqual(len(p.data.item(0)[1]), len(labels))
        with self.assertRaises(decision.DataError):  # at the checkpoint's own 24, they collapse
            decision.prepare(path, self.tiny.tokenizer(), self.tiny.manifest_bytes, self.tiny.encoder_bytes,
                             self.tiny.agent_bytes, max_len=128, head_max_len=24)

    def test_many_options_share_the_question_budget(self):
        """Laya's rule: when the options don't fit, each is cut to (budget - 16) // options tokens, at least 4."""
        tok = self.tiny.tokenizer()
        opts = [f"red {'a b c d e f g' if i % 2 else 'x y z'}" for i in range(6)]
        _, markers, distinct = tok.head("choice", "classify", opts, 24)
        self.assertEqual(np.diff(markers).tolist(), [4] * 5)
        self.assertEqual(distinct, 2)

    def test_round_trip_sniffing_and_checks(self):
        d = self.tiny.dataset(40)
        path = d.save(self.tiny.home / "d.npz")
        back = load_dataset(path)
        self.assertIsInstance(back, decision.DecisionDataset)
        for name in ("train_ids", "train_mask", "train_target", "val_type"):
            np.testing.assert_array_equal(getattr(back, name), getattr(d, name))
        back.check(self.tiny.spec(seq_len=d.train_ids.shape[1]))
        with self.assertRaises(ValueError):
            back.check(self.tiny.spec(seq_len=16))
        self.assertEqual(back.arch(), self.tiny.arch)

    def test_malformed_items_are_refused(self):
        d = self.tiny.dataset(40)
        for change in ({"train_target": d.train_target * 2}, {"train_mask": np.zeros_like(d.train_mask)},
                       {"train_type": d.train_type + 5}, {"agent": b"{"}):
            bad = decision.DecisionDataset(**{**{k: getattr(d, k) for k in (
                "train_ids", "train_len", "train_mask", "val_ids", "val_len", "val_mask", "manifest", "config", "agent",
                "train_target", "train_type", "val_target", "val_type")}, **change})
            with self.assertRaises(ValueError, msg=list(change)):
                bad.check_shape()

    def test_a_job_takes_only_its_kind_of_data(self):
        d = self.tiny.dataset(40)
        path = d.save(self.tiny.home / "d.npz")
        sha = file_sha256(path)
        job = job_of(self.tiny.spec(seq_len=d.train_ids.shape[1]), data_source={"kind": "tokens", "sha256": sha})
        job.attach_data(load_dataset(path), sha)
        from tests.finetune_fixtures import TinyModel
        tm = TinyModel(self.tiny.home / "llm")
        tpath = tm.dataset().save(self.tiny.home / "t.npz")
        tsha = file_sha256(tpath)
        with self.assertRaises(InvalidJob):
            job_of(self.tiny.spec(), data_source={"kind": "tokens", "sha256": tsha}).attach_data(load_dataset(tpath), tsha)

    def test_prepare_reads_csv_and_json_lines_and_names_the_options(self):
        rows = self.tiny.rows(30)
        jl = self.tiny.home / "rows.jsonl"
        jl.write_text("\n".join(json.dumps({"text": t, "label": lab}) for t, lab in rows))
        p = decision.prepare(jl, self.tiny.tokenizer(), self.tiny.manifest_bytes, self.tiny.encoder_bytes,
                             self.tiny.agent_bytes, labels=["red", "green", "blue"])
        self.assertEqual(p.labels, ["red", "green", "blue"])
        self.assertEqual(len(p.data.train_len) + len(p.data.val_len), 30)
        _, markers, target, _ = p.data.item(0)
        self.assertEqual(len(markers), 3)
        self.assertEqual(target.sum(), 1.0)
        bad = self.tiny.home / "bad.csv"
        bad.write_text("text,label\nred a,red\n,blue\n")
        with self.assertRaises(decision.DataError):
            decision.prepare(bad, self.tiny.tokenizer(), self.tiny.manifest_bytes, self.tiny.encoder_bytes,
                             self.tiny.agent_bytes)


@needs_torch
class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tiny = TinyLaya(tmpdir(self)).use(self)
        self.data = self.tiny.dataset(40)
        self.spec = self.tiny.spec(seq_len=self.data.train_ids.shape[1])

    def theta(self, scale=0.05, seed=3):
        """Adapters that change the model: B drawn at random."""
        rng = np.random.default_rng(seed)
        t = self.spec.init_params()
        t[self.spec.parts[0]] = rng.normal(0, scale, self.spec.n_adapter)
        return t

    def merged(self, theta) -> dict:
        """W + alpha/rank · B·A for every adapted weight, float32, by name."""
        s, b, out, i = self.spec, theta[self.spec.parts[0]], {}, 0
        for (name, rows, _), a in zip(s.modules(), s.lora_a()):
            w = self.tiny.tensors[s.weight_of(name)]
            base = np.frombuffer(w[2], dtype="<f2").astype(np.float64).reshape(w[1])
            out[s.weight_of(name)] = (base + s.scaling * b[i:i + rows * s.rank].reshape(rows, s.rank) @ a).astype(
                np.float32)
            i += rows * s.rank
        return out

    @needs_hf
    def test_matches_transformers_and_pytorchs_layers(self):
        """The ops a unit runs, forward, against ModernBertModel and nn.TransformerEncoderLayer built as Laya's
        DecisionModel is: with the base's weights, and with adapters (against their merged weights)."""
        for theta, extra in ((self.spec.init_params(), None), (self.theta(), None)):
            if extra is None and theta.any():
                extra = self.merged(theta)
            ref = self.tiny.torch_reference(extra)
            probs = decision.probabilities(self.spec, self.data, theta)
            for i, p in enumerate(probs):
                ids, markers, _, qtype = self.data.item(i, val=True)
                z = ref(ids, markers, qtype)
                want = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
                np.testing.assert_allclose(p, want, atol=2e-6)

    def test_gradients_match_autograd(self):
        """Every adapter's gradient from the written-out backward against autograd through the forward ops."""
        from vibehash.decision_program import DecisionProgram
        from vibehash.decision_tracer import DecisionExecutor
        from vibehash.program import split_theta

        theta = self.theta()
        rt = finetune.runtime(self.spec, self.data, "cpu")
        prog = DecisionProgram(self.spec, self.data, 1e-3, 1, 0)
        for i in (0, 3):
            st = {k: torch.tensor(v, requires_grad=k.startswith("B.")) for k, v in split_theta(self.spec, theta).items()}
            node = prog.forward(i)
            DecisionExecutor(rt, self.spec, prog, room=0).run_atoms(node, 0, len(node.atoms), st, ())
            _, _, target, _ = self.data.item(i)
            loss = -(torch.from_numpy(target) * torch.log_softmax(st["s.z"][:, 0], 0)).sum()
            loss.backward()
            got = {k: torch.tensor(v) for k, v in split_theta(self.spec, theta).items()}
            with torch.no_grad():
                micro = prog.micro(0, 0, i, 1)
                DecisionExecutor(rt, self.spec, prog, room=0).run_atoms(micro, 0, len(micro.atoms), got, ())
            names = [n for n, _, _ in self.spec.modules()]
            self.assertEqual(set(got), {f"{p}.{n}" for n in names for p in "BmvG"}, "an item leaves only theta and G")
            leaves = {k: t for k, t in st.items() if k.startswith("B.")}
            for n in names:
                np.testing.assert_allclose(got[f"G.{n}"].numpy(), leaves[f"B.{n}"].grad.numpy(), rtol=2e-4,
                                           atol=2e-7, err_msg=n)

    def test_partial_targets_train_only_what_they_name(self):
        for targets in (("scorer",), ("head_o", "scorer"), ("wo",)):
            spec = self.tiny.spec(seq_len=self.spec.seq_len, targets=targets)
            t = finetune.train_segment(spec, self.data, spec.init_params(), 7, 0, 2, 1e-2, 4)
            self.assertTrue(np.isfinite(t).all() and np.abs(t[spec.parts[0]]).max() > 0, targets)

    def test_a_model_without_head_layers(self):
        tiny = TinyLaya(self.tiny.home / "h0", head_layers=0).use(self)
        data = tiny.dataset(30)
        spec = tiny.spec(seq_len=data.train_ids.shape[1], targets=("qkv", "o", "wi", "wo", "scorer"))
        t = finetune.train_segment(spec, data, spec.init_params(), 7, 0, 2, 1e-2, 4)
        self.assertTrue(np.isfinite(t).all() and t[spec.parts[0]].any())
        if HAVE_HF:
            ref = tiny.torch_reference()
            for i, p in enumerate(decision.probabilities(spec, data, spec.init_params())[:3]):
                ids, markers, _, qtype = data.item(i, val=True)
                z = ref(ids, markers, qtype)
                np.testing.assert_allclose(p, np.exp(z - z.max()) / np.exp(z - z.max()).sum(), atol=2e-6)


@needs_torch
class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tiny = TinyLaya(tmpdir(self)).use(self)
        self.data = self.tiny.dataset(60)
        self.spec = self.tiny.spec(seq_len=self.data.train_ids.shape[1])

    def test_deterministic_and_resumable(self):
        s, d = self.spec, self.data
        a = finetune.train_segment(s, d, s.init_params(), 7, 0, 6, 1e-2, 4)
        np.testing.assert_array_equal(a, finetune.train_segment(s, d, s.init_params(), 7, 0, 6, 1e-2, 4))
        half = finetune.train_segment(s, d, s.init_params(), 7, 0, 3, 1e-2, 4)
        np.testing.assert_array_equal(a, finetune.train_segment(s, d, half, 7, 3, 3, 1e-2, 4))
        self.assertTrue(np.all(a[s.parts[2]] >= 0) and np.any(a[s.parts[1]]))
        self.assertFalse(np.array_equal(a, finetune.train_segment(s, d, s.init_params(), 8, 0, 6, 1e-2, 4)))

    def test_adams_bound_holds(self):
        s = self.spec
        a = finetune.train_segment(s, self.data, s.init_params(), 7, 0, 6, 1e-2, 4)
        self.assertTrue(s.sane(s.init_params(), a, 1e-2, 6, 1.0))

    def test_training_learns(self):
        """The toy task (the label is the one colour word among random letters), measured on 2026-11-18: at lr 3e-3 the
        training items' loss goes 1.114 to 0.237 in four rounds of two averaged units, every one of them right."""
        d = self.tiny.dataset(120)
        s = self.tiny.spec(seq_len=d.train_ids.shape[1])
        theta = s.init_params()
        before, acc0 = finetune.evaluate(s, d, theta)
        for r in range(5):
            theta = s.fresh(np.mean([finetune.train_segment(s, d, theta, 10 * r + u, 0, 8, 3e-3, 8) for u in range(2)],
                                    axis=0))
        after, acc1 = finetune.evaluate(s, d, theta)
        self.assertLess(after, before - 0.2, (before, after, acc0, acc1))
        self.assertGreater(acc1, acc0)
        report = decision.calibration(decision.probabilities(s, d, theta),
                                      [d.item(i, val=True)[2] for i in range(len(d.val_len))])
        self.assertEqual(report["items"], len(d.val_len))
        self.assertAlmostEqual(report["log_loss"], after, places=6)

    def test_checks_catch_doctored_results(self):
        from vibehash.verification import reproduces

        s, d = self.spec, self.data
        honest = finetune.train_segment(s, d, s.init_params(), 7, 0, 4, 1e-2, 4)
        replay = finetune.train_segment(s, d, s.init_params(), 7, 0, 4, 1e-2, 4)
        self.assertTrue(reproduces(honest, replay, s.init_params(), 1e-4, s))
        bad = honest.copy()
        bad[s.parts[0]] *= 1.5
        self.assertFalse(reproduces(bad, replay, s.init_params(), 1e-4, s))
        moments = honest.copy()
        moments[s.parts[2]] *= 1.5
        self.assertFalse(reproduces(moments, replay, s.init_params(), 1e-4, s), "the moments are checked on their own")


@needs_torch
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tiny = TinyLaya(tmpdir(self)).use(self)
        self.data = self.tiny.dataset(40)
        self.spec = self.tiny.spec(seq_len=self.data.train_ids.shape[1])

    def export(self, theta, **kw):
        out = self.tiny.home / "out"
        info = decision.export_merged(self.spec, theta, self.tiny.manifest, out, {"job_id": "t"},
                                      read_file=lambda p: (self.tiny.folder / p).read_bytes(), **kw)
        return out, info

    def read(self, folder: Path) -> dict:
        raw = (folder / "model.safetensors").read_bytes()
        n = int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8:8 + n])
        out = {}
        for name, e in header.items():
            if name == "__metadata__":
                continue
            b0, b1 = e["data_offsets"]
            dt = {"F16": "<f2", "F32": "<f4"}[e["dtype"]]
            out[name] = np.frombuffer(raw[8 + n + b0:8 + n + b1], dtype=dt).astype(np.float32).reshape(e["shape"])
        return out

    def test_it_is_a_laya_checkpoint_with_the_adapters_merged_in(self):
        rng = np.random.default_rng(5)
        theta = self.spec.init_params()
        theta[self.spec.parts[0]] = rng.normal(0, 0.05, self.spec.n_adapter)
        out, info = self.export(theta, float32=True)
        self.assertTrue(info["trained"])
        self.assertEqual(sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()),
                         sorted(["README.md", "config.json", "encoder/config.json", "finetune.json", "model.safetensors",
                                 "rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]))
        agent = json.loads((out / "rl_agent_config.json").read_text())
        self.assertEqual((agent["temperature"], agent["temperature_by_options"], agent["head_layers"]), ([1.0] * 3, {},
                                                                                                         2))
        self.assertEqual((agent["max_len"], agent["head_max_len"]), self.spec.budgets)
        w = self.read(out)
        self.assertEqual(w["temperature"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(set(w), set(self.tiny.tensors), "every tensor of the base, act head included")
        if HAVE_HF:
            ref = self.tiny.torch_reference({k: v for k, v in w.items()})
            for i, p in enumerate(decision.probabilities(self.spec, self.data, theta)):
                ids, markers, _, qtype = self.data.item(i, val=True)
                z = ref(ids, markers, qtype)
                np.testing.assert_allclose(p, np.exp(z - z.max()) / np.exp(z - z.max()).sum(), atol=2e-6)

    def test_in_the_bases_f16_unadapted_tensors_are_byte_for_byte(self):
        out, info = self.export(self.spec.init_params())
        self.assertFalse(info["trained"])
        w = self.read(out)
        for name, (dtype, shape, raw) in self.tiny.tensors.items():
            if name != "temperature":
                np.testing.assert_array_equal(w[name], np.frombuffer(raw, dtype="<f2" if dtype == "F16" else "<f4")
                                              .astype(np.float32).reshape(shape), err_msg=name)


@needs_torch
class ProtocolTests(unittest.TestCase):
    """A decision model's fine-tune through the same Coordinator a node runs, with cheaters, then replayed and
    rechecked, as test_finetune's fine-tunes are."""

    def test_rounds_train_pay_catch_cheats_and_replay(self):
        """Honest units are accepted and paid; no cheat is. A cheat that breaks Adam's bound (lazy, poisoner) is slashed
        on the node's own NumPy check. One that stays inside it (shortcut: half trained, half extrapolated) is rejected
        by its checkers, but the dispute that follows ends unchecked, since the referee has no arithmetic for a decision
        model's ops yet (referee.judge): it goes unpaid and keeps its stake. That gap is why these jobs are gated
        (ledger.DECISION_GATE); this test changes when the referee closes it."""
        from tests.protocol import Protocol
        from vibehash.coordinator import CoordinatorConfig
        from vibehash.replay import replay_block
        from vibehash.worker import HardwareProfile, Worker

        tiny = TinyLaya(tmpdir(self)).use(self)
        data = tiny.dataset(60)
        path = data.save(tiny.home / "data.npz")
        sha = file_sha256(path)
        behaviors = [("honest", "honest")] * 6 + [("lazy", "honest"), ("poisoner", "honest"), ("shortcut", "honest")]
        hw = HardwareProfile("gpu", 1e12, 16384, 1.0)
        workers = [Worker(f"w{i}", KeyPair.generate(f"w{i}".encode()), hw, *b, fp_noise=0) for i, b in enumerate(behaviors)]
        req = KeyPair.generate(b"req")
        chain = Chain({req.address: 1000 * VIBE}, stakes={w.address: 10 * VIBE for w in workers}, decision=True)
        proto = Protocol(chain, workers, CoordinatorConfig(), seed=3)
        job = job_of(tiny.spec(seq_len=data.train_ids.shape[1]), data_source={"kind": "tokens", "sha256": sha},
                     units_per_round=9, lr=3e-3, batch_size=4, steps_per_unit=8, memory_mb=2048, budget=100 * VIBE,
                     fee_per_round=10 * VIBE)
        job.attach_data(load_dataset(path), sha)
        proto.post_job(req, job)
        before = copy.deepcopy(proto.chain)
        reports = [proto.run_round() for _ in range(2)]
        by = {w.address: w for w in workers}
        for rep in reports:
            for res in rep.results:
                kind = by[res.worker].behavior
                want = {"honest": "accepted", "shortcut": "unchecked"}.get(kind, "failed_sanity")
                self.assertEqual(res.verdict.value, want, kind)
            self.assertTrue(np.isfinite(rep.val_loss))
            self.assertFalse(rep.params[job.spec.n_adapter:].any(), "the next round starts Adam afresh")
        for w in workers:
            stake = proto.chain.state.stake(w.address)
            if w.behavior in ("lazy", "poisoner"):
                self.assertLess(stake, 10 * VIBE, w.behavior)
            else:
                self.assertGreaterEqual(stake, 10 * VIBE, w.behavior)
        # A round's val_loss is its scorers' score of the model it started from, over six held-back items: too few to
        # show training, and on this toy set they swing either way. The items the units drew from show it.
        d = job.data
        train = dataclasses.replace(d, val_ids=d.train_ids, val_len=d.train_len, val_mask=d.train_mask,
                                    val_target=d.train_target, val_type=d.train_type)
        base = decision.evaluate(job.spec, train, job.spec.init_params())[0]
        self.assertLess(decision.evaluate(job.spec, train, reports[-1].params)[0], base - 0.01,
                        "the averaged units trained the model")

        class Recheck:
            def array(self, bid, job):
                return proto.checkpoints.array(bid, job)

            def bytes(self, bid):
                return proto.coord._get_trace(bid)

            def attach(self, j):
                j.attach_data(load_dataset(path), sha)
        chain, stats = copy.deepcopy(before), {}
        for rep in reports:
            self.assertEqual(replay_block(chain, rep.block, copy.deepcopy(rep.record), Recheck(), stats), [])
            chain.append(rep.block)
        self.assertGreater(stats["rechecked"], 5)
        self.assertEqual(stats["unchecked"], 0)


if __name__ == "__main__":
    unittest.main()
