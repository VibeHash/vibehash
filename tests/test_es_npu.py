"""vibehash/es_npu.py: Qwen3.5's twisted decode step compiled by OpenVINO. On OpenVINO's CPU in float32 it is
es_hybrid's engine: on test_es_hybrid's tiny random model, the same greedy answers, gaps of 0 on each other's answers
and the same log-probabilities. (The NPU itself, in fp16, is what scripts/evo_margin_xgpu.py --device npu measures.)
"""
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from tests.test_es import HAVE_TORCH

HAVE_OV = find_spec("openvino") is not None


@unittest.skipUnless(HAVE_TORCH and HAVE_OV, "needs PyTorch, transformers and OpenVINO")
class NpuEngineOnCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vibehash import es2_rounds, es_local, es_npu
        from vibehash.es_client import prompt_rows, round_factors, runtime2
        from tests.test_es_hybrid import tiny_hybrid

        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        found = es_local.find(str(tiny_hybrid(Path(cls.tmp.name) / "tiny35", seed=3)))
        es_local.register(found)
        cls.spec = es_local.spec2_for(found, "arithmetic", pairs=4, pairs_per_unit=2, problems=5, max_tokens=10,
                                      sigma=0.05, check=2, margin=1e-3, lp_tol=0.02)
        cls.rt = runtime2(cls.spec, "cpu")
        seed = 777
        cls.rows = prompt_rows(cls.rt, es2_rounds.items(cls.spec, seed))
        cls.fac = round_factors(cls.rt, cls.spec, seed)
        cls.members = [(j, s) for j in range(cls.spec.pairs) for s in (1, -1)]
        # n=3: fewer rows than the round's 40, so rows are replaced as they finish (the continuous batching)
        cls.eng = es_npu.Engine(cls.rt, cls.rt.linears, cls.spec.pairs * cls.spec.rank,
                                es_npu.cache_slots(cls.rows, cls.spec.max_tokens), n=3, device="CPU",
                                precision="f32", log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_it_answers_and_scores_as_es_hybrid(self):
        from vibehash import es_fast, es_hybrid, es_npu

        rt, rows, fac, spec, members = self.rt, self.rows, self.fac, self.spec, self.members
        self.assertEqual(self.eng.precision, "f32")
        want = es_fast.answer_members(rt, rt.linears, fac, members, rows, spec.max_tokens, spec.sigma)
        got = es_npu.answer(self.eng, rt, fac, members, rows, spec.max_tokens, spec.sigma, spec.rank)
        self.assertEqual(got, want)
        gaps = es_fast.gaps_members(rt, rt.linears, fac, members, rows, got, spec.sigma, max_tokens=spec.max_tokens)
        self.assertEqual(float(gaps.max()), 0.0)
        own, lps = es_npu.forced(self.eng, rt, fac, members, rows, want, spec.max_tokens, spec.sigma, spec.rank)
        self.assertEqual(float(own.max()), 0.0)
        _, ref = es_hybrid.forced(rt, rt.linears, fac, members, rows, want, spec.max_tokens, spec.sigma, spec.rank)
        self.assertLess(float((lps - ref).abs().max()), 1e-4)

    def test_a_twin_swap_is_caught(self):
        from vibehash import es_npu

        rt, rows, fac, spec, members = self.rt, self.rows, self.fac, self.spec, self.members
        got = es_npu.answer(self.eng, rt, fac, members, rows, spec.max_tokens, spec.sigma, spec.rank)
        differ = [q for q in range(len(rows)) if got[0][q] != got[1][q]]
        self.assertTrue(differ, "the twins answered alike everywhere")
        claimed, _ = es_npu.forced(self.eng, rt, fac, [members[1]], rows, [got[0]], spec.max_tokens, spec.sigma,
                                   spec.rank)
        self.assertTrue(all(float(claimed[0, q]) > 0 for q in differ))


if __name__ == "__main__":
    unittest.main()
