# Honest work on any hardware: checking Evo across machine kinds

November 2026. The plan, and what's built of it, for one goal: **no honest worker is turned down because its machine
rounds differently from its checker's.** It builds on docs/GPUS.md (what each GPU family does) and
docs/evo-margin-measurement.md (measuring the gaps).

## The problem

An Evo checker teacher-forces another worker's answers through the same twisted model and finds, for every token,
how far it is below the checker's own greedy choice: a gap, in logits. A claim passes when every gap is within the
job's `margin`. Two machines with the same arithmetic get the same bits, so their gaps are 0. Others don't:

- bf16 on CUDA and ROCm, float32 on Apple, Intel and the CPU (`es_client.runtime2`): up to 0.375 apart, measured;
- cuBLAS promises equal results only on the same architecture with the same number of SMs, and the same toolkit;
- a PyTorch default (bf16 split-K rounding) alone moves gaps by 0.25 on one card.

A cheat (answers its model didn't write) gives gaps of 3.5 or more. So there's plenty of room between honest and
cheating, but a single margin for every pair doesn't use it. On `vibehash-evo-1` (margin 0.1) an RTX 4070 SUPER
rejected a RunPod pod's honest work for gaps of 0.125-0.375 (HANDOFF.md, 2026-11-06). In a rule-2 round one
rejection makes a claim UNCHECKED: unpaid and out of the update, with no referee (`Coordinator._settle_es2`), so the
honest worker loses the round and nothing tells that apart from a checker turning work down on purpose.

## The plan

| | What | Rule change? | Status |
|---|---|---|---|
| A | Each worker registers its **number setup** (`hardware.numerics`): backend, Evo dtype, arch, SM/CU count, PyTorch and CUDA/ROCm version | no | **built** |
| G | Checkers send their **worst gap** with each verdict the gaps alone decided; the node counts them by pair of setups at `/network` (`gaps`) | no | **built** |
| F1 | AMD: matmuls on rocBLAS, not hipBLASLt (deterministic mode covers it; GPUS.md risk 3) | no | **built** |
| F2 | macOS 14: `install-gpu` installs PyTorch 2.14, which still uses the GPU there (GPUS.md risk 12) | no | **built** |
| C | **Pair checkers by setup**: a claim whose two drawn checkers don't share the worker's setup gets a third that does, when there is one, as referee | yes | **built** |
| D | **Margin by pair**: `margin` for the same setup (tight: their gaps are 0), `cross_margin` across setups | yes | **built** |
| E | **A matching machine referees**: a cross-setup rejection on gaps alone, up to twice `cross_margin`, is overruled when a same-setup checker approved | yes | **built** |
| F3 | Evo without bf16 split-K rounding (GPUS.md risk 2), as a spec field, after a two-card measurement | yes | after |
| F4 | An Evo self-test: a card that can't reproduce its own answers exactly (gap 0) doesn't take Evo work (GPUS.md risk 13) | no | **built** |
| H | An integer rule: int8 weights and activations, int32 sums, fixed-point nonlinearities. Bit-identical on every GPU, CPU, Mac and NPU, so the margin is 0 | yes | research |

C, D and E come together, behind one spec field (`cross_margin`), so jobs posted before it run exactly as they did,
and a node or miner without the code refuses a spec with it (as with `loop_stop`): nodes and miners upgrade before a
job uses it. `vibehash job template --evo` turns it on: `margin` 0.1 for alike machines and `cross_margin` 0.5
across (`es_local.V2_PAIRED_MARGIN`, `V2_CROSS_MARGIN`); `--cross-margin X` or `off` changes it.

A referee is one more check of the claim, so a paired job's checkers do up to half again as much checking: none
more for a claim whose worker's setup nobody else registered, or one of whose drawn checkers shares it.

Where each part is: pairing in `Coordinator._es_tasks`; the worker's setup goes to its checkers with the task
(`Coordinator.es_task`), and a checker picks its margin with `Spec2.margin_for` (`es_client.Es2Work.check_many`);
the referee is `Coordinator._overruled`, in `_settle_es2`. The census counts each gap against its pair's margin.

### Why pairing by setup is safe

A worker names its own setup, and could lie to be paired with a friend. That buys little:

- the two checkers that decide a claim are drawn from everyone, exactly as without pairing; the one chosen by setup
  is a third, added only when neither of them shares the worker's setup, and it only referees: if only it answered,
  the claim is unchecked and unpaid, and a drawn checker's rejection stands unless it's overruled
  (`Coordinator._judge_es2`);
- so a ring of friends that all name one made-up setup always gets a friend as referee, but still needs both drawn
  checkers to be friends to wave through a claim with a cheat's gaps, as now. (An earlier draft drew the first
  checker by setup and the second from everyone: a ring holding 20% of the checkers then had both of a claim's
  checkers 16% of the time, against 3.5% for two drawn from everyone);
- a worker can't leave its setup out to be held to the wider margin by everyone: a job with a cross margin refuses
  a registration without one;
- a rejection with a gap over twice `cross_margin` (1.0 for the template's 0.5; a cheat on Qwen3.5-0.8B sits 3.5
  or more below), or for anything but gaps (tokens that don't decode to the text, an
  answer that doesn't end right, a replay log-probability off), always stands;
- so a liar's friend can wave through only answers within twice `cross_margin` of the greedy choice: near-greedy work, which
  pays no better than honest work (workers are paid by accepted FLOPs, not by rewards);
- a worker that names a setup it doesn't have is held to that setup's tight margin, and gets its own work rejected.

### The census (G)

`GET /network` has `gaps`: `{"checks", "pairs": [{"worker", "checker", "same", "checks", "max", "p99", "near",
"over"}]}`, per pair of setups ("cuda bf16 sm_89 x24 cuda13.2 torch2.15.0"), worst first. `near` counts gaps of half
their job's margin or more, and `over` those over it. It's the node's own view of the verdicts sent to it, for a day,
and names no worker. A pair whose `p99` creeps toward the margin is the warning that a driver, PyTorch or card rounds
differently, before honest work is turned down. The same numbers decide `cross_margin` for the next job.

## NPUs

NPUs (Intel AI Boost, AMD XDNA, Qualcomm Hexagon) run int8 and fp16 through each vendor's own compiler (OpenVINO,
Ryzen AI, QNN), not PyTorch, with fixed shapes. fp16 can't meet a classic job's 1e-4, and Qwen3.5's recurrent state
needs float32, so an NPU is unlikely to land inside a bf16 checker's margin as it is. The steps:

1. On an NPU laptop, mine on its integrated GPU first (Intel Arc and AMD Radeon already work), and have it check a
   CUDA worker's dump (`scripts/evo_margin_xgpu.py check --device xpu`): a measurement GPUS.md doesn't have yet.
2. Add an NPU answering path to `evo_margin_xgpu.py`: the twist is low-rank (`spec.rank`), so one compiled model takes
   every member's factors as inputs. Measure its gaps against a CUDA checker.
3. If its worst gap is under half the cross margin, an `npu` setup (`hardware.BACKENDS` has it) that answers but
   doesn't check, until the census backs it.
4. If not, the integer rule (H) is where NPUs fit: int8 with int32 sums is what they're built for, and exact.

### Step 2, first measurement (2026-11-08, Intel AI Boost)

`vibehash/es_npu.py` is the answering path: es_hybrid's decode step as one pure function (every state and every
twisted linear's factors are inputs), compiled by OpenVINO 2026.4 for the NPU with fixed shapes (16 rows a step; about
9 minutes the first time, seconds from the compile cache). The shared prompt reading stays on the CPU in float32.
`scripts/evo_margin_xgpu.py answer --device npu` and `check --device npu` use it. On OpenVINO's CPU in float32 it gives
es_hybrid's answers and gaps exactly (tests/test_es_npu.py).

The NPU has fp16 and int8 only. As written, its honest answers were rejected outright: a float32 checker found them
up to 6.0 below its greedy choice, as far as a cheat. The cause was one place. Some Gated DeltaNet heads read out
values around 1e-5, which are fp16 subnormals. The NPU holds those to a few bits, and the gated RMSNorm after them
scales the error up to full size: the layer's output came out 3e-2 off float32, 20 times an Intel GPU's fp16. The fix
computes the read-out `QSCALE * sqrt(Dk)` times larger and takes the factor back out of the norm's epsilon, which
the norm otherwise ignores. The l2 norms and the gated norm are also written so that nothing squared leaves fp16's
range. Nothing else needed a change.

Qwen3.5-0.8B, the `vibe` mix, 2 pairs (4 members) x 8 problems, up to 128 tokens, one round, checked by the CPU in
float32 (the arithmetic of Intel, Apple and CPU checkers):

| | worst honest gap | false rejections at 0.1 / 0.375 / 0.5 / 0.75 / 1 | smallest cheat gap | false acceptances |
|---|---|---|---|---|
| before the fix | 6.01 | 4 / 4 / 4 / 4 / 4 | 4.30 | 0 |
| after | 0.065 | 0 / 0 / 0 / 0 / 0 | 3.96 | 0 |

One step's log-probabilities are now 0.047 off float32 (OpenVINO's Intel GPU in fp16: 0.069). The NPU also agrees
with itself: its own answers teacher-forced on it have gaps of exactly 0. So step 3's condition (worst gap under half
the cross margin) holds on this sample. The sample is small, though: a larger round, a CUDA bf16 checker (copy the
answers dump to a CUDA machine and run `check`), and the NPU as checker of others' work are still to measure.

A larger round turned it around. Same setup, 4 pairs (8 members) x 16 problems, up to 256 tokens:

| | worst honest gap | p99 (answer) | false rejections at 0.1 / 0.375 / 0.5 / 0.75 / 1 | smallest cheat gap | false acceptances |
|---|---|---|---|---|---|
| 8 x 16, 256 tokens | 0.522 | 0.313 | 4 / 1 / 1 / 0 / 0 | 4.26 | 0 |

The gaps sit in the long answers. Two problems ran to the 256-token cap for every member, and they hold every gap over
0.05 (0.52, 0.34, 0.23, 0.14, 0.14). Of the other 14 problems (2 to 19 tokens), all but one answer have gaps of 0,
and that one has 0.10. The error grows with the answer's length, which points at
the recurrent state: 18 DeltaNet layers carry it from token to token, and on the NPU it is updated in fp16. That
is a guess, not yet tested. So step 3's condition fails: an honest NPU worker would be rejected at the template's
`cross_margin` 0.5. The next things to try are keeping the state update in float32 on the host (the NPU computes
the step's inputs to it), or splitting the step so the state's part runs on the CPU, then measuring 256-token
answers again.

The NPU updated each DeltaNet layer's recurrent state itself and rounded it back to fp16 every token, so a 256-token
answer carried 256 roundings. The conv window and attention cache are only copied, so they don't add up. Now the step
outputs the update (k, delta, the decay d) and the host applies `S = S * d + k^T delta` in float32 (`es_npu`'s
default; `--npu-device-state` keeps the old path). The same round again:

| | worst honest gap | p99 (answer) | false rejections at 0.1 / 0.375 / 0.5 / 0.75 / 1 | smallest cheat gap | false acceptances |
|---|---|---|---|---|---|
| state in fp16 on the NPU | 0.522 | 0.313 | 4 / 1 / 1 / 0 / 0 | 4.26 | 0 |
| state in float32 on the host | 0.193 | 0.093 | 1 / 0 / 0 / 0 / 0 | 4.07 | 0 |

The worst gap is under half the cross margin (0.25), so step 3's condition holds again on this round. It still sits
mostly in the 256-token answers (0.19, 0.07, 0.06), but much lower. The host update adds a little time per step.
