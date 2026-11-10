# Evo gold scoring and pool tasks

## Gold scoring

Before `gold` (`es2.Spec2.gold`), a round scored its members on the replay targets: earlier rounds' answers the line
already got right. A round could only make it surer of those and never learned from a problem it got wrong; with
`lenient`, a right answer without its tags became a target and taught dropping them. Every scoreboard of that time tied
the original model (vibe-08b-2 on Qwen3.5-0.8B, the Qwen2.5 line).

With `gold`, members are scored on writing the round's own problems' right answers (`<answer>gold</answer>` for the
short-answer tasks; every task but `es_tasks.WRITTEN`, whose problems keep replay targets). Measured on
Qwen2.5-0.5B-Instruct (2026-11-09, RTX 4070 SUPER, 64 pairs x 16 problems, lr 5e-4, decay 1/256, 10 tasks mixed, 100
rounds, 200 held-out problems a task, paired): the trained tasks 36.9% to 52.4% right, 7 tasks never trained 33.5% to
41.9%, 2.4% from the base. What it learned is the job's answer format, not knowledge: asked for only a letter, the base
answers A to 84-100% of five multiple-choice tasks' problems, and the trained line picks. On Qwen's own tests, which ask
for reasoning first, it ties the original (scripts/vibe_bench.py, 200 problems: MMLU-Redux -1.0 ± 7.7, MMLU-Pro
+1.0 ± 5.0, GSM8K -1.5 ± 6.6). Pool tasks score in the benchmarks' own format.

## Pool tasks

A pool task is a set of training problems, each with a whole target reply, shipped inside the `vibehash` package
(`vibehash/data/`) and pinned by its sha256 in `es_tasks.POOLS`. A `gold` Evo job that names one scores each
round's members on how likely they are to write the target replies (`es2_rounds.gold_text`), with
`target_tokens` of room for them (`Spec2.target_tokens`). Members still write their own short answer to each problem
(16 tokens by default), so the answer check can still tell one member's model from another's. The targets cost the
scoring pass, not decoding, so a pool round costs a fraction of a round that writes long answers.

```
vibehash job template --evo Qwen/Qwen3.5-0.8B --tasks vibe_reason --id vibe-08b-reason-1 --rounds 200 --out job.json
vibehash post-job --spec job.json --budget 10
```

`--tasks vibe_reason` turns on `gold`, sets `target_tokens` 1024, answers of 16 tokens and no replay targets
(every problem is a gold one). Nodes and workers need 0.5.28 or later.

### vibe_reason

Built by `scripts/build_evo_pool.py` (2026-11-09). Every prompt is asked the way `scripts/vibe_bench.py` asks
Qwen's benchmarks: multiple choice says "Think step by step, briefly, then end your reply with "The answer is X"";
math says "Please reason step by step, and put your final answer within \boxed{}".

| Source (training split only) | Target reply | Problems kept |
|---|---|---|
| GSM8K train (es_tasks' training part) | Qwen3.5-4B's greedy reply, right and finished | 140 of 150 |
| MMLU validation + dev (not its test split) | Qwen3.5-0.8B's own right reply of median length, 2+ of 8 twisted copies right | 181 of 250 |
| MedMCQA train | the same | 78 of 100 |
| AQuA train | the same | 56 of 100 |

455 problems, 645 training rows, 500 held-out rows (GSM8K, MATH, MedMCQA, AQuA and MMLU's held-out tenth); 0.3 MB.
MATH is left out of the targets for now: Qwen3.5-4B's MATH replies run long enough that writing them on a 12 GB card
took about 15 s a problem, and the gradient test above didn't move MATH-500.

Problems the untwisted Qwen3.5-0.8B gets wrong appear twice. The "eval" rows are held-out problems of the same
sources, without targets.

Why these targets, measured with gradient steps toward a 363-problem version of this pool (LoRA, 2026-11-09):

- The model's own **shortest** right replies made replies 20-35% shorter and did not raise accuracy (on multiple
  choice the shortest right reply is often a lucky guess). Right replies of **median** length: a tie on accuracy
  (+1.0 ± 3.8 over 670 held-out problems), shorter replies.
- Qwen3.5-4B's replies for every source: GSM8K +9.0 ± 7.1 held out, but multiple choice took on the 4B's longer
  replies without its knowledge and slipped.
- The mix above: +4.5 ± 3.9 over the 670 held-out problems. On Qwen's own benchmarks (scripts/vibe_bench.py, 200
  items each, paired against the official model): MMLU-Pro 29.0 to 35.5 (+6.5 ± 6.4), MMMLU +5.0 ± 7.4, SuperGPQA
  +4.5 ± 6.1, GSM8K +1.5 ± 6.1, C-Eval +0.5, MMLU-Redux -0.5, MATH-500 -0.5, IFEval strict -1.5 ± 6.3; no change
  outside its interval but MMLU-Pro's; replies 10-25% shorter on every benchmark, cut-off replies down on all of them
  (GSM8K 14% to 4%, SuperGPQA 42% to 28.5%).

That model was made with gradient steps, not by Evo: the pool job is the test of whether Evo gets there too.

Licenses: GSM8K and MATH (MIT), MMLU (MIT), MedMCQA and AQuA (Apache-2.0); the replies are Qwen3.5-4B's and
Qwen3.5-0.8B's (Apache-2.0). SciQ (CC BY-NC) and ARC (CC BY-SA) are left out of the shipped file.
