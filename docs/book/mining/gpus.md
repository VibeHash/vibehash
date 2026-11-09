# GPUs

This page covers mining on a graphics card: installing GPU support, checking what your computer can use, which cards work, how much memory each kind of job needs, and why an NVIDIA card and a Mac can check each other's work.

## Install GPU support

AI is trained on graphics cards, and Vibe Hash uses them through PyTorch. Setup offers to install the right PyTorch for your card. To do that step on its own:

```bash
vibehash install-gpu
```

It reads your card and driver, checks which PyTorch builds are published for your driver and Python version, installs the newest one that fits (a download of up to 3 GB), and self-tests it. If it says your driver is too old, update it from your card maker's website and run it again. Add `--yes` to skip the question before downloading.

Without a GPU, or without PyTorch, the worker trains on the CPU. It's slower, but it's correct, and it's paid at the same rate per unit of work.

## See what your computer can use

```bash
vibehash devices
```

This lists the CPUs and GPUs this computer can train on, and what each registers as with the network (such as `RTX 4060, 8 GB (nvidia sm_89)`). The network counts machine kinds for its public page, never with an address attached.

For a fuller check, run the self test. In a minute or so, with no wallet needed and nothing spent, it says whether this computer can mine and what it can take:

```bash
vibehash selftest
```

It checks your software, every GPU's self-test and free memory, trains one real unit of the public network's demo job on the device the worker would use and checks it the way verifiers do, checks the node, your clock and download speed, your wallet, and disk space. It ends with what your computer can take and the command to run next. `--device cuda:1` tests a particular GPU, and `--model qwen3-0.6b` also trains one real fine-tune unit (a 1.5 GB download, asked first). The menu has it as **Check this computer**, and the desktop app as its **System check** tab.

Workers use the first GPU they find. `--device cpu|cuda|cuda:1|xpu|mps` picks another. One worker uses one GPU; for more, see [Several GPUs](worker.md#several-gpus).

## Which cards work

| Hardware | Status |
|---|---|
| **NVIDIA RTX 20-series and newer** | Mines everything: classic jobs, Evo Hash and fine-tunes. The tested path. RTX 50-series needs a recent driver (570 or newer); `install-gpu` picks a build that has it. |
| **Older NVIDIA** (GTX 10-series, Titan V, V100) | Works, but PyTorch 2.14 is the last version for these cards. Evo runs slowly. |
| **AMD on Linux** (Radeon RX 7000 and 9000, Instinct, Ryzen AI Max) | Runs the same code as NVIDIA, through PyTorch's ROCm build. RX 6000 works unofficially. Not yet measured against NVIDIA checkers. |
| **AMD on Windows** | Mines on the CPU for now: PyTorch's AMD builds are Linux-only. |
| **Intel Arc and Core Ultra graphics** | Classic jobs and fine-tunes; Evo in a slower mode. |
| **Apple M-series Macs** | Like Intel. The newest PyTorch needs macOS 15 to use the GPU; on macOS 14, `install-gpu` installs an older PyTorch that still does. |
| **NPUs** (Intel AI Boost, AMD XDNA, Qualcomm) | Not used for mining yet. On an NPU laptop, mine on its built-in Intel or AMD graphics. |
| **CPU only** | Works. Slow but correct, and the reference that classic jobs are checked against. |

DeepSeek jobs are narrower: they need an NVIDIA card with 8 GB or more. See [DeepSeek jobs](deepseek.md).

## How much graphics memory

The worker reports your graphics memory, and you're only given jobs that fit in it.

| Work | Memory |
|---|---|
| Classic jobs (small models on spreadsheets) | Small; each job states its own need |
| Fine-tune of Qwen3-0.6B | About 3.5 GB (measured) |
| Fine-tune of Qwen3-4B | About 11 GB (measured) |
| The public Evo network | A GPU of 8 GB or more |
| DeepSeek jobs | NVIDIA, 8 GB or more, plus 16 GB of system memory |

## Different hardware, same answers

Chips round numbers very slightly differently, so a GPU's result and a CPU's replay never match bit for bit. Vibe Hash is built for that:

- **Shortcuts are off.** Workers turn off GPU modes that trade accuracy for speed, such as NVIDIA's TF32, and use repeatable calculations.
- **Checks allow for rounding.** A classic job's replay must match within a tolerance hundreds of times wider than the differences measured between machines, but too tight to let skipped work through.
- **Evo matches like with like.** Evo Hash compares the model's word choices, where the gaps between kinds of hardware are bigger. Each worker tells the network its number setup (kind of chip, number format, PyTorch version), and Evo jobs can use a tight margin between matching machines, a wider one across kinds, and a matching machine as referee. Before taking an Evo job, a worker also checks that its card reproduces its own answers exactly.

NVIDIA's RTX 40-series is the tested path; other families are supported by PyTorch but less measured.
