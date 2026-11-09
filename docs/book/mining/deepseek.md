# DeepSeek jobs

This page covers DeepSeek jobs: what they train, what your computer needs, the big model files they download and when those are deleted, and how to opt in.

## What they train

The first DeepSeek job is **think shorter**, a pilot. It fine-tunes the top layers of **DeepSeek-V4.1-Flash** to reach correct answers with much less thinking, learning from short human-written solutions to grade-school math problems (GSM8K), written in the model's own thinking format.

Because only the top layers are trained, one ordinary gaming PC can take part, instead of a team of big servers.

## What your computer needs

| | Needed |
|---|---|
| Graphics card | NVIDIA, **8 GB or more** (RTX 20-series or newer) |
| Memory | **16 GB** (32 GB is faster) |
| Disk | About **40 GB free on an SSD** |
| Download | About **30 GB**, once |

If your PC can't take DeepSeek work, the worker says why and **keeps earning on other jobs**. Nothing is lost by opting in on a computer that turns out too small.

## Opting in

DeepSeek work is off unless you say yes, because of the download. Without a yes, the worker skips DeepSeek jobs and tells you how to opt in.

From the command line:

```bash
vibehash worker --deepseek yes
```

Your answer is remembered, so the next plain `vibehash worker` does the same. `--deepseek no` turns it off again. You can also answer in advance during setup with `vibehash setup --deepseek yes` (or `no`), which is handy for [unattended machines](../getting-started/first-run.md#running-with-nobody-at-the-keyboard).

Or run `vibehash` and pick **Mine the DeepSeek job** under **Mine**. It explains what it needs and asks before starting.

## The model files

The about 30 GB of model files go into the folder you chose in setup's **Space for big models** step (by default, the drive with the most room), within the space you allowed (40 GB unless you said otherwise).

- **Asked first.** Nothing big is downloaded until you've opted in, and never past your cap. If the files won't fit under the cap or on the disk, the job is refused rather than filling your drive, and other jobs carry on.
- **Not permanent.** A job's model files are deleted when the job ends, or after 7 days without mining it, whichever comes first.
- **Unless you keep them.** If you chose to keep model files after a job ends (to save downloading them again on a slow link), they stay until you clear them. Kept files still make way for your cap.
- **No half-downloads left behind.** Interrupted downloads resume, and ones nobody resumes are cleaned up.

To change where they go, how much space they may use, or whether they're kept, run setup again with `--models-dir`, `--models-gb` or `--keep-models` (see [vibehash setup](../reference/setup.md)).

## vibehash cache

To see what's stored, where, how big, and why:

```bash
vibehash cache
```

To remove it all, or just one job's files:

```bash
vibehash cache clear
```
```bash
vibehash cache clear JOB
```

`vibehash cache sweep` removes files no job has used for a week, and stray partial downloads. In the menu this is **Settings > Model files**; in the desktop app, **Settings > Storage**, where you can also set the cap. Pause mining before clearing; the next job downloads what it needs again.

Every flag is on [vibehash cache](../reference/cache.md).
