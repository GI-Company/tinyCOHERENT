# Rung 2: scaling to ~222k params on real text

**Status: IN PROGRESS.** Training was still running (53,500 / 100,000
steps) when this was last updated. The attribution findings below are
from mid-training checkpoints and are explicitly not final — see "Open
question" at the end. This file will be updated once training completes
and the checkpoint is re-tested.

## Goal

The known-answer and degradation results (`DEGRADE.md`) verified
attribution faithfulness at the original ~8k-param scale on a synthetic
task and a 400-sentence template corpus. Rung 2 tests whether that
faithfulness survives a real scale-up: more parameters, a real corpus,
longer training. Per the standing rule for this project: every time the
architecture or scale changes, the attribution gates get re-run, and if
they fail, this is "a small model," not a glass-box one, until proven
otherwise.

## Configuration

| Setting | Value |
|---|---|
| d_model | 64 |
| n_layers | 4 |
| n_heads | 4 |
| ff_mult | 2 |
| max_seq_len | 128 |
| vocab_size | 96 |
| **param count** | **222,400** (verified via `tc_param_count`, not just estimated) |

Kept as a separate binary (`train_scale.c`) and a separate weights file
(`build/model_scale.bin`) from the verified 8k oracle (`train.c` /
`build/model.bin`) on purpose — the oracle and its gates must never be
touched by this.

## Corpus

TinyStories (GPT-4 variant), sourced from an existing local copy at
`~/gort_corpus/nano/tinystories/TinyStoriesV2-GPT4-valid.txt` — no
external download needed. Subset to the first ~4M characters at story
boundaries (5,007 stories), `<|endoftext|>` separators stripped, common
non-ASCII punctuation (curly quotes, em-dashes) normalized to ASCII
before `tc_sanitize_text` would otherwise blank them to spaces. Written
to `data/tinystories_subset.txt`. Same 85/15 train/val split on a
sentence boundary as the oracle uses.

## Two real bugs caught before they wasted hours of compute

1. **stdio full-buffering.** Redirecting stdout to a log file switches C's
   stdio from line-buffered to fully-buffered, so `printf` output sat in
   an internal buffer indefinitely — the first run showed *zero* output
   after 33 seconds of actual training. Fixed with
   `setvbuf(stdout, NULL, _IOLBF, 0)` at the top of `main()`.
2. **Validation loss scaling with corpus size.** `eval_val_loss` originally
   tiled the *entire* val region into non-overlapping chunks every report
   interval — fine at the oracle's ~2KB val set (~44 chunks), but at this
   corpus's ~600KB val set that's ~6,000 chunks per check, costing ~22s
   every 500 steps. Across a 100k-step run with `report_every=500`, that's
   roughly an extra hour of pure evaluation overhead on top of training
   time. Fixed by capping the number of chunks evaluated (300 during
   training, 2000 for the final check) and striding evenly across the
   whole val region so the sample stays representative.

## Training progress (snapshot, not final)

```
step      1  train_loss 4.6858  val_loss 4.0540   (chance level: ln(96)=4.56)
step    500  train_loss 1.4915  val_loss 1.4355
step   1000  train_loss 1.3751  val_loss 1.2747
...
step  28500  train_loss 0.7395  val_loss 0.9168
...
step  53500  train_loss 0.7417  val_loss 0.8924
```

Loss dropped from chance level to ~0.86-0.89 and continues to decline
slowly; val loss tracks train loss closely throughout with no widening
gap — no overfitting signal so far. Rate holds at ~90-95ms/step
(~2.5-2.7 hours total estimated for the full 100k steps).

A mid-run generation sample at step 15,000 (from a genuinely held-out
prompt): given `" Then they went to the bathroom to wash "`, continued
`"in his secred and she eded what to do.\n\nOnce upon a time, th"` — not
fluent yet, but already reproducing TinyStories' own story-restart
convention correctly.

## faithcheck: a real engineering fix, and an unresolved finding

**Engineering fix (separate from the finding below).** `faithcheck.c`
originally shared one cache, sized to the generative model's config,
between the generative and embedder models. That's a live memory-safety
bug the moment the two models have different configs — exactly the
situation here, since no embedder has been trained at this scale yet.
Fixed: the embedder argument is now optional, each model gets its own
correctly-sized cache, and a prompt override was added (argv[3]) since a
prompt hardcoded for one corpus is not a fair test of a model trained on
a different one. **Regression-checked**: re-run against the original 8k
oracle + its embedder, byte-for-byte identical results to before the
refactor (same correlations, same deletion-curve numbers) — the fix
changed nothing about the case that already worked.

**The finding.** `faithcheck` run against `model_scale.bin` currently
reports `FAITHCHECK FAIL`, at two checkpoints tested so far:

| k | causal occlusion @ 28.5% trained | @ 52% trained |
|---|---|---|
| 1 | ok | ok |
| 2 | INVERTED | INVERTED |
| 3 | INVERTED | ok |
| 5 | INVERTED | INVERTED |

Randomization check passes cleanly at both checkpoints (corr = -0.166,
-0.123 — well inside the ±0.5 threshold), so the attribution is reading
something real from the trained weights, not producing noise
indistinguishable from an untrained model. The failure is specifically
in the deletion curve at k≥2: occluding the top-k "important" positions
together does not consistently hurt more than occluding k random
positions.

Ruled out: this is not the out-of-distribution-baseline problem from the
known-answer investigation. The same failure pattern held on a prompt
drawn from the model's own held-out TinyStories val split, not just the
oracle's hardcoded template-corpus prompt.

**Two live hypotheses, neither confirmed:**
1. **Undertrained model.** Loss is still declining; a half-converged
   model's causal structure may not be consolidated enough yet for
   multi-position occlusion to rank cleanly, even though single-position
   (k=1) ranking already works and randomization already passes.
2. **Greedy top-k redundancy.** A known, general limitation of
   deletion-curve tests: selecting the top-k positions by *individual*
   importance can pick several redundant, adjacent characters (e.g.
   within one word) whose *combined* occlusion doesn't hurt proportionally
   more, while a random k-subset is more likely to hit independent parts
   of the sentence. Plausibly more pronounced on real, information-dense
   text than on the oracle's short repetitive templates.

The k=3 flip from INVERTED to ok between the two checkpoints is a mild
signal toward hypothesis 1, but one flipped data point is not a trend --
this session's degradation sweep already showed how much single-checkpoint
deletion-curve numbers can move from noise alone.

## Open question

Does `faithcheck` pass once training actually completes? That's the
test that distinguishes the two hypotheses above: a clean pass at
convergence supports "just needed more training"; a persistent failure
at convergence means the deletion-curve methodology itself needs
rethinking for real text, not more training steps. Re-run:

```
./build/faithcheck build/model_scale.bin - "<a fresh held-out TinyStories prompt>"
```

against the final checkpoint and update this document with the result
before treating rung 2 as either a pass or a fail.
