# Rung 2: scaling to ~222k params on real text

**Status: COMPLETE.** Training finished all 100,000 steps (2.95 hours).
**Verdict: real language-modeling ability achieved; NOT a verified
glass-box model yet.** The generative model learns TinyStories-style text
correctly, but its causal-occlusion attribution fails its own
faithfulness gate at multi-position evaluation (k≥2), and this persists
at full convergence — ruling out "just needs more training." See
"Resolution" at the end for what's confirmed and what's still open.

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

## Training progress (final)

```
step      1  train_loss 4.6858  val_loss 4.0540   (chance level: ln(96)=4.56)
step    500  train_loss 1.4915  val_loss 1.4355
step   1000  train_loss 1.3751  val_loss 1.2747
...
step  28500  train_loss 0.7395  val_loss 0.9168
...
step  52000  train_loss 0.8557  val_loss 0.8908
...
step 100000  (final)             val_loss 0.8831 over the full 599,865-byte held-out set
trained 100000 steps x batch 8 on 100-token chunks in 10616.37s (2.95 hours)
```

Loss dropped from chance level to ~0.88 and effectively plateaued by
around step 50,000 — the last 48,000 steps (roughly half the run) moved
val loss by less than 0.01. Val loss tracked train loss closely
throughout with no widening gap — no overfitting signal at any point.
Rate held at ~90-95ms/step.

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

**The finding.** `faithcheck` run against `model_scale.bin` reports
`FAITHCHECK FAIL` at all three checkpoints tested, including the final,
fully-converged one:

| k | @ 28.5% trained | @ 52% trained | **@ 100% (final)** |
|---|---|---|---|
| 1 | ok | ok | **ok** |
| 2 | INVERTED | INVERTED | **INVERTED** |
| 3 | INVERTED | ok | **ok** |
| 5 | INVERTED | INVERTED | **INVERTED** |

Randomization check passes cleanly at every checkpoint (corr = -0.166,
-0.123, **-0.077** — well inside the ±0.5 threshold and, if anything,
getting cleaner with more training), so the attribution is reading
something real from the trained weights throughout, not producing noise
indistinguishable from an untrained model. The failure is specifically
in the deletion curve at k=2 and k=5: occluding the top-k "important"
positions together does not consistently hurt more than occluding k
random positions.

Ruled out: this is not the out-of-distribution-baseline problem from the
known-answer investigation (same failure on a genuinely held-out
TinyStories prompt, not just a template-corpus prompt). It is also now
**not the undertrained-model hypothesis**: val loss had already
plateaued by the 52% checkpoint (0.8908 there vs. 0.8831 at the true
final step — the last 48,000 steps moved it by less than 0.01), and the
k=2/k=5 failures persisted unchanged regardless.

## Resolution

**Confirmed:** real language-modeling ability was achieved at this scale
(loss dropped from chance level to 0.88, the model correctly reproduces
TinyStories' story-restart convention, no overfitting at any point).
**Confirmed:** causal-occlusion attribution is reading real, trained
structure (randomization check passes cleanly at every checkpoint, and
single-position (k=1) ranking works throughout).
**Confirmed ruled out:** the multi-position deletion-curve failure is not
an artifact of an out-of-distribution test prompt, and not a symptom of
undertraining — more training (52%→100%, essentially flat val loss)
did not change the outcome.

**Standing hypothesis, not yet confirmed:** greedy top-k redundancy in
the deletion-curve test itself. Selecting the top-k positions by
*individual* importance can pick several redundant, adjacent characters
(e.g. multiple letters within one word) whose *combined* occlusion
doesn't hurt proportionally more than occluding just one of them, while a
random k-subset is more likely to land on genuinely independent parts of
the sentence. This is a known, general limitation of naive top-k deletion
curves in the broader attribution literature, not specific to a bug in
this codebase -- but it hasn't been directly demonstrated here yet.

**What would actually confirm or refute it:** a deletion-curve variant
that selects a *diverse* top-k (e.g. greedily excluding candidates whose
position is within a few characters of one already selected) instead of
a naive top-k, re-run against this same checkpoint and prompt. If that
passes where the naive version fails, the redundancy hypothesis is
confirmed and the fix belongs in the test methodology, not the
attribution mechanism. That work is not done yet -- **rung 2 should be
treated as "real language modeling, glass-box status unresolved" until
it is.**
