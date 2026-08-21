# Attribution faithfulness: known-answer task and controlled degradation

This records the evidence behind the claim that `tc_explain_causal` (causal
occlusion attribution on the generative head) is faithful — that its
importance scores reflect genuine causal structure in the trained model,
not an artifact of the method or the input's surface shape. Everything
here follows the same standard as `gradcheck`: don't trust a mechanism
until a test that could come back negative has run and passed.

## 1. Known-answer task: 100%, with a real lesson attached

**Task.** 8-character sequences of random digits. Positions 0,1,2,4,5,6 are
independent noise. Position 3 is a random digit too, but the final
character is forced to equal position 3 — a pure positional copy, with no
value-based shortcut available. Ground truth for "what mattered" is true
by construction, not inferred.

**First attempt failed, correctly.** The model was trained for 3,000 steps
and reached 11.0% accuracy (chance = 10.0%) — it hadn't learned the task.
The test aborted rather than scoring attribution against a model with
nothing to attribute. Extending training to 40,000+ steps reached 100%
copy accuracy, but was unstable (oscillating 100% ↔ 15% even in the final
10,000 steps) until a plateau-triggered learning-rate decay was added
(explore at the original lr, decay once accuracy stops improving).

**The real finding: occlusion baseline choice matters, a lot.** With a
stably-trained, 100%-accurate model, causal occlusion using a **space**
token as the substitution baseline recovered position 3 as the top-ranked
position in only 29.3% of held-out examples — worse than the same test run
against an *unstable* model (73.0%). The cause: this task's entire
training distribution is digits `0`–`9`. Space is wildly out-of-distribution
for it, not neutral — occluding with it produces effects from unfamiliarity,
not from genuine causal importance. Switching to a Monte-Carlo baseline
(average the occlusion effect over several random *in-distribution* digit
substitutions instead of one fixed token) resolved it completely:

```
position 3 is the top-ranked position in 300/300 held-out examples (100.0%)
mean importance at position 3:        7.2372 bits
mean importance at every other position: 0.0000 bits
```

`tc_explain_causal` now takes an explicit `occlude_id` parameter (was
hardcoded to space) because of this. Callers on natural-language text
still correctly use space; anything on a different distribution must pass
something from that distribution.

Reproduce: `make known_answer`

## 2. Multi-seed noise sweep — the main result

Label noise: the training target is corrupted to a random wrong digit with
probability `noise_p`, independently per example. Same architecture, same
occlusion methodology as above, jitter held at 0. Every condition run with
3 seeds (seed 0 reproduces the original single-seed sweep exactly, seeds
1–2 vary both weight init and training data order).

| noise_p | top-1 recovery (mean ± std) | range | final accuracy (mean ± std) | range |
|---|---|---|---|---|
| 0.00 | **1.000 ± 0.000** | 1.000–1.000 | 0.958 ± 0.073 | 0.874–1.000 |
| 0.05 | **0.972 ± 0.002** | 0.970–0.973 | 0.793 ± 0.295 | 0.452–0.964 |
| 0.10 | **0.938 ± 0.016** | 0.920–0.950 | 0.851 ± 0.122 | 0.710–0.926 |
| 0.20 | **0.869 ± 0.011** | 0.857–0.877 | 0.581 ± 0.229 | 0.404–0.840 |
| 0.30 | **0.787 ± 0.009** | 0.780–0.797 | 0.730 ± 0.034 | 0.696–0.764 |

**This is the main scientific result of the synthetic campaign: top-1
attribution recovery degrades gracefully and monotonically under label
noise, and does so with almost no seed-to-seed variance** (std ≤ 0.016 at
every noise level). Never collapses toward the ~14% chance floor (1-in-7
candidate positions) even at 30% label noise.

**Accuracy-vs-attribution decoupling.** Final training accuracy, measured
on the exact same models, is far noisier — its std is 5–20x larger than
top-1 recovery's at every noise level, and it is *not* monotonic in
noise_p (p=0.05's single-seed accuracy alone ranges 45.2%–96.4%). This
means: whether a specific training run got lucky is a noisy, unreliable
signal; whether the trained model's causal mechanism is correctly
identifiable by occlusion is a stable one. Judge this method by top-1
recovery, not by how well any one run happened to converge.

**Worst condition tested** (noise_p=0.30, jitter_q=0.20, joint): top-1
recovery 0.636 ± 0.094 (range 0.527–0.693). Still well above chance, but
this is the one condition where real seed-dependent instability shows up,
not just measurement noise — flagged rather than smoothed over.

Reproduce: `for p in 0.00 0.05 0.10 0.20 0.30; do for s in 0 1 2; do ./build/degrade_test $p 0.0 1 $s; done; done`

## 3. Caveats (read before citing any number above)

**Ratio (mean importance at source ÷ mean importance at strongest other
position) is not a reliable point estimate.** At noise_p=0.00 alone, ratio
ranges from 81.8 to 2682.9 across three seeds — a 33x spread on the
*easiest* condition tested, because the denominator approaches zero near
perfect performance and tiny absolute differences blow up the ratio.
Treat any single ratio value in this codebase's output as order-of-
magnitude only, never as a precise or seed-to-seed comparable number.

**Capacity-axis results are provisional — single-seed except where noted.**
- Half-width (`d_model=8` vs. the default 16) on the *clean* task: **confirmed
  robust**, 1.000 ± 0.000 top-1 recovery and accuracy across 3 seeds.
- Quarter-width (`d_model=4`) failing the clean task outright (29.6%
  accuracy, a capacity cliff between ½ and ¼ width): **single-seed,
  unreplicated.**
- Half-width apparently *outperforming* full width under moderate
  degradation (noise_p=0.20, jitter_q=0.10): **does not survive
  replication.** 3-seed half-width top-1 recovery is 0.811 ± 0.025
  (range 0.783–0.830); full width's single available data point (0.797)
  falls inside that range. No detectable width effect at moderate
  degradation was actually established — the original single-seed
  comparison was noise. (A fully symmetric check would need full width
  replicated too, which this round didn't do.)

Do not cite the capacity cliff or any width-under-noise interaction as
established without further multi-seed replication. The noise-axis
top-1-recovery result above does not depend on the capacity findings and
stands on its own.
