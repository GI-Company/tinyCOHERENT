# TinyCoherent

A from-scratch, pure-C test of one question: can a tiny (~8k parameter)
hybrid language-model specialist run on ordinary consumer hardware while
staying **glass-box** — not just emitting explanations, but explanations
that are verified against tests that could come back negative, the same
way this codebase's hand-derived backward pass is verified against
numerical gradients rather than trusted on sight.

## What's here

- **Core specialist** (`tcmodel.c`/`.h`): a 2-layer hybrid block —
  gated linear recurrence (RWKV-lineage token mixer) + tiny causal
  self-attention + gated channel-mix FFN — with a hand-written forward
  and backward pass verified against finite differences (`make gradcheck`).
- **Registry + DAG** (`registry.c`, `dag.c`): multiple named model
  instances, and a minimal graph runner chaining them (generate → embed →
  retrieve → explain).
- **Embedder** (`embed_train.c`): the same architecture, a different head
  (attention-pooling instead of vocab projection), trained with a
  self-supervised contrastive objective.
- **RAG** (`vectorstore.c`): brute-force cosine retrieval over the
  embedder's output, wired into the DAG as a normal node.
- **Attribution verification** (`glassbox.c`, `faithcheck.c`,
  `known_answer.c`, `degrade_test.c`): the actual point of this project.
  See below.

## The result that matters

Most systems that call themselves "explainable" or "glass-box" emit an
attribution and stop there — the explanation is never itself tested. This
project's attribution methods are:

- **Gradient-checked** at the mechanism level (`make gradcheck`,
  `make embed_gradcheck`) — the same discipline applied to the model's
  own training.
- **Faithfulness-gated** (`make faithcheck`) — a randomization sanity
  check (attribution should NOT correlate with an untrained model's) and
  a deletion-curve test (occluding the positions attribution calls most
  important should hurt more than occluding random ones), both pass/fail
  like a regression suite.
- **Verified against ground truth** (`make known_answer`) — a synthetic
  task where the causally important position is true by construction.
  Causal occlusion attribution recovers it in **300/300 held-out
  examples**, with every other position's importance rounding to
  **0.0000 bits**.
- **Stress-tested under controlled degradation**, with multi-seed
  replication distinguishing real effects from optimization noise — see
  [`DEGRADE.md`](DEGRADE.md). Headline result: attribution's top-1
  recovery degrades gracefully and monotonically under label noise (100%
  → 78.7% as noise rises from 0% to 30%) with almost no seed-to-seed
  variance (std ≤ 0.016 throughout), even though the underlying model's
  raw training accuracy on the same runs is noisy and non-monotonic.
  Read `DEGRADE.md` for what's established vs. still provisional — not
  every number in this project's history has held up to replication
  (the ratio metric and some capacity-axis claims explicitly haven't),
  and that document says so plainly rather than rounding up.

## Scaling milestones

- **Oracle (~8k params, `d_model=16`)**: fast finite-difference baseline on synthetic templates.
- **Rung 2 (~222k params, `d_model=64`)**: [`SCALE_200k.md`](SCALE_200k.md) — trained on TinyStories, discovered and resolved greedy top-k deletion curve redundancy.
- **Rung 3 (~3.45M params, `d_model=256`)**: [`SCALE_3M.md`](SCALE_3M.md) — Apple Accelerate BLAS acceleration, verified faithful attribution under diverse deletion curves and weight randomization.
- **Rung 4 (~3.95M params, subword BPE, GCD multi-core)**: [`SCALE_RUNG4.md`](SCALE_RUNG4.md) — pure-C byte-level BPE tokenizer ($V=2048$), multi-core parallel batch training, eliminates character redundancy so naive top-$k$ deletion curves pass cleanly at every $k$, and interactive glass-box chat with word-level causal and input-gradient attribution.
- **Rung 5 (Grounded Glass-Box RAG, Mechanistic Head Attribution & Visual Studio)**: causal mediation loss shifts across all 32 attention heads ($L=4, H=8$), in-memory grounded RAG with exact Evidence Grounding Ratio & hallucination detection, and an interactive dark-mode visual studio web dashboard (`make ui`).
- **Milestone 1 Alignment (Instruction SFT & Conversational Glass-Box)**: prompt-masked loss training over 3,500 dialogue pairs, stop sequence delimiter monitoring (`<|endoftext|>`, `User:`), and query-focused attribution in chat (`make train_sft`, `make chat_sft`).
- **Milestone 2 Steering (Latent Activation Steering & Concept Vectors)**: in-engine latent activation injection ($x_{t, l^*} \leftarrow x_{t, l^*} + \alpha \hat{\mathbf{v}}_{l^*}$), concept vector bank (`joy`, `danger`, `magic`, `nature`), vocabulary projection inspection ($\Delta \mathbf{z} = W_{\text{embed}} \cdot \hat{\mathbf{v}}$), zero-overhead equivalence at $\alpha=0$, and REPL/Web studio interactive controls (`make extract_steer`, `make test_steer`).
- **Milestone 3 Scaling (Rung 6 ~21.5M Parameter Systems Milestone)**: [`SCALE_RUNG6.md`](SCALE_RUNG6.md) — 900-step Rung 6 checkpoint (21,534,208 parameters, $D=512, L=6$), standardized held-out loss 2.1637, Apple Accelerate SIMD BLAS throughput ~400 tok/s on Apple Silicon, causal attribution gates survive continue ($1.95\times$), generation not yet coherent.

## Building and running

```
make gradcheck        # verify the hand-derived backward pass (unmasked & masked loss)
make embed_gradcheck  # same, for the embedder's attention-pooling head
make train             # train the generative specialist, train/val split
make train_scale       # train Rung 4 scale model with BPE and multi-core GCD
make train_rung6       # train Rung 6 21.5M scale model with BLAS and multi-core GCD
make train_sft         # instruction fine-tuning on dialogue pairs with prompt masking
make extract_steer     # extract curated concept steering vectors into binary bank
make test_steer        # verify activation steering (zero-shift equivalence & monotonicity)
make embed_train       # contrastively train the embedder
make chat              # interactive glass-box terminal chat (model_rung3.bin)
make chat_rung4        # interactive glass-box terminal chat with BPE (model_rung4.bin)
make chat_rung6        # interactive glass-box terminal chat with Rung 6 (model_rung6.bin)
make chat_sft          # interactive conversational assistant with query attribution & steering
make ui                # launch the interactive visual glass-box web studio with steering controls
make dag                # 2-node DAG demo: generate -> explain
make rag                 # 3-node DAG demo: retrieve -> generate -> explain
make faithcheck            # attribution faithfulness gate (pass/fail)
make faithcheck_rung4      # attribution faithfulness gate for Rung 4 BPE model
make faithcheck_rung6      # attribution faithfulness gate for Rung 6 21.5M model
make known_answer           # ground-truth attribution test
make degrade_test            # ./build/degrade_test <noise_p> <jitter_q> <capacity_divisor> [seed]
```

See [`DEGRADE.md`](DEGRADE.md), [`SCALE_200k.md`](SCALE_200k.md), [`SCALE_3M.md`](SCALE_3M.md), [`SCALE_RUNG4.md`](SCALE_RUNG4.md), and [`SCALE_RUNG6.md`](SCALE_RUNG6.md) for experimental methodology, results, and caveats.
