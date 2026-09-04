# Rung 6 Systems Milestone: 21.5M Parameter Architecture, BLAS Acceleration & 900-Step Empirical Analysis

## 1. Executive Summary

> **Current Status**: 900-step Rung 6 checkpoint, val ~2.08 (standardized held-out loss: 2.1637), faithcheck still pass, generation not yet coherent.

Rung 6 represents a pure systems milestone for TinyCoherent (GLASSBOX):
- **Systems Demonstration**: A 21,534,208 parameter hybrid recurrent-attention model ($D=512, L=6, H=8, V=2048, \text{ff\_mult}=2, T=128$) running entirely in pure C with zero third-party ML framework dependencies.
- **Hardware Acceleration**: Replaced scalar C loops in `matvec` and `matvec_backward` with native Apple Accelerate SIMD BLAS calls (`cblas_sgemv`, `cblas_sger`), yielding a sustained throughput of **~400 tokens/second** during 8-worker parallel training on Apple Silicon.
- **Attribution Gate Stability**: Causal occlusion deletion curves survived loss reduction from the high-7s down into the low-2s without collapsing ($1.95\times$ impact vs random baseline).
- **Linguistic Reality**: Total training exposure across 900 steps is **720,000 tokens**—still an early warmup. The model exhibits broken syntax and mode-collapsing completions; it is **not** a fluent language model and is **not** ready for production export (GGUF deferred).

---

## 2. Checkpoint Provenance & Frozen Artifact

To ensure exact reproducibility and prevent silent overwrites, the 900-step checkpoint has been frozen:

- **File Path**: `build/model_rung6_step900.bin`
- **SHA-256 Hash**: `ce50612d4ab4c431673478fd844f625319f2490fbeac6c3fad94ba0db59f6ce1`
- **Parameter Count**: 21,534,208 FP32 floats (86.14 MB)
- **Configuration**: $V=2048, D=512, L=6, H=8, \text{ff\_mult}=2, T_{\text{max}}=128$

---

## 3. Standardized Validation Benchmark Protocol

### The Need for Protocol Standardization
Earlier intermediate logging sampled varying chunk counts ($N=25$ vs $N=50$ vs $N=100$), which altered sampling stride across the validation split and introduced artificial variance (e.g. 3.51 vs 2.61). 

### Standardized Benchmark Definition
All evaluation is now locked to a single deterministic recipe implemented in `src/train_scale.c`:
- **Slice**: Exactly the held-out 15% split of `data/tinystories_subset.txt` (278,872 tokens).
- **Chunk Length**: 100 tokens.
- **Sample Count**: Exactly **100 fixed chunks (10,000 held-out tokens)** evaluated at deterministic strides.

Under this locked benchmark protocol:
- **`model_rung6_step900.bin` Exact Held-Out Loss**: **2.1637**

---

## 4. Official Qualitative Evaluation: Before vs. After

To prevent cherry-picking, evaluation is locked to **two official benchmark prompts** decoded deterministically at **Greedy ($T=0$)**, temperature 0, fixed seed:

### Benchmark Pair 1
- **Prompt**: `"Timmy found a shiny red ball in the garden."`
- **Step 300 Output**:
  > `"Timmy found a shiny red ball in the garden. I a little bird to the bird had very happy to the ground away the dog the "`
- **Step 900 Output (Greedy $T=0$)**:
  > `"Timmy found a shiny red ball in the garden. The bird was very happy and said, "I will help you like the park. The bird was very "`

### Benchmark Pair 2
- **Prompt**: `"The puppy was very hungry, so he"`
- **Step 300 Output**:
  > `"The puppy was very hungry, so he a big happy and the  the  named there was a big tree home the bird the "`
- **Step 900 Output (Greedy $T=0$)**:
  > `"The puppy was very hungry, so he was very happy. The bird was very happy and said, "I will help you like the park. The"`

### Empirical Diagnostic
1. **Progress**: The model reduced exact repetitive unigram stuttering (*"to the bird had very happy to the ground away the dog the"* $\to$ grammatical phrases).
2. **Failure Mode**: Both prompts collapse greedily into the identical high-frequency memorized n-gram pattern (*"The bird was very happy and said, 'I will help you like the park. The..."*). 
3. **Verdict**: The model has acquired local subword bigram/trigram transition probabilities but possesses **zero plot maintenance or multi-sentence narrative grounding**.

---

## 5. Mechanistic Faithfulness & Attribution Stability

A central research question in glass-box scaling is whether mechanistic interpretability collapses as loss drops:

| Checkpoint | Training Exposure | Held-Out Val Loss | Causal Deletion Ratio ($k=1..5$) | Random Weight Correlation | Steering Identity ($\alpha=0$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Rung 4 (3.9M)** | ~1.5M tokens | ~1.85 | $1.85\times$ avg impact | $\text{corr} = -0.077$ (PASS) | Bit-identical |
| **Rung 6 (300 step)** | 240k tokens | ~2.60 | $2.04\times$ avg impact | $\text{corr} = +0.164$ (PASS) | Bit-identical |
| **Rung 6 (900 step)** | 720k tokens | **2.1637** | **$1.95\times$ avg impact** | $\text{corr} = +0.160$ (PASS) | Bit-identical |

### Analysis
- **Attribution Stability**: Attribution did not immediately degrade as loss moved from 7.7 down to 2.16. The deletion ratio shifted slightly ($2.04\times \to 1.95\times$), which represents normal single-run sample variation rather than an attribution breakdown.
- **Latent Steering Invariance**: $\alpha = 0.0$ residual injection remains mathematically identical to unsteered baseline inference ($\max |\Delta z| = 0.00000000$).

---

## 6. Systems Scaling & Architecture Specifications

| Parameter | Specification |
| :--- | :--- |
| **Hidden Dimension ($D$)** | 512 |
| **Number of Layers ($L$)** | 6 |
| **Attention Heads ($H$)** | 8 (head dimension $D_h = 64$) |
| **FFN Expansion Ratio** | 2 ($FF = 1024$) |
| **Vocabulary Size ($V$)** | 2,048 byte-level BPE subwords |
| **Context Horizon ($T$)** | 128 tokens |
| **Total Parameters** | **21,534,208** (86.14 MB FP32) |
| **Linear Algebra Engine** | Apple Accelerate BLAS (`cblas_sgemv`, `cblas_sger`) |
| **Parallel Execution** | Grand Central Dispatch (`dispatch_apply`), 8 worker cores |
| **Observed Throughput** | **~396–410 tokens/second** |

---

## 7. Next Research Directions (GGUF Deferred)

Exporting to GGUF format for `llama.cpp` is explicitly deferred. Deploying an undertrained 21.5M checkpoint alongside fluent production LLMs does not advance the project's scientific value.

### Priority Research Agenda:
1. **Extended Token Scaling**: Train across tens of millions of tokens using the standardized evaluation recipe.
2. **The Faithfulness vs. Fluency Curve**: Periodically evaluate `faithcheck` deletion ratios across training checkpoints to plot **validation loss vs. causal deletion ratio**. The primary scientific inquiry is: *Does mechanistic glass-box attribution remain faithful as the language model transitions from topic soup to true narrative fluency?*
3. **Planted-Fact Grounding**: Implement verified retrieval benchmarks before attempting model distribution.
