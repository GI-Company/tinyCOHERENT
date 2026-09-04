# Rung 4: Subword BPE Tokenization, Multi-Core Scaling & Glass-Box Attribution

## 1. Executive Summary

Rung 4 represents a fundamental evolutionary leap for TinyCoherent (GLASSBOX):
- **Subword BPE Tokenization ($V=2048$)**: Pure-C byte-level Byte Pair Encoding replacing 96-ASCII character encoding. Achieves **2.15x–2.28x token compression** on TinyStories.
- **Solving the Attribution Redundancy Confound**: In character-level models, adjacent characters within words exhibited collinear attribution, requiring non-maximum suppression (`min_gap`) to avoid hurting deletion curves. In Rung 4 BPE, semantic concepts ("Once", "little", "girl") are atomic tokens; naive top-$k$ deletion curves now pass cleanly at **every** $k \in \{1, 2, 3, 5\}$ with no gap skipping required (`min_gap=1`).
- **Multi-Core Parallel Training**: Grand Central Dispatch (`dispatch_apply`) parallel batch computation delivering **$>4,200$ tokens/second** on Apple Silicon without third-party frameworks.
- **Architectural Headroom**: Core engine expanded to support hidden dimension up to $D=1024$ and FFN dimension up to $4096$ with dynamic cache allocation.
- **Glass-Box Interactive Chat**: Streaming subword dialogue engine with real-time token surprise telemetry, causal occlusion attribution (`/explain`), and analytical gradient attribution (`/grad`).

---

## 2. Tokenization: Byte-Level BPE in Pure C

### Redundancy Problem at Character Scale
In Rungs 1 through 3, models operated over 96 ASCII characters. While mathematically pure, character-level modeling suffers from:
1. **Collinear Attribution**: Individual characters in a word (e.g. `l`, `o`, `g`) provide redundant predictive signals. Occluding `o` alone often yields negligible loss shift because `l` and `g` strongly reconstruct the word.
2. **Context Horizon**: A context length of $T=128$ characters only covers 20–25 English words.

### Byte-Level BPE Engine
- Trained 1,792 merges starting from 256 byte primitives ($V=2048$).
- Linear-time $O(N)$ piece-wise encoder in pure C (`src/bpe.c`), encoding 4 MB of text in $<0.05$ seconds with zero heap allocations during the tokenization loop.
- Full lossless invertibility (`bpe_decode`) reconstructs original text byte-for-byte.

---

## 3. Scaling & Multi-Core Batch Parallelism

### Architecture Configurations

| Parameter | Rung 1 (Oracle) | Rung 2 | Rung 3 | Rung 4 |
| :--- | :--- | :--- | :--- | :--- |
| **Vocab ($V$)** | 96 chars | 96 chars | 96 chars | **2048 BPE subwords** |
| **Dimension ($D$)** | 64 | 128 | 256 | **256** (scale-ready to 1024) |
| **Layers ($L$)** | 1 | 2 | 4 | **4** (up to 8) |
| **Heads ($H$)** | 2 | 4 | 4 | **8** |
| **Parameters** | 8,448 | 222,080 | 3,445,504 | **3,945,216** (or 21.5M at $D=512, L=6$) |
| **Throughput** | 1,200 tok/s | 3,100 tok/s | 3,800 tok/s | **4,250+ tok/s** (GCD 8-worker) |
| **Effective Context**| ~20 words | ~20 words | ~20 words | **~60–80 words** |

### Grand Central Dispatch Parallelism
Each training step executes $B=8$ forward and backward passes concurrently on separate worker threads:
```c
dispatch_apply(batch_size, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t b) {
    tc_paramset_zero(grads_ptr[b]);
    tc_forward(p, caches_ptr[b], ids, chunk_len, targets, &loss);
    tc_backward(p, grads_ptr[b], caches_ptr[b], ids, chunk_len, targets);
    losses_ptr[b] = loss;
});
```
Gradients are atomically accumulated and updated using Adam with a cosine learning rate schedule.

---

## 4. Falsifiable Faithfulness Gate (`faithcheck`)

A model is never declared a glass-box without passing the falsifiability battery:

```bash
./build/faithcheck build/model_rung4.bin -
```

### Deletion Curve Verification
Occluding top-$k$ BPE tokens identified by causal attribution shifts loss significantly more than occluding random tokens:
- **$k=1$**: Top occlusion shift $= +0.3361$, Random shift $= +0.1329$ (**$2.53\times$ impact**, PASS)
- **$k=2$**: Top occlusion shift $= +0.7913$, Random shift $= +0.4268$ (**$1.85\times$ impact**, PASS)
- **$k=3$**: Top occlusion shift $= +1.0543$, Random shift $= +0.5131$ (**$2.05\times$ impact**, PASS)
- **$k=5$**: Top occlusion shift $= +1.8176$, Random shift $= +1.0142$ (**$1.79\times$ impact**, PASS)

### Randomization Sanity Check
- Random-initialization attribution correlation: $\text{corr}(\text{trained}, \text{random}) = -0.054$ (**PASS**, threshold $< 0.50$).
- Proves that the explanation reflects learned model parameters rather than input token distribution.

---

## 5. Interactive Chat & Interpretability REPL

Run the glass-box conversational interface:
```bash
make chat_rung4
```
Features:
1. **Streaming Generation**: Word/subword generation in real time.
2. **Token Surprise Telemetry**: Calibrated negative log-likelihood per generated token.
3. **`/explain`**: On-demand causal occlusion attribution across all prompt tokens.
4. **`/grad`**: Analytical input-gradient attribution ($\nabla_{x_0} \mathcal{L}$).
5. **`/color on`**: Real-time ANSI color highlighting of surprise tokens.
