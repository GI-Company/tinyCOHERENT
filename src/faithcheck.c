/* Attribution faithfulness gate, in the same spirit as gradcheck: don't
 * trust that an "explanation" means anything just because it's emitted --
 * run tests that could come back negative, exactly the way gradcheck
 * doesn't trust hand-derived backprop until finite differences agree.
 *
 * Two tests, run against every attribution method this repo claims:
 *   1. Randomization sanity check -- attribution on the trained model vs.
 *      a freshly random-initialized one should NOT correlate. If it does,
 *      the "explanation" is reading the input's shape, not the model's
 *      learned weights.
 *   2. Deletion curve -- occluding the positions an attribution method
 *      calls most important should hurt more than occluding random
 *      positions, at every k tested. If it doesn't, the ranking isn't
 *      causally meaningful.
 *
 * Three methods are checked:
 *   - embedder occlusion (tc_explain_embedding)      -- gates the exit code
 *   - generative causal occlusion (tc_explain_causal) -- gates the exit code
 *   - generative surprise (tc_explain_generate)     -- diagnostic only,
 *     included because it's known NOT to be attribution (see glassbox.h);
 *     printed so the failure is visible and reproducible, but it does not
 *     fail the build the way a real attribution regression would. */
#include "tcmodel.h"
#include "tokenizer.h"
#include "glassbox.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int encode(const char *text, int *ids, int cap) {
    char buf[512];
    strncpy(buf, text, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    tc_sanitize_text(buf);
    int len = (int)strlen(buf);
    if (len > cap) len = cap;
    for (int i = 0; i < len; i++) ids[i] = tc_encode_char(buf[i]);
    return len;
}

static unsigned int xr(unsigned int *s) {
    unsigned int x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

static float pearson_corr(const float *a, const float *b, int n) {
    float ma = 0, mb = 0;
    for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    float cov = 0, va = 0, vb = 0;
    for (int i = 0; i < n; i++) {
        float da = a[i] - ma, db = b[i] - mb;
        cov += da * db; va += da * da; vb += db * db;
    }
    float denom = sqrtf(va * vb);
    return denom < 1e-9f ? 0.0f : cov / denom;
}

/* Sorts indices 0..n-1 by importance[idx] descending (simple insertion sort, n is tiny). */
static void argsort_desc(const float *importance, int n, int *out_idx) {
    for (int i = 0; i < n; i++) out_idx[i] = i;
    for (int i = 1; i < n; i++) {
        int key = out_idx[i];
        float kv = importance[key];
        int j = i - 1;
        while (j >= 0 && importance[out_idx[j]] < kv) { out_idx[j + 1] = out_idx[j]; j--; }
        out_idx[j + 1] = key;
    }
}

/* All metric functions share this exact signature so they can be passed
 * through one function pointer type without a cast -- casting a function
 * pointer to a type whose parameters don't match its definition is
 * undefined behavior in C, not just a style issue. */
typedef float (*MetricFn)(const TCParamSet *p, TCCache *c, const int *ids, int T,
                           const int *occ_positions, int k, const void *baseline);

typedef struct { const int *targets; float base_loss; } LossBaseline;

/* Mean loss over all T positions with the given k positions occluded
 * (space token), minus the unoccluded baseline. baseline points to a
 * LossBaseline. */
static float loss_shift(const TCParamSet *p, TCCache *c, const int *ids, int T,
                         const int *occ_positions, int k, const void *baseline) {
    const LossBaseline *bl = (const LossBaseline *)baseline;
    int space_id = tc_encode_char(' ');
    int tmp[256];
    memcpy(tmp, ids, sizeof(int) * (size_t)T);
    for (int i = 0; i < k; i++) tmp[occ_positions[i]] = space_id;
    float loss;
    tc_forward(p, c, tmp, T, bl->targets, &loss);
    return loss - bl->base_loss;
}

/* 1 - cosine(original embedding, embedding with k positions occluded).
 * baseline points to the original (unoccluded) embedding, d_model floats. */
static float embed_shift(const TCParamSet *p, TCCache *c, const int *ids, int T,
                          const int *occ_positions, int k, const void *baseline) {
    const float *base_vec = (const float *)baseline;
    int D = p->cfg.d_model;
    int space_id = tc_encode_char(' ');
    int tmp[256];
    memcpy(tmp, ids, sizeof(int) * (size_t)T);
    for (int i = 0; i < k; i++) tmp[occ_positions[i]] = space_id;
    float vec[256];
    tc_embed(p, c, tmp, T, vec);
    return 1.0f - tc_cosine(base_vec, vec, D);
}

static int run_deletion_curve(const char *name, const int *ranked_idx, int n_positions,
                               const TCParamSet *p, TCCache *c, const int *ids, int T,
                               MetricFn metric, const void *baseline, unsigned int seed, int gates) {
    int ks[] = { 1, 2, 3, 5 };
    int n_ks = 4;
    int all_pass = 1;
    printf("  [%s] deletion curve%s:\n", name, gates ? "" : " (diagnostic only, does not gate)");
    unsigned int rng = seed;
    for (int ki = 0; ki < n_ks; ki++) {
        int k = ks[ki];
        if (k > n_positions - 1) continue;
        float top_effect = metric(p, c, ids, T, ranked_idx, k, baseline);

        float rand_total = 0.0f;
        int trials = 20;
        for (int t = 0; t < trials; t++) {
            int rand_positions[8];
            for (int i = 0; i < k; i++) rand_positions[i] = (int)(xr(&rng) % (unsigned int)n_positions);
            rand_total += metric(p, c, ids, T, rand_positions, k, baseline);
        }
        float rand_effect = rand_total / trials;
        int pass = top_effect > rand_effect;
        if (!pass) all_pass = 0;
        printf("    k=%d  top=%.4f  random_avg=%.4f  %s\n", k, top_effect, rand_effect,
               pass ? "ok" : "INVERTED");
    }
    printf("  [%s] deletion curve: %s\n", name, all_pass ? "PASS" : "FAIL");
    return all_pass;
}

int main(int argc, char **argv) {
    const char *gen_path = argc > 1 ? argv[1] : "build/model.bin";
    const char *emb_path = argc > 2 ? argv[2] : "build/embedder.bin";

    TCParamSet *gen = tc_paramset_load(gen_path);
    TCParamSet *emb = tc_paramset_load(emb_path);
    if (!gen || !emb) { fprintf(stderr, "could not load %s / %s\n", gen_path, emb_path); return 1; }
    TCConfig cfg = gen->cfg;
    TCCache *cache = tc_cache_create(cfg);

    TCParamSet *gen_random = tc_paramset_create(cfg);
    tc_paramset_init_random(gen_random, 9999);
    TCParamSet *emb_random = tc_paramset_create(cfg);
    tc_paramset_init_random(emb_random, 9999);

    const char *prompt = "the cat sat on the mat quietly, and the dog ran across the log slowly";
    int ids[256];
    int T = encode(prompt, ids, cfg.max_seq_len);
    int T_use = T - 1;
    const int *targets = ids + 1;
    printf("faithcheck prompt: \"%.*s\"\n\n", T, prompt);

    int overall_pass = 1;

    /* --- surprise (diagnostic only, expected to fail) --- */
    {
        TCGlassBoxStep steps[256];
        int n = tc_explain_generate(gen, cache, ids, T, steps);
        float imp[256];
        for (int i = 0; i < n; i++) imp[i] = steps[i].surprise;
        int ranked[256];
        argsort_desc(imp, n, ranked);
        LossBaseline bl;
        bl.targets = targets;
        tc_forward(gen, cache, ids, T_use, targets, &bl.base_loss);
        printf("== surprise (generative head, known non-attribution) ==\n");
        run_deletion_curve("surprise", ranked, n, gen, cache, ids, T_use,
                            loss_shift, &bl, 12345, 0);
        printf("\n");
    }

    /* --- causal occlusion (generative head, real attribution) --- */
    {
        TCCausalStep steps[256];
        int n = tc_explain_causal(gen, cache, ids, T, tc_encode_char(' '), steps);
        float imp[256];
        for (int i = 0; i < n; i++) imp[i] = steps[i].importance;
        int ranked[256];
        argsort_desc(imp, n, ranked);
        LossBaseline bl;
        bl.targets = targets;
        tc_forward(gen, cache, ids, T_use, targets, &bl.base_loss);
        printf("== causal occlusion (generative head) ==\n");
        int pass_del = run_deletion_curve("causal", ranked, n, gen, cache, ids, T_use,
                                           loss_shift, &bl, 23456, 1);
        overall_pass &= pass_del;

        TCCausalStep steps_r[256];
        tc_explain_causal(gen_random, cache, ids, T, tc_encode_char(' '), steps_r);
        float imp_r[256];
        for (int i = 0; i < n; i++) imp_r[i] = steps_r[i].importance;
        float corr = pearson_corr(imp, imp_r, n);
        int pass_rand = fabsf(corr) < 0.5f;
        overall_pass &= pass_rand;
        printf("  [causal] randomization check: corr(trained, random) = %.3f  %s\n\n",
               corr, pass_rand ? "PASS" : "FAIL");
    }

    /* --- embedder occlusion (real attribution, per the original design) --- */
    {
        TCEmbedOcclusionStep steps[256];
        int n = tc_explain_embedding(emb, cache, ids, T, steps);
        float imp[256];
        for (int i = 0; i < n; i++) imp[i] = steps[i].importance;
        int ranked[256];
        argsort_desc(imp, n, ranked);
        float base_vec[256];
        tc_embed(emb, cache, ids, T, base_vec);
        printf("== embedder occlusion ==\n");
        int pass_del = run_deletion_curve("embed", ranked, n, emb, cache, ids, T,
                                           embed_shift, base_vec, 34567, 1);
        overall_pass &= pass_del;

        TCEmbedOcclusionStep steps_r[256];
        tc_explain_embedding(emb_random, cache, ids, T, steps_r);
        float imp_r[256];
        for (int i = 0; i < n; i++) imp_r[i] = steps_r[i].importance;
        float corr = pearson_corr(imp, imp_r, n);
        int pass_rand = fabsf(corr) < 0.5f;
        overall_pass &= pass_rand;
        printf("  [embed] randomization check: corr(trained, random) = %.3f  %s\n\n",
               corr, pass_rand ? "PASS" : "FAIL");
    }

    printf("=====================================\n");
    printf(overall_pass ? "FAITHCHECK PASS (embedder + causal occlusion gate; surprise is diagnostic only)\n"
                         : "FAITHCHECK FAIL\n");

    tc_cache_free(cache);
    tc_paramset_free(gen_random);
    tc_paramset_free(emb_random);
    tc_paramset_free(gen);
    tc_paramset_free(emb);
    return overall_pass ? 0 : 1;
}
