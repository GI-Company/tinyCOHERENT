/* Controlled-degradation sweep for the known-answer attribution test.
 * Same copy task as known_answer.c, generalized along three axes:
 *
 *   noise_p   -- probability the training target is corrupted to a random
 *                (possibly coincidentally-correct) wrong digit instead of
 *                truly copying the source position.
 *   jitter_q  -- probability the TRUE source position for a given example
 *                is 2 or 4 instead of 3 (50/50 between them). Ground truth
 *                is tracked per-example; "top-1 recovery" means attribution
 *                found THAT example's real source, not always position 3.
 *   capacity  -- d_model divisor (1, 2, or 4), n_heads held fixed at 2.
 *
 * Training schedule is FIXED across all conditions (same step count, same
 * fixed-step lr decay) rather than accuracy-triggered: under label noise
 * the achievable accuracy is capped below the old 95% trigger by
 * construction, so an accuracy-triggered decay would never fire in noisy
 * conditions and confound "how noisy" with "how long optimization stayed
 * in exploration mode." A fixed schedule keeps the swept parameter the
 * only thing that varies between runs.
 *
 * Usage: ./degrade_test <noise_p> <jitter_q> <capacity_divisor>
 * Prints one CSV line: noise_p,jitter_q,capacity_divisor,d_model,param_count,
 *   decay_step,final_accuracy,top1_recovery,mean_importance_src,
 *   mean_importance_strongest_other,ratio,leak_fraction */
#include "tcmodel.h"
#include "optim.h"
#include "tokenizer.h"
#include "glassbox.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define L 8
#define ALPHABET 10
#define BATCH 8
#define LEAK_THRESHOLD 0.1f

static unsigned int xr(unsigned int *s) {
    unsigned int x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}
static float xrf(unsigned int *s) { return (float)(xr(s) % 1000000) / 1000000.0f; }

/* Fills ids[0..L-1]. *out_src is the TRUE source position for this
 * example (3 normally, or 2/4 under jitter). ids[L-1] copies ids[*out_src]
 * unless label noise fires, in which case it's replaced with a random
 * (possibly coincidentally matching) digit instead. */
static void gen_sequence(int *ids, unsigned int *rng, float noise_p, float jitter_q, int *out_src) {
    for (int i = 0; i < L; i++)
        ids[i] = tc_encode_char((char)('0' + xr(rng) % ALPHABET));
    int src = 3;
    if (xrf(rng) < jitter_q) src = (xr(rng) % 2 == 0) ? 2 : 4;
    *out_src = src;
    if (xrf(rng) < noise_p)
        ids[L - 1] = tc_encode_char((char)('0' + xr(rng) % ALPHABET));
    else
        ids[L - 1] = ids[src];
}

static void scale_grad(TCParamSet *grad, float s) {
    for (int i = 0; i < grad->n_floats; i++) grad->buf[i] *= s;
}

static float copy_accuracy(const TCParamSet *p, TCCache *c, unsigned int *rng, int n_eval,
                            float noise_p, float jitter_q) {
    int correct = 0;
    int V = p->cfg.vocab_size;
    for (int e = 0; e < n_eval; e++) {
        int ids[L], src;
        gen_sequence(ids, rng, noise_p, jitter_q, &src);
        float loss;
        tc_forward(p, c, ids, L - 1, ids + 1, &loss);
        const float *probs = c->probs + (size_t)(L - 2) * V;
        int argmax = 0;
        for (int i = 1; i < V; i++) if (probs[i] > probs[argmax]) argmax = i;
        if (argmax == ids[L - 1]) correct++;
    }
    return (float)correct / n_eval;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <noise_p> <jitter_q> <capacity_divisor>\n", argv[0]);
        return 1;
    }
    float noise_p = strtof(argv[1], NULL);
    float jitter_q = strtof(argv[2], NULL);
    int cap_div = atoi(argv[3]);

    TCConfig cfg = tc_default_config();
    cfg.d_model = cfg.d_model / cap_div;
    if (cfg.d_model < cfg.n_heads) cfg.d_model = cfg.n_heads; /* keep divisibility sane at the floor */
    if (cfg.d_model % cfg.n_heads != 0) cfg.d_model -= cfg.d_model % cfg.n_heads;

    TCParamSet *p = tc_paramset_create(cfg);
    tc_paramset_init_random(p, 555);
    TCParamSet *grad = tc_paramset_create(cfg);
    TCCache *cache = tc_cache_create(cfg);
    TCAdam *adam = tc_adam_create(cfg, 5e-3f);

    unsigned int train_rng = 1;
    unsigned int eval_rng = 999999;

    /* Plateau-triggered lr decay instead of a fixed step or fixed accuracy
     * bar: under jitter, accuracy is capped below 100% for a structural
     * reason (which of positions 2/3/4 is the true source is an
     * unobservable per-example coin flip -- the model has no signal to
     * resolve it from, so its best strategy tops out around 1-0.9*q), and
     * under label noise similarly capped near 1-noise_p. Neither a fixed
     * step count nor a fixed target accuracy is right across the sweep;
     * decaying once progress stalls (whatever level that turns out to be)
     * is the one criterion that adapts to both. */
    int max_explore_steps = 35000, post_decay_steps = 12000, check_every = 2000;
    float prev_acc = -1.0f;
    int stable_checks = 0, decayed = 0, step = 0;
    for (; step <= max_explore_steps && !decayed; step++) {
        if (step > 0) {
            for (int b = 0; b < BATCH; b++) {
                int ids[L], src;
                gen_sequence(ids, &train_rng, noise_p, jitter_q, &src);
                float loss;
                tc_forward(p, cache, ids, L - 1, ids + 1, &loss);
                tc_backward(p, grad, cache, ids, L - 1, ids + 1);
            }
            scale_grad(grad, 1.0f / BATCH);
            tc_adam_step(adam, p, grad);
        }
        if (step > 0 && step % check_every == 0) {
            unsigned int probe_rng = eval_rng;
            float acc = copy_accuracy(p, cache, &probe_rng, 150, noise_p, jitter_q);
            if (prev_acc >= 0.0f && acc > 0.3f && fabsf(acc - prev_acc) < 0.03f) stable_checks++;
            else stable_checks = 0;
            prev_acc = acc;
            if (stable_checks >= 2) decayed = 1;
        }
    }
    adam->lr *= 0.2f;
    for (int extra = 0; extra < post_decay_steps; extra++) {
        for (int b = 0; b < BATCH; b++) {
            int ids[L], src;
            gen_sequence(ids, &train_rng, noise_p, jitter_q, &src);
            float loss;
            tc_forward(p, cache, ids, L - 1, ids + 1, &loss);
            tc_backward(p, grad, cache, ids, L - 1, ids + 1);
        }
        scale_grad(grad, 1.0f / BATCH);
        tc_adam_step(adam, p, grad);
    }

    float final_acc = copy_accuracy(p, cache, &eval_rng, 500, noise_p, jitter_q);

    int n_eval = 300, occ_trials = 5;
    int top1_correct = 0, leak_count = 0;
    float sum_importance_src = 0.0f, sum_importance_strongest_other = 0.0f;

    for (int e = 0; e < n_eval; e++) {
        int ids[L], src;
        gen_sequence(ids, &eval_rng, noise_p, jitter_q, &src);

        float avg_importance[L - 1] = {0};
        int n = 0;
        for (int trial = 0; trial < occ_trials; trial++) {
            int occ_digit = tc_encode_char((char)('0' + xr(&eval_rng) % ALPHABET));
            TCCausalStep steps[L];
            n = tc_explain_causal(p, cache, ids, L, occ_digit, steps);
            for (int i = 0; i < n; i++) avg_importance[i] += steps[i].importance;
        }
        for (int i = 0; i < n; i++) avg_importance[i] /= occ_trials;

        int best = 0;
        for (int i = 1; i < n; i++) if (avg_importance[i] > avg_importance[best]) best = i;
        if (best == src) top1_correct++;

        float strongest_other = -1e30f;
        int any_leak = 0;
        for (int i = 0; i < n; i++) {
            if (i == src) continue;
            if (avg_importance[i] > strongest_other) strongest_other = avg_importance[i];
            if (avg_importance[i] > LEAK_THRESHOLD) any_leak = 1;
        }
        sum_importance_src += avg_importance[src];
        sum_importance_strongest_other += strongest_other;
        if (any_leak) leak_count++;
    }

    float mean_src = sum_importance_src / n_eval;
    float mean_other = sum_importance_strongest_other / n_eval;
    float ratio = mean_other > 1e-6f ? mean_src / mean_other : INFINITY;
    float leak_frac = (float)leak_count / n_eval;

    printf("%.2f,%.2f,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.2f,%.4f\n",
           noise_p, jitter_q, cap_div, cfg.d_model, tc_param_count(cfg), step,
           final_acc, (float)top1_correct / n_eval, mean_src, mean_other, ratio, leak_frac);

    tc_adam_free(adam);
    tc_cache_free(cache);
    tc_paramset_free(grad);
    tc_paramset_free(p);
    return 0;
}
