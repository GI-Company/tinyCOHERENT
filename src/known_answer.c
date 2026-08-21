/* Known-answer attribution test: the strongest evidence available for
 * causal occlusion, because ground-truth importance isn't inferred or
 * assumed -- it's true by construction.
 *
 * Synthetic task: sequences of 8 digit characters. Positions 0,1,2,4,5,6
 * are independent random noise, uninformative and uncorrelated with
 * anything. Position 3 is a random digit too, but position 7 (the final
 * character) is FORCED to equal position 3 -- a pure positional copy, not
 * a value-based shortcut, so the model has no cue except "attend back to
 * index 3" to solve it. If a trained model learns this task (verified by
 * copy accuracy on held-out sequences), causal occlusion attribution on
 * the final prediction step MUST concentrate importance on position 3.
 * If it doesn't, the attribution is broken and this test says so --
 * no held-out val loss curve, no qualitative spot check, can give that
 * kind of unambiguous answer. */
#include "tcmodel.h"
#include "optim.h"
#include "tokenizer.h"
#include "glassbox.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define L 8          /* sequence length */
#define SRC_POS 3    /* the position that determines the final character */
#define ALPHABET 10  /* digits '0'-'9' */
#define BATCH 8

static unsigned int xr(unsigned int *s) {
    unsigned int x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

/* Fills ids[0..L-1]: random digits everywhere except ids[L-1], which is
 * forced to copy ids[SRC_POS]. */
static void gen_sequence(int *ids, unsigned int *rng) {
    for (int i = 0; i < L; i++)
        ids[i] = tc_encode_char((char)('0' + xr(rng) % ALPHABET));
    ids[L - 1] = ids[SRC_POS];
}

static void scale_grad(TCParamSet *grad, float s) {
    for (int i = 0; i < grad->n_floats; i++) grad->buf[i] *= s;
}

/* Fraction of a batch where the model's argmax prediction at the final
 * step (predicting ids[L-1] from context ids[0..L-2]) matches the true
 * copy target -- the plain "did it learn the task" signal, since mean
 * loss is dominated by the irreducible noise floor on every other step. */
static float copy_accuracy(const TCParamSet *p, TCCache *c, unsigned int *rng, int n_eval) {
    int correct = 0;
    int V = p->cfg.vocab_size;
    for (int e = 0; e < n_eval; e++) {
        int ids[L];
        gen_sequence(ids, rng);
        float loss;
        tc_forward(p, c, ids, L - 1, ids + 1, &loss);
        const float *probs = c->probs + (size_t)(L - 2) * V;
        int argmax = 0;
        for (int i = 1; i < V; i++) if (probs[i] > probs[argmax]) argmax = i;
        if (argmax == ids[L - 1]) correct++;
    }
    return (float)correct / n_eval;
}

int main(void) {
    TCConfig cfg = tc_default_config();
    TCParamSet *p = tc_paramset_create(cfg);
    tc_paramset_init_random(p, 555);
    TCParamSet *grad = tc_paramset_create(cfg);
    TCCache *cache = tc_cache_create(cfg);
    TCAdam *adam = tc_adam_create(cfg, 5e-3f);

    unsigned int train_rng = 1;
    unsigned int eval_rng = 999999; /* disjoint stream from training */

    printf("known-answer task: %d-char sequences, position %d determines the final character\n", L, SRC_POS);
    printf("training (fresh random sequences every step, no fixed corpus)...\n");

    /* The sparse signal here (1 of 7 loss terms per step actually carries
     * information, the rest is irreducible noise) takes ~15-20k steps at
     * a exploratory learning rate just to find the right region at all.
     * Once found, that same learning rate overshoots a narrow solution --
     * accuracy oscillates 100% <-> ~15% even after 50k+ steps at constant
     * lr. Decaying lr once the task is first solved lets it explore early
     * and settle late, instead of forcing one lr to do both jobs. */
    int num_steps = 40000;
    int decayed = 0;
    for (int step = 1; step <= num_steps; step++) {
        float step_loss = 0.0f;
        for (int b = 0; b < BATCH; b++) {
            int ids[L];
            gen_sequence(ids, &train_rng);
            float loss;
            tc_forward(p, cache, ids, L - 1, ids + 1, &loss);
            tc_backward(p, grad, cache, ids, L - 1, ids + 1);
            step_loss += loss;
        }
        scale_grad(grad, 1.0f / BATCH);
        tc_adam_step(adam, p, grad);
        if (step % 1000 == 0 || step == 1) {
            unsigned int probe_rng = eval_rng; /* don't perturb the real eval stream */
            float acc = copy_accuracy(p, cache, &probe_rng, 200);
            if (!decayed && acc >= 0.95f) {
                adam->lr *= 0.2f;
                decayed = 1;
                printf("  step %5d  copy accuracy %.1f%% >= 95%% -- decaying lr to %.5f\n", step, 100.0f * acc, adam->lr);
            }
            printf("  step %5d  train_loss %.4f  held-out copy accuracy %.1f%%\n",
                   step, step_loss / BATCH, 100.0f * acc);
        }
    }

    float final_acc = copy_accuracy(p, cache, &eval_rng, 500);
    printf("\nfinal held-out copy accuracy: %.1f%% over 500 fresh sequences (chance = %.1f%%)\n",
           100.0f * final_acc, 100.0f / ALPHABET);
    if (final_acc < 0.9f) {
        printf("model did not learn the task well enough to test attribution against -- aborting\n");
        return 1;
    }

    /* --- the actual test: does causal occlusion find position 3? ---
     * Occlusion baseline: average over several random in-distribution
     * digit substitutions per position, not one fixed token. A single
     * fixed substitution can (a) coincidentally match the original digit
     * at some position, silently reading as "no effect" there, and (b),
     * if out-of-distribution (space was tried first here and made things
     * worse, not better -- see the header comment for tc_explain_causal),
     * produce large effects from unfamiliarity rather than genuine
     * causal importance. Averaging over random real digits fixes both. */
    printf("\n--- known-answer attribution test ---\n");
    int n_eval = 300;
    int occ_trials = 5;
    int top1_correct = 0;
    float importance_at_src = 0.0f, importance_at_other = 0.0f;
    int n_other = 0;

    for (int e = 0; e < n_eval; e++) {
        int ids[L];
        gen_sequence(ids, &eval_rng);

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
        if (best == SRC_POS) top1_correct++;

        for (int i = 0; i < n; i++) {
            if (i == SRC_POS) importance_at_src += avg_importance[i];
            else { importance_at_other += avg_importance[i]; n_other++; }
        }
    }
    importance_at_src /= n_eval;
    importance_at_other /= n_other;

    printf("position %d (the true answer source) is the top-ranked position in %d/%d held-out examples (%.1f%%)\n",
           SRC_POS, top1_correct, n_eval, 100.0f * top1_correct / n_eval);
    printf("mean importance at position %d: %.4f bits\n", SRC_POS, importance_at_src);
    printf("mean importance at every other position: %.4f bits\n", importance_at_other);
    printf("ratio: %.1fx\n", importance_at_other > 1e-6f ? importance_at_src / importance_at_other : INFINITY);

    int pass = (top1_correct >= (int)(0.9f * n_eval)) && (importance_at_src > 3.0f * importance_at_other);
    printf("\n%s\n", pass ? "KNOWN-ANSWER TEST PASS" : "KNOWN-ANSWER TEST FAIL");

    tc_adam_free(adam);
    tc_cache_free(cache);
    tc_paramset_free(grad);
    tc_paramset_free(p);
    return pass ? 0 : 1;
}
