/* Numerical gradient check for tc_embed/tc_embed_backward (the attention-
 * pooling head), isolated from the contrastive loss that will drive it.
 * Uses a dummy scalar loss L = dot(embed(ids), target) against a fixed
 * random target vector, so d(L)/d(out_vec) = target exactly -- any
 * backward bug in the pooling math (weighted-sum, softmax, L2-normalize)
 * or in feeding tc_encode_backward will show up here before the much
 * harder-to-debug contrastive loss is layered on top. */
#include "tcmodel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    TCConfig cfg;
    cfg.vocab_size = 16;
    cfg.d_model = 8;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.ff_mult = 1;
    cfg.max_seq_len = 8;

    TCParamSet *p = tc_paramset_create(cfg);
    tc_paramset_init_random(p, 99);
    TCParamSet *grad = tc_paramset_create(cfg);
    TCCache *c = tc_cache_create(cfg);
    int D = cfg.d_model;

    int T = 6;
    int ids[6];
    unsigned int s = 11;
    for (int t = 0; t < T; t++) {
        s = s * 1664525u + 1013904223u;
        ids[t] = s % cfg.vocab_size;
    }

    float target[8];
    for (int i = 0; i < D; i++) {
        s = s * 1664525u + 1013904223u;
        target[i] = ((float)(s % 1000) / 1000.0f) * 2 - 1;
    }

    float out_vec[8];
    tc_embed(p, c, ids, T, out_vec);
    float base_loss = 0.0f;
    for (int i = 0; i < D; i++) base_loss += out_vec[i] * target[i];
    printf("param count = %d, base loss (dot vs fixed target) = %f\n", p->n_floats, base_loss);

    tc_paramset_zero(grad);
    tc_embed_backward(p, grad, c, ids, T, target); /* d(loss)/d(out_vec) = target */

    const float eps = 1e-3f;
    const float atol = 2e-3f, rtol = 3e-2f;
    int stride = p->n_floats / 80 > 0 ? p->n_floats / 80 : 1;
    int checked = 0, failed = 0;
    float worst_margin = -1e30f; /* see gradcheck.c: must start below any real margin, not at 0 */

    for (int i = 0; i < p->n_floats; i += stride) {
        float orig = p->buf[i];

        p->buf[i] = orig + eps;
        float ov[8];
        tc_embed(p, c, ids, T, ov);
        float lp = 0.0f; for (int k = 0; k < D; k++) lp += ov[k] * target[k];

        p->buf[i] = orig - eps;
        tc_embed(p, c, ids, T, ov);
        float lm = 0.0f; for (int k = 0; k < D; k++) lm += ov[k] * target[k];

        p->buf[i] = orig;

        float numeric = (lp - lm) / (2 * eps);
        float analytic = grad->buf[i];
        float diff = fabsf(numeric - analytic);
        float tol = atol + rtol * fabsf(analytic);
        float margin = diff - tol;
        checked++;
        if (margin > worst_margin) worst_margin = margin;
        if (margin > 0.0f) {
            failed++;
            printf("  MISMATCH idx=%d numeric=%.6f analytic=%.6f diff=%.6f tol=%.6f\n",
                   i, numeric, analytic, diff, tol);
        }
    }

    printf("checked %d params, %d failed tolerance, worst margin = %.6f\n", checked, failed, worst_margin);
    printf(failed == 0 ? "EMBED GRADCHECK PASS\n" : "EMBED GRADCHECK FAIL\n");

    tc_cache_free(c);
    tc_paramset_free(grad);
    tc_paramset_free(p);
    return failed == 0 ? 0 : 1;
}
