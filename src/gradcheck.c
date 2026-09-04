/* Standalone numerical gradient check for tc_backward. Perturbs a sample
 * of parameters across the whole flat buffer, compares (f(x+e)-f(x-e))/2e
 * against the analytic gradient from tc_backward, and reports the worst
 * relative error. This is the actual correctness proof for the hand
 * written backprop in tcmodel.c -- treat any run above tolerance as a bug. */
#include "tcmodel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float run_loss(const TCParamSet *p, TCCache *c, const int *ids, int T, const int *targets) {
    float loss;
    tc_forward(p, c, ids, T, targets, &loss);
    return loss;
}

int main(void) {
    TCConfig cfg;
    cfg.vocab_size = 16;
    cfg.d_model = 8;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.ff_mult = 1;
    cfg.max_seq_len = 8;

    TCParamSet *p = tc_paramset_create(cfg);
    tc_paramset_init_random(p, 42);
    TCParamSet *grad = tc_paramset_create(cfg);
    TCCache *c = tc_cache_create(cfg);

    int T = 6;
    int ids[6], targets[6];
    unsigned int s = 7;
    for (int t = 0; t < T; t++) {
        s = s * 1664525u + 1013904223u;
        ids[t] = s % cfg.vocab_size;
        s = s * 1664525u + 1013904223u;
        targets[t] = s % cfg.vocab_size;
    }

    float base_loss;
    tc_forward(p, c, ids, T, targets, &base_loss);
    tc_paramset_zero(grad);
    tc_backward(p, grad, c, ids, T, targets);

    printf("param count = %d, base loss = %f\n", p->n_floats, base_loss);

    /* atol+rtol tolerance (as in np.allclose), not pure relative error:
     * near-zero analytic gradients make relative error blow up on float32
     * rounding noise alone, which is not a backprop bug. */
    const float eps = 1e-3f;
    const float atol = 2e-3f, rtol = 3e-2f;
    int stride = p->n_floats / 80 > 0 ? p->n_floats / 80 : 1;
    /* |diff| - (atol+rtol*|analytic|), largest (closest to failing) over all checks.
     * Must start below any real margin, not at 0 -- a passing check always has
     * margin < 0, so starting at 0 silently clamps this to "0.000000" on every
     * clean run and hides how close the closest call actually was. */
    float worst_margin = -1e30f;
    int checked = 0, failed = 0;

    for (int i = 0; i < p->n_floats; i += stride) {
        float orig = p->buf[i];
        p->buf[i] = orig + eps;
        float lp = run_loss(p, c, ids, T, targets);
        p->buf[i] = orig - eps;
        float lm = run_loss(p, c, ids, T, targets);
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

    printf("--- standard unmasked gradcheck ---\n");
    printf("checked %d params, %d failed tolerance, worst margin = %.6f\n", checked, failed, worst_margin);

    /* --- test case 2: masked targets (prefix and alternating) --- */
    printf("--- masked targets gradcheck (prefix & alternating mask) ---\n");
    int masked_targets[6] = {-1, -1, -1, targets[3], -1, targets[5]};
    float base_loss_masked;
    tc_forward(p, c, ids, T, masked_targets, &base_loss_masked);
    tc_paramset_zero(grad);
    tc_backward(p, grad, c, ids, T, masked_targets);

    printf("masked targets base loss = %f (active targets at t=3, 5)\n", base_loss_masked);
    float worst_margin_masked = -1e30f;
    int checked_masked = 0, failed_masked = 0;

    for (int i = 0; i < p->n_floats; i += stride) {
        float orig = p->buf[i];
        p->buf[i] = orig + eps;
        float lp = run_loss(p, c, ids, T, masked_targets);
        p->buf[i] = orig - eps;
        float lm = run_loss(p, c, ids, T, masked_targets);
        p->buf[i] = orig;

        float numeric = (lp - lm) / (2 * eps);
        float analytic = grad->buf[i];
        float diff = fabsf(numeric - analytic);
        float tol = atol + rtol * fabsf(analytic);
        float margin = diff - tol;
        checked_masked++;
        if (margin > worst_margin_masked) worst_margin_masked = margin;
        if (margin > 0.0f) {
            failed_masked++;
            printf("  MASKED MISMATCH idx=%d numeric=%.6f analytic=%.6f diff=%.6f tol=%.6f\n",
                   i, numeric, analytic, diff, tol);
        }
    }

    printf("checked %d params (masked), %d failed tolerance, worst margin = %.6f\n",
           checked_masked, failed_masked, worst_margin_masked);

    int total_failed = failed + failed_masked;
    printf(total_failed == 0 ? "GRADCHECK PASS\n" : "GRADCHECK FAIL\n");

    tc_cache_free(c);
    tc_paramset_free(grad);
    tc_paramset_free(p);
    return total_failed == 0 ? 0 : 1;
}
