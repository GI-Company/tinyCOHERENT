#include "optim.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

TCAdam *tc_adam_create(TCConfig cfg, float lr, float wd) {
    TCAdam *a = calloc(1, sizeof(TCAdam));
    a->m = tc_paramset_create(cfg);
    a->v = tc_paramset_create(cfg);
    a->t = 0;
    a->lr = lr; a->wd = wd; a->beta1 = 0.9f; a->beta2 = 0.999f; a->eps = 1e-8f;
    return a;
}

void tc_adam_free(TCAdam *a) {
    if (!a) return;
    tc_paramset_free(a->m);
    tc_paramset_free(a->v);
    free(a);
}

void tc_adam_step(TCAdam *a, TCParamSet *params, TCParamSet *grad) {
    a->t++;
    float b1t = 1.0f - powf(a->beta1, (float)a->t);
    float b2t = 1.0f - powf(a->beta2, (float)a->t);
    int n = params->n_floats;
    for (int i = 0; i < n; i++) {
        float g = grad->buf[i];
        a->m->buf[i] = a->beta1 * a->m->buf[i] + (1 - a->beta1) * g;
        a->v->buf[i] = a->beta2 * a->v->buf[i] + (1 - a->beta2) * g * g;
        float mhat = a->m->buf[i] / b1t;
        float vhat = a->v->buf[i] / b2t;
        params->buf[i] -= a->lr * (mhat / (sqrtf(vhat) + a->eps) + a->wd * params->buf[i]);
    }
    tc_paramset_zero(grad);
}
