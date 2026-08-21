#ifndef TC_OPTIM_H
#define TC_OPTIM_H

#include "tcmodel.h"

typedef struct {
    TCParamSet *m;
    TCParamSet *v;
    int t;
    float lr, beta1, beta2, eps;
} TCAdam;

TCAdam *tc_adam_create(TCConfig cfg, float lr);
void tc_adam_free(TCAdam *a);
/* Applies one Adam step: params -= update, using grad (already accumulated
 * by the caller, e.g. mean over a batch), then zeroes grad for reuse. */
void tc_adam_step(TCAdam *a, TCParamSet *params, TCParamSet *grad);

#endif
