#pragma once
#include "tcmodel.h"

#ifdef __cplusplus
extern "C" {
#endif

int  tc_metal_available(void);
int  tc_metal_init(const TCConfig *cfg);
void tc_metal_shutdown(void);

/* Zero-copy wrap of existing malloc'd slabs (storageModeShared). */
int  tc_metal_bind_params(TCParamSet *p, TCParamSet *g);
int  tc_metal_bind_cache (TCCache *c);
void tc_metal_register_ptr(void *ptr, size_t size);

int  tc_metal_forward    (const TCParamSet *p, TCCache **caches,
                          const int *ids, int B, int T,
                          const int *targets, float *out_losses);
int  tc_metal_backward   (const TCParamSet *p, TCParamSet **grads,
                          TCCache **caches,
                          const int *ids, int B, int T,
                          const int *targets);

/* Primitive escape hatches used while porting layer-by-layer. */
void tc_metal_gemm_nt(float *C, const float *A, const float *B, int M, int N, int K);
void tc_metal_rmsnorm_fwd(const float *x, const float *g, float *rms, float *xhat, float *y, int T, int D, float eps);
void tc_metal_swiglu_fwd(const float *hgate, const float *hup, float *hsilu, int T, int FFD);
void tc_metal_wkv_fwd(const float *k, const float *v, const float *r_sig, const float *decay, float *state, float *wkv, int T, int D);
void tc_metal_attn_fwd(const float *Q, const float *K, const float *V, float *attn_w, float *attn_ctx, int T, int D, int H, int n_kv_heads);
void tc_metal_ce_fwd(const float *logits, const int *targets, float *probs, float *loss_out, int *n_active_out, int T, int V);
void tc_metal_ce_bwd(const float *probs, const int *targets, float *dlogits, int n_active, float weight, int T, int V, const float *logits);
void tc_metal_rope_fwd(const float *q, const float *k, float *q_out, float *k_out, int T, int D, int H, int n_kv_heads);
void tc_metal_rope_bwd(const float *dq, const float *dk, float *dq_out, float *dk_out, int T, int D, int H, int n_kv_heads);
void tc_metal_add_fwd(const float *a, const float *b, float *out, int T, int D);
void tc_metal_tmix_fwd(const float *x, const float *mix_w, float *out, int T, int D);
void tc_metal_tmix_bwd(const float *dout, const float *x, const float *mix_w, float *dx, float *dmix_w, int T, int D);
void tc_metal_embed_fwd(const float *embed, const int *ids, float *out, int T, int D, int V);
void tc_metal_embed_bwd(const float *dout, const int *ids, float *dembed, int T, int D, int V);
void tc_metal_rmsnorm_bwd(const float *dy, const float *xhat, const float *gamma, const float *rms, int T, int D, float *dgamma, float *dx);
void tc_metal_swiglu_bwd(const float *dhsilu, const float *hgate, const float *hup, float *dhgate, float *dhup, int T, int FFD);
void tc_metal_wkv_bwd(const float *dwkv, const float *k, const float *v, const float *r_sig, const float *decay, const float *state, float *dstate, float *drpre, float *dk, float *dv, float *ddecay, int T, int D);
void tc_metal_attn_bwd(const float *dctx, const float *attn_w, const float *Q, const float *K, const float *V, float *dQ, float *dK, float *dV, int T, int D, int H, int n_kv_heads);
void tc_metal_commit_and_wait(void);

#ifdef __cplusplus
}
#endif
