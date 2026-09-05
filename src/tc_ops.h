#ifndef TC_OPS_H
#define TC_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Computes C = A * B^T
 * A is [M x K], B is [N x K]. C is [M x N].
 * All matrices are row-major.
 */
void tc_gemm_nt(float *C, const float *A, const float *B, int M, int N, int K);

/* Computes y = W * x
 * W is [rows x cols], x is [cols], y is [rows].
 * All are row-major. Overwrites y.
 */
void tc_gemv(float *y, const float *W, const float *x, int rows, int cols);

/* Computes y += W^T * x
 * W is [rows x cols], x is [rows], y is [cols].
 * All are row-major. Accumulates into y.
 */
void tc_gemv_t_add(float *y, const float *W, const float *x, int rows, int cols);


/* Computes W = W + alpha * (g * x^T)
 * W is [rows x cols], x is [cols], g is [rows].
 */
void tc_ger(float *W, const float *x, const float *g, int rows, int cols, float alpha);

/* Computes RMSNorm forward pass over a sequence of length T.
 * x: [T x D], gamma: [D], rms: [T], xhat: [T x D], y: [T x D]
 */
void tc_rmsnorm_fwd(const float *x, const float *g, float *rms, float *xhat, float *y, int T, int D, float eps);

/* Computes RMSNorm backward pass over a sequence of length T.
 * dy: [T x D], xhat: [T x D], gamma: [D], rms: [T]
 * dgamma (accumulates): [D], dx (accumulates): [T x D]
 */
void tc_rmsnorm_bwd(const float *dy, const float *xhat, const float *gamma, const float *rms, int T, int D, float *dgamma, float *dx);

/* Computes SwiGLU forward pass over a sequence of length T.
 * hgate: [T x FFD], hup: [T x FFD], hsilu: [T x FFD]
 */
void tc_swiglu_fwd(const float *hgate, const float *hup, float *hsilu, int T, int FFD);

/* Computes SwiGLU backward pass over a sequence of length T.
 * dhsilu: [T x FFD], hgate: [T x FFD], hup: [T x FFD]
 * dhgate (accumulates): [T x FFD], dhup (accumulates): [T x FFD]
 */
void tc_swiglu_bwd(const float *dhsilu, const float *hgate, const float *hup, float *dhgate, float *dhup, int T, int FFD);

/* Computes WKV forward over sequence T.
 * k, v, r_sig, state, wkv: [T x D]. decay: [D].
 */
void tc_wkv_fwd(const float *k, const float *v, const float *r_sig, const float *decay, float *state, float *wkv, int T, int D);

/* Computes WKV backward over sequence T.
 * dwkv, k, v, r_sig, state, dstate, drpre, dk, dv: [T x D]. decay, ddecay: [D].
 * ddecay is accumulated if not NULL.
 */
void tc_wkv_bwd(const float *dwkv, const float *k, const float *v, const float *r_sig, const float *decay, const float *state, float *dstate, float *drpre, float *dk, float *dv, float *ddecay, int T, int D);

/* Computes self-attention forward pass over sequence T. */
void tc_attn_fwd(const float *Q, const float *K, const float *V, float *attn_w, float *attn_ctx, int T, int D, int H, int n_kv_heads);

/* Computes self-attention backward pass over sequence T. */
void tc_attn_bwd(const float *dctx, const float *attn_w, const float *Q, const float *K, const float *V, float *dQ, float *dK, float *dV, int T, int D, int H, int n_kv_heads);

/* Computes CE loss forward (softmax + cross entropy). */
void tc_ce_fwd(const float *logits, const int *targets, float *probs, float *loss_out, int *n_active_out, int T, int V);

/* Computes CE loss backward (dlogits). */
void tc_ce_bwd(const float *probs, const int *targets, float *dlogits, int n_active, float weight, int T, int V);

#ifdef __cplusplus
}
#endif

#endif
