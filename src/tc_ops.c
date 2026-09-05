#include "tc_ops.h"
#include <stddef.h>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include "tc_metal.h"
#endif

int tc_use_metal_gemm = 0;

void tc_gemm_nt(float *C, const float *A, const float *B, int M, int N, int K) {
#if defined(__APPLE__)
    if (tc_use_metal_gemm) {
        tc_metal_gemm_nt(C, A, B, M, N, K);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K, 1.0f,
                    A, K,
                    B, K,
                    0.0f, C, N);
    }
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[(size_t)m * K + k] * B[(size_t)n * K + k];
            }
            C[(size_t)m * N + n] = sum;
        }
    }
#endif
}

void tc_gemv(float *y, const float *W, const float *x, int rows, int cols) {
#if defined(__APPLE__) && defined(ACCELERATE_NEW_LAPACK)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0f, W, cols, x, 1, 0.0f, y, 1);
#else
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const float *row = W + (size_t)r * cols;
        for (int c = 0; c < cols; c++) {
            sum += row[c] * x[c];
        }
        y[r] = sum;
    }
#endif
}

void tc_gemv_t_add(float *y, const float *W, const float *x, int rows, int cols) {
#if defined(__APPLE__) && defined(ACCELERATE_NEW_LAPACK)
    cblas_sgemv(CblasRowMajor, CblasTrans, rows, cols, 1.0f, W, cols, x, 1, 1.0f, y, 1);
#else
    for (int r = 0; r < rows; r++) {
        float xr = x[r];
        const float *row = W + (size_t)r * cols;
        for (int c = 0; c < cols; c++) {
            y[c] += row[c] * xr;
        }
    }
#endif
}

void tc_ger(float *W, const float *x, const float *g, int rows, int cols, float alpha) {
#if defined(__APPLE__) && defined(ACCELERATE_NEW_LAPACK)
    cblas_sger(CblasRowMajor, rows, cols, alpha, g, 1, x, 1, W, cols);
#else
    for (int r = 0; r < rows; r++) {
        float gr = g[r] * alpha;
        float *row = W + (size_t)r * cols;
        for (int c = 0; c < cols; c++) {
            row[c] += gr * x[c];
        }
    }
#endif
}

#include <math.h>

void tc_rmsnorm_fwd(const float *x, const float *g, float *rms, float *xhat, float *y, int T, int D, float eps) {
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * D;
        float *xhatt = xhat + (size_t)t * D;
        float *yt = y + (size_t)t * D;
        
        float var = 0.0f;
        for (int i = 0; i < D; i++) var += xt[i] * xt[i];
        var /= D;
        float r = 1.0f / sqrtf(var + eps);
        rms[t] = r;
        for (int i = 0; i < D; i++) {
            float xh = xt[i] * r;
            xhatt[i] = xh;
            yt[i] = g[i] * xh;
        }
    }
}

void tc_rmsnorm_bwd(const float *dy, const float *xhat, const float *gamma, const float *rms, int T, int D, float *dgamma, float *dx) {
    for (int t = 0; t < T; t++) {
        const float *dyt = dy + (size_t)t * D;
        const float *xhatt = xhat + (size_t)t * D;
        float *dxt = dx ? (dx + (size_t)t * D) : NULL;
        float r = rms[t];
        
        float sum2 = 0.0f;
        float dxhatt[1024]; // Assuming max D=1024
        for (int i = 0; i < D; i++) {
            if (dgamma) dgamma[i] += dyt[i] * xhatt[i];
            dxhatt[i] = dyt[i] * gamma[i];
            sum2 += dxhatt[i] * xhatt[i];
        }
        if (dxt) {
            for (int i = 0; i < D; i++) {
                dxt[i] += r * (dxhatt[i] - xhatt[i] * sum2 / D);
            }
        }
    }
}

void tc_swiglu_fwd(const float *hgate, const float *hup, float *hsilu, int T, int FFD) {
    for (int t = 0; t < T; t++) {
        const float *g = hgate + (size_t)t * FFD;
        const float *u = hup + (size_t)t * FFD;
        float *s = hsilu + (size_t)t * FFD;
        for (int i = 0; i < FFD; i++) {
            float x = g[i];
            s[i] = (x / (1.0f + expf(-x))) * u[i];
        }
    }
}

void tc_swiglu_bwd(const float *dhsilu, const float *hgate, const float *hup, float *dhgate, float *dhup, int T, int FFD) {
    for (int t = 0; t < T; t++) {
        const float *dhs = dhsilu + (size_t)t * FFD;
        const float *g = hgate + (size_t)t * FFD;
        const float *u = hup + (size_t)t * FFD;
        float *dg = dhgate + (size_t)t * FFD;
        float *du = dhup + (size_t)t * FFD;
        for (int i = 0; i < FFD; i++) {
            float x = g[i];
            float sig = 1.0f / (1.0f + expf(-x));
            float swish = x * sig;
            dg[i] += dhs[i] * u[i] * (swish + sig * (1.0f - swish));
            du[i] += dhs[i] * swish;
        }
    }
}

void tc_wkv_fwd(const float *k, const float *v, const float *r_sig, const float *decay, float *state, float *wkv, int T, int D) {
    for (int t = 0; t < T; t++) {
        const float *kt = k + (size_t)t * D;
        const float *vt = v + (size_t)t * D;
        const float *rs = r_sig + (size_t)t * D;
        const float *stprev = t > 0 ? state + (size_t)(t - 1) * D : NULL;
        float *st = state + (size_t)t * D;
        float *wk = wkv + (size_t)t * D;
        for (int i = 0; i < D; i++) {
            float dc = 1.0f / (1.0f + expf(-decay[i]));
            float sp = stprev ? stprev[i] : 0.0f;
            st[i] = dc * sp + (1.0f - dc) * (kt[i] * vt[i]);
            wk[i] = rs[i] * st[i];
        }
    }
}

void tc_wkv_bwd(const float *dwkv, const float *k, const float *v, const float *r_sig, const float *decay, const float *state, float *dstate, float *drpre, float *dk, float *dv, float *ddecay, int T, int D) {
    for (int t = T - 1; t >= 0; t--) {
        const float *rs = r_sig + (size_t)t * D;
        const float *st = state + (size_t)t * D;
        const float *dw = dwkv + (size_t)t * D;
        float *dr = drpre + (size_t)t * D;
        float *dkt = dk + (size_t)t * D;
        float *dvt = dv + (size_t)t * D;
        const float *kt = k + (size_t)t * D;
        const float *vt = v + (size_t)t * D;
        for (int i = 0; i < D; i++) {
            float drs_i = dw[i] * st[i];
            float dst_i = dw[i] * rs[i] + dstate[(size_t)t * D + i];
            dr[i] = drs_i * rs[i] * (1.0f - rs[i]);
            float dc = 1.0f / (1.0f + expf(-decay[i]));
            float sp = t > 0 ? state[(size_t)(t - 1) * D + i] : 0.0f;
            float dgate = 1.0f - dc;
            dkt[i] = dst_i * dgate * vt[i];
            dvt[i] = dst_i * dgate * kt[i];
            if (t > 0) dstate[(size_t)(t - 1) * D + i] += dst_i * dc;
            if (ddecay) {
                ddecay[i] += dst_i * (sp - kt[i] * vt[i]) * dc * (1.0f - dc);
            }
        }
    }
}

void tc_attn_fwd(const float *Q, const float *K, const float *V, float *attn_w, float *attn_ctx, int T, int D, int H, int n_kv_heads) {
    int Dh = D / H;
    int kv_D = (D / H) * n_kv_heads;
    int kv_Dh = D / H;
    float invsqrt_dh = 1.0f / sqrtf((float)Dh);
    for (int h = 0; h < H; h++) {
        int kv_h = h / (H / n_kv_heads);
        for (int t = 0; t < T; t++) {
            float scores[4096];
            float maxs = -1e30f;
            for (int u = 0; u <= t; u++) {
                const float *qh = Q + (size_t)t * D + h * Dh;
                const float *kh = K + (size_t)u * kv_D + kv_h * kv_Dh;
                float s = 0.0f;
                for (int i = 0; i < Dh; i++) s += qh[i] * kh[i];
                s *= invsqrt_dh;
                scores[u] = s;
                if (s > maxs) maxs = s;
            }
            float sum = 0.0f;
            float *wrow = attn_w + ((size_t)h * T + t) * T;
            for (int u = 0; u <= t; u++) {
                float w = expf(scores[u] - maxs);
                wrow[u] = w;
                sum += w;
            }
            for (int u = 0; u <= t; u++) wrow[u] /= sum;
            float *ctx = attn_ctx + (size_t)t * D + h * Dh;
            for (int i = 0; i < Dh; i++) ctx[i] = 0.0f;
            for (int u = 0; u <= t; u++) {
                const float *vh = V + (size_t)u * kv_D + kv_h * kv_Dh;
                float w = wrow[u];
                for (int i = 0; i < Dh; i++) ctx[i] += w * vh[i];
            }
        }
    }
}

void tc_attn_bwd(const float *dctx, const float *attn_w, const float *Q, const float *K, const float *V, float *dQ, float *dK, float *dV, int T, int D, int H, int n_kv_heads) {
    int Dh = D / H;
    int kv_D = (D / H) * n_kv_heads;
    int kv_Dh = D / H;
    float invsqrt_dh = 1.0f / sqrtf((float)Dh);
    for (int h = 0; h < H; h++) {
        int kv_h = h / (H / n_kv_heads);
        for (int t = 0; t < T; t++) {
            const float *dctx_h = dctx + (size_t)t * D + h * Dh;
            const float *wrow = attn_w + ((size_t)h * T + t) * T;
            float dw[4096];
            for (int u = 0; u <= t; u++) {
                const float *vh = V + (size_t)u * kv_D + kv_h * kv_Dh;
                float *dvh = dV + (size_t)u * kv_D + kv_h * kv_Dh;
                float s = 0.0f;
                for (int i = 0; i < Dh; i++) { s += dctx_h[i] * vh[i]; dvh[i] += wrow[u] * dctx_h[i]; }
                dw[u] = s;
            }
            float dot = 0.0f;
            for (int u = 0; u <= t; u++) dot += wrow[u] * dw[u];
            float *dqh = dQ + (size_t)t * D + h * Dh;
            for (int u = 0; u <= t; u++) {
                float ds = wrow[u] * (dw[u] - dot) * invsqrt_dh;
                const float *kh = K + (size_t)u * kv_D + kv_h * kv_Dh;
                float *dkh = dK + (size_t)u * kv_D + kv_h * kv_Dh;
                const float *qh = Q + (size_t)t * D + h * Dh;
                for (int i = 0; i < Dh; i++) { dqh[i] += ds * kh[i]; dkh[i] += ds * qh[i]; }
            }
        }
    }
}

void tc_ce_fwd(const float *logits, const int *targets, float *probs, float *loss_out, int *n_active_out, int T, int V) {
    float total_loss = 0.0f;
    int n_active = 0;
    for (int t = 0; t < T; t++) {
        const float *logit = logits + (size_t)t * V;
        float *prob = probs + (size_t)t * V;
        float maxv = -1e30f;
        for (int i = 0; i < V; i++) if (logit[i] > maxv) maxv = logit[i];
        float sum = 0.0f;
        for (int i = 0; i < V; i++) {
            prob[i] = expf(logit[i] - maxv);
            sum += prob[i];
        }
        for (int i = 0; i < V; i++) prob[i] /= sum;
        
        if (targets && targets[t] >= 0) {
            float pt = prob[targets[t]];
            if (pt < 1e-9f) pt = 1e-9f;
            total_loss += -logf(pt);
            n_active++;
        }
    }
    if (loss_out) *loss_out = total_loss;
    if (n_active_out) *n_active_out = n_active;
}

void tc_ce_bwd(const float *probs, const int *targets, float *dlogits, int n_active, float weight, int T, int V) {
    float inv_n = n_active > 0 ? (weight / (float)n_active) : 0.0f;
    for (int t = 0; t < T; t++) {
        float *dlogit = dlogits + (size_t)t * V;
        const float *prob = probs + (size_t)t * V;
        int target = targets ? targets[t] : -1;
        if (target >= 0) {
            for (int i = 0; i < V; i++) {
                dlogit[i] = prob[i] * inv_n;
            }
            dlogit[target] -= inv_n;
        } else {
            for (int i = 0; i < V; i++) {
                dlogit[i] = 0.0f;
            }
        }
    }
}
