#include "tcmodel.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define LN_EPS 1e-5f

#if defined(__APPLE__) && defined(ACCELERATE_NEW_LAPACK)
#include <Accelerate/Accelerate.h>
#endif

/* Several hot loops use fixed-size stack buffers sized for models
 * up to D=1024, FF=4096 instead of alloca/VLA. */
static void tc_config_check(TCConfig cfg) {
    if (cfg.max_seq_len <= 0 || cfg.max_seq_len > 4096) {
        printf("tc_config_check FAILED: max_seq_len=%d, d_model=%d, n_layers=%d\n", cfg.max_seq_len, cfg.d_model, cfg.n_layers);
    }
    assert(cfg.d_model > 0 && cfg.d_model <= TC_MAX_D_MODEL);
    assert(cfg.ff_mult > 0 && cfg.ff_mult * cfg.d_model <= TC_MAX_FF_DIM);
    assert(cfg.max_seq_len > 0 && cfg.max_seq_len <= 4096);
    assert(cfg.vocab_size > 0 && cfg.vocab_size <= 4096);
    assert(cfg.n_heads > 0 && cfg.d_model % cfg.n_heads == 0);
    assert(cfg.n_layers > 0);
}

TCConfig tc_default_config(void) {
    TCConfig c;
    c.vocab_size = 96;   /* ASCII 32..127 */
    c.d_model = 16;
    c.n_layers = 2;
    c.n_heads = 2;
    c.n_kv_heads = 2;
    c.ff_mult = 1;
    c.max_seq_len = 64;
    return c;
}

/* ---------------------------------------------------------------------
 * Parameter layout: one function walks the same sequence of "take a
 * chunk of N floats" steps whether buf is NULL (counting pass) or a
 * real allocated buffer (assigning pass). This keeps the count and the
 * pointer assignment mechanically impossible to drift apart.
 * ------------------------------------------------------------------- */

static int layout_layer(TCLayerView *lv, float *buf, int off, TCConfig cfg) {
    int D = cfg.d_model;
    int FF = cfg.ff_mult * D;
    int kv_D = (D / cfg.n_heads) * cfg.n_kv_heads;
#define TAKE(field, n) do { lv->field = buf ? buf + off : NULL; off += (n); } while (0)
    TAKE(ln1_gamma, D);
    TAKE(tm_mix_k, D); TAKE(tm_mix_v, D); TAKE(tm_mix_r, D);
    TAKE(tm_Wk, D * D); TAKE(tm_Wv, D * D); TAKE(tm_Wr, D * D);
    TAKE(tm_decay, D);
    TAKE(tm_Wo, D * D);

    TAKE(ln2_gamma, D);
    TAKE(at_Wq, D * D); TAKE(at_Wk, kv_D * D); TAKE(at_Wv, kv_D * D); TAKE(at_Wo, D * D);

    TAKE(ln3_gamma, D);
    TAKE(cm_mix_gate, D); TAKE(cm_mix_up, D);
    TAKE(cm_Wgate, D * FF); TAKE(cm_Wup, D * FF); TAKE(cm_Wdown, FF * D);
#undef TAKE
    return off;
}

static int layout_all(TCParamSet *ps, float *buf, TCConfig cfg) {
    int off = 0;
    int D = cfg.d_model, V = cfg.vocab_size;
#define TAKE(field, n) do { ps->field = buf ? buf + off : NULL; off += (n); } while (0)
    TAKE(embed, V * D);
    TAKE(ln_f_gamma, D);
    TAKE(pool_w, D);
    TAKE(mtp_head1, D * D);  /* MTP t+2 projection */
    TAKE(mtp_head2, D * D);  /* MTP t+3 projection */
#undef TAKE
    for (int l = 0; l < cfg.n_layers; l++)
        off = layout_layer(&ps->layers[l], buf, off, cfg);
    return off;
}

int tc_param_count(TCConfig cfg) {
    TCParamSet tmp;
    tmp.layers = calloc(cfg.n_layers, sizeof(TCLayerView));
    int n = layout_all(&tmp, NULL, cfg);
    free(tmp.layers);
    return n;
}

TCParamSet *tc_paramset_create(TCConfig cfg) {
    tc_config_check(cfg);
    TCParamSet *ps = calloc(1, sizeof(TCParamSet));
    ps->cfg = cfg;
    ps->layers = calloc(cfg.n_layers, sizeof(TCLayerView));
    int n = layout_all(ps, NULL, cfg);
    ps->n_floats = n;
    posix_memalign((void**)&ps->buf, 16384, (size_t)n * sizeof(float));
    memset(ps->buf, 0, (size_t)n * sizeof(float));
    layout_all(ps, ps->buf, cfg);
    return ps;
}

void tc_paramset_free(TCParamSet *ps) {
    if (!ps) return;
    free(ps->buf);
    free(ps->layers);
    free(ps);
}

void tc_paramset_zero(TCParamSet *ps) {
    memset(ps->buf, 0, (size_t)ps->n_floats * sizeof(float));
}

static float randf(unsigned int *s) {
    /* xorshift32 */
    unsigned int x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return (float)(x % 1000000) / 1000000.0f; /* [0,1) */
}

void tc_paramset_init_random(TCParamSet *ps, unsigned int seed) {
    unsigned int s = seed ? seed : 1;
    int D = ps->cfg.d_model;
    float wscale = 1.0f / sqrtf((float)D);

    for (int i = 0; i < ps->cfg.vocab_size * D; i++)
        ps->embed[i] = (randf(&s) * 2 - 1) * 0.1f;
    for (int i = 0; i < D; i++) { ps->ln_f_gamma[i] = 1.0f; }
    for (int i = 0; i < D; i++) ps->pool_w[i] = (randf(&s) * 2 - 1) * wscale;
    int DD = D * D;
    for (int i = 0; i < DD; i++) {
        ps->mtp_head1[i] = (randf(&s) * 2 - 1) * wscale;
        ps->mtp_head2[i] = (randf(&s) * 2 - 1) * wscale;
    }

    for (int l = 0; l < ps->cfg.n_layers; l++) {
        TCLayerView *lv = &ps->layers[l];
        for (int i = 0; i < D; i++) {
            lv->ln1_gamma[i] = 1.0f;
            lv->ln2_gamma[i] = 1.0f;
            lv->ln3_gamma[i] = 1.0f;
            /* start mostly "pass through current token" and moderate decay */
            lv->tm_mix_k[i] = 1.0f; lv->tm_mix_v[i] = 1.0f; lv->tm_mix_r[i] = 1.0f;
            lv->tm_decay[i] = 0.0f; /* sigmoid(0)=0.5 */
            lv->cm_mix_gate[i] = 1.0f; lv->cm_mix_up[i] = 1.0f;
        }
        int DD = D * D, FFD = ps->cfg.ff_mult * D;
        int kv_DD = D * ((D / ps->cfg.n_heads) * ps->cfg.n_kv_heads);
        for (int i = 0; i < DD; i++) {
            lv->tm_Wk[i] = (randf(&s) * 2 - 1) * wscale;
            lv->tm_Wv[i] = (randf(&s) * 2 - 1) * wscale;
            lv->tm_Wr[i] = (randf(&s) * 2 - 1) * wscale;
            lv->tm_Wo[i] = (randf(&s) * 2 - 1) * wscale;
            lv->at_Wq[i] = (randf(&s) * 2 - 1) * wscale;
            lv->at_Wo[i] = (randf(&s) * 2 - 1) * wscale;
        }
        for (int i = 0; i < kv_DD; i++) {
            lv->at_Wk[i] = (randf(&s) * 2 - 1) * wscale;
            lv->at_Wv[i] = (randf(&s) * 2 - 1) * wscale;
        }
        for (int i = 0; i < D * FFD; i++) {
            lv->cm_Wgate[i] = (randf(&s) * 2 - 1) * wscale;
            lv->cm_Wup[i] = (randf(&s) * 2 - 1) * wscale;
        }
        for (int i = 0; i < FFD * D; i++)
            lv->cm_Wdown[i] = (randf(&s) * 2 - 1) * wscale;
    }
}

int tc_paramset_save(const TCParamSet *ps, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t wrote = fwrite(&ps->cfg, sizeof(TCConfig), 1, f);
    wrote += fwrite(ps->buf, sizeof(float), (size_t)ps->n_floats, f);
    fclose(f);
    return wrote == (size_t)(1 + ps->n_floats) ? 0 : -1;
}

TCParamSet *tc_paramset_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    TCConfig cfg;
    if (fread(&cfg, sizeof(TCConfig), 1, f) != 1) { fclose(f); return NULL; }
    if (cfg.max_seq_len <= 0 || cfg.max_seq_len > 4096) cfg.max_seq_len = 128;
    TCParamSet *ps = tc_paramset_create(cfg);
    size_t n = fread(ps->buf, sizeof(float), (size_t)ps->n_floats, f);
    fclose(f);
    if (n != (size_t)ps->n_floats) { tc_paramset_free(ps); return NULL; }
    return ps;
}

/* ---------------------------------------------------------------------
 * Cache allocation (mirrors the shapes used in forward/backward).
 * ------------------------------------------------------------------- */

static int layout_cache_layer(TCLayerCache *lc, float *buf, int off, TCConfig cfg) {
    int D = cfg.d_model, T = cfg.max_seq_len;
    int FF = cfg.ff_mult * D;
    int kv_D = (D / cfg.n_heads) * cfg.n_kv_heads;
#define TAKE(field, n) do { lc->field = buf ? buf + off : NULL; off += (n); } while (0)
    TAKE(x0, T * D);
    TAKE(ln1_rms, T); TAKE(ln1_out, T * D); TAKE(ln1_xhat, T * D);
    TAKE(xk, T * D); TAKE(xv, T * D); TAKE(xr, T * D);
    TAKE(k, T * D); TAKE(v, T * D); TAKE(r_sig, T * D);
    TAKE(state, T * D); TAKE(wkv, T * D); TAKE(tm_out, T * D); TAKE(resid1, T * D);

    TAKE(ln2_rms, T); TAKE(ln2_out, T * D); TAKE(ln2_xhat, T * D);
    TAKE(Q, T * D); TAKE(K, T * kv_D); TAKE(V, T * kv_D);
    TAKE(attn_w, cfg.n_heads * T * T);
    TAKE(attn_ctx, T * D); TAKE(attn_out, T * D); TAKE(resid2, T * D);

    TAKE(ln3_rms, T); TAKE(ln3_out, T * D); TAKE(ln3_xhat, T * D);
    TAKE(cmxgate, T * D); TAKE(cmxup, T * D);
    TAKE(h_gate, T * FF); TAKE(h_up, T * FF); TAKE(h_silu, T * FF);
    TAKE(cm_out, T * D); TAKE(resid3, T * D);
#undef TAKE
    return off;
}

static int layout_cache_all(TCCache *c, float *buf, TCConfig cfg) {
    int off = 0;
    int D = cfg.d_model, T = cfg.max_seq_len, V = cfg.vocab_size;
#define TAKE(field, n) do { c->field = buf ? buf + off : NULL; off += (n); } while (0)
    TAKE(ln_f_rms, T); TAKE(ln_f_out, T * D); TAKE(ln_f_xhat, T * D);
    TAKE(logits, T * V); TAKE(probs, T * V);
    TAKE(mtp_h1, T * D); TAKE(mtp_h2, T * D);
    TAKE(mtp_probs1, T * V); TAKE(mtp_probs2, T * V);
    TAKE(pool_scores, T); TAKE(pool_weights, T); TAKE(pool_raw, D);
#undef TAKE
    for (int l = 0; l < cfg.n_layers; l++)
        off = layout_cache_layer(&c->layers[l], buf, off, cfg);
    return off;
}

TCCache *tc_cache_create(TCConfig cfg) {
    TCCache *c = calloc(1, sizeof(TCCache));
    c->cfg = cfg;
    c->T = cfg.max_seq_len;
    c->layers = calloc(cfg.n_layers, sizeof(TCLayerCache));
    int n = layout_cache_all(c, NULL, cfg);
    c->n_floats = n;
    posix_memalign((void**)&c->buf, 16384, (size_t)n * sizeof(float));
    memset(c->buf, 0, (size_t)n * sizeof(float));
    layout_cache_all(c, c->buf, cfg);
    return c;
}

void tc_cache_free(TCCache *c) {
    if (!c) return;
    free(c->buf);
    free(c->layers);
    free(c);
}

/* ---------------------------------------------------------------------
 * Small math helpers. Vectors are D-wide float* unless noted.
 * ------------------------------------------------------------------- */

static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

#include "tc_ops.h"

/* y = W x, W row-major [out x in] */
static void matvec(const float *W, const float *x, float *y, int out, int in) {
    tc_gemv(y, W, x, out, in);
}

/* Accumulates dW += dy (x) x^T, dx += W^T dy. dx must already hold valid
 * data to accumulate into (caller zeroes once per position). */
static void matvec_backward(const float *W, float *dW, const float *x, float *dx,
                             const float *dy, int out, int in) {
    if (dW) {
        tc_ger(dW, x, dy, out, in, 1.0f);
    }
    if (dx) {
        tc_gemv_t_add(dx, W, dy, out, in);
    }
}


static void rope_forward(float *q, float *k, int t, int D, int n_heads, int n_kv_heads) {
    int Dh = D / n_heads;
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < Dh; i += 2) {
            float theta = (float)t * powf(10000.0f, -(float)i / (float)Dh);
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            int idx0 = h * Dh + i;
            int idx1 = idx0 + 1;
            float q0 = q[idx0], q1 = q[idx1];
            q[idx0] = q0 * cos_t - q1 * sin_t;
            q[idx1] = q1 * cos_t + q0 * sin_t;
        }
    }
    int kv_Dh = Dh;
    for (int h = 0; h < n_kv_heads; h++) {
        for (int i = 0; i < kv_Dh; i += 2) {
            float theta = (float)t * powf(10000.0f, -(float)i / (float)kv_Dh);
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            int idx0 = h * kv_Dh + i;
            int idx1 = idx0 + 1;
            float k0 = k[idx0], k1 = k[idx1];
            k[idx0] = k0 * cos_t - k1 * sin_t;
            k[idx1] = k1 * cos_t + k0 * sin_t;
        }
    }
}

static void rope_backward(float *dq, float *dk, int t, int D, int n_heads, int n_kv_heads) {
    int Dh = D / n_heads;
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < Dh; i += 2) {
            float theta = (float)t * powf(10000.0f, -(float)i / (float)Dh);
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            int idx0 = h * Dh + i;
            int idx1 = idx0 + 1;
            float dq0 = dq[idx0], dq1 = dq[idx1];
            dq[idx0] = dq0 * cos_t + dq1 * sin_t;
            dq[idx1] = dq1 * cos_t - dq0 * sin_t;
        }
    }
    int kv_Dh = Dh;
    for (int h = 0; h < n_kv_heads; h++) {
        for (int i = 0; i < kv_Dh; i += 2) {
            float theta = (float)t * powf(10000.0f, -(float)i / (float)kv_Dh);
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            int idx0 = h * kv_Dh + i;
            int idx1 = idx0 + 1;
            float dk0 = dk[idx0], dk1 = dk[idx1];
            dk[idx0] = dk0 * cos_t + dk1 * sin_t;
            dk[idx1] = dk1 * cos_t - dk0 * sin_t;
        }
    }
}

/* ---------------------------------------------------------------------
 * Forward
 * ------------------------------------------------------------------- */

/* Shared backward tail (defined further down, alongside tc_backward);
 * forward-declared here so any head's backward (generative, pooling, ...)
 * can call it regardless of definition order in this file. */
static void tc_encode_backward(const TCParamSet *p, TCParamSet *grad, const TCCache *c,
                                const int *ids, int T, const float *dln_f_out,
                                float *out_dx0);

/* Runs the shared layer stack + final LN, filling every cache field up to
 * and including c->ln_f_out. Both tc_forward (which adds the tied-embedding
 * head + softmax + loss) and tc_embed (which mean-pools instead) build on
 * this so the two specialist "heads" can never drift out of sync with the
 * body that produces them. Supports single-head ablation and latent steering. */
static void tc_encode_ex(const TCParamSet *p, TCCache *c, const int *ids, int T,
                         int ablate_layer, int ablate_head,
                         const TCSteerConfig *steer) {
    TCConfig cfg = p->cfg;
    int D = cfg.d_model, H = cfg.n_heads, Dh = D / H, FFD = cfg.ff_mult * D;
    float invsqrt_dh = 1.0f / sqrtf((float)Dh);
    c->T = T;

    float *x = malloc((size_t)T * D * sizeof(float));
    for (int t = 0; t < T; t++)
        memcpy(x + (size_t)t * D, p->embed + (size_t)ids[t] * D, D * sizeof(float));

    for (int l = 0; l < cfg.n_layers; l++) {
        TCLayerView *lv = &p->layers[l];
        TCLayerCache *lc = &c->layers[l];
        lc->T = T;
        memcpy(lc->x0, x, (size_t)T * D * sizeof(float));

        /* --- gated linear recurrence (time-mix) --- */
        tc_rmsnorm_fwd(x, lv->ln1_gamma, lc->ln1_rms, lc->ln1_xhat, lc->ln1_out, T, D, LN_EPS);
        for (int t = 0; t < T; t++) {
            const float *cur = lc->ln1_out + (size_t)t * D;
            const float *prev = t > 0 ? lc->ln1_out + (size_t)(t - 1) * D : NULL;
            float *xk = lc->xk + (size_t)t * D, *xv = lc->xv + (size_t)t * D, *xr = lc->xr + (size_t)t * D;
            for (int i = 0; i < D; i++) {
                float mk = sigmoidf_(lv->tm_mix_k[i]);
                float mv = sigmoidf_(lv->tm_mix_v[i]);
                float mr = sigmoidf_(lv->tm_mix_r[i]);
                float p_ = prev ? prev[i] : 0.0f;
                xk[i] = mk * cur[i] + (1 - mk) * p_;
                xv[i] = mv * cur[i] + (1 - mv) * p_;
                xr[i] = mr * cur[i] + (1 - mr) * p_;
            }
        }
        tc_gemm_nt(lc->k, lc->xk, lv->tm_Wk, T, D, D);
        tc_gemm_nt(lc->v, lc->xv, lv->tm_Wv, T, D, D);
        tc_gemm_nt(lc->r_sig, lc->xr, lv->tm_Wr, T, D, D);
        for (int i = 0; i < T * D; i++) {
            lc->r_sig[i] = sigmoidf_(lc->r_sig[i]);
        }
        
        tc_wkv_fwd(lc->k, lc->v, lc->r_sig, lv->tm_decay, lc->state, lc->wkv, T, D);

        tc_gemm_nt(lc->tm_out, lc->wkv, lv->tm_Wo, T, D, D);
        for (int i = 0; i < T * D; i++) {
            lc->resid1[i] = lc->x0[i] + lc->tm_out[i];
        }

        /* --- tiny causal self-attention --- */
        int kv_D = (D / H) * cfg.n_kv_heads;
        int kv_Dh = D / H;
        tc_rmsnorm_fwd(lc->resid1, lv->ln2_gamma, lc->ln2_rms, lc->ln2_xhat, lc->ln2_out, T, D, LN_EPS);
        tc_gemm_nt(lc->Q, lc->ln2_out, lv->at_Wq, T, D, D);
        tc_gemm_nt(lc->K, lc->ln2_out, lv->at_Wk, T, kv_D, D);
        tc_gemm_nt(lc->V, lc->ln2_out, lv->at_Wv, T, kv_D, D);
        for (int t = 0; t < T; t++) {
            rope_forward(lc->Q + (size_t)t * D, lc->K + (size_t)t * kv_D, t, D, H, cfg.n_kv_heads);
        }

        tc_attn_fwd(lc->Q, lc->K, lc->V, lc->attn_w, lc->attn_ctx, T, D, H, cfg.n_kv_heads);

        tc_gemm_nt(lc->attn_out, lc->attn_ctx, lv->at_Wo, T, D, D);
        for (int i = 0; i < T * D; i++) {
            lc->resid2[i] = lc->resid1[i] + lc->attn_out[i];
        }

        /* --- SwiGLU FFN --- */
        tc_rmsnorm_fwd(lc->resid2, lv->ln3_gamma, lc->ln3_rms, lc->ln3_xhat, lc->ln3_out, T, D, LN_EPS);
        for (int t = 0; t < T; t++) {
            const float *cur = lc->ln3_out + (size_t)t * D;
            const float *prev = t > 0 ? lc->ln3_out + (size_t)(t - 1) * D : NULL;
            float *cmxgate = lc->cmxgate + (size_t)t * D, *cmxup = lc->cmxup + (size_t)t * D;
            for (int i = 0; i < D; i++) {
                float mgate = sigmoidf_(lv->cm_mix_gate[i]);
                float mup = sigmoidf_(lv->cm_mix_up[i]);
                float p_ = prev ? prev[i] : 0.0f;
                cmxgate[i] = mgate * cur[i] + (1 - mgate) * p_;
                cmxup[i] = mup * cur[i] + (1 - mup) * p_;
            }
        }
        tc_gemm_nt(lc->h_gate, lc->cmxgate, lv->cm_Wgate, T, FFD, D);
        tc_gemm_nt(lc->h_up, lc->cmxup, lv->cm_Wup, T, FFD, D);
        tc_swiglu_fwd(lc->h_gate, lc->h_up, lc->h_silu, T, FFD);
        tc_gemm_nt(lc->cm_out, lc->h_silu, lv->cm_Wdown, T, D, FFD);
        for (int i = 0; i < T * D; i++) {
            lc->resid3[i] = lc->resid2[i] + lc->cm_out[i];
        }

        memcpy(x, lc->resid3, (size_t)T * D * sizeof(float));

        /* Latent activation steering injection */
        if (steer && steer->layer == l && steer->vec && fabsf(steer->alpha) > 1e-7f) {
            float a = steer->alpha;
            const float *v = steer->vec;
            for (int t = 0; t < T; t++) {
                float *xt = x + (size_t)t * D;
                float *r3 = lc->resid3 + (size_t)t * D;
                for (int i = 0; i < D; i++) {
                    float delta = a * v[i];
                    xt[i] += delta;
                    r3[i] += delta;
                }
            }
        }
    }

    /* --- final LN --- */
    tc_rmsnorm_fwd(x, p->ln_f_gamma, c->ln_f_rms, c->ln_f_xhat, c->ln_f_out, T, D, LN_EPS);
    free(x);
}

static void tc_encode(const TCParamSet *p, TCCache *c, const int *ids, int T) {
    tc_encode_ex(p, c, ids, T, -1, -1, NULL);
}

/* Generative head: tied-embedding projection + softmax + (optional) loss. */
void tc_forward_ex(const TCParamSet *p, TCCache *c, const int *ids, int T,
                   const int *targets, float *out_loss, int ablate_layer, int ablate_head) {
    tc_encode_ex(p, c, ids, T, ablate_layer, ablate_head, NULL);
    int V = p->cfg.vocab_size, D = p->cfg.d_model;
    float total_loss = 0.0f;
    int n_active = 0;
    for (int t = 0; t < T; t++) {
        float *logit = c->logits + (size_t)t * V;
        matvec(p->embed, c->ln_f_out + (size_t)t * D, logit, V, D);
    }
    tc_ce_fwd(c->logits, targets, c->probs, &total_loss, &n_active, T, V);
    if (targets && out_loss) *out_loss = n_active > 0 ? (total_loss / (float)n_active) : 0.0f;
}

void tc_forward(const TCParamSet *p, TCCache *c, const int *ids, int T,
                 const int *targets, float *out_loss) {
    tc_forward_ex(p, c, ids, T, targets, out_loss, -1, -1);
}

/* Forward with contrastive (unlikelihood) loss.
 * targets_neg[t] >= 0 means "penalize if model predicts this token at position t".
 * ul_weight scales the penalty relative to the standard loss. */
void tc_forward_ul(const TCParamSet *p, TCCache *c, const int *ids, int T,
                   const int *targets, const int *targets_neg, float ul_weight,
                   float *out_loss) {
    tc_forward_ex(p, c, ids, T, targets, out_loss, -1, -1);
    if (!targets_neg || !out_loss || ul_weight <= 0.0f) return;
    int V = p->cfg.vocab_size;
    int n_active = 0;
    for (int t = 0; t < T; t++) if (targets[t] >= 0) n_active++;
    if (n_active == 0) return;
    float ul_total = 0.0f;
    for (int t = 0; t < T; t++) {
        if (targets_neg[t] < 0) continue;
        const float *prob = c->probs + (size_t)t * V;
        float p_neg = prob[targets_neg[t]];
        if (p_neg > 1.0f - 1e-9f) p_neg = 1.0f - 1e-9f;
        ul_total += -logf(1.0f - p_neg);
    }
    *out_loss += ul_weight * ul_total / (float)n_active;
}

/* Multi-Token Prediction forward.
 * After standard forward, project ln_f_out through mtp_head1 (D×D) and
 * mtp_head2 (D×D), then tied-embedding logits → softmax → CE loss for
 * t+2 and t+3 targets. Caches mtp_h1/h2 and mtp_probs1/2 for backward. */
void tc_forward_mtp(const TCParamSet *p, TCCache *c, const int *ids, int T,
                    const int *targets, const int *targets_t2, const int *targets_t3,
                    float mtp_weight, float *out_loss) {
    /* Standard t+1 forward (fills c->probs, c->ln_f_out, etc.) */
    tc_forward_ex(p, c, ids, T, targets, out_loss, -1, -1);
    if (mtp_weight <= 0.0f || !out_loss) return;

    int V = p->cfg.vocab_size, D = p->cfg.d_model;
    int n_active = 0;
    for (int t = 0; t < T; t++) if (targets && targets[t] >= 0) n_active++;
    if (n_active == 0) return;

    /* Process each MTP head: project, logits, softmax, CE */
    const float *heads[2] = { p->mtp_head1, p->mtp_head2 };
    float *h_bufs[2] = { c->mtp_h1, c->mtp_h2 };
    float *prob_bufs[2] = { c->mtp_probs1, c->mtp_probs2 };
    const int *tgt_arrays[2] = { targets_t2, targets_t3 };

    for (int head = 0; head < 2; head++) {
        if (!tgt_arrays[head]) continue;
        const float *W = heads[head];
        float *h = h_bufs[head];
        float *probs_out = prob_bufs[head];

        float *logits = malloc((size_t)T * V * sizeof(float));
        for (int t = 0; t < T; t++) {
            /* h[t] = ln_f_out[t] @ W^T (D×D) */
            matvec(W, c->ln_f_out + (size_t)t * D, h + (size_t)t * D, D, D);

            /* logits[t] = h[t] @ embed^T (V×D) */
            float *logit = logits + (size_t)t * V;
            matvec(p->embed, h + (size_t)t * D, logit, V, D);
        }

        float mtp_loss = 0.0f;
        int mtp_active = 0;
        tc_ce_fwd(logits, tgt_arrays[head], prob_bufs[head], &mtp_loss, &mtp_active, T, V);

        if (mtp_active > 0)
            *out_loss += mtp_weight * mtp_loss / (float)mtp_active;
        free(logits);
    }
}

/* Generative head with latent activation steering */
void tc_forward_steered(const TCParamSet *p, TCCache *c, const int *ids, int T,
                        const int *targets, float *out_loss, const TCSteerConfig *steer) {
    tc_encode_ex(p, c, ids, T, -1, -1, steer);
    int V = p->cfg.vocab_size, D = p->cfg.d_model;
    float total_loss = 0.0f;
    int n_active = 0;
    for (int t = 0; t < T; t++) {
        float *logit = c->logits + (size_t)t * V;
        matvec(p->embed, c->ln_f_out + (size_t)t * D, logit, V, D);
    }
    tc_ce_fwd(c->logits, targets, c->probs, &total_loss, &n_active, T, V);
    if (targets && out_loss) *out_loss = n_active > 0 ? (total_loss / (float)n_active) : 0.0f;
}

/* Embedding head: take the LAST position's final hidden state and
 * L2-normalize it. Same body as tc_forward, different head -- the "second
 * specialist type" is a reinterpretation of the pooled representation, not
 * a different network. Mean-pooling over all positions was tried first and
 * measured worse: on this causal architecture the last position is the
 * only one that has attended to (via the recurrence + causal attention)
 * the entire sequence, so it acts as a natural summary; averaging in every
 * early, low-context position dilutes that signal. Empirically, mean pool
 * put an unrelated gibberish string at cosine 0.87-0.95 to real sentences
 * (no separation at all); last-token pool put the same gibberish string at
 * -0.1 to -0.18 against real sentences while same-template sentences
 * cluster at 0.98-0.99. */
void tc_embed(const TCParamSet *p, TCCache *c, const int *ids, int T, float *out_vec) {
    tc_encode(p, c, ids, T);
    int D = p->cfg.d_model;

    float maxs = -1e30f;
    for (int t = 0; t < T; t++) {
        float s = 0.0f;
        const float *h = c->ln_f_out + (size_t)t * D;
        for (int i = 0; i < D; i++) s += p->pool_w[i] * h[i];
        c->pool_scores[t] = s;
        if (s > maxs) maxs = s;
    }
    float sum = 0.0f;
    for (int t = 0; t < T; t++) { c->pool_weights[t] = expf(c->pool_scores[t] - maxs); sum += c->pool_weights[t]; }
    for (int t = 0; t < T; t++) c->pool_weights[t] /= sum;

    for (int i = 0; i < D; i++) c->pool_raw[i] = 0.0f;
    for (int t = 0; t < T; t++) {
        float w = c->pool_weights[t];
        const float *h = c->ln_f_out + (size_t)t * D;
        for (int i = 0; i < D; i++) c->pool_raw[i] += w * h[i];
    }
    float norm = 0.0f;
    for (int i = 0; i < D; i++) norm += c->pool_raw[i] * c->pool_raw[i];
    norm = sqrtf(norm) + 1e-8f;
    c->pool_norm = norm;
    for (int i = 0; i < D; i++) out_vec[i] = c->pool_raw[i] / norm;
}

void tc_embed_backward(const TCParamSet *p, TCParamSet *grad, const TCCache *c,
                        const int *ids, int T, const float *d_out_vec) {
    int D = p->cfg.d_model;
    float out_vec[TC_MAX_D_MODEL], d_raw[TC_MAX_D_MODEL];
    for (int i = 0; i < D; i++) out_vec[i] = c->pool_raw[i] / c->pool_norm;

    /* L2-normalize backward: out = raw/norm => d_raw = (d_out - out*<d_out,out>)/norm */
    float dot = 0.0f;
    for (int i = 0; i < D; i++) dot += d_out_vec[i] * out_vec[i];
    for (int i = 0; i < D; i++) d_raw[i] = (d_out_vec[i] - out_vec[i] * dot) / c->pool_norm;

    /* weighted-sum backward: pool_raw = sum_t weights[t]*ln_f_out[t] */
    float *dln_f_out = calloc((size_t)T * D, sizeof(float));
    float d_weights[4096];
    for (int t = 0; t < T; t++) {
        const float *h = c->ln_f_out + (size_t)t * D;
        float s = 0.0f;
        for (int i = 0; i < D; i++) s += d_raw[i] * h[i];
        d_weights[t] = s;
        float *dh = dln_f_out + (size_t)t * D;
        for (int i = 0; i < D; i++) dh[i] += c->pool_weights[t] * d_raw[i];
    }

    /* softmax backward over the T-length weight distribution */
    float wdot = 0.0f;
    for (int t = 0; t < T; t++) wdot += c->pool_weights[t] * d_weights[t];
    float d_scores[4096];
    for (int t = 0; t < T; t++) d_scores[t] = c->pool_weights[t] * (d_weights[t] - wdot);

    /* scores[t] = dot(pool_w, ln_f_out[t]) */
    for (int t = 0; t < T; t++) {
        const float *h = c->ln_f_out + (size_t)t * D;
        float *dh = dln_f_out + (size_t)t * D;
        for (int i = 0; i < D; i++) {
            grad->pool_w[i] += d_scores[t] * h[i];
            dh[i] += d_scores[t] * p->pool_w[i];
        }
    }

    tc_encode_backward(p, grad, c, ids, T, dln_f_out, NULL);
    free(dln_f_out);
}

float tc_cosine(const float *a, const float *b, int D) {
    float dot = 0.0f;
    for (int i = 0; i < D; i++) dot += a[i] * b[i];
    return dot; /* both inputs are expected L2-normalized already */
}

/* ---------------------------------------------------------------------
 * Backward. dc mirrors TCCache's shapes but holds d(loss)/d(activation);
 * built once per call via tc_cache_create and zeroed at entry.
 * ------------------------------------------------------------------- */

/* Shared tail of every backward pass: given d(loss)/d(ln_f_out) as a
 * T x D array (one head's job to produce), backprops through the final LN,
 * every layer, and the input embedding lookup. Any head (generative
 * softmax, pooling, future heads) can drive this once it has reduced its
 * own loss down to a gradient on the encoder's output. */
static void tc_encode_backward(const TCParamSet *p, TCParamSet *grad, const TCCache *c,
                                const int *ids, int T, const float *dln_f_out,
                                float *out_dx0) {
    TCConfig cfg = p->cfg;
    int D = cfg.d_model, H = cfg.n_heads, Dh = D / H, FFD = cfg.ff_mult * D;

    TCCache *dc = tc_cache_create(cfg);
    dc->T = T;
    float *dx_final = calloc((size_t)T * D, sizeof(float)); /* grad w.r.t. final block output */

    tc_rmsnorm_bwd(dln_f_out, c->ln_f_xhat, p->ln_f_gamma, c->ln_f_rms, T, D,
                   grad ? grad->ln_f_gamma : NULL, dx_final);

    float *dx_next = dx_final; /* grad w.r.t. output of layer l (resid3) */

    for (int l = cfg.n_layers - 1; l >= 0; l--) {
        TCLayerView *lv = &p->layers[l];
        TCLayerView *glv = grad ? &grad->layers[l] : NULL;
        const TCLayerCache *lc = &c->layers[l];
        TCLayerCache *dlc = &dc->layers[l];

        /* resid3 = resid2 + cm_out */
        memcpy(dlc->resid2, dx_next, (size_t)T * D * sizeof(float));
        memcpy(dlc->cm_out, dx_next, (size_t)T * D * sizeof(float));

        for (int t = 0; t < T; t++) {
            float dhsilu[TC_MAX_FF_DIM] = {0};
            matvec_backward(lv->cm_Wdown, glv ? glv->cm_Wdown : NULL, lc->h_silu + (size_t)t * FFD, dhsilu, dlc->cm_out + (size_t)t * D, D, FFD);
            
            float dhgate[TC_MAX_FF_DIM] = {0};
            float dhup[TC_MAX_FF_DIM] = {0};
            tc_swiglu_bwd(dhsilu, lc->h_gate + (size_t)t * FFD, lc->h_up + (size_t)t * FFD, dhgate, dhup, 1, FFD);

            float dcmxgate[TC_MAX_D_MODEL] = {0};
            float dcmxup[TC_MAX_D_MODEL] = {0};
            matvec_backward(lv->cm_Wgate, glv ? glv->cm_Wgate : NULL, lc->cmxgate + (size_t)t * D, dcmxgate, dhgate, FFD, D);
            matvec_backward(lv->cm_Wup, glv ? glv->cm_Wup : NULL, lc->cmxup + (size_t)t * D, dcmxup, dhup, FFD, D);

            /* token-shift split */
            for (int i = 0; i < D; i++) {
                float mgate = sigmoidf_(lv->cm_mix_gate[i]);
                float mup = sigmoidf_(lv->cm_mix_up[i]);
                float cur_i = lc->ln3_out[(size_t)t * D + i];
                float prev_i = t > 0 ? lc->ln3_out[(size_t)(t - 1) * D + i] : 0.0f;
                dlc->ln3_out[(size_t)t * D + i] += dcmxgate[i] * mgate + dcmxup[i] * mup;
                if (t > 0) {
                    dlc->ln3_out[(size_t)(t - 1) * D + i] += dcmxgate[i] * (1 - mgate) + dcmxup[i] * (1 - mup);
                }
                if (glv) {
                    float dmgate = dcmxgate[i] * (cur_i - prev_i);
                    float dmup = dcmxup[i] * (cur_i - prev_i);
                    glv->cm_mix_gate[i] += dmgate * mgate * (1 - mgate);
                    glv->cm_mix_up[i] += dmup * mup * (1 - mup);
                }
            }
        }
        tc_rmsnorm_bwd(dlc->ln3_out, lc->ln3_xhat, lv->ln3_gamma, lc->ln3_rms, T, D,
                       glv ? glv->ln3_gamma : NULL, dlc->resid2);

        /* resid2 = resid1 + attn_out */
        memcpy(dlc->resid1, dlc->resid2, (size_t)T * D * sizeof(float));
        memcpy(dlc->attn_out, dlc->resid2, (size_t)T * D * sizeof(float));

        for (int t = 0; t < T; t++)
            matvec_backward(lv->at_Wo, glv ? glv->at_Wo : NULL, lc->attn_ctx + (size_t)t * D,
                             dlc->attn_ctx + (size_t)t * D, dlc->attn_out + (size_t)t * D, D, D);

        tc_attn_bwd(dlc->attn_ctx, lc->attn_w, lc->Q, lc->K, lc->V, dlc->Q, dlc->K, dlc->V, T, D, H, cfg.n_kv_heads);

        for (int t = T - 1; t >= 0; t--) {
            rope_backward(dlc->Q + (size_t)t * D, dlc->K + (size_t)t * (D/H * cfg.n_kv_heads), t, D, H, cfg.n_kv_heads);
            float *dln2 = dlc->ln2_out + (size_t)t * D;
            matvec_backward(lv->at_Wq, glv ? glv->at_Wq : NULL, lc->ln2_out + (size_t)t * D, dln2, dlc->Q + (size_t)t * D, D, D);
            matvec_backward(lv->at_Wk, glv ? glv->at_Wk : NULL, lc->ln2_out + (size_t)t * D, dln2, dlc->K + (size_t)t * (D/H * cfg.n_kv_heads), (D/H * cfg.n_kv_heads), D);
            matvec_backward(lv->at_Wv, glv ? glv->at_Wv : NULL, lc->ln2_out + (size_t)t * D, dln2, dlc->V + (size_t)t * (D/H * cfg.n_kv_heads), (D/H * cfg.n_kv_heads), D);
        }
        tc_rmsnorm_bwd(dlc->ln2_out, lc->ln2_xhat, lv->ln2_gamma, lc->ln2_rms, T, D,
                       glv ? glv->ln2_gamma : NULL, dlc->resid1);

        /* resid1 = x0 + tm_out */
        memcpy(dlc->x0, dlc->resid1, (size_t)T * D * sizeof(float));
        memcpy(dlc->tm_out, dlc->resid1, (size_t)T * D * sizeof(float));

        for (int t = 0; t < T; t++)
            matvec_backward(lv->tm_Wo, glv ? glv->tm_Wo : NULL, lc->wkv + (size_t)t * D,
                             dlc->wkv + (size_t)t * D, dlc->tm_out + (size_t)t * D, D, D);

        float *drpre = dlc->h_gate; /* alias, size T*D, free after CM bwd */
        float *dk = dlc->h_up;      /* alias */
        float *dv = dlc->h_silu;    /* alias */
        float ddecay[TC_MAX_D_MODEL] = {0};
        tc_wkv_bwd(dlc->wkv, lc->k, lc->v, lc->r_sig, lv->tm_decay, lc->state, dlc->state, drpre, dk, dv, glv ? ddecay : NULL, T, D);
        
        if (glv) {
            for (int i = 0; i < D; i++) glv->tm_decay[i] += ddecay[i];
        }

        for (int t = T - 1; t >= 0; t--) {
            float dxr_lin[TC_MAX_D_MODEL] = {0};
            matvec_backward(lv->tm_Wr, glv ? glv->tm_Wr : NULL, lc->xr + (size_t)t * D, dxr_lin, drpre + (size_t)t * D, D, D);

            float dxk_lin[TC_MAX_D_MODEL] = {0}, dxv_lin[TC_MAX_D_MODEL] = {0};
            matvec_backward(lv->tm_Wk, glv ? glv->tm_Wk : NULL, lc->xk + (size_t)t * D, dxk_lin, dk + (size_t)t * D, D, D);
            matvec_backward(lv->tm_Wv, glv ? glv->tm_Wv : NULL, lc->xv + (size_t)t * D, dxv_lin, dv + (size_t)t * D, D, D);

            float cur_k[TC_MAX_D_MODEL], cur_v[TC_MAX_D_MODEL], cur_r[TC_MAX_D_MODEL];
            float prev_k[TC_MAX_D_MODEL] = {0}, prev_v[TC_MAX_D_MODEL] = {0}, prev_r[TC_MAX_D_MODEL] = {0};
            for (int i = 0; i < D; i++) {
                cur_k[i] = lc->ln1_out[(size_t)t * D + i];
                cur_v[i] = cur_k[i]; cur_r[i] = cur_k[i];
                if (t > 0) { prev_k[i] = lc->ln1_out[(size_t)(t - 1) * D + i]; prev_v[i] = prev_k[i]; prev_r[i] = prev_k[i]; }
            }
            for (int i = 0; i < D; i++) {
                float mk = sigmoidf_(lv->tm_mix_k[i]);
                float mv = sigmoidf_(lv->tm_mix_v[i]);
                float mr = sigmoidf_(lv->tm_mix_r[i]);
                dlc->ln1_out[(size_t)t * D + i] += dxk_lin[i] * mk + dxv_lin[i] * mv + dxr_lin[i] * mr;
                if (t > 0) {
                    dlc->ln1_out[(size_t)(t - 1) * D + i] +=
                        dxk_lin[i] * (1 - mk) + dxv_lin[i] * (1 - mv) + dxr_lin[i] * (1 - mr);
                }
                if (glv) {
                    float dmk = dxk_lin[i] * (cur_k[i] - prev_k[i]);
                    float dmv = dxv_lin[i] * (cur_v[i] - prev_v[i]);
                    float dmr = dxr_lin[i] * (cur_r[i] - prev_r[i]);
                    glv->tm_mix_k[i] += dmk * mk * (1 - mk);
                    glv->tm_mix_v[i] += dmv * mv * (1 - mv);
                    glv->tm_mix_r[i] += dmr * mr * (1 - mr);
                }
            }
        }
        tc_rmsnorm_bwd(dlc->ln1_out, lc->ln1_xhat, lv->ln1_gamma, lc->ln1_rms, T, D,
                       glv ? glv->ln1_gamma : NULL, dlc->x0);

        if (l > 0) {
            dx_next = dc->layers[l - 1].resid3; /* alias: previous layer's output grad */
            memcpy(dx_next, dlc->x0, (size_t)T * D * sizeof(float));
        } else {
            if (out_dx0) {
                memcpy(out_dx0, dlc->x0, (size_t)T * D * sizeof(float));
            }
            if (grad) {
                /* d(embed) for the input tokens */
                for (int t = 0; t < T; t++) {
                    float *derow = grad->embed + (size_t)ids[t] * D;
                    for (int i = 0; i < D; i++) derow[i] += dlc->x0[(size_t)t * D + i];
                }
            }
        }
    }

    free(dx_final);
    tc_cache_free(dc);
}

/* Generative head backward: supports masked targets (targets[t] < 0) and optional out_dx0 */
void tc_backward_ex(const TCParamSet *p, TCParamSet *grad, const TCCache *c,
                    const int *ids, int T, const int *targets, float *out_dx0) {
    int D = p->cfg.d_model, V = p->cfg.vocab_size;
    float *dln_f_out = calloc((size_t)T * D, sizeof(float));

    int n_active = 0;
    for (int t = 0; t < T; t++) {
        if (targets[t] >= 0) n_active++;
    }
    float *dlogits = calloc((size_t)T * V, sizeof(float));
    tc_ce_bwd(c->probs, targets, dlogits, n_active, 1.0f, T, V);

    for (int t = 0; t < T; t++) {
        float *dlogit = dlogits + (size_t)t * V;
        float *dlnf = dln_f_out + (size_t)t * D;
        for (int o = 0; o < V; o++) {
            float dlo = dlogit[o];
            const float *erow = p->embed + (size_t)o * D;
            if (grad) {
                float *derow = grad->embed + (size_t)o * D;
                const float *xin = c->ln_f_out + (size_t)t * D;
                for (int i = 0; i < D; i++) {
                    derow[i] += dlo * xin[i];
                }
            }
            for (int i = 0; i < D; i++) {
                dlnf[i] += erow[i] * dlo;
            }
        }
    }
    free(dlogits);
    tc_encode_backward(p, grad, c, ids, T, dln_f_out, out_dx0);
    free(dln_f_out);
}

void tc_backward(const TCParamSet *p, TCParamSet *grad, const TCCache *c,
                 const int *ids, int T, const int *targets) {
    tc_backward_ex(p, grad, c, ids, T, targets, NULL);
}

/* Backward with contrastive (unlikelihood) gradients.
 * For negative targets, the gradient is d/dlogit[-log(1-p_neg)] = p_neg/(1-p_neg) * softmax_jacobian.
 * Simplified: dlogit[neg] += ul_weight * p_neg / (1-p_neg) / n_active,
 *             dlogit[all] -= ul_weight * p_neg^2 / (1-p_neg) / n_active */
void tc_backward_ul(const TCParamSet *p, TCParamSet *grad, const TCCache *c,
                    const int *ids, int T, const int *targets,
                    const int *targets_neg, float ul_weight) {
    int D = p->cfg.d_model, V = p->cfg.vocab_size;
    float *dln_f_out = calloc((size_t)T * D, sizeof(float));

    int n_active = 0;
    for (int t = 0; t < T; t++) {
        if (targets[t] >= 0) n_active++;
    }
    float inv_n = n_active > 0 ? (1.0f / (float)n_active) : 0.0f;

    float *dlogits = calloc((size_t)T * V, sizeof(float));
    tc_ce_bwd(c->probs, targets, dlogits, n_active, 1.0f, T, V);

    for (int t = 0; t < T; t++) {
        float *dlogit = dlogits + (size_t)t * V;
        const float *prob = c->probs + (size_t)t * V;

        /* Unlikelihood repulsive gradient */
        if (targets_neg && targets_neg[t] >= 0 && ul_weight > 0.0f) {
            int neg = targets_neg[t];
            float p_neg = prob[neg];
            if (p_neg > 1.0f - 1e-7f) p_neg = 1.0f - 1e-7f;
            float scale = ul_weight * inv_n * p_neg / (1.0f - p_neg);
            /* d/dlogit_j of -log(1-p_neg) through softmax:
             * = scale * (1[j==neg] - p[j]) */
            for (int i = 0; i < V; i++)
                dlogit[i] += scale * (-prob[i]);
            dlogit[neg] += scale;
        }

        float *dlnf = dln_f_out + (size_t)t * D;
        for (int o = 0; o < V; o++) {
            float dlo = dlogit[o];
            if (fabsf(dlo) < 1e-12f) continue;
            const float *erow = p->embed + (size_t)o * D;
            if (grad) {
                float *derow = grad->embed + (size_t)o * D;
                const float *xin = c->ln_f_out + (size_t)t * D;
                for (int i = 0; i < D; i++)
                    derow[i] += dlo * xin[i];
            }
            for (int i = 0; i < D; i++)
                dlnf[i] += erow[i] * dlo;
        }
    }
    free(dlogits);
    tc_encode_backward(p, grad, c, ids, T, dln_f_out, NULL);
    free(dln_f_out);
}

/* Multi-Token Prediction backward.
 * Backpropagates standard CE and MTP CE losses.
 * dLogits for all three heads are computed. MTP heads backprop through
 * their respective projection matrices, and all gradients sum into dln_f_out
 * before passing through the shared encoder. */
void tc_backward_mtp(const TCParamSet *p, TCParamSet *grad, const TCCache *c,
                     const int *ids, int T, const int *targets,
                     const int *targets_t2, const int *targets_t3,
                     float mtp_weight) {
    int D = p->cfg.d_model, V = p->cfg.vocab_size;
    float *dln_f_out = calloc((size_t)T * D, sizeof(float));

    int n_active = 0;
    for (int t = 0; t < T; t++) if (targets && targets[t] >= 0) n_active++;
    float *dlogits = calloc((size_t)T * V, sizeof(float));
    tc_ce_bwd(c->probs, targets, dlogits, n_active, 1.0f, T, V);

    /* 1. Standard t+1 backward */
    for (int t = 0; t < T; t++) {
        float *dlogit = dlogits + (size_t)t * V;
        float *dlnf = dln_f_out + (size_t)t * D;
        for (int o = 0; o < V; o++) {
            float dlo = dlogit[o];
            if (fabsf(dlo) < 1e-12f) continue;
            const float *erow = p->embed + (size_t)o * D;
            if (grad) {
                float *derow = grad->embed + (size_t)o * D;
                const float *xin = c->ln_f_out + (size_t)t * D;
                for (int i = 0; i < D; i++) derow[i] += dlo * xin[i];
            }
            for (int i = 0; i < D; i++) dlnf[i] += erow[i] * dlo;
        }
    }
    free(dlogits);

    /* 2. MTP backward (t+2, t+3) */
    if (mtp_weight > 0.0f) {
        float *W_grads[2] = { grad ? grad->mtp_head1 : NULL, grad ? grad->mtp_head2 : NULL };
        const float *W_vals[2] = { p->mtp_head1, p->mtp_head2 };
        const float *h_bufs[2] = { c->mtp_h1, c->mtp_h2 };
        const float *prob_bufs[2] = { c->mtp_probs1, c->mtp_probs2 };
        const int *tgt_arrays[2] = { targets_t2, targets_t3 };

        for (int head = 0; head < 2; head++) {
            if (!tgt_arrays[head]) continue;
            int mtp_active = 0;
            for (int t = 0; t < T; t++) if (tgt_arrays[head][t] >= 0) mtp_active++;
            if (mtp_active == 0) continue;

            float *dlogits = calloc((size_t)T * V, sizeof(float));
            tc_ce_bwd(prob_bufs[head], tgt_arrays[head], dlogits, mtp_active, mtp_weight, T, V);

            const float *W = W_vals[head];
            float *dW = W_grads[head];
            const float *h_buf = h_bufs[head];

            for (int t = 0; t < T; t++) {
                float *dlogit = dlogits + (size_t)t * V;
                float dh[1024]; /* D_MAX */
                memset(dh, 0, D * sizeof(float));
                for (int o = 0; o < V; o++) {
                    float dlo = dlogit[o];
                    if (fabsf(dlo) < 1e-12f) continue;
                    const float *erow = p->embed + (size_t)o * D;
                    if (grad) {
                        float *derow = grad->embed + (size_t)o * D;
                        const float *hin = h_buf + (size_t)t * D;
                        for (int i = 0; i < D; i++) derow[i] += dlo * hin[i];
                    }
                    for (int i = 0; i < D; i++) dh[i] += erow[i] * dlo;
                }

                /* c) Backprop through MTP projection: h = ln_f_out @ W^T */
                float *dlnf = dln_f_out + (size_t)t * D;
                const float *xin = c->ln_f_out + (size_t)t * D;
                for (int out_idx = 0; out_idx < D; out_idx++) {
                    float dho = dh[out_idx];
                    if (fabsf(dho) < 1e-12f) continue;
                    const float *wrow = W + (size_t)out_idx * D;
                    if (dW) {
                        float *dwrow = dW + (size_t)out_idx * D;
                        for (int in_idx = 0; in_idx < D; in_idx++)
                            dwrow[in_idx] += dho * xin[in_idx];
                    }
                    for (int in_idx = 0; in_idx < D; in_idx++)
                        dlnf[in_idx] += wrow[in_idx] * dho;
                }
            }
        }
    }

    tc_encode_backward(p, grad, c, ids, T, dln_f_out, NULL);
    free(dln_f_out);
}

void tc_input_grad(const TCParamSet *p, const TCCache *c,
                   const int *ids, int T, const int *targets, float *out_dx0) {
    tc_backward_ex(p, NULL, c, ids, T, targets, out_dx0);
}

static unsigned int xrand(unsigned int *s) {
    unsigned int x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

int tc_generate(const TCParamSet *p, TCCache *c, int *ids, int prompt_len,
                 int max_new_tokens, float temperature, unsigned int *rng_state) {
    int T = prompt_len;
    int V = p->cfg.vocab_size;
    unsigned int local_rng = rng_state ? *rng_state : 1;

    for (int step = 0; step < max_new_tokens && T < p->cfg.max_seq_len; step++) {
        tc_forward(p, c, ids, T, NULL, NULL);
        const float *logit = c->logits + (size_t)(T - 1) * V;

        int next;
        if (temperature <= 1e-6f) {
            next = 0;
            float best = logit[0];
            for (int i = 1; i < V; i++) if (logit[i] > best) { best = logit[i]; next = i; }
        } else {
            float scaled[4096], maxv = -1e30f;
            for (int i = 0; i < V; i++) { scaled[i] = logit[i] / temperature; if (scaled[i] > maxv) maxv = scaled[i]; }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) { scaled[i] = expf(scaled[i] - maxv); sum += scaled[i]; }
            float r = (float)(xrand(&local_rng) % 1000000) / 1000000.0f * sum;
            float acc = 0.0f;
            next = V - 1;
            for (int i = 0; i < V; i++) { acc += scaled[i]; if (r <= acc) { next = i; break; } }
        }
        ids[T] = next;
        T++;
    }
    if (rng_state) *rng_state = local_rng;
    return T;
}
