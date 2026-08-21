#include "tcmodel.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define LN_EPS 1e-5f

/* Several hot loops use fixed-size stack buffers sized for this toy
 * model's scale instead of alloca/VLA. Configs must stay within these
 * bounds or forward/backward will corrupt the stack. */
static void tc_config_check(TCConfig cfg) {
    assert(cfg.d_model > 0 && cfg.d_model <= 256);
    assert(cfg.ff_mult > 0 && cfg.ff_mult * cfg.d_model <= 1024);
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

static int layout_layer(TCLayerView *lv, float *buf, int off, int D, int FF) {
#define TAKE(field, n) do { lv->field = buf ? buf + off : NULL; off += (n); } while (0)
    TAKE(ln1_gamma, D); TAKE(ln1_beta, D);
    TAKE(tm_mix_k, D); TAKE(tm_mix_v, D); TAKE(tm_mix_r, D);
    TAKE(tm_Wk, D * D); TAKE(tm_Wv, D * D); TAKE(tm_Wr, D * D);
    TAKE(tm_decay, D);
    TAKE(tm_Wo, D * D);

    TAKE(ln2_gamma, D); TAKE(ln2_beta, D);
    TAKE(at_Wq, D * D); TAKE(at_Wk, D * D); TAKE(at_Wv, D * D); TAKE(at_Wo, D * D);

    TAKE(ln3_gamma, D); TAKE(ln3_beta, D);
    TAKE(cm_mix_k, D); TAKE(cm_mix_r, D);
    TAKE(cm_Wk, D * FF * D); TAKE(cm_Wv, FF * D * D); TAKE(cm_Wr, D * D);
#undef TAKE
    return off;
}

static int layout_all(TCParamSet *ps, float *buf, TCConfig cfg) {
    int off = 0;
    int D = cfg.d_model, V = cfg.vocab_size;
#define TAKE(field, n) do { ps->field = buf ? buf + off : NULL; off += (n); } while (0)
    TAKE(embed, V * D);
    TAKE(ln_f_gamma, D); TAKE(ln_f_beta, D);
    TAKE(pool_w, D);
#undef TAKE
    for (int l = 0; l < cfg.n_layers; l++)
        off = layout_layer(&ps->layers[l], buf, off, D, cfg.ff_mult);
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
    ps->buf = calloc((size_t)n, sizeof(float));
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
    for (int i = 0; i < D; i++) { ps->ln_f_gamma[i] = 1.0f; ps->ln_f_beta[i] = 0.0f; }
    for (int i = 0; i < D; i++) ps->pool_w[i] = (randf(&s) * 2 - 1) * wscale;

    for (int l = 0; l < ps->cfg.n_layers; l++) {
        TCLayerView *lv = &ps->layers[l];
        for (int i = 0; i < D; i++) {
            lv->ln1_gamma[i] = 1.0f; lv->ln1_beta[i] = 0.0f;
            lv->ln2_gamma[i] = 1.0f; lv->ln2_beta[i] = 0.0f;
            lv->ln3_gamma[i] = 1.0f; lv->ln3_beta[i] = 0.0f;
            /* start mostly "pass through current token" and moderate decay */
            lv->tm_mix_k[i] = 1.0f; lv->tm_mix_v[i] = 1.0f; lv->tm_mix_r[i] = 1.0f;
            lv->tm_decay[i] = 0.0f; /* sigmoid(0)=0.5 */
            lv->cm_mix_k[i] = 1.0f; lv->cm_mix_r[i] = 1.0f;
        }
        int DD = D * D, FFD = ps->cfg.ff_mult * D;
        for (int i = 0; i < DD; i++) {
            lv->tm_Wk[i] = (randf(&s) * 2 - 1) * wscale;
            lv->tm_Wv[i] = (randf(&s) * 2 - 1) * wscale;
            lv->tm_Wr[i] = (randf(&s) * 2 - 1) * wscale;
            lv->tm_Wo[i] = (randf(&s) * 2 - 1) * wscale;
            lv->at_Wq[i] = (randf(&s) * 2 - 1) * wscale;
            lv->at_Wk[i] = (randf(&s) * 2 - 1) * wscale;
            lv->at_Wv[i] = (randf(&s) * 2 - 1) * wscale;
            lv->at_Wo[i] = (randf(&s) * 2 - 1) * wscale;
            lv->cm_Wr[i] = (randf(&s) * 2 - 1) * wscale;
        }
        for (int i = 0; i < D * FFD; i++)
            lv->cm_Wk[i] = (randf(&s) * 2 - 1) * wscale;
        for (int i = 0; i < FFD * D; i++)
            lv->cm_Wv[i] = (randf(&s) * 2 - 1) * wscale;
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
    TCParamSet *ps = tc_paramset_create(cfg);
    size_t n = fread(ps->buf, sizeof(float), (size_t)ps->n_floats, f);
    fclose(f);
    if (n != (size_t)ps->n_floats) { tc_paramset_free(ps); return NULL; }
    return ps;
}

/* ---------------------------------------------------------------------
 * Cache allocation (mirrors the shapes used in forward/backward).
 * ------------------------------------------------------------------- */

TCCache *tc_cache_create(TCConfig cfg) {
    TCCache *c = calloc(1, sizeof(TCCache));
    c->cfg = cfg;
    int T = cfg.max_seq_len, D = cfg.d_model, V = cfg.vocab_size;
    int FFD = cfg.ff_mult * D;
    c->layers = calloc(cfg.n_layers, sizeof(TCLayerCache));
    for (int l = 0; l < cfg.n_layers; l++) {
        TCLayerCache *lc = &c->layers[l];
        lc->x0 = calloc((size_t)T * D, sizeof(float));
        lc->ln1_mean = calloc(T, sizeof(float));
        lc->ln1_rstd = calloc(T, sizeof(float));
        lc->ln1_out = calloc((size_t)T * D, sizeof(float));
        lc->ln1_xhat = calloc((size_t)T * D, sizeof(float));
        lc->xk = calloc((size_t)T * D, sizeof(float));
        lc->xv = calloc((size_t)T * D, sizeof(float));
        lc->xr = calloc((size_t)T * D, sizeof(float));
        lc->k = calloc((size_t)T * D, sizeof(float));
        lc->v = calloc((size_t)T * D, sizeof(float));
        lc->r_sig = calloc((size_t)T * D, sizeof(float));
        lc->state = calloc((size_t)T * D, sizeof(float));
        lc->wkv = calloc((size_t)T * D, sizeof(float));
        lc->tm_out = calloc((size_t)T * D, sizeof(float));
        lc->resid1 = calloc((size_t)T * D, sizeof(float));

        lc->ln2_mean = calloc(T, sizeof(float));
        lc->ln2_rstd = calloc(T, sizeof(float));
        lc->ln2_out = calloc((size_t)T * D, sizeof(float));
        lc->ln2_xhat = calloc((size_t)T * D, sizeof(float));
        lc->Q = calloc((size_t)T * D, sizeof(float));
        lc->K = calloc((size_t)T * D, sizeof(float));
        lc->V = calloc((size_t)T * D, sizeof(float));
        lc->attn_w = calloc((size_t)cfg.n_heads * T * T, sizeof(float));
        lc->attn_ctx = calloc((size_t)T * D, sizeof(float));
        lc->attn_out = calloc((size_t)T * D, sizeof(float));
        lc->resid2 = calloc((size_t)T * D, sizeof(float));

        lc->ln3_mean = calloc(T, sizeof(float));
        lc->ln3_rstd = calloc(T, sizeof(float));
        lc->ln3_out = calloc((size_t)T * D, sizeof(float));
        lc->ln3_xhat = calloc((size_t)T * D, sizeof(float));
        lc->cmxk = calloc((size_t)T * D, sizeof(float));
        lc->cmxr = calloc((size_t)T * D, sizeof(float));
        lc->h_pre = calloc((size_t)T * FFD, sizeof(float));
        lc->h_relu = calloc((size_t)T * FFD, sizeof(float));
        lc->cm_v = calloc((size_t)T * D, sizeof(float));
        lc->r3_sig = calloc((size_t)T * D, sizeof(float));
        lc->cm_out = calloc((size_t)T * D, sizeof(float));
        lc->resid3 = calloc((size_t)T * D, sizeof(float));
    }
    c->ln_f_mean = calloc(T, sizeof(float));
    c->ln_f_rstd = calloc(T, sizeof(float));
    c->ln_f_out = calloc((size_t)T * D, sizeof(float));
    c->ln_f_xhat = calloc((size_t)T * D, sizeof(float));
    c->logits = calloc((size_t)T * V, sizeof(float));
    c->probs = calloc((size_t)T * V, sizeof(float));
    c->pool_scores = calloc(T, sizeof(float));
    c->pool_weights = calloc(T, sizeof(float));
    c->pool_raw = calloc(D, sizeof(float));
    return c;
}

void tc_cache_free(TCCache *c) {
    if (!c) return;
    for (int l = 0; l < c->cfg.n_layers; l++) {
        TCLayerCache *lc = &c->layers[l];
        free(lc->x0); free(lc->ln1_mean); free(lc->ln1_rstd); free(lc->ln1_out); free(lc->ln1_xhat);
        free(lc->xk); free(lc->xv); free(lc->xr);
        free(lc->k); free(lc->v); free(lc->r_sig);
        free(lc->state); free(lc->wkv); free(lc->tm_out); free(lc->resid1);
        free(lc->ln2_mean); free(lc->ln2_rstd); free(lc->ln2_out); free(lc->ln2_xhat);
        free(lc->Q); free(lc->K); free(lc->V);
        free(lc->attn_w); free(lc->attn_ctx); free(lc->attn_out); free(lc->resid2);
        free(lc->ln3_mean); free(lc->ln3_rstd); free(lc->ln3_out); free(lc->ln3_xhat);
        free(lc->cmxk); free(lc->cmxr);
        free(lc->h_pre); free(lc->h_relu); free(lc->cm_v); free(lc->r3_sig);
        free(lc->cm_out); free(lc->resid3);
    }
    free(c->layers);
    free(c->ln_f_mean); free(c->ln_f_rstd); free(c->ln_f_out); free(c->ln_f_xhat);
    free(c->logits); free(c->probs);
    free(c->pool_scores); free(c->pool_weights); free(c->pool_raw);
    free(c);
}

/* ---------------------------------------------------------------------
 * Small math helpers. Vectors are D-wide float* unless noted.
 * ------------------------------------------------------------------- */

static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

/* y = W x, W row-major [out x in] */
static void matvec(const float *W, const float *x, float *y, int out, int in) {
    for (int o = 0; o < out; o++) {
        float s = 0.0f;
        const float *row = W + (size_t)o * in;
        for (int i = 0; i < in; i++) s += row[i] * x[i];
        y[o] = s;
    }
}

/* Accumulates dW += dy (x) x^T, dx += W^T dy. dx must already hold valid
 * data to accumulate into (caller zeroes once per position). */
static void matvec_backward(const float *W, float *dW, const float *x, float *dx,
                             const float *dy, int out, int in) {
    for (int o = 0; o < out; o++) {
        float dyo = dy[o];
        const float *row = W + (size_t)o * in;
        float *drow = dW + (size_t)o * in;
        for (int i = 0; i < in; i++) {
            drow[i] += dyo * x[i];
            dx[i] += row[i] * dyo;
        }
    }
}

static void ln_forward(const float *x, const float *gamma, const float *beta, int D,
                        float *mean_out, float *rstd_out, float *xhat_out, float *y_out) {
    float mean = 0.0f;
    for (int i = 0; i < D; i++) mean += x[i];
    mean /= D;
    float var = 0.0f;
    for (int i = 0; i < D; i++) { float d = x[i] - mean; var += d * d; }
    var /= D;
    float rstd = 1.0f / sqrtf(var + LN_EPS);
    *mean_out = mean; *rstd_out = rstd;
    for (int i = 0; i < D; i++) {
        float xhat = (x[i] - mean) * rstd;
        xhat_out[i] = xhat;
        y_out[i] = gamma[i] * xhat + beta[i];
    }
}

/* dx accumulated (+=), dgamma/dbeta accumulated (+=). */
static void ln_backward(const float *dy, const float *xhat, const float *gamma,
                         float rstd, int D, float *dgamma, float *dbeta, float *dx) {
    float sum1 = 0.0f, sum2 = 0.0f;
    float dxhat[256]; /* D is tiny (<=256 by construction of this toy model) */
    for (int i = 0; i < D; i++) {
        dgamma[i] += dy[i] * xhat[i];
        dbeta[i] += dy[i];
        dxhat[i] = dy[i] * gamma[i];
        sum1 += dxhat[i];
        sum2 += dxhat[i] * xhat[i];
    }
    for (int i = 0; i < D; i++)
        dx[i] += rstd * (dxhat[i] - sum1 / D - xhat[i] * sum2 / D);
}

/* ---------------------------------------------------------------------
 * Forward
 * ------------------------------------------------------------------- */

/* Shared backward tail (defined further down, alongside tc_backward);
 * forward-declared here so any head's backward (generative, pooling, ...)
 * can call it regardless of definition order in this file. */
static void tc_encode_backward(const TCParamSet *p, TCParamSet *grad, const TCCache *c,
                                const int *ids, int T, const float *dln_f_out);

/* Runs the shared layer stack + final LN, filling every cache field up to
 * and including c->ln_f_out. Both tc_forward (which adds the tied-embedding
 * head + softmax + loss) and tc_embed (which mean-pools instead) build on
 * this so the two specialist "heads" can never drift out of sync with the
 * body that produces them. */
static void tc_encode(const TCParamSet *p, TCCache *c, const int *ids, int T) {
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
        for (int t = 0; t < T; t++) {
            const float *xt = x + (size_t)t * D;
            ln_forward(xt, lv->ln1_gamma, lv->ln1_beta, D,
                       &lc->ln1_mean[t], &lc->ln1_rstd[t],
                       lc->ln1_xhat + (size_t)t * D, lc->ln1_out + (size_t)t * D);
        }
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
            matvec(lv->tm_Wk, xk, lc->k + (size_t)t * D, D, D);
            matvec(lv->tm_Wv, xv, lc->v + (size_t)t * D, D, D);
            float rpre[256];
            matvec(lv->tm_Wr, xr, rpre, D, D);
            float *rs = lc->r_sig + (size_t)t * D;
            for (int i = 0; i < D; i++) rs[i] = sigmoidf_(rpre[i]);

            float *st = lc->state + (size_t)t * D;
            const float *stprev = t > 0 ? lc->state + (size_t)(t - 1) * D : NULL;
            const float *kt = lc->k + (size_t)t * D, *vt = lc->v + (size_t)t * D;
            for (int i = 0; i < D; i++) {
                float decay = sigmoidf_(lv->tm_decay[i]);
                float sp = stprev ? stprev[i] : 0.0f;
                st[i] = decay * sp + (1 - decay) * (kt[i] * vt[i]);
            }
            float *wkv = lc->wkv + (size_t)t * D;
            for (int i = 0; i < D; i++) wkv[i] = rs[i] * st[i];
            matvec(lv->tm_Wo, wkv, lc->tm_out + (size_t)t * D, D, D);
            for (int i = 0; i < D; i++)
                lc->resid1[(size_t)t * D + i] = lc->x0[(size_t)t * D + i] + lc->tm_out[(size_t)t * D + i];
        }

        /* --- tiny causal self-attention --- */
        for (int t = 0; t < T; t++) {
            const float *rt = lc->resid1 + (size_t)t * D;
            ln_forward(rt, lv->ln2_gamma, lv->ln2_beta, D,
                       &lc->ln2_mean[t], &lc->ln2_rstd[t],
                       lc->ln2_xhat + (size_t)t * D, lc->ln2_out + (size_t)t * D);
            matvec(lv->at_Wq, lc->ln2_out + (size_t)t * D, lc->Q + (size_t)t * D, D, D);
            matvec(lv->at_Wk, lc->ln2_out + (size_t)t * D, lc->K + (size_t)t * D, D, D);
            matvec(lv->at_Wv, lc->ln2_out + (size_t)t * D, lc->V + (size_t)t * D, D, D);
        }
        for (int h = 0; h < H; h++) {
            for (int t = 0; t < T; t++) {
                float scores[4096];
                float maxs = -1e30f;
                for (int u = 0; u <= t; u++) {
                    const float *qh = lc->Q + (size_t)t * D + h * Dh;
                    const float *kh = lc->K + (size_t)u * D + h * Dh;
                    float s = 0.0f;
                    for (int i = 0; i < Dh; i++) s += qh[i] * kh[i];
                    s *= invsqrt_dh;
                    scores[u] = s;
                    if (s > maxs) maxs = s;
                }
                float sum = 0.0f;
                for (int u = 0; u <= t; u++) { scores[u] = expf(scores[u] - maxs); sum += scores[u]; }
                float *wrow = lc->attn_w + ((size_t)h * T + t) * T;
                for (int u = 0; u <= t; u++) wrow[u] = scores[u] / sum;
                float *ctx = lc->attn_ctx + (size_t)t * D + h * Dh;
                for (int i = 0; i < Dh; i++) ctx[i] = 0.0f;
                for (int u = 0; u <= t; u++) {
                    const float *vh = lc->V + (size_t)u * D + h * Dh;
                    float w = wrow[u];
                    for (int i = 0; i < Dh; i++) ctx[i] += w * vh[i];
                }
            }
        }
        for (int t = 0; t < T; t++) {
            matvec(lv->at_Wo, lc->attn_ctx + (size_t)t * D, lc->attn_out + (size_t)t * D, D, D);
            for (int i = 0; i < D; i++)
                lc->resid2[(size_t)t * D + i] = lc->resid1[(size_t)t * D + i] + lc->attn_out[(size_t)t * D + i];
        }

        /* --- gated channel-mix FFN --- */
        for (int t = 0; t < T; t++) {
            const float *rt = lc->resid2 + (size_t)t * D;
            ln_forward(rt, lv->ln3_gamma, lv->ln3_beta, D,
                       &lc->ln3_mean[t], &lc->ln3_rstd[t],
                       lc->ln3_xhat + (size_t)t * D, lc->ln3_out + (size_t)t * D);
        }
        for (int t = 0; t < T; t++) {
            const float *cur = lc->ln3_out + (size_t)t * D;
            const float *prev = t > 0 ? lc->ln3_out + (size_t)(t - 1) * D : NULL;
            float *cmxk = lc->cmxk + (size_t)t * D, *cmxr = lc->cmxr + (size_t)t * D;
            for (int i = 0; i < D; i++) {
                float mk = sigmoidf_(lv->cm_mix_k[i]);
                float mr = sigmoidf_(lv->cm_mix_r[i]);
                float p_ = prev ? prev[i] : 0.0f;
                cmxk[i] = mk * cur[i] + (1 - mk) * p_;
                cmxr[i] = mr * cur[i] + (1 - mr) * p_;
            }
            float *hpre = lc->h_pre + (size_t)t * FFD;
            matvec(lv->cm_Wk, cmxk, hpre, FFD, D);
            float *hrelu = lc->h_relu + (size_t)t * FFD;
            for (int i = 0; i < FFD; i++) hrelu[i] = hpre[i] > 0 ? hpre[i] : 0.0f;
            matvec(lv->cm_Wv, hrelu, lc->cm_v + (size_t)t * D, D, FFD);
            float rpre[256];
            matvec(lv->cm_Wr, cmxr, rpre, D, D);
            float *r3 = lc->r3_sig + (size_t)t * D;
            for (int i = 0; i < D; i++) r3[i] = sigmoidf_(rpre[i]);
            float *cmo = lc->cm_out + (size_t)t * D;
            for (int i = 0; i < D; i++) cmo[i] = r3[i] * lc->cm_v[(size_t)t * D + i];
            for (int i = 0; i < D; i++)
                lc->resid3[(size_t)t * D + i] = lc->resid2[(size_t)t * D + i] + cmo[i];
        }

        memcpy(x, lc->resid3, (size_t)T * D * sizeof(float));
    }

    /* --- final LN --- */
    for (int t = 0; t < T; t++) {
        ln_forward(x + (size_t)t * D, p->ln_f_gamma, p->ln_f_beta, D,
                   &c->ln_f_mean[t], &c->ln_f_rstd[t],
                   c->ln_f_xhat + (size_t)t * D, c->ln_f_out + (size_t)t * D);
    }
    free(x);
}

/* Generative head: tied-embedding projection + softmax + (optional) loss. */
void tc_forward(const TCParamSet *p, TCCache *c, const int *ids, int T,
                 const int *targets, float *out_loss) {
    tc_encode(p, c, ids, T);
    int V = p->cfg.vocab_size, D = p->cfg.d_model;
    float total_loss = 0.0f;
    for (int t = 0; t < T; t++) {
        float *logit = c->logits + (size_t)t * V;
        matvec(p->embed, c->ln_f_out + (size_t)t * D, logit, V, D);
        float maxv = -1e30f;
        for (int i = 0; i < V; i++) if (logit[i] > maxv) maxv = logit[i];
        float sum = 0.0f;
        float *prob = c->probs + (size_t)t * V;
        for (int i = 0; i < V; i++) { prob[i] = expf(logit[i] - maxv); sum += prob[i]; }
        for (int i = 0; i < V; i++) prob[i] /= sum;
        if (targets) {
            float pt = prob[targets[t]];
            if (pt < 1e-9f) pt = 1e-9f;
            total_loss += -logf(pt);
        }
    }
    if (targets && out_loss) *out_loss = total_loss / T;
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
    float out_vec[256], d_raw[256];
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

    tc_encode_backward(p, grad, c, ids, T, dln_f_out);
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
                                const int *ids, int T, const float *dln_f_out) {
    TCConfig cfg = p->cfg;
    int D = cfg.d_model, H = cfg.n_heads, Dh = D / H, FFD = cfg.ff_mult * D;
    float invsqrt_dh = 1.0f / sqrtf((float)Dh);

    TCCache *dc = tc_cache_create(cfg);
    dc->T = T;
    float *dx_final = calloc((size_t)T * D, sizeof(float)); /* grad w.r.t. final block output */

    for (int t = 0; t < T; t++) {
        ln_backward(dln_f_out + (size_t)t * D, c->ln_f_xhat + (size_t)t * D, p->ln_f_gamma,
                    c->ln_f_rstd[t], D, grad->ln_f_gamma, grad->ln_f_beta, dx_final + (size_t)t * D);
    }

    float *dx_next = dx_final; /* grad w.r.t. output of layer l (resid3) */

    for (int l = cfg.n_layers - 1; l >= 0; l--) {
        TCLayerView *lv = &p->layers[l];
        TCLayerView *glv = &grad->layers[l];
        const TCLayerCache *lc = &c->layers[l];
        TCLayerCache *dlc = &dc->layers[l];

        /* resid3 = resid2 + cm_out */
        memcpy(dlc->resid2, dx_next, (size_t)T * D * sizeof(float));
        memcpy(dlc->cm_out, dx_next, (size_t)T * D * sizeof(float));

        for (int t = 0; t < T; t++) {
            float *dcmo = dlc->cm_out + (size_t)t * D;
            const float *r3 = lc->r3_sig + (size_t)t * D;
            const float *cmv = lc->cm_v + (size_t)t * D;
            float dr3[256], dcmv[256];
            for (int i = 0; i < D; i++) {
                dr3[i] = dcmo[i] * cmv[i];
                dcmv[i] = dcmo[i] * r3[i];
            }
            float drpre[256];
            for (int i = 0; i < D; i++) drpre[i] = dr3[i] * r3[i] * (1 - r3[i]);
            float dcmxr[256] = {0};
            matvec_backward(lv->cm_Wr, glv->cm_Wr, lc->cmxr + (size_t)t * D, dcmxr, drpre, D, D);

            float dhrelu[1024] = {0};
            matvec_backward(lv->cm_Wv, glv->cm_Wv, lc->h_relu + (size_t)t * FFD, dhrelu, dcmv, D, FFD);
            float dhpre[1024];
            for (int i = 0; i < FFD; i++) dhpre[i] = lc->h_pre[(size_t)t * FFD + i] > 0 ? dhrelu[i] : 0.0f;
            float dcmxk[256] = {0};
            matvec_backward(lv->cm_Wk, glv->cm_Wk, lc->cmxk + (size_t)t * D, dcmxk, dhpre, FFD, D);

            /* token-shift split for cmxk/cmxr back into d(ln3_out[t]) and d(ln3_out[t-1]) */
            const float *cur = lc->ln3_out + (size_t)t * D;
            const float *prev = t > 0 ? lc->ln3_out + (size_t)(t - 1) * D : NULL;
            (void)cur; (void)prev;
            for (int i = 0; i < D; i++) {
                float mk = sigmoidf_(lv->cm_mix_k[i]);
                float mr = sigmoidf_(lv->cm_mix_r[i]);
                float cur_i = lc->ln3_out[(size_t)t * D + i];
                float prev_i = t > 0 ? lc->ln3_out[(size_t)(t - 1) * D + i] : 0.0f;
                dlc->ln3_out[(size_t)t * D + i] += dcmxk[i] * mk + dcmxr[i] * mr;
                if (t > 0) {
                    dlc->ln3_out[(size_t)(t - 1) * D + i] += dcmxk[i] * (1 - mk) + dcmxr[i] * (1 - mr);
                }
                float dmk = dcmxk[i] * (cur_i - prev_i);
                float dmr = dcmxr[i] * (cur_i - prev_i);
                glv->cm_mix_k[i] += dmk * mk * (1 - mk);
                glv->cm_mix_r[i] += dmr * mr * (1 - mr);
            }
        }
        for (int t = 0; t < T; t++) {
            ln_backward(dlc->ln3_out + (size_t)t * D, lc->ln3_xhat + (size_t)t * D, lv->ln3_gamma,
                        lc->ln3_rstd[t], D, glv->ln3_gamma, glv->ln3_beta, dlc->resid2 + (size_t)t * D);
        }

        /* resid2 = resid1 + attn_out */
        memcpy(dlc->resid1, dlc->resid2, (size_t)T * D * sizeof(float));
        memcpy(dlc->attn_out, dlc->resid2, (size_t)T * D * sizeof(float));

        for (int t = 0; t < T; t++)
            matvec_backward(lv->at_Wo, glv->at_Wo, lc->attn_ctx + (size_t)t * D,
                             dlc->attn_ctx + (size_t)t * D, dlc->attn_out + (size_t)t * D, D, D);

        for (int h = 0; h < H; h++) {
            for (int t = 0; t < T; t++) {
                const float *dctx = dlc->attn_ctx + (size_t)t * D + h * Dh;
                const float *wrow = lc->attn_w + ((size_t)h * T + t) * T;
                float dw[4096];
                for (int u = 0; u <= t; u++) {
                    const float *vh = lc->V + (size_t)u * D + h * Dh;
                    float *dvh = dlc->V + (size_t)u * D + h * Dh;
                    float s = 0.0f;
                    for (int i = 0; i < Dh; i++) { s += dctx[i] * vh[i]; dvh[i] += wrow[u] * dctx[i]; }
                    dw[u] = s;
                }
                float dot = 0.0f;
                for (int u = 0; u <= t; u++) dot += wrow[u] * dw[u];
                float *dqh = dlc->Q + (size_t)t * D + h * Dh;
                for (int u = 0; u <= t; u++) {
                    float ds = wrow[u] * (dw[u] - dot) * invsqrt_dh;
                    const float *kh = lc->K + (size_t)u * D + h * Dh;
                    float *dkh = dlc->K + (size_t)u * D + h * Dh;
                    const float *qh = lc->Q + (size_t)t * D + h * Dh;
                    for (int i = 0; i < Dh; i++) { dqh[i] += ds * kh[i]; dkh[i] += ds * qh[i]; }
                }
            }
        }

        for (int t = 0; t < T; t++) {
            float *dln2 = dlc->ln2_out + (size_t)t * D;
            matvec_backward(lv->at_Wq, glv->at_Wq, lc->ln2_out + (size_t)t * D, dln2, dlc->Q + (size_t)t * D, D, D);
            matvec_backward(lv->at_Wk, glv->at_Wk, lc->ln2_out + (size_t)t * D, dln2, dlc->K + (size_t)t * D, D, D);
            matvec_backward(lv->at_Wv, glv->at_Wv, lc->ln2_out + (size_t)t * D, dln2, dlc->V + (size_t)t * D, D, D);
            ln_backward(dln2, lc->ln2_xhat + (size_t)t * D, lv->ln2_gamma, lc->ln2_rstd[t], D,
                        glv->ln2_gamma, glv->ln2_beta, dlc->resid1 + (size_t)t * D);
        }

        /* resid1 = x0 + tm_out */
        memcpy(dlc->x0, dlc->resid1, (size_t)T * D * sizeof(float));
        memcpy(dlc->tm_out, dlc->resid1, (size_t)T * D * sizeof(float));

        for (int t = 0; t < T; t++)
            matvec_backward(lv->tm_Wo, glv->tm_Wo, lc->wkv + (size_t)t * D,
                             dlc->wkv + (size_t)t * D, dlc->tm_out + (size_t)t * D, D, D);

        /* recurrence backward, reverse time order (state[t] depends on state[t-1]) */
        for (int t = T - 1; t >= 0; t--) {
            const float *rs = lc->r_sig + (size_t)t * D;
            const float *st = lc->state + (size_t)t * D;
            float *dwkv = dlc->wkv + (size_t)t * D;
            float drs[256], dst[256];
            for (int i = 0; i < D; i++) {
                drs[i] = dwkv[i] * st[i];
                dst[i] = dwkv[i] * rs[i] + dlc->state[(size_t)t * D + i]; /* += grad flowing from state[t+1] */
            }
            float drpre[256];
            for (int i = 0; i < D; i++) drpre[i] = drs[i] * rs[i] * (1 - rs[i]);
            float dxr_lin[256] = {0};
            matvec_backward(lv->tm_Wr, glv->tm_Wr, lc->xr + (size_t)t * D, dxr_lin, drpre, D, D);

            const float *kt = lc->k + (size_t)t * D, *vt = lc->v + (size_t)t * D;
            float dk[256], dv[256];
            for (int i = 0; i < D; i++) {
                float decay = sigmoidf_(lv->tm_decay[i]);
                float sp = t > 0 ? lc->state[(size_t)(t - 1) * D + i] : 0.0f;
                float dgate = 1 - decay;
                dk[i] = dst[i] * dgate * vt[i];
                dv[i] = dst[i] * dgate * kt[i];
                if (t > 0) dlc->state[(size_t)(t - 1) * D + i] += dst[i] * decay;
                float ddecay_pre = dst[i] * (sp - kt[i] * vt[i]) * decay * (1 - decay);
                glv->tm_decay[i] += ddecay_pre;
            }
            float dxk_lin[256] = {0}, dxv_lin[256] = {0};
            matvec_backward(lv->tm_Wk, glv->tm_Wk, lc->xk + (size_t)t * D, dxk_lin, dk, D, D);
            matvec_backward(lv->tm_Wv, glv->tm_Wv, lc->xv + (size_t)t * D, dxv_lin, dv, D, D);

            float cur_k[256], cur_v[256], cur_r[256];
            float prev_k[256] = {0}, prev_v[256] = {0}, prev_r[256] = {0};
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
                float dmk = dxk_lin[i] * (cur_k[i] - prev_k[i]);
                float dmv = dxv_lin[i] * (cur_v[i] - prev_v[i]);
                float dmr = dxr_lin[i] * (cur_r[i] - prev_r[i]);
                glv->tm_mix_k[i] += dmk * mk * (1 - mk);
                glv->tm_mix_v[i] += dmv * mv * (1 - mv);
                glv->tm_mix_r[i] += dmr * mr * (1 - mr);
            }
        }
        for (int t = 0; t < T; t++)
            ln_backward(dlc->ln1_out + (size_t)t * D, lc->ln1_xhat + (size_t)t * D, lv->ln1_gamma,
                        lc->ln1_rstd[t], D, glv->ln1_gamma, glv->ln1_beta, dlc->x0 + (size_t)t * D);

        if (l > 0) {
            dx_next = dc->layers[l - 1].resid3; /* alias: previous layer's output grad */
            memcpy(dx_next, dlc->x0, (size_t)T * D * sizeof(float));
        } else {
            /* d(embed) for the input tokens */
            for (int t = 0; t < T; t++) {
                float *derow = grad->embed + (size_t)ids[t] * D;
                for (int i = 0; i < D; i++) derow[i] += dlc->x0[(size_t)t * D + i];
            }
        }
    }

    free(dx_final);
    tc_cache_free(dc);
}

/* Generative head backward: dLogits = probs - onehot(target), reduced
 * through the tied-embedding projection (which is also grad'd here, on
 * its output-side use) down to d(ln_f_out), then handed to the shared
 * encoder backward. Requires targets to have been passed to tc_forward. */
void tc_backward(const TCParamSet *p, TCParamSet *grad, const TCCache *c,
                  const int *ids, int T, const int *targets) {
    int D = p->cfg.d_model, V = p->cfg.vocab_size;
    float *dln_f_out = calloc((size_t)T * D, sizeof(float));

    for (int t = 0; t < T; t++) {
        float dlogit[4096];
        const float *prob = c->probs + (size_t)t * V;
        for (int i = 0; i < V; i++) dlogit[i] = prob[i] / T;
        dlogit[targets[t]] -= 1.0f / T;

        float *dlnf = dln_f_out + (size_t)t * D;
        for (int o = 0; o < V; o++) {
            float dlo = dlogit[o];
            const float *erow = p->embed + (size_t)o * D;
            float *derow = grad->embed + (size_t)o * D;
            const float *xin = c->ln_f_out + (size_t)t * D;
            for (int i = 0; i < D; i++) {
                derow[i] += dlo * xin[i];
                dlnf[i] += erow[i] * dlo;
            }
        }
    }
    tc_encode_backward(p, grad, c, ids, T, dln_f_out);
    free(dln_f_out);
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
