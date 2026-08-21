#ifndef TC_MODEL_H
#define TC_MODEL_H

/* TinyCoherent core specialist: one ~8k-param hybrid block in pure C.
 *
 * Block per layer: LN -> gated linear recurrence (RWKV-lineage token
 * mixer, NOT the literal RWKV-v6 WKV kernel -- see README note) -> resid
 *               -> LN -> tiny causal self-attention               -> resid
 *               -> LN -> gated channel-mix FFN                    -> resid
 *
 * All parameters for a model live in one contiguous float buffer
 * (TCParamSet.buf); everything else (TCLayerView, embed, final LN) is a
 * named pointer *into* that buffer. Gradients, and Adam's m/v moments,
 * are each a separate TCParamSet built with the identical layout, so a
 * flat loop over buf[0..n_floats) works for zeroing, Adam, and save/load.
 */

#include <stddef.h>

typedef struct {
    int vocab_size;   /* number of distinct token ids */
    int d_model;      /* hidden width D */
    int n_layers;     /* number of hybrid blocks */
    int n_heads;      /* attention heads, must divide d_model */
    int ff_mult;      /* channel-mix hidden expansion ratio */
    int max_seq_len;  /* activation buffer capacity (chunk length cap) */
} TCConfig;

TCConfig tc_default_config(void);

typedef struct {
    float *ln1_gamma, *ln1_beta;
    float *tm_mix_k, *tm_mix_v, *tm_mix_r; /* token-shift interpolation logits */
    float *tm_Wk, *tm_Wv, *tm_Wr;          /* D x D */
    float *tm_decay;                        /* per-channel decay logit, D */
    float *tm_Wo;                           /* D x D */

    float *ln2_gamma, *ln2_beta;
    float *at_Wq, *at_Wk, *at_Wv, *at_Wo;  /* D x D */

    float *ln3_gamma, *ln3_beta;
    float *cm_mix_k, *cm_mix_r;
    float *cm_Wk;  /* D x (ff_mult*D) */
    float *cm_Wv;  /* (ff_mult*D) x D */
    float *cm_Wr;  /* D x D */
} TCLayerView;

typedef struct {
    TCConfig cfg;
    float *buf;       /* owned flat parameter/grad/moment storage */
    int n_floats;
    float *embed;      /* vocab_size x d_model, tied with output head */
    float *ln_f_gamma; /* d_model */
    float *ln_f_beta;  /* d_model */
    float *pool_w;     /* d_model, learned attention-pooling query (embedder head only) */
    TCLayerView *layers;
} TCParamSet;

TCParamSet *tc_paramset_create(TCConfig cfg);
void tc_paramset_free(TCParamSet *ps);
void tc_paramset_zero(TCParamSet *ps);
/* Small random init in [-scale,scale]; decay/mix logits get a biased init
 * so the recurrence starts near "remember a little, mostly pass through". */
void tc_paramset_init_random(TCParamSet *ps, unsigned int seed);

/* Per-layer forward/backward caches for one chunk of length T (T<=max_seq_len). */
typedef struct {
    int T;
    float *x0;               /* T x D, block input */
    float *ln1_mean, *ln1_rstd; /* T */
    float *ln1_out;           /* T x D */
    float *ln1_xhat;          /* T x D, normalized pre-affine (needed for LN backward) */
    float *xk, *xv, *xr;      /* T x D each, token-shift mixed */
    float *k, *v, *r_sig;     /* T x D each */
    float *state;             /* T x D, recurrent state after step t */
    float *wkv;                /* T x D */
    float *tm_out;             /* T x D, after Wo */
    float *resid1;             /* T x D */

    float *ln2_mean, *ln2_rstd;
    float *ln2_out;             /* T x D */
    float *ln2_xhat;             /* T x D */
    float *Q, *K, *V;           /* T x D each */
    float *attn_w;              /* n_heads x T x T (causal softmax weights) */
    float *attn_ctx;            /* T x D, pre-Wo attention output (concat heads) */
    float *attn_out;            /* T x D, after Wo */
    float *resid2;               /* T x D */

    float *ln3_mean, *ln3_rstd;
    float *ln3_out;               /* T x D */
    float *ln3_xhat;               /* T x D */
    float *cmxk, *cmxr;           /* T x D */
    float *h_pre, *h_relu;        /* T x (ff_mult*D) */
    float *cm_v;                   /* T x D */
    float *r3_sig;                  /* T x D */
    float *cm_out;                   /* T x D */
    float *resid3;                    /* T x D, block output */
} TCLayerCache;

typedef struct {
    TCConfig cfg;
    int T;
    TCLayerCache *layers;   /* n_layers */
    float *ln_f_mean, *ln_f_rstd; /* T */
    float *ln_f_out;                /* T x D */
    float *ln_f_xhat;                /* T x D */
    float *logits;                   /* T x vocab_size */
    float *probs;                     /* T x vocab_size, softmax(logits) */

    float *pool_scores;  /* T, raw attention-pool scores before softmax */
    float *pool_weights;  /* T, softmax(pool_scores) */
    float *pool_raw;       /* D, weighted sum of ln_f_out before L2-normalize */
    float pool_norm;        /* ||pool_raw|| + eps, cached for backward */
} TCCache;

TCCache *tc_cache_create(TCConfig cfg);
void tc_cache_free(TCCache *c);

/* Forward pass over token ids[0..T-1]; fills cache (T <= cfg.max_seq_len).
 * If targets != NULL (next-token ids, length T), also computes mean
 * cross-entropy loss over all T positions and returns it via *out_loss. */
void tc_forward(const TCParamSet *p, TCCache *c, const int *ids, int T,
                 const int *targets, float *out_loss);

/* Backward pass: requires targets to have been passed to tc_forward
 * (uses cache->probs and targets to seed dLogits). Accumulates gradients
 * into grad (caller should zero it first for a fresh accumulation). */
void tc_backward(const TCParamSet *p, TCParamSet *grad, const TCCache *c,
                   const int *ids, int T, const int *targets);

int tc_param_count(TCConfig cfg);

/* Autoregressive generation: extends ids (length prompt_len on entry, must
 * be < cfg.max_seq_len) in place, greedy if temperature<=0 else sampling
 * from softmax(logits/temperature). Stops early at cfg.max_seq_len. Uses
 * c as scratch (must be sized for this model's cfg). Returns new length. */
int tc_generate(const TCParamSet *p, TCCache *c, int *ids, int prompt_len,
                 int max_new_tokens, float temperature, unsigned int *rng_state);

/* Embedder head: attention-pool ln_f_out with a single learned query
 * (p->pool_w) -- softmax over positions, weighted sum, L2-normalize. This
 * only discriminates on content once pool_w (and ideally the body) has
 * been trained; a random pool_w pools close to uniformly and inherits
 * whatever bias the body already has. See tc_embed_backward and the
 * contrastive training loop in embed_train.c. out_vec must hold
 * cfg.d_model floats; on return it is L2-normalized (ready for tc_cosine).
 * Fills c->pool_scores/pool_weights/pool_raw/pool_norm for tc_embed_backward. */
void tc_embed(const TCParamSet *p, TCCache *c, const int *ids, int T, float *out_vec);

/* Backward for tc_embed: given d(loss)/d(out_vec) (D floats, out_vec as
 * returned by the tc_embed call that populated this exact cache c),
 * backprops through L2-normalize -> attention-pool -> encoder body,
 * accumulating into grad (including grad->pool_w). */
void tc_embed_backward(const TCParamSet *p, TCParamSet *grad, const TCCache *c,
                        const int *ids, int T, const float *d_out_vec);

/* Cosine similarity; assumes both vectors are already L2-normalized (as
 * tc_embed's output is), so this is just a dot product. */
float tc_cosine(const float *a, const float *b, int D);

/* Binary format: TCConfig header followed by n_floats raw floats.
 * Returns 0 on success, -1 on I/O error, -2 on config mismatch. */
int tc_paramset_save(const TCParamSet *ps, const char *path);
/* Allocates and returns a new TCParamSet loaded from path, or NULL on error. */
TCParamSet *tc_paramset_load(const char *path);

#endif
