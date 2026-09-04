#ifndef TC_GLASSBOX_H
#define TC_GLASSBOX_H

#include "tcmodel.h"

#define TC_TOPK 3

typedef struct {
    int token_id;    /* actual next token */
    char ch;
    float prob;       /* probability the model assigned to the actual next char */
    float surprise;    /* -log2(prob), bits of surprise */
    int topk_ids[TC_TOPK];
    char topk_chars[TC_TOPK];
    float topk_probs[TC_TOPK];
} TCGlassBoxStep;

/* Forward pass over ids[0..T-2] predicting ids[1..T-1] (byte-level next-char
 * prediction). Fills one TCGlassBoxStep per predicted position (T-1 of them)
 * into out_steps (caller-allocated, at least T-1 entries) and returns T-1. */
int tc_explain_generate(const TCParamSet *p, TCCache *c, const int *ids, int T,
                         TCGlassBoxStep *out_steps);

void tc_glassbox_print(const TCGlassBoxStep *steps, int n);

typedef struct {
    char ch;
    float importance; /* 1 - cosine(base embedding, embedding with this char occluded) */
} TCEmbedOcclusionStep;

/* Occlusion attribution for the embedder head: replace each position in
 * turn with a neutral token (space) and measure how far the pooled,
 * L2-normalized embedding moves. out_steps must hold at least T entries. */
int tc_explain_embedding(const TCParamSet *p, TCCache *c, const int *ids, int T,
                          TCEmbedOcclusionStep *out_steps);
void tc_embed_glassbox_print(const TCEmbedOcclusionStep *steps, int n);

typedef struct {
    char ch;
    float importance; /* increase in mean loss over positions (i, T) when position i is occluded */
} TCCausalStep;

/* Real causal attribution for the generative head, as opposed to surprise
 * (tc_explain_generate): surprise at position t measures how hard t was to
 * predict -- a confidence/calibration signal about t itself. This measures
 * how much removing t hurts the model's ability to predict what comes
 * AFTER t, which is what "this position mattered" should actually mean.
 * Forward pass over ids[0..T-2] predicting ids[1..T-1], T-1 steps. */
/* occlude_id is the token substituted at each occluded position. Callers on
 * natural-language text use space (the corpus's most neutral common
 * token); callers on other distributions (e.g. a digits-only synthetic
 * task) MUST pass something in that distribution -- an out-of-distribution
 * substitution can produce large effects simply from being unfamiliar,
 * unrelated to genuine causal importance. See known_answer.c for a case
 * where this mattered. */
int tc_explain_causal(const TCParamSet *p, TCCache *c, const int *ids, int T,
                       int occlude_id, TCCausalStep *out_steps);
void tc_causal_glassbox_print(const TCCausalStep *steps, int n);

typedef struct {
    char ch;
    float importance; /* Input x Grad: |dot(x0, d(loss)/d(x0))| */
    float grad_norm;  /* ||d(loss)/d(x0)||_2 */
} TCGradStep;

/* Fast analytical attribution for the generative head: computes input embedding
 * gradients in a single forward + backward pass O(1) instead of O(T) occlusion passes.
 * ids has length T; forward pass evaluates ids[0..T-2] predicting ids[1..T-1] (T-1 steps).
 * out_steps must hold at least T-1 entries. Returns T-1. */
int tc_explain_input_grad(const TCParamSet *p, TCCache *c, const int *ids, int T,
                          TCGradStep *out_steps);
void tc_grad_glassbox_print(const TCGradStep *steps, int n);

/* Mechanistic interpretability: evaluates causal importance of every attention head
 * by ablating each head individually and recording the resulting loss shift.
 * ids has length T; evaluates predicting targets ids[1..T-1] (T-1 steps).
 * out_heads must hold at least p->cfg.n_layers * p->cfg.n_heads entries.
 * Returns total number of heads evaluated. */
typedef struct {
    int layer;
    int head;
    float importance; /* Loss shift when this head is ablated: L_ablated - L_base */
} TCHeadImportance;

int tc_explain_heads(const TCParamSet *p, TCCache *c, const int *ids, int T,
                     TCHeadImportance *out_heads);

#endif
