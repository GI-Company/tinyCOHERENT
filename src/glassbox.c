#include "glassbox.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static char printable(char c) { return (c == '\n') ? ' ' : c; }

int tc_explain_generate(const TCParamSet *p, TCCache *c, const int *ids, int T,
                         TCGlassBoxStep *out_steps) {
    if (T < 2) return 0;
    int T_use = T - 1;
    const int *targets = ids + 1;
    int V = p->cfg.vocab_size;

    float loss;
    tc_forward(p, c, ids, T_use, targets, &loss);

    for (int t = 0; t < T_use; t++) {
        const float *prob = c->probs + (size_t)t * V;
        TCGlassBoxStep *step = &out_steps[t];
        step->token_id = targets[t];
        step->ch = tc_decode_id(targets[t]);
        step->prob = prob[targets[t]];
        step->surprise = -log2f(fmaxf(step->prob, 1e-9f));

        int best[TC_TOPK];
        for (int k = 0; k < TC_TOPK; k++) best[k] = -1;
        for (int i = 0; i < V; i++) {
            for (int k = 0; k < TC_TOPK; k++) {
                if (best[k] == -1 || prob[i] > prob[best[k]]) {
                    for (int j = TC_TOPK - 1; j > k; j--) best[j] = best[j - 1];
                    best[k] = i;
                    break;
                }
            }
        }
        for (int k = 0; k < TC_TOPK; k++) {
            step->topk_ids[k] = best[k];
            step->topk_chars[k] = tc_decode_id(best[k]);
            step->topk_probs[k] = prob[best[k]];
        }
    }
    return T_use;
}

void tc_glassbox_print(const TCGlassBoxStep *steps, int n) {
    printf("%-6s %-6s %-8s %-10s  %s\n", "char", "p", "surprise", "", "top-3 alternatives");
    for (int t = 0; t < n; t++) {
        const TCGlassBoxStep *s = &steps[t];
        printf("'%c'    %.3f  %6.2f bits            ", printable(s->ch), s->prob, s->surprise);
        for (int k = 0; k < TC_TOPK; k++) {
            printf("'%c':%.2f ", printable(s->topk_chars[k]), s->topk_probs[k]);
        }
        printf("\n");
    }
}

int tc_explain_embedding(const TCParamSet *p, TCCache *c, const int *ids, int T,
                          TCEmbedOcclusionStep *out_steps) {
    int D = p->cfg.d_model;
    float base[TC_MAX_D_MODEL], occluded_vec[TC_MAX_D_MODEL];
    tc_embed(p, c, ids, T, base);

    int *occ_ids = malloc(sizeof(int) * (size_t)T);
    memcpy(occ_ids, ids, sizeof(int) * (size_t)T);
    int space_id = tc_encode_char(' ');

    for (int t = 0; t < T; t++) {
        int orig = occ_ids[t];
        occ_ids[t] = space_id;
        tc_embed(p, c, occ_ids, T, occluded_vec);
        occ_ids[t] = orig;

        out_steps[t].ch = tc_decode_id(orig);
        out_steps[t].importance = 1.0f - tc_cosine(base, occluded_vec, D);
    }
    free(occ_ids);
    return T;
}

int tc_explain_causal(const TCParamSet *p, TCCache *c, const int *ids, int T,
                       int occlude_id, TCCausalStep *out_steps) {
    if (T < 2) return 0;
    int T_use = T - 1;
    const int *targets = ids + 1;
    int V = p->cfg.vocab_size;

    float loss;
    tc_forward(p, c, ids, T_use, targets, &loss);
    float *base_loss_at = malloc(sizeof(float) * (size_t)T_use);
    for (int t = 0; t < T_use; t++)
        base_loss_at[t] = -log2f(fmaxf(c->probs[(size_t)t * V + targets[t]], 1e-9f));

    int *occ_ids = malloc(sizeof(int) * (size_t)T_use);
    memcpy(occ_ids, ids, sizeof(int) * (size_t)T_use);

    for (int i = 0; i < T_use; i++) {
        int orig = occ_ids[i];
        occ_ids[i] = occlude_id;
        tc_forward(p, c, occ_ids, T_use, targets, &loss);
        occ_ids[i] = orig;

        float sum_base = 0.0f, sum_occ = 0.0f;
        int cnt = 0;
        for (int t = i + 1; t < T_use; t++) {
            sum_base += base_loss_at[t];
            sum_occ += -log2f(fmaxf(c->probs[(size_t)t * V + targets[t]], 1e-9f));
            cnt++;
        }
        out_steps[i].ch = tc_decode_id(orig);
        out_steps[i].importance = cnt > 0 ? (sum_occ - sum_base) / cnt : 0.0f;
    }
    free(occ_ids);
    free(base_loss_at);
    return T_use;
}

void tc_causal_glassbox_print(const TCCausalStep *steps, int n) {
    printf("%-6s %-10s  %s\n", "char", "importance", "(downstream loss increase, bits, if this char is occluded)");
    for (int t = 0; t < n; t++) {
        const TCCausalStep *s = &steps[t];
        int bar_len = (int)(s->importance / 0.3f * 30.0f);
        if (bar_len > 30) bar_len = 30;
        if (bar_len < 0) bar_len = 0;
        printf("'%c'    %+.4f      ", printable(s->ch), s->importance);
        for (int k = 0; k < bar_len; k++) putchar('#');
        printf("\n");
    }
}

void tc_embed_glassbox_print(const TCEmbedOcclusionStep *steps, int n) {
    printf("%-6s %-10s  %s\n", "char", "importance", "(embedding shift if this char is occluded)");
    for (int t = 0; t < n; t++) {
        const TCEmbedOcclusionStep *s = &steps[t];
        int bar_len = (int)(s->importance / 0.5f * 30.0f);
        if (bar_len > 30) bar_len = 30;
        if (bar_len < 0) bar_len = 0;
        printf("'%c'    %.4f      ", printable(s->ch), s->importance);
        for (int k = 0; k < bar_len; k++) putchar('#');
        printf("\n");
    }
}

int tc_explain_input_grad(const TCParamSet *p, TCCache *c, const int *ids, int T,
                          TCGradStep *out_steps) {
    if (T < 2) return 0;
    int T_use = T - 1;
    const int *targets = ids + 1;
    int D = p->cfg.d_model;

    float loss;
    tc_forward(p, c, ids, T_use, targets, &loss);

    float *dx0 = calloc((size_t)T_use * D, sizeof(float));
    tc_input_grad(p, c, ids, T_use, targets, dx0);

    for (int t = 0; t < T_use; t++) {
        int tok = ids[t];
        const float *emb = p->embed + (size_t)tok * D;
        const float *g = dx0 + (size_t)t * D;

        float dot = 0.0f;
        float norm_sq = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += emb[d] * g[d];
            norm_sq += g[d] * g[d];
        }
        out_steps[t].ch = tc_decode_id(tok);
        out_steps[t].grad_norm = sqrtf(norm_sq);
        out_steps[t].importance = fabsf(dot);
    }
    free(dx0);
    return T_use;
}

void tc_grad_glassbox_print(const TCGradStep *steps, int n) {
    printf("%-6s %-12s %-12s  %s\n", "char", "Input x Grad", "Grad Norm", "(analytical gradient attribution)");
    float max_imp = 1e-6f;
    for (int t = 0; t < n; t++) if (steps[t].importance > max_imp) max_imp = steps[t].importance;

    for (int t = 0; t < n; t++) {
        const TCGradStep *s = &steps[t];
        int bar_len = (int)((s->importance / max_imp) * 30.0f);
        if (bar_len > 30) bar_len = 30;
        if (bar_len < 0) bar_len = 0;
        printf("'%c'    %.6f     %.6f     ", printable(s->ch), s->importance, s->grad_norm);
        for (int k = 0; k < bar_len; k++) putchar('#');
        printf("\n");
    }
}

int tc_explain_heads(const TCParamSet *p, TCCache *c, const int *ids, int T,
                     TCHeadImportance *out_heads) {
    if (T < 2 || !out_heads) return 0;
    int T_use = T - 1;
    const int *targets = ids + 1;

    float base_loss;
    tc_forward(p, c, ids, T_use, targets, &base_loss);

    int count = 0;
    for (int l = 0; l < p->cfg.n_layers; l++) {
        for (int h = 0; h < p->cfg.n_heads; h++) {
            float ablated_loss;
            tc_forward_ex(p, c, ids, T_use, targets, &ablated_loss, l, h);
            out_heads[count].layer = l;
            out_heads[count].head = h;
            out_heads[count].importance = ablated_loss - base_loss;
            count++;
        }
    }
    return count;
}
