#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <strings.h>
#include "steer.h"

#define TC_STEER_MAGIC 0x54534354  /* "TCST" in little-endian */
#define TC_STEER_VERSION 1

TCSteerBank *tc_steer_bank_create(void) {
    TCSteerBank *bank = calloc(1, sizeof(TCSteerBank));
    if (!bank) return NULL;
    bank->capacity = 8;
    bank->count = 0;
    bank->vectors = calloc(bank->capacity, sizeof(TCSteerVector));
    return bank;
}

void tc_steer_bank_free(TCSteerBank *bank) {
    if (!bank) return;
    for (int i = 0; i < bank->count; i++) {
        if (bank->vectors[i].vec) {
            free(bank->vectors[i].vec);
            bank->vectors[i].vec = NULL;
        }
    }
    free(bank->vectors);
    free(bank);
}

int tc_steer_bank_add(TCSteerBank *bank, const char *name, const char *desc,
                      int layer, int dim, const float *vec) {
    if (!bank || !name || !vec || dim <= 0) return -1;

    /* Check if already exists; if so, overwrite */
    for (int i = 0; i < bank->count; i++) {
        if (strcasecmp(bank->vectors[i].name, name) == 0) {
            if (bank->vectors[i].dim != dim) {
                free(bank->vectors[i].vec);
                bank->vectors[i].vec = malloc(dim * sizeof(float));
            }
            strncpy(bank->vectors[i].description, desc ? desc : "", sizeof(bank->vectors[i].description) - 1);
            bank->vectors[i].layer = layer;
            bank->vectors[i].dim = dim;
            memcpy(bank->vectors[i].vec, vec, dim * sizeof(float));
            return i;
        }
    }

    if (bank->count >= bank->capacity) {
        int new_cap = bank->capacity * 2;
        TCSteerVector *new_vecs = realloc(bank->vectors, new_cap * sizeof(TCSteerVector));
        if (!new_vecs) return -1;
        bank->vectors = new_vecs;
        bank->capacity = new_cap;
    }

    TCSteerVector *sv = &bank->vectors[bank->count];
    memset(sv, 0, sizeof(TCSteerVector));
    strncpy(sv->name, name, sizeof(sv->name) - 1);
    strncpy(sv->description, desc ? desc : "", sizeof(sv->description) - 1);
    sv->layer = layer;
    sv->dim = dim;
    sv->vec = malloc(dim * sizeof(float));
    if (!sv->vec) return -1;
    memcpy(sv->vec, vec, dim * sizeof(float));

    int idx = bank->count;
    bank->count++;
    return idx;
}

const TCSteerVector *tc_steer_bank_find(const TCSteerBank *bank, const char *name) {
    if (!bank || !name) return NULL;
    for (int i = 0; i < bank->count; i++) {
        if (strcasecmp(bank->vectors[i].name, name) == 0) {
            return &bank->vectors[i];
        }
    }
    return NULL;
}

int tc_steer_extract_direction(const TCParamSet *p, TCCache *c,
                               const int *pos_ids, int pos_len,
                               const int *neg_ids, int neg_len,
                               int layer, float *out_vec) {
    if (!p || !c || !pos_ids || pos_len <= 0 || !neg_ids || neg_len <= 0 || !out_vec) {
        return -1;
    }
    if (layer < 0 || layer >= p->cfg.n_layers) return -1;

    int D = p->cfg.d_model;
    float *pos_mean = calloc(D, sizeof(float));
    float *neg_mean = calloc(D, sizeof(float));
    if (!pos_mean || !neg_mean) {
        free(pos_mean);
        free(neg_mean);
        return -1;
    }

    /* Positive forward */
    tc_forward(p, c, pos_ids, pos_len, NULL, NULL);
    const float *pos_act = c->layers[layer].resid3;
    for (int t = 0; t < pos_len; t++) {
        const float *xt = pos_act + (size_t)t * D;
        for (int i = 0; i < D; i++) pos_mean[i] += xt[i];
    }
    for (int i = 0; i < D; i++) pos_mean[i] /= (float)pos_len;

    /* Negative forward */
    tc_forward(p, c, neg_ids, neg_len, NULL, NULL);
    const float *neg_act = c->layers[layer].resid3;
    for (int t = 0; t < neg_len; t++) {
        const float *xt = neg_act + (size_t)t * D;
        for (int i = 0; i < D; i++) neg_mean[i] += xt[i];
    }
    for (int i = 0; i < D; i++) neg_mean[i] /= (float)neg_len;

    /* Difference vector */
    double norm_sq = 0.0;
    for (int i = 0; i < D; i++) {
        float diff = pos_mean[i] - neg_mean[i];
        out_vec[i] = diff;
        norm_sq += (double)diff * (double)diff;
    }

    double norm = sqrt(norm_sq);
    if (norm > 1e-9) {
        float inv = (float)(1.0 / norm);
        for (int i = 0; i < D; i++) out_vec[i] *= inv;
    } else {
        for (int i = 0; i < D; i++) out_vec[i] = 0.0f;
    }

    free(pos_mean);
    free(neg_mean);
    return 0;
}

typedef struct {
    int token;
    float delta;
} TokenDelta;

static int cmp_token_delta_desc(const void *a, const void *b) {
    float da = ((const TokenDelta *)a)->delta;
    float db = ((const TokenDelta *)b)->delta;
    if (da > db) return -1;
    if (da < db) return 1;
    return 0;
}

void tc_steer_inspect_boost(const TCParamSet *p, const float *vec, int top_k,
                            int *out_pos_tokens, float *out_pos_deltas,
                            int *out_neg_tokens, float *out_neg_deltas) {
    if (!p || !vec || top_k <= 0) return;
    int V = p->cfg.vocab_size, D = p->cfg.d_model;

    TokenDelta *deltas = malloc(V * sizeof(TokenDelta));
    if (!deltas) return;

    for (int v = 0; v < V; v++) {
        const float *erow = p->embed + (size_t)v * D;
        float dot = 0.0f;
        for (int i = 0; i < D; i++) dot += erow[i] * vec[i];
        deltas[v].token = v;
        deltas[v].delta = dot;
    }

    qsort(deltas, V, sizeof(TokenDelta), cmp_token_delta_desc);

    int k_pos = top_k < V ? top_k : V;
    for (int i = 0; i < k_pos; i++) {
        if (out_pos_tokens) out_pos_tokens[i] = deltas[i].token;
        if (out_pos_deltas) out_pos_deltas[i] = deltas[i].delta;
    }

    for (int i = 0; i < k_pos; i++) {
        int tail_idx = V - 1 - i;
        if (out_neg_tokens) out_neg_tokens[i] = deltas[tail_idx].token;
        if (out_neg_deltas) out_neg_deltas[i] = deltas[tail_idx].delta;
    }

    free(deltas);
}

int tc_steer_save_bank(const TCSteerBank *bank, const char *filepath) {
    if (!bank || !filepath) return -1;
    FILE *f = fopen(filepath, "wb");
    if (!f) return -1;

    unsigned int magic = TC_STEER_MAGIC;
    unsigned int version = TC_STEER_VERSION;
    int count = bank->count;

    if (fwrite(&magic, sizeof(unsigned int), 1, f) != 1 ||
        fwrite(&version, sizeof(unsigned int), 1, f) != 1 ||
        fwrite(&count, sizeof(int), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    for (int i = 0; i < count; i++) {
        const TCSteerVector *sv = &bank->vectors[i];
        if (fwrite(sv->name, sizeof(sv->name), 1, f) != 1 ||
            fwrite(sv->description, sizeof(sv->description), 1, f) != 1 ||
            fwrite(&sv->layer, sizeof(int), 1, f) != 1 ||
            fwrite(&sv->dim, sizeof(int), 1, f) != 1 ||
            fwrite(sv->vec, sizeof(float), sv->dim, f) != (size_t)sv->dim) {
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;
}

TCSteerBank *tc_steer_load_bank(const char *filepath) {
    if (!filepath) return NULL;
    FILE *f = fopen(filepath, "rb");
    if (!f) return NULL;

    unsigned int magic = 0, version = 0;
    int count = 0;

    if (fread(&magic, sizeof(unsigned int), 1, f) != 1 ||
        fread(&version, sizeof(unsigned int), 1, f) != 1 ||
        fread(&count, sizeof(int), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    if (magic != TC_STEER_MAGIC || version != TC_STEER_VERSION || count < 0) {
        fclose(f);
        return NULL;
    }

    TCSteerBank *bank = tc_steer_bank_create();
    if (!bank) { fclose(f); return NULL; }

    for (int i = 0; i < count; i++) {
        char name[64];
        char desc[128];
        int layer = 0, dim = 0;

        if (fread(name, sizeof(name), 1, f) != 1 ||
            fread(desc, sizeof(desc), 1, f) != 1 ||
            fread(&layer, sizeof(int), 1, f) != 1 ||
            fread(&dim, sizeof(int), 1, f) != 1) {
            tc_steer_bank_free(bank);
            fclose(f);
            return NULL;
        }

        if (dim <= 0 || dim > 4096) {
            tc_steer_bank_free(bank);
            fclose(f);
            return NULL;
        }

        float *vec = malloc(dim * sizeof(float));
        if (!vec) {
            tc_steer_bank_free(bank);
            fclose(f);
            return NULL;
        }

        if (fread(vec, sizeof(float), dim, f) != (size_t)dim) {
            free(vec);
            tc_steer_bank_free(bank);
            fclose(f);
            return NULL;
        }

        tc_steer_bank_add(bank, name, desc, layer, dim, vec);
        free(vec);
    }

    fclose(f);
    return bank;
}
