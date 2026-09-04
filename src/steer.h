#ifndef TC_STEER_H
#define TC_STEER_H

#include "tcmodel.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Individual latent steering direction vector */
typedef struct {
    char name[64];
    char description[128];
    int layer;        /* target layer for residual injection (0 <= layer < n_layers) */
    int dim;          /* d_model */
    float *vec;       /* normalized unit direction vector (dim floats) */
} TCSteerVector;

/* Bank of latent steering vectors */
typedef struct {
    int count;
    int capacity;
    TCSteerVector *vectors;
} TCSteerBank;

/* Allocate and free steering banks */
TCSteerBank *tc_steer_bank_create(void);
void tc_steer_bank_free(TCSteerBank *bank);

/* Add a steering vector to the bank. vec is copied. Returns vector index or -1 on failure. */
int tc_steer_bank_add(TCSteerBank *bank, const char *name, const char *desc,
                      int layer, int dim, const float *vec);

/* Find vector by name (case-insensitive). Returns pointer or NULL. */
const TCSteerVector *tc_steer_bank_find(const TCSteerBank *bank, const char *name);

/* Extract a normalized difference direction from positive and negative prompt token sequences.
 * out_vec must have space for cfg.d_model floats.
 * The direction is extracted by taking the difference of mean hidden states at the specified layer:
 *   d = mean(resid3_pos) - mean(resid3_neg), normalized to unit length.
 * Returns 0 on success, -1 on error. */
int tc_steer_extract_direction(const TCParamSet *p, TCCache *c,
                               const int *pos_ids, int pos_len,
                               const int *neg_ids, int neg_len,
                               int layer, float *out_vec);

/* Inspect vocabulary token projections (mechanistic interpretability).
 * Computes Delta_z = W_embed * vec (dimension V).
 * Fills top_k tokens with the highest positive logit boost and top_k tokens
 * with the strongest negative inhibition. */
void tc_steer_inspect_boost(const TCParamSet *p, const float *vec, int top_k,
                            int *out_pos_tokens, float *out_pos_deltas,
                            int *out_neg_tokens, float *out_neg_deltas);

/* Serialization to and from binary files */
int tc_steer_save_bank(const TCSteerBank *bank, const char *filepath);
TCSteerBank *tc_steer_load_bank(const char *filepath);

#ifdef __cplusplus
}
#endif

#endif /* TC_STEER_H */
