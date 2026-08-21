#include "vectorstore.h"
#include "tcmodel.h" /* tc_cosine */
#include <stdlib.h>
#include <string.h>

TCVectorStore *tc_vs_create(int dim) {
    TCVectorStore *vs = calloc(1, sizeof(TCVectorStore));
    vs->dim = dim;
    vs->capacity = 8;
    vs->entries = calloc((size_t)vs->capacity, sizeof(TCVSEntry));
    return vs;
}

void tc_vs_free(TCVectorStore *vs) {
    if (!vs) return;
    for (int i = 0; i < vs->count; i++) {
        free(vs->entries[i].text);
        free(vs->entries[i].embedding);
    }
    free(vs->entries);
    free(vs);
}

void tc_vs_add(TCVectorStore *vs, const char *text, const float *embedding) {
    if (vs->count == vs->capacity) {
        vs->capacity *= 2;
        vs->entries = realloc(vs->entries, (size_t)vs->capacity * sizeof(TCVSEntry));
    }
    TCVSEntry *e = &vs->entries[vs->count++];
    e->text = strdup(text);
    e->embedding = malloc((size_t)vs->dim * sizeof(float));
    memcpy(e->embedding, embedding, (size_t)vs->dim * sizeof(float));
}

int tc_vs_search(const TCVectorStore *vs, const float *query_embedding, int k, TCVSHit *out_hits) {
    if (k > vs->count) k = vs->count;
    for (int i = 0; i < k; i++) out_hits[i] = (TCVSHit){ -1, -2.0f };

    for (int i = 0; i < vs->count; i++) {
        float score = tc_cosine(query_embedding, vs->entries[i].embedding, vs->dim);
        for (int j = 0; j < k; j++) {
            if (score > out_hits[j].score) {
                for (int m = k - 1; m > j; m--) out_hits[m] = out_hits[m - 1];
                out_hits[j] = (TCVSHit){ i, score };
                break;
            }
        }
    }
    return k;
}
