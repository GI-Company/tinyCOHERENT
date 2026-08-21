#ifndef TC_VECTORSTORE_H
#define TC_VECTORSTORE_H

/* Flat in-memory vector store, brute-force cosine search. Fine up to the
 * tens-of-thousands-of-vectors scale the architecture doc calls out;
 * nothing here assumes more than that. */

typedef struct {
    char *text;        /* owned copy */
    float *embedding;  /* owned copy, dim floats, expected L2-normalized */
} TCVSEntry;

typedef struct {
    TCVSEntry *entries;
    int count, capacity;
    int dim;
} TCVectorStore;

TCVectorStore *tc_vs_create(int dim);
void tc_vs_free(TCVectorStore *vs);

/* Copies both text and embedding into the store. */
void tc_vs_add(TCVectorStore *vs, const char *text, const float *embedding);

typedef struct {
    int idx;
    float score; /* cosine similarity, assumes normalized embeddings */
} TCVSHit;

/* Fills out_hits (caller-allocated, >= k entries) with the top-k matches by
 * cosine similarity, best first. Returns the number of hits (min(k, count)). */
int tc_vs_search(const TCVectorStore *vs, const float *query_embedding, int k, TCVSHit *out_hits);

#endif
