#ifndef TC_BPE_H
#define TC_BPE_H

#include <stddef.h>

typedef struct {
    int p0;
    int p1;
    int new_id;
} BPEMerge;

typedef struct {
    int vocab_size;
    int num_merges;
    BPEMerge *merges;

    /* Hash table for (p0, p1) -> rank */
    int *hash_keys_p0;
    int *hash_keys_p1;
    int *hash_values_rank;
    int *hash_values_new_id;
    int hash_cap;

    /* Token to byte string mapping for decoding */
    unsigned char **token_bytes;
    int *token_lens;
} BPETokenizer;

/* Load BPE merges from a text file created by train_bpe.py */
BPETokenizer *bpe_load(const char *path);
void bpe_free(BPETokenizer *bpe);

/* Encode text into BPE token IDs. Returns number of tokens written (<= max_tokens). */
int bpe_encode(const BPETokenizer *bpe, const char *text, int *out_tokens, int max_tokens);

/* Decode BPE token IDs back into text. Returns number of bytes written. Null-terminated. */
int bpe_decode(const BPETokenizer *bpe, const int *tokens, int num_tokens, char *out_text, int max_bytes);

#endif
