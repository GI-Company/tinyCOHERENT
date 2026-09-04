#include "bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline unsigned int hash_pair(int p0, int p1, int cap) {
    unsigned int h = ((unsigned int)p0 * 2654435761u) ^ ((unsigned int)p1 * 2246822519u);
    return h & (unsigned int)(cap - 1);
}

static void hash_insert(BPETokenizer *bpe, int p0, int p1, int rank, int new_id) {
    unsigned int idx = hash_pair(p0, p1, bpe->hash_cap);
    while (bpe->hash_keys_p0[idx] != -1) {
        idx = (idx + 1) & (unsigned int)(bpe->hash_cap - 1);
    }
    bpe->hash_keys_p0[idx] = p0;
    bpe->hash_keys_p1[idx] = p1;
    bpe->hash_values_rank[idx] = rank;
    bpe->hash_values_new_id[idx] = new_id;
}

static int hash_lookup(const BPETokenizer *bpe, int p0, int p1, int *out_new_id) {
    unsigned int idx = hash_pair(p0, p1, bpe->hash_cap);
    while (bpe->hash_keys_p0[idx] != -1) {
        if (bpe->hash_keys_p0[idx] == p0 && bpe->hash_keys_p1[idx] == p1) {
            if (out_new_id) *out_new_id = bpe->hash_values_new_id[idx];
            return bpe->hash_values_rank[idx];
        }
        idx = (idx + 1) & (unsigned int)(bpe->hash_cap - 1);
    }
    return -1; /* not found */
}

BPETokenizer *bpe_load(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;

    int vocab_size = 0, num_merges = 0;
    if (fscanf(f, "%d %d", &vocab_size, &num_merges) != 2) {
        fclose(f);
        return NULL;
    }

    BPETokenizer *bpe = calloc(1, sizeof(BPETokenizer));
    bpe->vocab_size = vocab_size;
    bpe->num_merges = num_merges;
    bpe->merges = calloc((size_t)num_merges, sizeof(BPEMerge));

    /* Hash table cap is next power of 2 >= 4 * num_merges */
    int cap = 1024;
    while (cap < num_merges * 4) cap *= 2;
    bpe->hash_cap = cap;
    bpe->hash_keys_p0 = malloc(sizeof(int) * (size_t)cap);
    bpe->hash_keys_p1 = malloc(sizeof(int) * (size_t)cap);
    bpe->hash_values_rank = malloc(sizeof(int) * (size_t)cap);
    bpe->hash_values_new_id = malloc(sizeof(int) * (size_t)cap);
    for (int i = 0; i < cap; i++) bpe->hash_keys_p0[i] = -1;

    /* Read merges and insert into hash table */
    for (int i = 0; i < num_merges; i++) {
        int p0, p1, new_id;
        if (fscanf(f, "%d %d %d", &p0, &p1, &new_id) != 3) {
            bpe_free(bpe);
            fclose(f);
            return NULL;
        }
        bpe->merges[i].p0 = p0;
        bpe->merges[i].p1 = p1;
        bpe->merges[i].new_id = new_id;
        hash_insert(bpe, p0, p1, i, new_id);
    }
    fclose(f);

    /* Construct decoding table */
    bpe->token_bytes = calloc((size_t)vocab_size, sizeof(unsigned char *));
    bpe->token_lens = calloc((size_t)vocab_size, sizeof(int));

    /* 0..255 are base bytes */
    for (int i = 0; i < 256; i++) {
        bpe->token_lens[i] = 1;
        bpe->token_bytes[i] = malloc(1);
        bpe->token_bytes[i][0] = (unsigned char)i;
    }

    /* Merges */
    for (int i = 0; i < num_merges; i++) {
        int p0 = bpe->merges[i].p0;
        int p1 = bpe->merges[i].p1;
        int nid = bpe->merges[i].new_id;
        if (nid >= 0 && nid < vocab_size && p0 < vocab_size && p1 < vocab_size) {
            int len0 = bpe->token_lens[p0];
            int len1 = bpe->token_lens[p1];
            bpe->token_lens[nid] = len0 + len1;
            bpe->token_bytes[nid] = malloc((size_t)(len0 + len1));
            memcpy(bpe->token_bytes[nid], bpe->token_bytes[p0], (size_t)len0);
            memcpy(bpe->token_bytes[nid] + len0, bpe->token_bytes[p1], (size_t)len1);
        }
    }

    return bpe;
}

void bpe_free(BPETokenizer *bpe) {
    if (!bpe) return;
    free(bpe->merges);
    free(bpe->hash_keys_p0);
    free(bpe->hash_keys_p1);
    free(bpe->hash_values_rank);
    free(bpe->hash_values_new_id);
    if (bpe->token_bytes) {
        for (int i = 0; i < bpe->vocab_size; i++) {
            free(bpe->token_bytes[i]);
        }
        free(bpe->token_bytes);
    }
    free(bpe->token_lens);
    free(bpe);
}

static int is_word_char(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

static int is_space_char(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static int encode_piece(const BPETokenizer *bpe, const unsigned char *src, int len, int *out, int max_out) {
    if (len <= 0 || max_out <= 0) return 0;
    if (len == 1) {
        out[0] = src[0];
        return 1;
    }
    int tokens[256];
    if (len > 250) len = 250;
    int n = len;
    for (int i = 0; i < n; i++) tokens[i] = src[i];

    while (n >= 2) {
        int best_rank = -1;
        int best_idx = -1;
        int best_new_id = -1;

        for (int i = 0; i < n - 1; i++) {
            int new_id;
            int rank = hash_lookup(bpe, tokens[i], tokens[i + 1], &new_id);
            if (rank != -1 && (best_rank == -1 || rank < best_rank)) {
                best_rank = rank;
                best_idx = i;
                best_new_id = new_id;
            }
        }

        if (best_rank == -1) break;

        int p0 = tokens[best_idx];
        int p1 = tokens[best_idx + 1];
        int w = 0;
        for (int r = 0; r < n; ) {
            if (r < n - 1 && tokens[r] == p0 && tokens[r + 1] == p1) {
                tokens[w++] = best_new_id;
                r += 2;
            } else {
                tokens[w++] = tokens[r++];
            }
        }
        n = w;
    }

    int to_copy = n < max_out ? n : max_out;
    for (int i = 0; i < to_copy; i++) out[i] = tokens[i];
    return to_copy;
}

int bpe_encode(const BPETokenizer *bpe, const char *text, int *out_tokens, int max_tokens) {
    if (!bpe || !text || max_tokens < 1) return 0;

    size_t in_len = strlen(text);
    if (in_len == 0) return 0;

    int total = 0;
    size_t i = 0;
    while (i < in_len && total < max_tokens) {
        size_t start = i;
        unsigned char c = (unsigned char)text[i];
        if (is_word_char(c)) {
            while (i < in_len && is_word_char((unsigned char)text[i]) && (i - start) < 64) i++;
        } else if (is_space_char(c)) {
            while (i < in_len && is_space_char((unsigned char)text[i]) && (i - start) < 64) i++;
        } else {
            i++;
        }
        int piece_len = (int)(i - start);
        int written = encode_piece(bpe, (const unsigned char *)text + start, piece_len,
                                   out_tokens + total, max_tokens - total);
        total += written;
    }
    return total;
}

int bpe_decode(const BPETokenizer *bpe, const int *tokens, int num_tokens, char *out_text, int max_bytes) {
    if (!bpe || !tokens || !out_text || max_bytes < 1) return 0;

    int written = 0;
    for (int i = 0; i < num_tokens; i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < bpe->vocab_size && bpe->token_bytes[tok]) {
            int len = bpe->token_lens[tok];
            if (written + len >= max_bytes) len = max_bytes - written - 1;
            if (len <= 0) break;
            memcpy(out_text + written, bpe->token_bytes[tok], (size_t)len);
            written += len;
        }
    }
    out_text[written] = '\0';
    return written;
}
