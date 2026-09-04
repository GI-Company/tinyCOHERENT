#include "tcmodel.h"
#include "bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(len + 1);
    size_t n = fread(buf, 1, len, f);
    fclose(f);
    buf[n] = 0;
    *out_len = (long)n;
    return buf;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "build/model_rung6_step900.bin";
    const char *corpus_path = "data/tinystories_subset.txt";
    const char *bpe_path = "data/bpe_merges.txt";

    TCParamSet *p = tc_paramset_load(model_path);
    if (!p) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }

    BPETokenizer *bpe = bpe_load(bpe_path);
    if (!bpe) { fprintf(stderr, "failed to load bpe\n"); return 1; }

    long raw_len;
    char *text = read_file(corpus_path, &raw_len);
    int *tokens = malloc(sizeof(int) * (size_t)raw_len);
    long num_tokens = bpe_encode(bpe, text, tokens, (int)raw_len);
    free(text);

    long split_at = (long)(num_tokens * 0.85);
    long val_len = num_tokens - split_at;
    const int *val_tokens = tokens + split_at;

    TCCache *c = tc_cache_create(p->cfg);

    int chunk_len = 100;
    int fixed_n_chunks = 100; /* exactly 10,000 tokens evaluated */
    long stride = (val_len - 1) / (chunk_len * fixed_n_chunks);
    if (stride < 1) stride = 1;

    float total_loss = 0.0f;
    for (int i = 0; i < fixed_n_chunks; i++) {
        long off = (long)i * stride * chunk_len;
        if (off + chunk_len >= val_len) break;
        float loss;
        tc_forward(p, c, val_tokens + off, chunk_len, val_tokens + off + 1, &loss);
        total_loss += loss;
    }
    float mean_val_loss = total_loss / (float)fixed_n_chunks;

    printf("MODEL: %s (D=%d, L=%d, params=%d)\n", model_path, p->cfg.d_model, p->cfg.n_layers, p->n_floats);
    printf("Fixed Held-Out Protocol: %d chunks x %d tokens = %d tokens\n", fixed_n_chunks, chunk_len, fixed_n_chunks * chunk_len);
    printf("Exact Validation Loss: %.4f\n", mean_val_loss);

    tc_cache_free(c);
    tc_paramset_free(p);
    bpe_free(bpe);
    free(tokens);
    return 0;
}
