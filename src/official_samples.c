#include "tcmodel.h"
#include "bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void run_greedy_sample(TCParamSet *p, TCCache *c, BPETokenizer *bpe, const char *prompt, int max_new) {
    int ids[128];
    int plen = bpe_encode(bpe, prompt, ids, 128);
    unsigned int rng = 0;
    int tot_len = tc_generate(p, c, ids, plen, max_new, 0.0f, &rng);
    char out_buf[1024];
    bpe_decode(bpe, ids, tot_len, out_buf, sizeof(out_buf));
    printf("Prompt:    \"%s\"\n", prompt);
    printf("Greedy T=0: \"%s\"\n\n", out_buf);
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "build/model_rung6_step900.bin";
    TCParamSet *p = tc_paramset_load(model_path);
    if (!p) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }
    BPETokenizer *bpe = bpe_load("data/bpe_merges.txt");
    if (!bpe) { fprintf(stderr, "failed to load bpe\n"); return 1; }
    TCCache *c = tc_cache_create(p->cfg);

    printf("=== OFFICIAL GREEDY EVALUATION (T=0) ===\n");
    printf("Model: %s (D=%d, L=%d, params=%d)\n\n", model_path, p->cfg.d_model, p->cfg.n_layers, p->n_floats);

    run_greedy_sample(p, c, bpe, "Timmy found a shiny red ball in the garden.", 40);
    run_greedy_sample(p, c, bpe, "The puppy was very hungry, so he", 40);

    tc_cache_free(c);
    tc_paramset_free(p);
    bpe_free(bpe);
    return 0;
}
