/* Embedder milestone: the "second specialist type" is the same trained
 * hybrid-block network, just read through a different head -- mean-pool
 * + L2-normalize the final hidden state instead of projecting to vocab
 * logits (see tc_encode/tc_forward/tc_embed in tcmodel.c). This proves
 * the pooled representation actually discriminates between sentences
 * (same-template sentences should be closer to each other than to
 * unrelated gibberish) and demonstrates the embedder's own glass-box
 * method: occlusion on cosine similarity. */
#include "tcmodel.h"
#include "tokenizer.h"
#include "glassbox.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int encode(const char *text, int *ids, int cap) {
    char buf[512];
    strncpy(buf, text, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    tc_sanitize_text(buf);
    int len = (int)strlen(buf);
    if (len > cap) len = cap;
    for (int i = 0; i < len; i++) ids[i] = tc_encode_char(buf[i]);
    return len;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "build/model.bin";
    TCParamSet *p = tc_paramset_load(model_path);
    if (!p) {
        fprintf(stderr, "could not load %s -- run `make train` first\n", model_path);
        return 1;
    }
    TCCache *cache = tc_cache_create(p->cfg);
    int D = p->cfg.d_model;

    const char *sentences[] = {
        "the cat sat on the mat quietly.",
        "the dog ran across the log quickly.",
        "the fox jumped over the hill happily.",
        "xqz plmr wvbn tkjh gfds aqwe zxcv.",
    };
    const char *labels[] = { "s1 (template)", "s2 (template)", "s3 (template)", "s4 (gibberish)" };
    int n = 4;

    float embeds[4][256];
    for (int i = 0; i < n; i++) {
        int ids[256];
        int len = encode(sentences[i], ids, p->cfg.max_seq_len);
        tc_embed(p, cache, ids, len, embeds[i]);
    }

    printf("sentences:\n");
    for (int i = 0; i < n; i++) printf("  %s: \"%s\"\n", labels[i], sentences[i]);

    printf("\npairwise cosine similarity:\n%-16s", "");
    for (int j = 0; j < n; j++) printf("%-16s", labels[j]);
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%-16s", labels[i]);
        for (int j = 0; j < n; j++) {
            float sim = tc_cosine(embeds[i], embeds[j], D);
            printf("%-16.3f", sim);
        }
        printf("\n");
    }

    printf("\n--- embedder glass-box: occlusion on cosine similarity ---\n");
    printf("sentence: \"%s\"\n\n", sentences[0]);
    int ids0[256];
    int len0 = encode(sentences[0], ids0, p->cfg.max_seq_len);
    TCEmbedOcclusionStep steps[256];
    int ns = tc_explain_embedding(p, cache, ids0, len0, steps);
    tc_embed_glassbox_print(steps, ns);

    tc_cache_free(cache);
    tc_paramset_free(p);
    return 0;
}
