/* Milestone driver: tokenize a small corpus, train one ~8k-param
 * TinyCoherent specialist on CPU, save/reload the weights, and print a
 * glass-box (per-character surprise + top-3) trace on a genuinely
 * held-out prompt.
 *
 * The corpus is split into train (first ~85%) and val (last ~15%, split
 * on a sentence boundary so no sentence spans both halves) once, up
 * front. Training only ever samples chunks from the train region; val
 * loss is computed separately over the whole val region every
 * report_every steps and printed next to train loss. Until val loss can
 * go the wrong way, a train loss number alone isn't evidence of
 * anything except that the optimizer is doing its job. */
#include "tcmodel.h"
#include "optim.h"
#include "tokenizer.h"
#include "glassbox.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static void scale_grad(TCParamSet *grad, float s) {
    for (int i = 0; i < grad->n_floats; i++) grad->buf[i] *= s;
}

/* Full, deterministic, non-overlapping tiling of val_tokens -- not a random
 * subsample -- so the number means the same thing on every call. */
static float eval_val_loss(const TCParamSet *p, TCCache *cache, const int *val_tokens, long val_len, int chunk_len) {
    float total = 0.0f;
    int n_chunks = 0;
    long off = 0;
    while (off + chunk_len < val_len) {
        float loss;
        tc_forward(p, cache, val_tokens + off, chunk_len, val_tokens + off + 1, &loss);
        total += loss;
        n_chunks++;
        off += chunk_len;
    }
    return n_chunks > 0 ? total / n_chunks : -1.0f;
}

int main(int argc, char **argv) {
    const char *corpus_path = argc > 1 ? argv[1] : "data/corpus.txt";
    const char *model_path = "build/model.bin";

    long len;
    char *text = read_file(corpus_path, &len);
    tc_sanitize_text(text);
    int *tokens = malloc(sizeof(int) * len);
    for (long i = 0; i < len; i++) tokens[i] = tc_encode_char(text[i]);
    printf("corpus: %s (%ld bytes)\n", corpus_path, len);

    /* Split on a '.' boundary near the 85% mark so no sentence straddles
     * train and val. */
    long split_at = (long)(len * 0.85);
    while (split_at < len && text[split_at] != '.') split_at++;
    if (split_at < len) split_at++;
    if (split_at >= len || split_at < 1) split_at = len; /* no '.' found: degrade to no held-out set */
    long train_len = split_at, val_len = len - split_at;
    int *train_tokens = tokens;
    int *val_tokens = tokens + split_at;
    printf("split: %ld train bytes / %ld val bytes (held out, never trained on)\n", train_len, val_len);

    TCConfig cfg = tc_default_config();
    printf("config: vocab=%d d_model=%d n_layers=%d n_heads=%d ff_mult=%d max_seq_len=%d\n",
           cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.ff_mult, cfg.max_seq_len);
    printf("param count = %d (target band ~7500-9000)\n", tc_param_count(cfg));

    TCParamSet *p = tc_paramset_create(cfg);
    tc_paramset_init_random(p, 1234);
    TCParamSet *grad = tc_paramset_create(cfg);
    TCCache *cache = tc_cache_create(cfg);
    TCAdam *adam = tc_adam_create(cfg, 5e-3f, 0.0f);

    int chunk_len = 48, batch_size = 4, num_steps = 3000, report_every = 200;
    srand(7);
    clock_t t0 = clock();
    float running_loss = 0.0f;
    for (int step = 1; step <= num_steps; step++) {
        float step_loss = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            long max_off = train_len - chunk_len - 1;
            long off = rand() % max_off;
            int *ids = train_tokens + off;
            int *targets = train_tokens + off + 1;
            float loss;
            tc_forward(p, cache, ids, chunk_len, targets, &loss);
            tc_backward(p, grad, cache, ids, chunk_len, targets);
            step_loss += loss;
        }
        scale_grad(grad, 1.0f / batch_size);
        tc_adam_step(adam, p, grad);
        step_loss /= batch_size;
        running_loss = (step == 1) ? step_loss : 0.98f * running_loss + 0.02f * step_loss;
        if (step % report_every == 0 || step == 1) {
            float val_loss = eval_val_loss(p, cache, val_tokens, val_len, chunk_len);
            printf("step %5d  train_loss %.4f (running avg %.4f)  val_loss %.4f\n",
                   step, step_loss, running_loss, val_loss);
        }
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("trained %d steps x batch %d on %d-token chunks in %.2fs on one CPU core\n",
           num_steps, batch_size, chunk_len, secs);
    float final_val_loss = eval_val_loss(p, cache, val_tokens, val_len, chunk_len);
    printf("final val_loss %.4f over %ld held-out bytes (never trained on)\n", final_val_loss, val_len);

    if (tc_paramset_save(p, model_path) == 0)
        printf("saved weights to %s\n", model_path);
    else
        fprintf(stderr, "failed to save weights\n");

    TCParamSet *p2 = tc_paramset_load(model_path);
    if (!p2) { fprintf(stderr, "failed to reload weights\n"); return 1; }
    float loss_orig, loss_reloaded;
    tc_forward(p, cache, val_tokens, chunk_len, val_tokens + 1, &loss_orig);
    tc_forward(p2, cache, val_tokens, chunk_len, val_tokens + 1, &loss_reloaded);
    printf("save/load check (on held-out data): loss before=%.6f after reload=%.6f (should match)\n",
           loss_orig, loss_reloaded);

    printf("\n--- glass-box trace on a genuinely held-out prompt (from the val split) ---\n");
    int plen = 55;
    if (plen > val_len) plen = (int)val_len;
    if (plen > cfg.max_seq_len) plen = cfg.max_seq_len;
    char pbuf[256];
    for (int i = 0; i < plen; i++) pbuf[i] = tc_decode_id(val_tokens[i]);
    pbuf[plen] = 0;
    int pids[256];
    for (int i = 0; i < plen; i++) pids[i] = val_tokens[i];

    TCGlassBoxStep steps[256];
    int n = tc_explain_generate(p2, cache, pids, plen, steps);
    printf("prompt: \"%.*s\"\n\n", plen, pbuf);
    tc_glassbox_print(steps, n);

    tc_paramset_free(p2);
    tc_adam_free(adam);
    tc_cache_free(cache);
    tc_paramset_free(grad);
    tc_paramset_free(p);
    free(tokens);
    free(text);
    return 0;
}
