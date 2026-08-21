/* Rung-2 scale driver: same mechanism as train.c (train/val split on a
 * sentence boundary, held-out glass-box prompt, save/load check), but at
 * a larger config and with periodic checkpointing for longer runs. Kept
 * as a SEPARATE binary and a separate config/weights file (build/model_scale.bin)
 * on purpose -- train.c and build/model.bin are the verified ~8k oracle
 * and must never be touched by this. Every attribution gate (gradcheck,
 * faithcheck, known_answer, degrade_test) still targets the tiny oracle
 * unless explicitly pointed at this run's output. */
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

/* Evaluates up to max_chunks non-overlapping chunks, evenly strided across
 * the whole val region -- not just a prefix -- so cost stays bounded
 * regardless of val set size while still sampling representatively. Tiling
 * the *entire* val set (the original approach, fine at ~2KB val scale)
 * cost ~22s per call alone at ~600KB val scale here -- roughly an hour of
 * pure eval overhead across a 100k-step run with report_every=500. */
static float eval_val_loss(const TCParamSet *p, TCCache *cache, const int *val_tokens, long val_len,
                            int chunk_len, int max_chunks) {
    long total_possible = (val_len - 1) / chunk_len;
    if (total_possible < 1) return -1.0f;
    long n_chunks = total_possible < max_chunks ? total_possible : max_chunks;
    long stride = total_possible / n_chunks;
    if (stride < 1) stride = 1;

    float total = 0.0f;
    long used = 0;
    for (long i = 0; i < n_chunks; i++) {
        long off = i * stride * chunk_len;
        if (off + chunk_len >= val_len) break;
        float loss;
        tc_forward(p, cache, val_tokens + off, chunk_len, val_tokens + off + 1, &loss);
        total += loss;
        used++;
    }
    return used > 0 ? total / used : -1.0f;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0); /* line-buffer even when redirected to a file/log, so a long run's progress is visible live */
    const char *corpus_path = argc > 1 ? argv[1] : "data/corpus.txt";
    const char *model_path = "build/model_scale.bin";
    int num_steps = argc > 2 ? atoi(argv[2]) : 20000;

    long len;
    char *text = read_file(corpus_path, &len);
    tc_sanitize_text(text);
    int *tokens = malloc(sizeof(int) * len);
    for (long i = 0; i < len; i++) tokens[i] = tc_encode_char(text[i]);
    printf("corpus: %s (%ld bytes)\n", corpus_path, len);

    long split_at = (long)(len * 0.85);
    while (split_at < len && text[split_at] != '.') split_at++;
    if (split_at < len) split_at++;
    if (split_at >= len || split_at < 1) split_at = len;
    long train_len = split_at, val_len = len - split_at;
    int *train_tokens = tokens;
    int *val_tokens = tokens + split_at;
    printf("split: %ld train bytes / %ld val bytes (held out, never trained on)\n", train_len, val_len);

    TCConfig cfg;
    cfg.vocab_size = 96; cfg.d_model = 64; cfg.n_layers = 4;
    cfg.n_heads = 4; cfg.ff_mult = 2; cfg.max_seq_len = 128;
    printf("config: vocab=%d d_model=%d n_layers=%d n_heads=%d ff_mult=%d max_seq_len=%d\n",
           cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.ff_mult, cfg.max_seq_len);
    printf("param count = %d\n", tc_param_count(cfg));

    int chunk_len = 100;
    if (train_len - chunk_len - 1 < 1) { fprintf(stderr, "corpus too small for chunk_len=%d\n", chunk_len); return 1; }

    TCParamSet *p = tc_paramset_create(cfg);
    tc_paramset_init_random(p, 1234);
    TCParamSet *grad = tc_paramset_create(cfg);
    TCCache *cache = tc_cache_create(cfg);
    TCAdam *adam = tc_adam_create(cfg, 3e-3f);

    int batch_size = 8, report_every = 500, checkpoint_every = 5000;
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
            float val_loss = eval_val_loss(p, cache, val_tokens, val_len, chunk_len, 300);
            double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
            printf("step %6d  train_loss %.4f (running avg %.4f)  val_loss %.4f  [%.1fs elapsed]\n",
                   step, step_loss, running_loss, val_loss, secs);
        }
        if (step % checkpoint_every == 0) {
            if (tc_paramset_save(p, model_path) == 0) printf("  checkpoint saved at step %d\n", step);
        }
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("trained %d steps x batch %d on %d-token chunks in %.2fs\n", num_steps, batch_size, chunk_len, secs);
    float final_val_loss = eval_val_loss(p, cache, val_tokens, val_len, chunk_len, 2000);
    printf("final val_loss %.4f over %ld held-out bytes\n", final_val_loss, val_len);

    if (tc_paramset_save(p, model_path) == 0) printf("saved weights to %s\n", model_path);
    else { fprintf(stderr, "failed to save weights\n"); return 1; }

    printf("\n--- glass-box trace on a genuinely held-out prompt (from the val split) ---\n");
    int plen = 60;
    if (plen > val_len) plen = (int)val_len;
    if (plen > cfg.max_seq_len) plen = cfg.max_seq_len;
    char pbuf[256];
    for (int i = 0; i < plen; i++) pbuf[i] = tc_decode_id(val_tokens[i]);
    pbuf[plen] = 0;
    int pids[256];
    for (int i = 0; i < plen; i++) pids[i] = val_tokens[i];

    TCGlassBoxStep steps[256];
    int n = tc_explain_generate(p, cache, pids, plen, steps);
    printf("prompt: \"%.*s\"\n\n", plen, pbuf);
    tc_glassbox_print(steps, n);

    tc_adam_free(adam);
    tc_cache_free(cache);
    tc_paramset_free(grad);
    tc_paramset_free(p);
    free(tokens);
    free(text);
    return 0;
}
