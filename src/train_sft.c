/* Supervised Fine-Tuning (SFT) & Conversational Alignment Driver
 * Trains TinyCoherent base checkpoint on dialogue exchanges using prompt-masked loss.
 */
#include "tcmodel.h"
#include "optim.h"
#include "tokenizer.h"
#include "bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    int *ids;
    int *targets;
    int T;
    int n_active;
} DialogueExample;

static char *read_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
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
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char *base_path = "build/model_rung4.bin";
    const char *data_path = "data/sft_dialogue.txt";
    const char *bpe_path = "data/bpe_merges.txt";
    const char *out_path = "build/model_sft.bin";
    int num_steps = 1500;
    int batch_size = 8;
    float base_lr = 1.5e-4f;
    float min_lr = 1.5e-5f;
    int warmup_steps = 100;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--base") && i + 1 < argc) base_path = argv[++i];
        else if (!strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else if (!strcmp(argv[i], "--merges") && i + 1 < argc) bpe_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) num_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) base_lr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch_size = atoi(argv[++i]);
    }

    printf("========================================================\n");
    printf("  TinyCoherent Supervised Fine-Tuning (SFT) Alignment\n");
    printf("========================================================\n");
    printf("Base checkpoint: %s\n", base_path);
    printf("Dialogue dataset: %s\n", data_path);
    printf("BPE merges:       %s\n", bpe_path);
    printf("Output target:    %s\n", out_path);
    printf("Steps: %d, Batch: %d, Base LR: %.2e\n", num_steps, batch_size, base_lr);

    /* Load Base Model */
    TCParamSet *p = tc_paramset_load(base_path);
    if (!p) {
        fprintf(stderr, "Failed to load base model from %s\n", base_path);
        return 1;
    }
    TCConfig cfg = p->cfg;
    printf("Loaded model: %d params (D=%d, L=%d, H=%d, V=%d, max_seq=%d)\n",
           p->n_floats, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.vocab_size, cfg.max_seq_len);

    /* Load Tokenizer */
    BPETokenizer *bpe = bpe_load(bpe_path);
    if (!bpe) {
        fprintf(stderr, "Failed to load BPE merges from %s\n", bpe_path);
        return 1;
    }

    /* Load Dialogue Dataset */
    long raw_len;
    char *raw_text = read_file(data_path, &raw_len);

    int cap_examples = 5000;
    DialogueExample *examples = malloc(sizeof(DialogueExample) * cap_examples);
    int num_examples = 0;

    char *cursor = raw_text;
    while (*cursor) {
        char *end_tag = strstr(cursor, "<|endoftext|>");
        if (!end_tag) break;

        size_t d_len = (size_t)(end_tag - cursor + strlen("<|endoftext|>"));
        char *d_str = malloc(d_len + 1);
        memcpy(d_str, cursor, d_len);
        d_str[d_len] = '\0';

        char *asst_pos = strstr(d_str, "Assistant:");
        if (asst_pos) {
            char *asst_start = asst_pos + strlen("Assistant:");
            while (*asst_start == ' ') asst_start++;
            size_t p_len = (size_t)(asst_start - d_str);

            char *p_str = malloc(p_len + 1);
            memcpy(p_str, d_str, p_len);
            p_str[p_len] = '\0';

            int p_toks[512], f_toks[512];
            int np = bpe_encode(bpe, p_str, p_toks, 512);
            int nf = bpe_encode(bpe, d_str, f_toks, 512);

            if (nf > 1 && nf <= cfg.max_seq_len && np > 0 && np < nf) {
                int T = nf - 1;
                DialogueExample *ex = &examples[num_examples++];
                ex->ids = malloc(sizeof(int) * T);
                ex->targets = malloc(sizeof(int) * T);
                ex->T = T;
                ex->n_active = 0;

                for (int t = 0; t < T; t++) {
                    ex->ids[t] = f_toks[t];
                    if (t < np - 1) {
                        ex->targets[t] = -1; /* Masked prompt token */
                    } else {
                        ex->targets[t] = f_toks[t + 1]; /* Active response token */
                        ex->n_active++;
                    }
                }

                if (num_examples >= cap_examples) {
                    free(p_str);
                    free(d_str);
                    break;
                }
            }
            free(p_str);
        }
        free(d_str);
        cursor = end_tag + strlen("<|endoftext|>");
        while (*cursor == '\n' || *cursor == '\r' || *cursor == ' ') cursor++;
    }
    free(raw_text);

    printf("Parsed %d dialogue examples for SFT (max sequence length = %d)\n",
           num_examples, cfg.max_seq_len);
    if (num_examples == 0) {
        fprintf(stderr, "Error: No valid dialogue examples parsed!\n");
        return 1;
    }

    /* Setup optimizer and worker caches */
    TCAdam *adam = tc_adam_create(cfg, base_lr);

    TCParamSet *grads[16];
    TCCache *caches[16];
    unsigned int thread_seeds[16];
    for (int b = 0; b < batch_size; b++) {
        grads[b] = tc_paramset_create(cfg);
        caches[b] = tc_cache_create(cfg);
        thread_seeds[b] = 42 + b * 1009;
    }

    TCParamSet **grads_ptr = grads;
    TCCache **caches_ptr = caches;
    unsigned int *seeds_ptr = thread_seeds;
    float batch_losses[16];
    float *losses_ptr = batch_losses;

    double t0 = get_time_sec();
    float running_loss = 0.0f;
    long total_tokens_trained = 0;

    printf("Beginning SFT training with prompt masking...\n");

    for (int step = 1; step <= num_steps; step++) {
        /* Learning rate schedule: linear warmup + cosine decay */
        float lr;
        if (step <= warmup_steps) {
            lr = base_lr * ((float)step / (float)warmup_steps);
        } else {
            float progress = (float)(step - warmup_steps) / (float)(num_steps - warmup_steps);
            lr = min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cosf(3.14159265f * progress));
        }
        adam->lr = lr;

#ifdef __APPLE__
        dispatch_apply(batch_size, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t b) {
            tc_paramset_zero(grads_ptr[b]);
            int idx = rand_r(&seeds_ptr[b]) % num_examples;
            const DialogueExample *ex = &examples[idx];
            float loss = 0.0f;
            tc_forward(p, caches_ptr[b], ex->ids, ex->T, ex->targets, &loss);
            tc_backward(p, grads_ptr[b], caches_ptr[b], ex->ids, ex->T, ex->targets);
            losses_ptr[b] = loss;
        });
#else
        for (int b = 0; b < batch_size; b++) {
            tc_paramset_zero(grads[b]);
            int idx = rand() % num_examples;
            const DialogueExample *ex = &examples[idx];
            float loss = 0.0f;
            tc_forward(p, caches[b], ex->ids, ex->T, ex->targets, &loss);
            tc_backward(p, grads[b], caches[b], ex->ids, ex->T, ex->targets);
            batch_losses[b] = loss;
        }
#endif

        /* Accumulate gradients across batch */
        float step_loss = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            step_loss += batch_losses[b];
            if (b > 0) {
                for (int i = 0; i < p->n_floats; i++) grads[0]->buf[i] += grads[b]->buf[i];
            }
        }
        step_loss /= (float)batch_size;

        /* Scale by 1 / batch_size */
        float inv_b = 1.0f / (float)batch_size;
        for (int i = 0; i < p->n_floats; i++) grads[0]->buf[i] *= inv_b;

        /* Gradient clipping: L2 norm <= 1.0 */
        float norm_sq = 0.0f;
        for (int i = 0; i < p->n_floats; i++) norm_sq += grads[0]->buf[i] * grads[0]->buf[i];
        float grad_norm = sqrtf(norm_sq);
        if (grad_norm > 1.0f) {
            float clip_scale = 1.0f / grad_norm;
            for (int i = 0; i < p->n_floats; i++) grads[0]->buf[i] *= clip_scale;
        }

        /* Optimizer step */
        tc_adam_step(adam, p, grads[0]);

        running_loss += step_loss;
        total_tokens_trained += batch_size * 35; /* approx response tokens */

        if (step == 1 || step % 100 == 0 || step == num_steps) {
            double elapsed = get_time_sec() - t0;
            float avg_loss = (step == 1) ? step_loss : (running_loss / (step % 100 == 0 ? 100 : (step % 100)));
            printf("  step %4d/%d  loss %6.4f  lr %.2e  |grad| %.3f  (%.1f steps/s)\n",
                   step, num_steps, avg_loss, lr, grad_norm, (double)step / elapsed);
            running_loss = 0.0f;
        }
    }

    double total_time = get_time_sec() - t0;
    printf("SFT training completed in %.2f seconds (~%ld active response tokens trained).\n",
           total_time, total_tokens_trained);

    /* Save fine-tuned checkpoint */
    printf("Saving aligned SFT model to %s...\n", out_path);
    tc_paramset_save(p, out_path);

    /* Verify round-trip load */
    TCParamSet *p_verify = tc_paramset_load(out_path);
    if (!p_verify) {
        fprintf(stderr, "Error: Saved model failed to verify!\n");
        return 1;
    }
    printf("Successfully verified fine-tuned model checkpoint (%d parameters).\n", p_verify->n_floats);
    tc_paramset_free(p_verify);

    /* Cleanup */
    for (int i = 0; i < num_examples; i++) {
        free(examples[i].ids);
        free(examples[i].targets);
    }
    free(examples);
    for (int b = 0; b < batch_size; b++) {
        tc_paramset_free(grads[b]);
        tc_cache_free(caches[b]);
    }
    tc_adam_free(adam);
    tc_paramset_free(p);
    bpe_free(bpe);

    return 0;
}
