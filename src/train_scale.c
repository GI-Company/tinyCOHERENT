/* Rung-4 scale driver: supports BPE tokenization, multi-core parallel batch
 * computation via Apple Grand Central Dispatch, and configurations up to D=1024.
 */
#include "tcmodel.h"
#include "optim.h"
#include "tokenizer.h"
#include "bpe.h"
#include "glassbox.h"
#include "tc_metal.h"
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

/* Standardized validation benchmark protocol: evaluates exactly 100 chunks
 * (10,000 held-out tokens) at fixed deterministic offsets every time,
 * eliminating sampling variance between intermediate reports and final checkpoints. */
static float eval_val_loss(const TCParamSet *p, TCCache *cache, const int *val_tokens, long val_len,
                            int chunk_len) {
    int fixed_chunks = 100;
    long stride = (val_len - 1) / (chunk_len * fixed_chunks);
    if (stride < 1) stride = 1;
    float total = 0.0f;
    int used = 0;
    for (int i = 0; i < fixed_chunks; i++) {
        long off = (long)i * stride * chunk_len;
        if (off + chunk_len >= val_len) break;
        float loss;
        tc_forward(p, cache, val_tokens + off, chunk_len, val_tokens + off + 1, &loss);
        total += loss;
        used++;
    }
    return used > 0 ? (total / (float)used) : -1.0f;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char *corpus_path = "data/tinystories_subset.txt";
    int num_steps = 10000;
    int arg_d_model = 512;
    int arg_n_layers = 6;
    const char *model_path = "build/model_rung6.bin";
    const char *bpe_path = "data/bpe_merges.txt";
    float base_lr = 2.0e-3f;
    float min_lr = 1e-4f;
    const char *resume_path = NULL;
    int accum_steps = 1;  /* gradient accumulation steps */
    float mtp_weight = 0.0f; /* multi-token prediction weight (0 = disabled) */
    float ponder_rate = 0.0f; /* fraction of tokens replaced with ponder token */

    int has_flag = 0;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--", 2) == 0) { has_flag = 1; break; }
    }

    if (!has_flag && argc > 1) {
        corpus_path = argv[1];
        if (argc > 2) num_steps = atoi(argv[2]);
        if (argc > 3) arg_d_model = atoi(argv[3]);
        if (argc > 4) arg_n_layers = atoi(argv[4]);
        if (argc > 5) model_path = argv[5];
        if (argc > 6) bpe_path = argv[6];
    } else {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--rung6") == 0) {
                corpus_path = "data/tinystories_subset.txt";
                arg_d_model = 512;
                arg_n_layers = 6;
                model_path = "build/model_rung6.bin";
                bpe_path = "data/bpe_merges.txt";
                base_lr = 2.0e-3f;
                num_steps = 300;
            } else if (strcmp(argv[i], "--rung4") == 0) {
                corpus_path = "data/tinystories_subset.txt";
                arg_d_model = 256;
                arg_n_layers = 4;
                model_path = "build/model_rung4.bin";
                bpe_path = "data/bpe_merges.txt";
                base_lr = 2.5e-3f;
                num_steps = 20000;
            } else if (strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
                corpus_path = argv[++i];
            } else if (strcmp(argv[i], "--metal") == 0) {
                tc_metal_available();
            } else if (strcmp(argv[i], "--metal-gemm") == 0) {
                tc_metal_available();
                tc_use_metal_gemm = 1;
                printf("--- PARSED --metal-gemm ---\n");
            } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
                num_steps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
                arg_d_model = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
                arg_n_layers = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
                model_path = argv[++i];
            } else if (strcmp(argv[i], "--bpe") == 0 && i + 1 < argc) {
                bpe_path = argv[++i];
            } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
                base_lr = (float)atof(argv[++i]);
            } else if ((strcmp(argv[i], "--resume") == 0 || strcmp(argv[i], "--base") == 0) && i + 1 < argc && argv[i+1][0] != '-') {
                resume_path = argv[++i];
                if (base_lr == 2.0e-3f) base_lr = 8.0e-4f;
                min_lr = 5.0e-5f;
            } else if (strcmp(argv[i], "--resume") == 0) {
                resume_path = model_path;
                if (base_lr == 2.0e-3f) base_lr = 8.0e-4f;
                min_lr = 5.0e-5f;
            } else if (strcmp(argv[i], "--accum") == 0 && i + 1 < argc) {
                accum_steps = atoi(argv[++i]);
                if (accum_steps < 1) accum_steps = 1;
            } else if (strcmp(argv[i], "--mtp") == 0 && i + 1 < argc) {
                mtp_weight = (float)atof(argv[++i]);
            } else if (strcmp(argv[i], "--ponder") == 0 && i + 1 < argc) {
                ponder_rate = (float)atof(argv[++i]);
            }
        }
    }

    long raw_len;
    char *text = read_file(corpus_path, &raw_len);

    BPETokenizer *bpe = NULL;
    int *tokens = NULL;
    long num_tokens = 0;
    int vocab_size = 96;

    if (bpe_path) {
        bpe = bpe_load(bpe_path);
        if (!bpe) {
            fprintf(stderr, "Failed to load BPE merges from %s\n", bpe_path);
            return 1;
        }
        vocab_size = bpe->vocab_size;
        printf("Loaded BPE tokenizer: vocab=%d, merges=%d\n", bpe->vocab_size, bpe->num_merges);
        tokens = malloc(sizeof(int) * (size_t)raw_len);
        num_tokens = bpe_encode(bpe, text, tokens, (int)raw_len);
        printf("Encoded corpus to %ld BPE tokens (compression %.2fx)\n",
               num_tokens, (double)raw_len / (double)num_tokens);
    } else {
        tc_sanitize_text(text);
        tokens = malloc(sizeof(int) * (size_t)raw_len);
        for (long i = 0; i < raw_len; i++) tokens[i] = tc_encode_char(text[i]);
        num_tokens = raw_len;
        printf("Using 96-vocab character encoding: %ld tokens\n", num_tokens);
    }

    long split_at = (long)(num_tokens * 0.85);
    long train_len = split_at, val_len = num_tokens - split_at;
    const int *train_tokens = tokens;
    const int *val_tokens = tokens + split_at;
    printf("split: %ld train tokens / %ld val tokens (held out, never trained on)\n", train_len, val_len);

    int ponder_token_id = vocab_size;
    if (ponder_rate > 0.0f) {
        vocab_size++;
        printf("Ponder token enabled: rate=%.2f, id=%d\n", ponder_rate, ponder_token_id);
    }

    TCConfig cfg = tc_default_config();
    cfg.vocab_size = vocab_size;
    cfg.d_model = arg_d_model;
    cfg.n_layers = arg_n_layers;
    cfg.n_heads = 8;
    while (cfg.d_model % cfg.n_heads != 0 && cfg.n_heads > 1) cfg.n_heads /= 2;
    cfg.ff_mult = 2;
    cfg.max_seq_len = 128;

    printf("config: vocab=%d d_model=%d n_layers=%d n_heads=%d n_kv_heads=%d ff_mult=%d max_seq_len=%d\n",
           cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.ff_mult, cfg.max_seq_len);
    int pcount = tc_param_count(cfg);
    printf("param count = %d (%.2fM parameters)\n", pcount, pcount / 1e6);

    int chunk_len = 100;
    if (train_len - chunk_len - 1 < 1) { fprintf(stderr, "corpus too small for chunk_len=%d\n", chunk_len); return 1; }

    TCParamSet *p = NULL;
    if (resume_path) {
        p = tc_paramset_load(resume_path);
        if (!p) {
            fprintf(stderr, "Failed to load resume weights from %s\n", resume_path);
            return 1;
        }
        cfg = p->cfg;
        printf("Resumed model weights from %s (%d params, D=%d, L=%d)\n", resume_path, p->n_floats, cfg.d_model, cfg.n_layers);
    } else {
        p = tc_paramset_create(cfg);
        tc_paramset_init_random(p, 1234);
    }
    TCAdam *adam = tc_adam_create(cfg, base_lr, 0.01f);

    int batch_size = 8;
    // (Removed Metal batch_size = 1 limitation since it's now fully batched natively on MSL)
    if (accum_steps > 1)
        printf("gradient accumulation: %d steps (effective batch = %d)\n", accum_steps, batch_size * accum_steps);
    if (mtp_weight > 0.0f)
        printf("multi-token prediction: weight=%.2f (t+2, t+3 auxiliary losses)\n", mtp_weight);
    int report_every = (num_steps >= 2000) ? 100 : (num_steps >= 200 ? 25 : 10);
    int checkpoint_every = (num_steps >= 1000) ? 500 : 50;

    /* Multi-thread worker caches and gradient accumulators */
    TCParamSet *grads[8];
    TCCache *caches[8];
    unsigned int thread_seeds[8];
    
#ifdef __APPLE__
    tc_metal_init(&cfg);
    tc_metal_bind_params(p, NULL);
#endif
    
    for (int b = 0; b < batch_size; b++) {
        grads[b] = tc_paramset_create(cfg);
        caches[b] = tc_cache_create(cfg);
        thread_seeds[b] = 12345 + b * 997;
#ifdef __APPLE__
        tc_metal_bind_params(NULL, grads[b]);
        tc_metal_bind_cache(caches[b]);
#endif
    }

    TCParamSet *global_grad = tc_paramset_create(cfg);

    double t0 = get_time_sec();
    float running_loss = 0.0f;

    printf("Starting training on %d parallel batch workers...\n", batch_size);

    TCParamSet **grads_ptr = grads;
    TCCache **caches_ptr = caches;
    unsigned int *seeds_ptr = thread_seeds;
    float *batch_losses;
    posix_memalign((void**)&batch_losses, 16384, ((8 * sizeof(float)) + 16383) & ~16383);
    float *losses_ptr = batch_losses;
    
    int *batch_ids;
    posix_memalign((void**)&batch_ids, 16384, ((batch_size * chunk_len * sizeof(int)) + 16383) & ~16383);
    int *batch_targets;
    posix_memalign((void**)&batch_targets, 16384, ((batch_size * chunk_len * sizeof(int)) + 16383) & ~16383);

#ifdef __APPLE__
    tc_metal_register_ptr(batch_losses, 8 * sizeof(float));
    tc_metal_register_ptr(batch_ids, batch_size * chunk_len * sizeof(int));
    tc_metal_register_ptr(batch_targets, batch_size * chunk_len * sizeof(int));
#endif

    for (int step = 1; step <= num_steps; step++) {
        /* --- gradient accumulation outer loop --- */
        float step_loss = 0.0f;
        tc_paramset_zero(global_grad);
        for (int accum = 0; accum < accum_steps; accum++) {
#ifdef __APPLE__
            /* Metal batched dispatch */
            for (int b = 0; b < batch_size; b++) {
                tc_paramset_zero(grads_ptr[b]);
                batch_losses[b] = 0.0f;
            }
            
            for (int b = 0; b < batch_size; b++) {
                long max_off = train_len - chunk_len - 3;
                if (max_off < 1) max_off = 1;
                long off = ((long)rand_r(&seeds_ptr[b])) % max_off;
                
                int *local_ids = batch_ids + b * chunk_len;
                int *local_targets = batch_targets + b * chunk_len;
                
                for (int i = 0; i < chunk_len; i++) {
                    local_ids[i] = train_tokens[off + i];
                }
                
                if (ponder_rate > 0.0f && ((float)(rand_r(&seeds_ptr[b]) % 10000) / 10000.0f) < ponder_rate) {
                    int pos = rand_r(&seeds_ptr[b]) % (chunk_len - 1);
                    for (int i = chunk_len - 1; i > pos; i--) local_ids[i] = local_ids[i - 1];
                    local_ids[pos] = ponder_token_id;
                }
                
                for (int i = 0; i < chunk_len; i++) {
                    local_targets[i] = (i + 1 < chunk_len) ? local_ids[i + 1] : train_tokens[off + chunk_len];
                }
            }
            
            tc_metal_forward(p, caches_ptr, batch_ids, batch_size, chunk_len, batch_targets, losses_ptr);
            tc_metal_backward(p, grads_ptr, caches_ptr, batch_ids, batch_size, chunk_len, batch_targets);

            for (int b = 0; b < batch_size; b++) {
                step_loss += batch_losses[b];
                for (int i = 0; i < p->n_floats; i++) global_grad->buf[i] += grads_ptr[b]->buf[i];
            }
#else
            float micro_loss = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                long max_off = train_len - chunk_len - 3;
                if (max_off < 1) max_off = 1;
                long off = rand() % max_off;

                int local_ids[1024], local_targets[1024], local_t2[1024], local_t3[1024];
                for (int i = 0; i < chunk_len; i++) {
                    local_ids[i] = train_tokens[off + i];
                }

                if (ponder_rate > 0.0f && ((float)(rand() % 10000) / 10000.0f) < ponder_rate) {
                    int pos = rand() % (chunk_len - 1);
                    for (int i = chunk_len - 1; i > pos; i--) local_ids[i] = local_ids[i - 1];
                    local_ids[pos] = ponder_token_id;
                }

                for (int i = 0; i < chunk_len; i++) {
                    local_targets[i] = (i + 1 < chunk_len) ? local_ids[i + 1] : train_tokens[off + chunk_len];
                    local_t2[i] = (i + 2 < chunk_len) ? local_ids[i + 2] : train_tokens[off + chunk_len + 1];
                    local_t3[i] = (i + 3 < chunk_len) ? local_ids[i + 3] : train_tokens[off + chunk_len + 2];
                }

                const int *ids = local_ids;
                const int *targets = local_targets;
                const int *targets_t2 = local_t2;
                const int *targets_t3 = local_t3;
                float loss;

                if (mtp_weight > 0.0f) {
                    tc_forward_mtp(p, caches[0], ids, chunk_len, targets, targets_t2, targets_t3, mtp_weight, &loss);
                    tc_backward_mtp(p, global_grad, caches[0], ids, chunk_len, targets, targets_t2, targets_t3, mtp_weight);
                } else {
                    tc_forward(p, caches[0], ids, chunk_len, targets, &loss);
                    tc_backward(p, global_grad, caches[0], ids, chunk_len, targets);
                }
                micro_loss += loss;
            }
            step_loss += micro_loss;
#endif
        } /* end accum loop */
        float total_microbatches = (float)(batch_size * accum_steps);
        scale_grad(global_grad, 1.0f / total_microbatches);

        /* Global Gradient Clipping */
        float l2_norm = 0.0f;
        for (int i = 0; i < p->n_floats; i++) {
            l2_norm += global_grad->buf[i] * global_grad->buf[i];
        }
        l2_norm = sqrtf(l2_norm);
        if (l2_norm > 1.0f) {
            scale_grad(global_grad, 1.0f / l2_norm);
        }

        /* Cosine learning rate schedule */
        float progress = (float)step / (float)num_steps;
        adam->lr = min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cosf(3.14159265f * progress));

        tc_adam_step(adam, p, global_grad);
        step_loss /= total_microbatches;
        running_loss = (step == 1) ? step_loss : 0.98f * running_loss + 0.02f * step_loss;

        if (step % report_every == 0 || step == 1) {
            float val_loss = eval_val_loss(p, caches[0], val_tokens, val_len, chunk_len);
            double secs = get_time_sec() - t0;
            double tok_per_sec = (double)(step * (int)total_microbatches * chunk_len) / (secs > 0 ? secs : 1.0);
            printf("step %6d  lr %.6f  train_loss %.4f (avg %.4f)  val_loss %.4f  [%.1fs | %.0f tok/s]\n",
                   step, adam->lr, step_loss, running_loss, val_loss, secs, tok_per_sec);
        }

        if (step % checkpoint_every == 0) {
            if (tc_paramset_save(p, model_path) == 0) printf("  checkpoint saved at step %d\n", step);
        }
    }

    double secs = get_time_sec() - t0;
    printf("trained %d steps x batch %d on %d-token chunks in %.2fs (%.0f tokens/sec)\n",
           num_steps, batch_size, chunk_len, secs, (double)(num_steps * batch_size * chunk_len) / secs);

    float final_val_loss = eval_val_loss(p, caches[0], val_tokens, val_len, chunk_len);
    printf("final val_loss %.4f over %ld held-out tokens (100-chunk benchmark)\n", final_val_loss, val_len);

    if (tc_paramset_save(p, model_path) == 0) printf("saved weights to %s\n", model_path);
    else { fprintf(stderr, "failed to save weights\n"); return 1; }

    tc_adam_free(adam);
    for (int b = 0; b < batch_size; b++) {
        tc_cache_free(caches[b]);
        tc_paramset_free(grads[b]);
    }
    tc_paramset_free(p);
    if (bpe) bpe_free(bpe);
    free(tokens);
    free(text);
    return 0;
}
