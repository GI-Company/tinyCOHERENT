/* Contrastive fine-tuning for the embedder head. Self-supervised, no
 * labels needed: split the training corpus into sentences, for each
 * anchor sentence make two independently-corrupted views (random token
 * substitution, a cheap stand-in for dropout-based augmentation), and
 * train so each view's embedding is closest to its own pair's embedding
 * and far from every other sentence's in the batch (in-batch-negatives
 * softmax / InfoNCE, same softmax-cross-entropy math already used and
 * verified for the generative head's vocab prediction).
 *
 * Warm-starts from the next-char-pretrained body (build/model.bin) so the
 * recurrence/attention machinery isn't learned from scratch -- only
 * pool_w starts untrained; the whole body is then free to drift under
 * this new objective. Saves to build/embedder.bin, a separate file from
 * the generative model since the two now diverge. */
#include "tcmodel.h"
#include "optim.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N_BATCH 8
#define MAX_SENT_LEN 60
#define MAX_SENTENCES 512

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

#define MAX_WORDS 10
#define MAX_WORD_LEN 16

typedef struct {
    int ids[MAX_SENT_LEN];
    int len;
    char text[MAX_SENT_LEN + 1];
    char content_words[MAX_WORDS][MAX_WORD_LEN]; /* non-stopword words, filled after stopword pass */
    int n_content_words;
} Sentence;

static int split_sentences(const char *text, Sentence *out, int cap) {
    int n = 0;
    const char *p = text;
    while (*p && n < cap) {
        const char *start = p;
        while (*p && *p != '.') p++;
        int raw_len = (int)(p - start);
        if (*p == '.') p++;
        while (*start == ' ') { start++; raw_len--; }
        if (raw_len > 3) {
            int len = raw_len < MAX_SENT_LEN - 1 ? raw_len : MAX_SENT_LEN - 1;
            for (int i = 0; i < len; i++) out[n].ids[i] = tc_encode_char(start[i]);
            out[n].len = len;
            memcpy(out[n].text, start, (size_t)len);
            out[n].text[len] = 0;
            out[n].n_content_words = 0;
            n++;
        }
    }
    return n;
}

/* Splits a sentence's text into lowercase words on spaces (already
 * lowercase/sanitized). */
static int split_words(const char *text, char words[][MAX_WORD_LEN], int cap) {
    int n = 0;
    const char *p = text;
    while (*p && n < cap) {
        while (*p == ' ') p++;
        if (!*p) break;
        int i = 0;
        while (*p && *p != ' ' && i < MAX_WORD_LEN - 1) words[n][i++] = *p++;
        while (*p && *p != ' ') p++; /* skip any overflow tail */
        words[n][i] = 0;
        if (i > 0) n++;
    }
    return n;
}

/* Global document-frequency was tried first and is wrong for this corpus:
 * a preposition bound 1:1 to one verb ("looked at", "jumped over", "slept
 * near") has, by construction, the EXACT same document frequency as that
 * verb -- no threshold on unigram frequency can tell them apart. Measured
 * on this corpus: "at"=0.218, "looked"=0.218; "over"=0.170, "jumped"=0.170;
 * "near"=0.155, "slept"=0.155; "across"=0.142, "ran"=0.142. A frequency
 * threshold that lets any of those verbs through as content necessarily
 * lets their bound preposition through too. A small curated closed-class
 * stopword list has no such blind spot and is standard practice for
 * exactly this reason -- it is also more general than the frequency
 * heuristic was, not less: it doesn't silently break on the next corpus
 * where a function word happens to be evenly distributed. */
static const char *STOPWORDS[] = {
    "a", "an", "the",
    "on", "in", "at", "by", "to", "of", "for", "with", "from",
    "near", "over", "under", "past", "across", "through",
    "into", "onto", "upon", "about", "against", "between", "among",
    "during", "before", "after", "behind", "beside", "since", "until",
    "and", "or", "but", "is", "was", "were", "be", "been", "this", "that",
    NULL
};

static int is_stopword(const char *w) {
    for (int i = 0; STOPWORDS[i]; i++) if (strcmp(STOPWORDS[i], w) == 0) return 1;
    return 0;
}

static void tag_content_words(Sentence *sentences, int n) {
    char sent_words[MAX_WORDS][MAX_WORD_LEN];
    for (int s = 0; s < n; s++) {
        int nw = split_words(sentences[s].text, sent_words, MAX_WORDS);
        sentences[s].n_content_words = 0;
        for (int w = 0; w < nw; w++) {
            if (!is_stopword(sent_words[w]) && sentences[s].n_content_words < MAX_WORDS)
                strcpy(sentences[s].content_words[sentences[s].n_content_words++], sent_words[w]);
        }
    }
}

static int shares_content_word(const Sentence *a, const Sentence *b) {
    for (int i = 0; i < a->n_content_words; i++)
        for (int j = 0; j < b->n_content_words; j++)
            if (strcmp(a->content_words[i], b->content_words[j]) == 0) return 1;
    return 0;
}

static unsigned int xr(unsigned int *s) {
    unsigned int x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}
static float xrf(unsigned int *s) { return (float)(xr(s) % 1000000) / 1000000.0f; }

static void corrupt(const int *src, int *dst, int T, float p, int vocab, unsigned int *rng) {
    for (int t = 0; t < T; t++) {
        dst[t] = src[t];
        if (xrf(rng) < p) dst[t] = (int)(xr(rng) % (unsigned int)vocab);
    }
}

/* Runs one contrastive batch: N anchors -> 2N corrupted views -> NxN
 * in-batch-negatives softmax-CE loss. Always accumulates into grad
 * (caller zeroes/steps). Returns the batch's mean loss. */
static float run_batch(const TCParamSet *p, TCParamSet *grad, TCCache **cacheA, TCCache **cacheB,
                        Sentence *sentences, int n_sentences, unsigned int *rng, float tau, float corrupt_p) {
    int D = p->cfg.d_model;
    int idxs[N_BATCH];
    for (int k = 0; k < N_BATCH; k++) idxs[k] = (int)(xr(rng) % (unsigned int)n_sentences);

    static int idsA[N_BATCH][MAX_SENT_LEN], idsB[N_BATCH][MAX_SENT_LEN];
    static float embA[N_BATCH][256], embB[N_BATCH][256];
    int lenA[N_BATCH], lenB[N_BATCH];

    for (int k = 0; k < N_BATCH; k++) {
        Sentence *sent = &sentences[idxs[k]];
        corrupt(sent->ids, idsA[k], sent->len, corrupt_p, p->cfg.vocab_size, rng);
        corrupt(sent->ids, idsB[k], sent->len, corrupt_p, p->cfg.vocab_size, rng);
        lenA[k] = sent->len; lenB[k] = sent->len;
        tc_embed(p, cacheA[k], idsA[k], lenA[k], embA[k]);
        tc_embed(p, cacheB[k], idsB[k], lenB[k], embB[k]);
    }

    /* False-negative masking: two different sentences that share a content
     * (non-stopword) word -- e.g. both mention "cat", or both end in "mat"
     * -- are excluded from row i's softmax denominator instead of being
     * forced apart as ordinary negatives. This is what stops the loss from
     * punishing same-template sentences for being similar; the true
     * positive (j==i) is never masked. */
    int mask[N_BATCH][N_BATCH];
    for (int i = 0; i < N_BATCH; i++)
        for (int j = 0; j < N_BATCH; j++)
            mask[i][j] = (i != j) && shares_content_word(&sentences[idxs[i]], &sentences[idxs[j]]);

    float sim[N_BATCH][N_BATCH];
    for (int i = 0; i < N_BATCH; i++)
        for (int j = 0; j < N_BATCH; j++)
            sim[i][j] = tc_cosine(embA[i], embB[j], D) / tau;

    float dsim[N_BATCH][N_BATCH];
    float loss = 0.0f;
    for (int i = 0; i < N_BATCH; i++) {
        float maxv = -1e30f;
        for (int j = 0; j < N_BATCH; j++) if (!mask[i][j] && sim[i][j] > maxv) maxv = sim[i][j];
        float probrow[N_BATCH], sumexp = 0.0f;
        for (int j = 0; j < N_BATCH; j++) {
            probrow[j] = mask[i][j] ? 0.0f : expf(sim[i][j] - maxv);
            sumexp += probrow[j];
        }
        for (int j = 0; j < N_BATCH; j++) probrow[j] /= sumexp;
        loss += -logf(fmaxf(probrow[i], 1e-9f));
        for (int j = 0; j < N_BATCH; j++)
            dsim[i][j] = mask[i][j] ? 0.0f : (probrow[j] - (j == i ? 1.0f : 0.0f)) / N_BATCH;
    }
    loss /= N_BATCH;

    static float dembA[N_BATCH][256], dembB[N_BATCH][256];
    memset(dembA, 0, sizeof(dembA));
    memset(dembB, 0, sizeof(dembB));
    for (int i = 0; i < N_BATCH; i++) {
        for (int j = 0; j < N_BATCH; j++) {
            float ds = dsim[i][j] / tau;
            for (int d = 0; d < D; d++) {
                dembA[i][d] += ds * embB[j][d];
                dembB[j][d] += ds * embA[i][d];
            }
        }
    }

    for (int k = 0; k < N_BATCH; k++) {
        tc_embed_backward(p, grad, cacheA[k], idsA[k], lenA[k], dembA[k]);
        tc_embed_backward(p, grad, cacheB[k], idsB[k], lenB[k], dembB[k]);
    }
    return loss;
}

int main(int argc, char **argv) {
    const char *corpus_path = "data/corpus.txt";
    const char *base_model_path = argc > 1 ? argv[1] : "build/model.bin";
    const char *out_path = "build/embedder.bin";

    long len;
    char *text = read_file(corpus_path, &len);
    tc_sanitize_text(text);
    Sentence sentences[MAX_SENTENCES];
    int n_sentences = split_sentences(text, sentences, MAX_SENTENCES);
    tag_content_words(sentences, n_sentences);
    printf("loaded %d sentences from %s\n", n_sentences, corpus_path);
    printf("example content words: \"%s\" -> [", sentences[0].text);
    for (int i = 0; i < sentences[0].n_content_words; i++) printf("%s%s", i ? "," : "", sentences[0].content_words[i]);
    printf("]\n");
    if (n_sentences < N_BATCH) { fprintf(stderr, "need at least %d sentences\n", N_BATCH); return 1; }

    /* Masking-rate observability: a lower contrastive loss can mean either
     * "the embedder got better" or "the task got easier" (fewer negatives
     * to separate) -- report the second so the first can't hide behind it. */
    {
        long pairs = 0, masked_pairs = 0;
        for (int i = 0; i < n_sentences; i++)
            for (int j = 0; j < n_sentences; j++) {
                if (i == j) continue;
                pairs++;
                if (shares_content_word(&sentences[i], &sentences[j])) masked_pairs++;
            }
        unsigned int mrng = 123;
        int degenerate_rows = 0, total_rows = 0;
        for (int trial = 0; trial < 5000; trial++) {
            int idxs[N_BATCH];
            for (int k = 0; k < N_BATCH; k++) idxs[k] = (int)(xr(&mrng) % (unsigned int)n_sentences);
            for (int i = 0; i < N_BATCH; i++) {
                int all_masked = 1;
                for (int j = 0; j < N_BATCH; j++) {
                    if (j == i) continue;
                    if (!shares_content_word(&sentences[idxs[i]], &sentences[idxs[j]])) { all_masked = 0; break; }
                }
                total_rows++;
                if (all_masked) degenerate_rows++;
            }
        }
        printf("masking rate: %.1f%% of all sentence pairs share a content word (excluded as negatives)\n",
               100.0 * masked_pairs / pairs);
        printf("degenerate rows: %.2f%% of batch rows (5000 simulated batches, N=%d) have every negative masked\n",
               100.0 * degenerate_rows / total_rows, N_BATCH);
    }

    TCParamSet *p = tc_paramset_load(base_model_path);
    if (!p) { fprintf(stderr, "could not load %s -- run `make train` first\n", base_model_path); return 1; }
    TCConfig cfg = p->cfg;
    TCParamSet *grad = tc_paramset_create(cfg);

    TCCache *cacheA[N_BATCH], *cacheB[N_BATCH];
    for (int k = 0; k < N_BATCH; k++) { cacheA[k] = tc_cache_create(cfg); cacheB[k] = tc_cache_create(cfg); }

    float tau = 0.1f, corrupt_p = 0.15f;
    unsigned int rng = 4242;

    /* Self-check: finite-difference the actual batch loss on a handful of
     * sampled params before trusting any real training. Same fixed rng
     * seed for +eps/-eps/base so the sampled batch is identical each time. */
    printf("\n--- gradient self-check on the real contrastive batch loss ---\n");
    {
        unsigned int check_seed = 777;
        const float eps = 1e-3f, atol = 3e-3f, rtol = 5e-2f;
        int sample_idx[10];
        for (int i = 0; i < 10; i++) sample_idx[i] = (i * (p->n_floats / 10)) % p->n_floats;

        unsigned int rr = check_seed;
        tc_paramset_zero(grad);
        run_batch(p, grad, cacheA, cacheB, sentences, n_sentences, &rr, tau, corrupt_p);
        float analytic[10];
        for (int i = 0; i < 10; i++) analytic[i] = grad->buf[sample_idx[i]];

        int failed = 0;
        for (int i = 0; i < 10; i++) {
            int idx = sample_idx[i];
            float orig = p->buf[idx];

            p->buf[idx] = orig + eps;
            unsigned int r1 = check_seed;
            TCParamSet *dummy = tc_paramset_create(cfg);
            float lp = run_batch(p, dummy, cacheA, cacheB, sentences, n_sentences, &r1, tau, corrupt_p);
            tc_paramset_free(dummy);

            p->buf[idx] = orig - eps;
            unsigned int r2 = check_seed;
            dummy = tc_paramset_create(cfg);
            float lm = run_batch(p, dummy, cacheA, cacheB, sentences, n_sentences, &r2, tau, corrupt_p);
            tc_paramset_free(dummy);

            p->buf[idx] = orig;
            float numeric = (lp - lm) / (2 * eps);
            float diff = fabsf(numeric - analytic[i]);
            float tol = atol + rtol * fabsf(analytic[i]);
            if (diff > tol) {
                failed++;
                printf("  MISMATCH idx=%d numeric=%.6f analytic=%.6f diff=%.6f tol=%.6f\n",
                       idx, numeric, analytic[i], diff, tol);
            }
        }
        printf("checked 10 params, %d failed\n", failed);
        if (failed > 0) { fprintf(stderr, "contrastive gradient self-check FAILED -- aborting training\n"); return 1; }
        printf("SELF-CHECK PASS -- proceeding to train\n\n");
    }

    TCAdam *adam = tc_adam_create(cfg, 3e-3f, 0.0f);
    int num_steps = 4000, report_every = 400;
    float running = -1.0f;
    clock_t t0 = clock();
    for (int step = 1; step <= num_steps; step++) {
        float loss = run_batch(p, grad, cacheA, cacheB, sentences, n_sentences, &rng, tau, corrupt_p);
        tc_adam_step(adam, p, grad);
        running = running < 0 ? loss : 0.98f * running + 0.02f * loss;
        if (step % report_every == 0 || step == 1)
            printf("step %5d  loss %.4f  (running avg %.4f)\n", step, loss, running);
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("trained %d contrastive steps (batch %d, %d views/step) in %.2fs\n",
           num_steps, N_BATCH, 2 * N_BATCH, secs);

    if (tc_paramset_save(p, out_path) == 0) printf("saved embedder to %s\n", out_path);
    else fprintf(stderr, "failed to save %s\n", out_path);

    for (int k = 0; k < N_BATCH; k++) { tc_cache_free(cacheA[k]); tc_cache_free(cacheB[k]); }
    tc_adam_free(adam);
    tc_paramset_free(grad);
    tc_paramset_free(p);
    free(text);
    return 0;
}
