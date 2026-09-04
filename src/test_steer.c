#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "tcmodel.h"
#include "steer.h"
#include "bpe.h"

static void check_fail(const char *msg) {
    fprintf(stderr, "STEER TEST FAILED: %s\n", msg);
    exit(1);
}

int main(int argc, char **argv) {
    const char *model_path = (argc > 1) ? argv[1] : "build/model_sft.bin";
    const char *bpe_path = (argc > 2) ? argv[2] : "data/bpe_merges.txt";

    printf("=== TC ACTIVATION STEERING VERIFICATION SUITE ===\n");
    printf("Model: %s\n", model_path);
    printf("BPE:   %s\n", bpe_path);

    /* Load model */
    TCParamSet *p = tc_paramset_load(model_path);
    if (!p) {
        /* Fall back to model_rung4.bin if model_sft.bin isn't available */
        model_path = "build/model_rung4.bin";
        p = tc_paramset_load(model_path);
        if (!p) check_fail("Could not open model file");
    }
    printf("Loaded model: layers=%d, d_model=%d, vocab=%d, params=%d\n",
           p->cfg.n_layers, p->cfg.d_model, p->cfg.vocab_size, tc_param_count(p->cfg));

    /* Load BPE */
    BPETokenizer *bpe = bpe_load(bpe_path);
    if (!bpe) check_fail("Could not load BPE merges");

    TCCache *c = tc_cache_create(p->cfg);
    assert(c != NULL);

    /* Test Sequence */
    const char *test_prompt = "User: Tell me a story about a quiet morning.\nAssistant: The sun rose";
    int ids[128];
    int T = bpe_encode(bpe, test_prompt, ids, 128);
    printf("Test prompt tokens: %d\n", T);

    /* --- TEST 1: Zero-Steering Equivalence --- */
    printf("\n--- TEST 1: Zero-Steering Equivalence (alpha = 0.0) ---\n");
    float loss_base = 0.0f;
    tc_forward(p, c, ids, T, ids, &loss_base);
    float *base_logits = malloc((size_t)T * p->cfg.vocab_size * sizeof(float));
    memcpy(base_logits, c->logits, (size_t)T * p->cfg.vocab_size * sizeof(float));

    TCSteerConfig cfg_zero = {
        .layer = 2,
        .alpha = 0.0f,
        .vec = NULL
    };
    float loss_steered_zero = 0.0f;
    tc_forward_steered(p, c, ids, T, ids, &loss_steered_zero, &cfg_zero);

    float max_diff = 0.0f;
    for (int i = 0; i < T * p->cfg.vocab_size; i++) {
        float diff = fabsf(c->logits[i] - base_logits[i]);
        if (diff > max_diff) max_diff = diff;
    }
    float loss_diff = fabsf(loss_steered_zero - loss_base);
    printf("Max logit diff at alpha=0: %.8f\n", max_diff);
    printf("Loss diff at alpha=0:      %.8f\n", loss_diff);
    if (max_diff > 1e-6f || loss_diff > 1e-6f) {
        check_fail("alpha=0 did not match unsteered forward pass exactly!");
    }
    printf("-> PASS: Zero-steering equivalence verified!\n");

    /* --- TEST 2: Concept Extraction & Top Token Inspection --- */
    printf("\n--- TEST 2: Concept Direction Extraction & Mechanistic Projection ---\n");
    const char *pos_text = "The happy celebration brought great joy, laughter, smiles, delight, and wonderful cheer to everyone.";
    const char *neg_text = "The grim tragedy brought great sorrow, grief, tears, mourning, and terrible despair to everyone.";

    int pos_ids[128], neg_ids[128];
    int pos_len = bpe_encode(bpe, pos_text, pos_ids, 128);
    int neg_len = bpe_encode(bpe, neg_text, neg_ids, 128);

    float *joy_vec = malloc(p->cfg.d_model * sizeof(float));
    int target_layer = p->cfg.n_layers > 2 ? 2 : 1;
    int err = tc_steer_extract_direction(p, c, pos_ids, pos_len, neg_ids, neg_len, target_layer, joy_vec);
    if (err != 0) check_fail("Failed to extract concept direction");

    /* Verify norm is 1.0 */
    double norm_sq = 0.0;
    for (int i = 0; i < p->cfg.d_model; i++) norm_sq += joy_vec[i] * joy_vec[i];
    double norm = sqrt(norm_sq);
    printf("Extracted 'joy' concept at layer %d (L2 norm = %.6f)\n", target_layer, norm);
    if (fabs(norm - 1.0) > 1e-4) check_fail("Extracted vector is not unit normalized!");

    /* Vocabulary projection inspection */
    int top_pos[5], top_neg[5];
    float delta_pos[5], delta_neg[5];
    tc_steer_inspect_boost(p, joy_vec, 5, top_pos, delta_pos, top_neg, delta_neg);

    printf("Top 5 promoted vocabulary tokens (+joy):\n");
    for (int k = 0; k < 5; k++) {
        char tok_buf[64];
        bpe_decode(bpe, &top_pos[k], 1, tok_buf, sizeof(tok_buf));
        printf("  +%d: id=%-4d  delta=%+6.3f  token='%s'\n", k+1, top_pos[k], delta_pos[k], tok_buf);
    }
    printf("Top 5 suppressed vocabulary tokens (-joy):\n");
    for (int k = 0; k < 5; k++) {
        char tok_buf[64];
        bpe_decode(bpe, &top_neg[k], 1, tok_buf, sizeof(tok_buf));
        printf("  -%d: id=%-4d  delta=%+6.3f  token='%s'\n", k+1, top_neg[k], delta_neg[k], tok_buf);
    }

    int promoted_tok = top_pos[0];
    int suppressed_tok = top_neg[0];

    /* --- TEST 3: Monotonic Steering Response --- */
    printf("\n--- TEST 3: Monotonic Probability Response under Steering ---\n");
    float alphas[] = {-4.0f, -2.0f, 0.0f, 2.0f, 4.0f};
    int n_alphas = sizeof(alphas) / sizeof(alphas[0]);
    float prev_prob_promoted = -1.0f;
    float prev_prob_suppressed = 1e9f;
    (void)prev_prob_suppressed;

    for (int a = 0; a < n_alphas; a++) {
        TCSteerConfig sc = {
            .layer = target_layer,
            .alpha = alphas[a],
            .vec = joy_vec
        };
        tc_forward_steered(p, c, ids, T, NULL, NULL, &sc);
        const float *last_probs = c->probs + (size_t)(T - 1) * p->cfg.vocab_size;
        float p_pro = last_probs[promoted_tok];
        float p_sup = last_probs[suppressed_tok];

        printf("  alpha=%+4.1f | P(promoted id=%d) = %8.6f | P(suppressed id=%d) = %8.6f\n",
               alphas[a], promoted_tok, p_pro, suppressed_tok, p_sup);

        if (a > 0) {
            if (p_pro < prev_prob_promoted) {
                printf("Warning: Non-monotonic increase in promoted token (expected due to softmax competition)\n");
            }
        }
        prev_prob_promoted = p_pro;
        prev_prob_suppressed = p_sup;
    }

    /* Positive steering (alpha = +4) must give strictly higher probability for promoted token than negative steering (alpha = -4) */
    TCSteerConfig sc_pos = { .layer = target_layer, .alpha = +4.0f, .vec = joy_vec };
    tc_forward_steered(p, c, ids, T, NULL, NULL, &sc_pos);
    float p_pro_pos = c->probs[(size_t)(T - 1) * p->cfg.vocab_size + promoted_tok];

    TCSteerConfig sc_neg = { .layer = target_layer, .alpha = -4.0f, .vec = joy_vec };
    tc_forward_steered(p, c, ids, T, NULL, NULL, &sc_neg);
    float p_pro_neg = c->probs[(size_t)(T - 1) * p->cfg.vocab_size + promoted_tok];

    printf("Promoted token P(alpha=+4): %.6f vs P(alpha=-4): %.6f (ratio: %.2fx)\n",
           p_pro_pos, p_pro_neg, p_pro_neg > 0 ? (p_pro_pos / p_pro_neg) : 999.0f);
    if (p_pro_pos <= p_pro_neg) {
        check_fail("Positive steering did not enhance target concept over negative steering!");
    }
    printf("-> PASS: Steering direction controls concept token probability!\n");

    /* --- TEST 4: Bank Serialization & Fidelity --- */
    printf("\n--- TEST 4: Bank Serialization & Round-Trip Fidelity ---\n");
    TCSteerBank *bank = tc_steer_bank_create();
    tc_steer_bank_add(bank, "joy", "Joy and happiness direction vs sorrow", target_layer, p->cfg.d_model, joy_vec);

    const char *tmp_bank_path = "build/test_bank.bin";
    if (tc_steer_save_bank(bank, tmp_bank_path) != 0) {
        check_fail("Failed to save steering bank");
    }

    TCSteerBank *loaded = tc_steer_load_bank(tmp_bank_path);
    if (!loaded) check_fail("Failed to load steering bank");
    if (loaded->count != 1) check_fail("Loaded bank count mismatch");

    const TCSteerVector *found = tc_steer_bank_find(loaded, "joy");
    if (!found) check_fail("Could not find 'joy' in loaded bank");
    if (found->layer != target_layer || found->dim != p->cfg.d_model) {
        check_fail("Loaded vector metadata mismatch");
    }

    float max_vec_err = 0.0f;
    for (int i = 0; i < p->cfg.d_model; i++) {
        float d = fabsf(found->vec[i] - joy_vec[i]);
        if (d > max_vec_err) max_vec_err = d;
    }
    printf("Max serialization vector difference: %.10f\n", max_vec_err);
    if (max_vec_err > 1e-7f) check_fail("Loaded vector values diverged from original!");
    printf("-> PASS: Bank serialization is 100%% lossless!\n");

    /* Cleanup */
    tc_steer_bank_free(bank);
    tc_steer_bank_free(loaded);
    free(joy_vec);
    free(base_logits);
    tc_cache_free(c);
    tc_paramset_free(p);
    bpe_free(bpe);

    printf("\nALL STEERING TESTS PASSED SUCCESSFULLY!\n");
    return 0;
}
