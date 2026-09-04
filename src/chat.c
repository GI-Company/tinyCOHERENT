/* Interactive Glass-Box Chat System for TinyCoherent
 *
 * Provides a conversational REPL with:
 *   - Streaming token-by-token generation
 *   - Live token surprise (perplexity/calibration) telemetry
 *   - On-demand causal occlusion attribution (/explain)
 *   - Fast analytical input gradient attribution (/grad)
 *   - Mechanistic attention head attribution (/heads)
 *   - Grounded Glass-Box RAG with hallucination attribution (/doc, /docs, /rag)
 *   - Dynamic model hot-swapping (/model <path>)
 *   - ANSI color-coded token surprise inspection (/color on|off)
 */
#include "tcmodel.h"
#include "tokenizer.h"
#include "bpe.h"
#include "glassbox.h"
#include "vectorstore.h"
#include "steer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define ANSI_RESET      "\033[0m"
#define ANSI_BOLD       "\033[1m"
#define ANSI_DIM        "\033[2m"
#define ANSI_CYAN       "\033[36m"
#define ANSI_GREEN      "\033[32m"
#define ANSI_YELLOW     "\033[33m"
#define ANSI_RED        "\033[31m"
#define ANSI_MAGENTA    "\033[35m"

#define ANSI_BG_GREEN   "\033[42;30m"
#define ANSI_BG_YELLOW  "\033[43;30m"
#define ANSI_BG_RED     "\033[41;37m"

typedef struct {
    TCParamSet *model;
    TCCache *cache;
    BPETokenizer *bpe;
    TCVectorStore *vs;
    char model_path[256];
    float temperature;
    int color_mode;
    unsigned int rng_state;

    /* Last turn storage for /explain, /grad, /heads */
    int last_ids[512];
    int last_prompt_len;
    int last_total_len;
    float last_avg_surprise;

    /* RAG grounding tracking */
    int last_rag_mode;
    int last_ctx_len;
    char last_retrieved_doc[512];

    /* SFT dialogue & query tracking */
    int is_sft;
    int last_query_start;
    int last_query_len;

    /* Activation Steering */
    TCSteerBank *steer_bank;
    TCSteerConfig current_steer;
    char current_concept[64];
} ChatState;

static unsigned int xrand_step(unsigned int *s) {
    unsigned int x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

static float rand_unif(unsigned int *s) {
    return (float)(xrand_step(s) >> 8) * (1.0f / 16777216.0f);
}

static void embed_text(ChatState *cs, const char *text, float *out_vec) {
    int ids[256];
    int len;
    if (cs->bpe && cs->model->cfg.vocab_size > 256) {
        len = bpe_encode(cs->bpe, text, ids, cs->model->cfg.max_seq_len);
    } else {
        char clean[512];
        strncpy(clean, text, sizeof(clean) - 1);
        clean[sizeof(clean) - 1] = 0;
        tc_sanitize_text(clean);
        len = (int)strlen(clean);
        if (len > cs->model->cfg.max_seq_len) len = cs->model->cfg.max_seq_len;
        for (int i = 0; i < len; i++) ids[i] = tc_encode_char(clean[i]);
    }
    if (len < 1) len = 1;
    tc_embed(cs->model, cs->cache, ids, len, out_vec);
}

static int load_model(ChatState *cs, const char *path) {
    TCParamSet *new_p = tc_paramset_load(path);
    if (!new_p) {
        printf(ANSI_RED "Error:" ANSI_RESET " Failed to load model from '%s'\n", path);
        return 0;
    }
    if (cs->cache) tc_cache_free(cs->cache);
    if (cs->model) tc_paramset_free(cs->model);
    if (cs->vs) tc_vs_free(cs->vs);

    cs->model = new_p;
    cs->cache = tc_cache_create(new_p->cfg);
    cs->vs = tc_vs_create(new_p->cfg.d_model);
    strncpy(cs->model_path, path, sizeof(cs->model_path) - 1);
    cs->model_path[sizeof(cs->model_path) - 1] = 0;

    printf(ANSI_GREEN "Loaded model:" ANSI_RESET " %s (%d params, d_model=%d, n_layers=%d, max_seq=%d)\n",
           path, new_p->n_floats, new_p->cfg.d_model, new_p->cfg.n_layers, new_p->cfg.max_seq_len);

    if (new_p->cfg.vocab_size > 256) {
        if (!cs->bpe) cs->bpe = bpe_load("data/bpe_merges.txt");
        if (cs->bpe) {
            printf(ANSI_GREEN "Tokenizer: " ANSI_RESET "BPE subword (vocab=%d, %d merges)\n",
                   cs->bpe->vocab_size, cs->bpe->num_merges);
        } else {
            printf(ANSI_YELLOW "Warning: " ANSI_RESET "Model has vocab=%d but data/bpe_merges.txt not found\n",
                   new_p->cfg.vocab_size);
        }
    } else {
        printf(ANSI_GREEN "Tokenizer: " ANSI_RESET "96-ASCII character encoding\n");
    }

    if (strstr(path, "sft") != NULL) {
        cs->is_sft = 1;
        printf(ANSI_CYAN "Alignment: " ANSI_RESET "Conversational SFT mode enabled (User/Assistant delimiters)\n");
    } else {
        cs->is_sft = 0;
    }

    if (cs->steer_bank) {
        tc_steer_bank_free(cs->steer_bank);
        cs->steer_bank = NULL;
    }
    cs->steer_bank = tc_steer_load_bank("data/steering_vectors.bin");
    cs->current_steer.layer = -1;
    cs->current_steer.alpha = 0.0f;
    cs->current_steer.vec = NULL;
    cs->current_concept[0] = 0;
    if (cs->steer_bank) {
        printf(ANSI_MAGENTA "Steering:  " ANSI_RESET "Loaded %d concept directions from data/steering_vectors.bin\n",
               cs->steer_bank->count);
    }
    return 1;
}

static void print_banner(void) {
    printf(ANSI_BOLD ANSI_CYAN "\n=======================================================\n");
    printf("   TinyCoherent Glass-Box Chat & Interpretability REPL\n");
    printf("=======================================================\n" ANSI_RESET);
    printf("Commands:\n");
    printf("  " ANSI_BOLD "/explain" ANSI_RESET "                - Show causal prompt attribution for last reply\n");
    printf("  " ANSI_BOLD "/grad" ANSI_RESET "                   - Show analytical input gradient attribution\n");
    printf("  " ANSI_BOLD "/heads" ANSI_RESET "                  - Mechanistic attention head ablation matrix\n");
    printf("  " ANSI_BOLD "/steer <concept> [alpha]" ANSI_RESET " - Inject latent direction vector (e.g. /steer joy 2.5)\n");
    printf("  " ANSI_BOLD "/steer off" ANSI_RESET "              - Disable activation steering\n");
    printf("  " ANSI_BOLD "/steer-list" ANSI_RESET "             - List available concept vectors in bank\n");
    printf("  " ANSI_BOLD "/steer-diff" ANSI_RESET "             - Show vocabulary logit shifts under steering\n");
    printf("  " ANSI_BOLD "/rag <query>" ANSI_RESET "            - Retrieve evidence and answer with Grounding Ratio\n");
    printf("  " ANSI_BOLD "/doc <text>" ANSI_RESET "             - Ingest passage into knowledge vector store\n");
    printf("  " ANSI_BOLD "/docs" ANSI_RESET "                   - List all passages in knowledge vector store\n");
    printf("  " ANSI_BOLD "/model <path>" ANSI_RESET "           - Hot-swap model checkpoint\n");
    printf("  " ANSI_BOLD "/temp <val>" ANSI_RESET "             - Set temperature (e.g. 0.0=greedy, 0.7=default)\n");
    printf("  " ANSI_BOLD "/color <on|off>" ANSI_RESET "         - Toggle ANSI surprise token coloring\n");
    printf("  " ANSI_BOLD "/clear" ANSI_RESET "                  - Reset conversation history\n");
    printf("  " ANSI_BOLD "/help" ANSI_RESET "                   - Print this command list\n");
    printf("  " ANSI_BOLD "/quit" ANSI_RESET "                   - Exit\n\n");
}

static int sample_token(const float *probs, int V, float temperature, unsigned int *rng) {
    if (temperature <= 0.01f) {
        int best = 0;
        float maxp = probs[0];
        for (int i = 1; i < V; i++) {
            if (probs[i] > maxp) { maxp = probs[i]; best = i; }
        }
        return best;
    }
    float scaled[4096];
    float maxs = -1e30f;
    float inv_t = 1.0f / temperature;
    for (int i = 0; i < V; i++) {
        float p = probs[i] < 1e-9f ? 1e-9f : probs[i];
        float s = logf(p) * inv_t;
        scaled[i] = s;
        if (s > maxs) maxs = s;
    }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        scaled[i] = expf(scaled[i] - maxs);
        sum += scaled[i];
    }
    float r = rand_unif(rng) * sum;
    float cum = 0.0f;
    for (int i = 0; i < V; i++) {
        cum += scaled[i];
        if (cum >= r) return i;
    }
    return V - 1;
}

static void show_attribution(const ChatState *cs, int use_gradient) {
    if (cs->last_total_len <= cs->last_prompt_len) {
        printf(ANSI_YELLOW "No active response to explain. Send a prompt first!\n" ANSI_RESET);
        return;
    }

    int T = cs->last_total_len;
    int plen = cs->last_prompt_len;
    const int *ids = cs->last_ids;

    int start_i = 0;
    int end_i = plen;
    if (cs->is_sft && cs->last_query_len > 0) {
        start_i = cs->last_query_start;
        end_i = cs->last_query_start + cs->last_query_len;
        if (end_i > plen) end_i = plen;
        printf(ANSI_BOLD "\n--- %s Attribution for User Query (len=%d) ---\n" ANSI_RESET,
               use_gradient ? "Analytical Gradient (Input x Grad)" : "Causal Occlusion", cs->last_query_len);
    } else {
        printf(ANSI_BOLD "\n--- %s Attribution for Prompt (len=%d) ---\n" ANSI_RESET,
               use_gradient ? "Analytical Gradient (Input x Grad)" : "Causal Occlusion", plen);
    }

    float importances[256] = {0};
    char chars[256] = {0};

    int mask_token = (cs->model->cfg.vocab_size > 256) ? 32 : tc_encode_char(' ');

    if (use_gradient) {
        TCGradStep steps[256];
        tc_explain_input_grad(cs->model, cs->cache, ids, T, steps);
        for (int i = 0; i < plen && i < T - 1; i++) {
            importances[i] = steps[i].importance;
            chars[i] = steps[i].ch;
        }
    } else {
        TCCausalStep steps[256];
        tc_explain_causal(cs->model, cs->cache, ids, T, mask_token, steps);
        for (int i = 0; i < plen && i < T - 1; i++) {
            importances[i] = steps[i].importance;
            chars[i] = steps[i].ch;
        }
    }

    if (cs->last_rag_mode && cs->last_ctx_len > 0) {
        float ctx_imp = 0.0f, query_imp = 0.0f;
        for (int i = 0; i < plen; i++) {
            if (i < cs->last_ctx_len) ctx_imp += importances[i];
            else query_imp += importances[i];
        }
        float tot = ctx_imp + query_imp;
        float g_pct = tot > 1e-6f ? (ctx_imp / tot * 100.0f) : 0.0f;
        printf(ANSI_BOLD "Retrieved Context: " ANSI_RESET "\"%s\"\n", cs->last_retrieved_doc);
        printf(ANSI_BOLD "Grounding Faithfulness: " ANSI_RESET);
        if (g_pct >= 50.0f) printf(ANSI_GREEN ANSI_BOLD "%.1f%% (HIGH EVIDENCE GROUNDING)" ANSI_RESET "\n\n", g_pct);
        else if (g_pct >= 25.0f) printf(ANSI_YELLOW "%.1f%% (MODERATE EVIDENCE GROUNDING)" ANSI_RESET "\n\n", g_pct);
        else printf(ANSI_RED ANSI_BOLD "%.1f%% (POTENTIAL PARAMETRIC HALLUCINATION)" ANSI_RESET "\n\n", g_pct);
    }

    float max_imp = 1e-6f;
    for (int i = start_i; i < end_i; i++) {
        if (importances[i] > max_imp) max_imp = importances[i];
    }

    int is_bpe = (cs->bpe && cs->model->cfg.vocab_size > 256);
    if (cs->last_rag_mode && cs->last_ctx_len > 0) {
        printf("%-5s %-18s %-12s  %s\n", "Type", is_bpe ? "Token" : "Char", "Importance", "Contribution bar");
    } else {
        printf("%-18s %-12s  %s\n", is_bpe ? "Token" : "Char", "Importance", "Contribution bar");
    }

    for (int i = start_i; i < end_i; i++) {
        char tok_str[64];
        if (is_bpe) {
            char piece[32] = {0};
            bpe_decode(cs->bpe, &ids[i], 1, piece, sizeof(piece));
            snprintf(tok_str, sizeof(tok_str), "\"%s\"", piece);
        } else {
            char ch = chars[i] ? chars[i] : tc_decode_id(ids[i]);
            char display_ch = (ch == ' ') ? '_' : (ch == '\n' ? '\\' : ch);
            snprintf(tok_str, sizeof(tok_str), "'%c'", display_ch);
        }
        float score = importances[i];
        int bar = (int)((score / max_imp) * 28.0f);
        if (bar > 28) bar = 28;
        if (bar < 0) bar = 0;

        const char *color = ANSI_RESET;
        if (score > max_imp * 0.75f) color = ANSI_RED ANSI_BOLD;
        else if (score > max_imp * 0.40f) color = ANSI_YELLOW;
        else if (score > max_imp * 0.15f) color = ANSI_GREEN;

        if (cs->last_rag_mode && cs->last_ctx_len > 0) {
            const char *type_lbl = (i < cs->last_ctx_len) ? ANSI_CYAN "[CTX]" ANSI_RESET : ANSI_DIM "[QRY]" ANSI_RESET;
            printf(" %s %-18s %s%+.6f" ANSI_RESET "   ", type_lbl, tok_str, color, score);
        } else {
            printf(" %-18s %s%+.6f" ANSI_RESET "   ", tok_str, color, score);
        }
        for (int b = 0; b < bar; b++) putchar('#');
        putchar('\n');
    }
    if (cs->is_sft && cs->last_query_len > 0) {
        printf(ANSI_DIM "Attribution shows which words in your question causally drove the assistant's answer.\n\n" ANSI_RESET);
    } else {
        printf(ANSI_DIM "Prompt token attribution: higher score indicates tokens that causally drove the generation.\n\n" ANSI_RESET);
    }
}

static void show_heads(const ChatState *cs) {
    if (cs->last_total_len < 2) {
        printf(ANSI_YELLOW "No active conversation to analyze heads. Send a prompt first!\n\n" ANSI_RESET);
        return;
    }
    TCHeadImportance heads[128];
    int n = tc_explain_heads(cs->model, cs->cache, cs->last_ids, cs->last_total_len, heads);
    (void)n;
    int L = cs->model->cfg.n_layers;
    int H = cs->model->cfg.n_heads;

    printf(ANSI_BOLD "\n--- Mechanistic Attention Head Attribution (L=%d, H=%d) ---\n" ANSI_RESET, L, H);
    printf("Head ablation loss shift ΔL (positive = critical head whose ablation hurts prediction):\n\n");
    printf("        ");
    for (int h = 0; h < H; h++) printf(" Head %-2d ", h);
    printf("\n");

    for (int l = 0; l < L; l++) {
        printf("Layer %d: ", l);
        for (int h = 0; h < H; h++) {
            float imp = heads[l * H + h].importance;
            const char *col = ANSI_RESET;
            if (imp > 0.010f) col = ANSI_RED ANSI_BOLD;
            else if (imp > 0.003f) col = ANSI_YELLOW;
            else if (imp < -0.005f) col = ANSI_CYAN;
            printf("%s%+0.3f  " ANSI_RESET, col, imp);
        }
        printf("\n");
    }
    printf(ANSI_DIM "\nKey: " ANSI_RED "High Causal Impact (>0.010)" ANSI_RESET " | "
           ANSI_YELLOW "Moderate (>0.003)" ANSI_RESET " | "
           ANSI_CYAN "Suppressor (< -0.005)" ANSI_RESET " | "
           ANSI_RESET "Neutral\n\n");
}

static void run_turn(ChatState *cs, const char *user_input) {
    TCConfig cfg = cs->model->cfg;
    int is_bpe = (cs->bpe && cfg.vocab_size > 256);
    int max_cap = cfg.max_seq_len;
    int max_prompt = max_cap - 45; /* reserve at least 45 tokens for generation */
    if (max_prompt < 10) max_prompt = 10;

    int ids[512];
    int plen = 0;
    cs->last_query_start = 0;
    cs->last_query_len = 0;

    if (is_bpe) {
        if (cs->is_sft && strncmp(user_input, "User:", 5) != 0) {
            char formatted[1024];
            snprintf(formatted, sizeof(formatted), "User: %s\nAssistant: ", user_input);
            int prefix_tokens[64];
            int pfx_len = bpe_encode(cs->bpe, "User: ", prefix_tokens, 64);
            int q_tokens[256];
            int q_len = bpe_encode(cs->bpe, user_input, q_tokens, 256);
            cs->last_query_start = pfx_len;
            cs->last_query_len = q_len;
            plen = bpe_encode(cs->bpe, formatted, ids, max_prompt);
        } else {
            plen = bpe_encode(cs->bpe, user_input, ids, max_prompt);
            cs->last_query_start = 0;
            cs->last_query_len = plen;
        }
    } else {
        char clean_input[512];
        strncpy(clean_input, user_input, sizeof(clean_input) - 1);
        clean_input[sizeof(clean_input) - 1] = 0;
        tc_sanitize_text(clean_input);

        int input_len = (int)strlen(clean_input);
        if (input_len < 1) return;

        int start_off = 0;
        if (input_len > max_prompt) start_off = input_len - max_prompt;
        for (int i = start_off; i < input_len && plen < max_prompt; i++) {
            ids[plen++] = tc_encode_char(clean_input[i]);
        }
        cs->last_query_start = 0;
        cs->last_query_len = plen;
    }

    if (plen < 1) return;

    cs->last_prompt_len = plen;
    int cur_len = plen;
    float sum_surprise = 0.0f;
    int gen_tokens = 0;

    if (cs->current_steer.vec && fabsf(cs->current_steer.alpha) > 1e-6f) {
        printf(ANSI_MAGENTA "[Steer: %s alpha=%+.1f @ L%d] " ANSI_RESET,
               cs->current_concept, cs->current_steer.alpha, cs->current_steer.layer);
    }
    printf(ANSI_BOLD ANSI_CYAN "Assistant: " ANSI_RESET);
    fflush(stdout);

    int V = cfg.vocab_size;
    int max_gen = is_bpe ? 45 : 50;
    char rolling[1024] = {0};

    while (cur_len < max_cap - 1 && gen_tokens < max_gen) {
        float loss;
        if (cs->current_steer.vec && fabsf(cs->current_steer.alpha) > 1e-6f) {
            tc_forward_steered(cs->model, cs->cache, ids, cur_len, NULL, &loss, &cs->current_steer);
        } else {
            tc_forward(cs->model, cs->cache, ids, cur_len, NULL, &loss);
        }
        const float *probs = cs->cache->probs + (size_t)(cur_len - 1) * V;

        int next_id = sample_token(probs, V, cs->temperature, &cs->rng_state);
        float p_next = probs[next_id];
        if (p_next < 1e-9f) p_next = 1e-9f;
        float surprise = -log2f(p_next);

        if (is_bpe) {
            char piece[128] = {0};
            bpe_decode(cs->bpe, &next_id, 1, piece, sizeof(piece));

            /* Stop sequences: end of text delimiter or next turn */
            if (strstr(piece, "<|") || strstr(piece, "<|endoftext|>") ||
                strstr(piece, "User:") || strstr(piece, "Assistant:")) {
                break;
            }

            strncat(rolling, piece, sizeof(rolling) - strlen(rolling) - 1);
            if (strstr(rolling, "<|") || strstr(rolling, "<|endoftext|>") ||
                strstr(rolling, "\nUser:") || strstr(rolling, "User:")) {
                break;
            }

            sum_surprise += surprise;
            gen_tokens++;

            if (cs->color_mode) {
                if (surprise < 1.0f) printf(ANSI_GREEN "%s" ANSI_RESET, piece);
                else if (surprise < 2.5f) printf(ANSI_YELLOW "%s" ANSI_RESET, piece);
                else printf(ANSI_RED "%s" ANSI_RESET, piece);
            } else {
                fputs(piece, stdout);
            }
            fflush(stdout);

            ids[cur_len++] = next_id;
            if (gen_tokens >= 15 && (strchr(piece, '\n') || (strchr(piece, '.') && gen_tokens >= 25))) {
                break;
            }
        } else {
            char ch = tc_decode_id(next_id);
            sum_surprise += surprise;
            gen_tokens++;

            if (cs->color_mode) {
                if (surprise < 1.0f) printf(ANSI_GREEN "%c" ANSI_RESET, ch);
                else if (surprise < 2.5f) printf(ANSI_YELLOW "%c" ANSI_RESET, ch);
                else printf(ANSI_RED "%c" ANSI_RESET, ch);
            } else {
                putchar(ch);
            }
            fflush(stdout);

            ids[cur_len++] = next_id;
            if (gen_tokens >= 20 && (ch == '\n' || (ch == '.' && gen_tokens >= 35))) {
                break;
            }
        }
    }
    putchar('\n');

    float avg_surprise = gen_tokens > 0 ? (sum_surprise / gen_tokens) : 0.0f;
    cs->last_total_len = cur_len;
    cs->last_avg_surprise = avg_surprise;
    memcpy(cs->last_ids, ids, sizeof(int) * (size_t)cur_len);

    const char *conf_label;
    if (avg_surprise < 2.0f) conf_label = ANSI_GREEN "High Confidence / Fluent" ANSI_RESET;
    else if (avg_surprise < 4.0f) conf_label = ANSI_YELLOW "Moderate Confidence" ANSI_RESET;
    else conf_label = ANSI_RED "High Uncertainty" ANSI_RESET;

    printf(ANSI_DIM " [avg surprise: %.2f bits/token - %s" ANSI_RESET, avg_surprise, conf_label);

    if (cs->last_rag_mode && cs->last_ctx_len > 0) {
        TCCausalStep steps[256];
        int mask_token = (cfg.vocab_size > 256) ? 32 : tc_encode_char(' ');
        tc_explain_causal(cs->model, cs->cache, ids, cur_len, mask_token, steps);
        float ctx_imp = 0.0f, query_imp = 0.0f;
        for (int i = 0; i < plen && i < cur_len - 1; i++) {
            if (i < cs->last_ctx_len) ctx_imp += steps[i].importance;
            else query_imp += steps[i].importance;
        }
        float tot = ctx_imp + query_imp;
        float g_pct = tot > 1e-6f ? (ctx_imp / tot * 100.0f) : 0.0f;
        const char *g_col = (g_pct >= 50.0f) ? ANSI_GREEN : ((g_pct >= 25.0f) ? ANSI_YELLOW : ANSI_RED);
        printf(ANSI_DIM " | Grounding: %s%.1f%%%s" ANSI_RESET, g_col, g_pct, ANSI_DIM);
    }

    printf(ANSI_DIM " | /explain for attribution]\n\n" ANSI_RESET);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char *initial_model = "build/model_rung4.bin";
    if (argc > 1) initial_model = argv[1];

    /* Fallback if model doesn't exist */
    FILE *test_f = fopen(initial_model, "rb");
    if (!test_f) {
        if (strcmp(initial_model, "build/model_rung3.bin") != 0) {
            initial_model = "build/model_rung3.bin";
        }
    } else {
        fclose(test_f);
    }

    ChatState cs;
    memset(&cs, 0, sizeof(cs));
    cs.temperature = 0.7f;
    cs.color_mode = 0;
    cs.rng_state = (unsigned int)time(NULL) ^ 0x9e3779b9;

    print_banner();

    if (!load_model(&cs, initial_model)) {
        if (!load_model(&cs, "build/model.bin")) {
            fprintf(stderr, "No valid model found in build/. Train one first with make train.\n");
            return 1;
        }
    }

    char line[512];
    while (1) {
        printf(ANSI_BOLD "User: " ANSI_RESET);
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;

        /* Strip trailing newline */
        size_t l = strlen(line);
        while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = 0;
        if (l == 0) continue;

        /* Check commands */
        if (line[0] == '/') {
            if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0) {
                printf(ANSI_CYAN "Goodbye!\n" ANSI_RESET);
                break;
            } else if (strcmp(line, "/help") == 0) {
                print_banner();
            } else if (strcmp(line, "/explain") == 0) {
                show_attribution(&cs, 0);
            } else if (strcmp(line, "/grad") == 0) {
                show_attribution(&cs, 1);
            } else if (strcmp(line, "/heads") == 0) {
                show_heads(&cs);
            } else if (strcmp(line, "/steer-list") == 0) {
                if (!cs.steer_bank || cs.steer_bank->count == 0) {
                    printf(ANSI_YELLOW "No steering bank loaded. Run 'make extract_steer' first.\n\n" ANSI_RESET);
                } else {
                    printf(ANSI_BOLD "\n=== Latent Concept Vector Bank (%d concepts) ===\n" ANSI_RESET, cs.steer_bank->count);
                    for (int i = 0; i < cs.steer_bank->count; i++) {
                        const TCSteerVector *sv = &cs.steer_bank->vectors[i];
                        int top_pos[3], top_neg[3];
                        float dpos[3], dneg[3];
                        tc_steer_inspect_boost(cs.model, sv->vec, 3, top_pos, dpos, top_neg, dneg);
                        printf(ANSI_CYAN "  [%s]" ANSI_RESET " (L%d): %s\n", sv->name, sv->layer, sv->description);
                        printf("     Promoted: ");
                        for (int k = 0; k < 3; k++) {
                            char tok[64];
                            if (cs.bpe) bpe_decode(cs.bpe, &top_pos[k], 1, tok, sizeof(tok));
                            else { tok[0] = tc_decode_id(top_pos[k]); tok[1] = 0; }
                            printf("'%s' (%+.2f) ", tok, dpos[k]);
                        }
                        printf("\n");
                    }
                    if (cs.current_steer.vec) {
                        printf(ANSI_MAGENTA "\nActive Steering: %s (alpha=%+.1f at Layer %d)\n\n" ANSI_RESET,
                               cs.current_concept, cs.current_steer.alpha, cs.current_steer.layer);
                    } else {
                        printf("\nActive Steering: None (use /steer <concept> [alpha] to activate)\n\n");
                    }
                }
            } else if (strcmp(line, "/steer off") == 0 || strcmp(line, "/steer none") == 0) {
                cs.current_steer.vec = NULL;
                cs.current_steer.alpha = 0.0f;
                cs.current_steer.layer = -1;
                cs.current_concept[0] = 0;
                printf(ANSI_GREEN "Activation steering disabled.\n\n" ANSI_RESET);
            } else if (strcmp(line, "/steer-diff") == 0) {
                if (!cs.current_steer.vec) {
                    printf(ANSI_YELLOW "No active steering. Activate with /steer <concept> first.\n\n" ANSI_RESET);
                } else {
                    printf(ANSI_BOLD "\n--- Mechanistic Logit Projection: %s (alpha=%+.1f) ---\n" ANSI_RESET,
                           cs.current_concept, cs.current_steer.alpha);
                    int top_pos[5], top_neg[5];
                    float dpos[5], dneg[5];
                    tc_steer_inspect_boost(cs.model, cs.current_steer.vec, 5, top_pos, dpos, top_neg, dneg);
                    printf(ANSI_GREEN "Top Promoted Tokens (Delta z > 0):\n" ANSI_RESET);
                    for (int k = 0; k < 5; k++) {
                        char tok[64];
                        if (cs.bpe) bpe_decode(cs.bpe, &top_pos[k], 1, tok, sizeof(tok));
                        else { tok[0] = tc_decode_id(top_pos[k]); tok[1] = 0; }
                        printf("  +%d: '%s'  delta = %+.3f  (steered shift = %+.3f logits)\n",
                               k+1, tok, dpos[k], cs.current_steer.alpha * dpos[k]);
                    }
                    printf(ANSI_RED "Top Suppressed Tokens (Delta z < 0):\n" ANSI_RESET);
                    for (int k = 0; k < 5; k++) {
                        char tok[64];
                        if (cs.bpe) bpe_decode(cs.bpe, &top_neg[k], 1, tok, sizeof(tok));
                        else { tok[0] = tc_decode_id(top_neg[k]); tok[1] = 0; }
                        printf("  -%d: '%s'  delta = %+.3f  (steered shift = %+.3f logits)\n",
                               k+1, tok, dneg[k], cs.current_steer.alpha * dneg[k]);
                    }
                    printf("\n");
                }
            } else if (strncmp(line, "/steer ", 7) == 0) {
                char cname[64] = {0};
                float alpha = 2.5f;
                int n_args = sscanf(line + 7, "%63s %f", cname, &alpha);
                if (n_args < 1) {
                    printf(ANSI_YELLOW "Usage: /steer <concept_name> [alpha]\n\n" ANSI_RESET);
                } else if (strcmp(cname, "off") == 0 || strcmp(cname, "none") == 0) {
                    cs.current_steer.vec = NULL;
                    cs.current_steer.alpha = 0.0f;
                    cs.current_steer.layer = -1;
                    cs.current_concept[0] = 0;
                    printf(ANSI_GREEN "Activation steering disabled.\n\n" ANSI_RESET);
                } else {
                    if (!cs.steer_bank) cs.steer_bank = tc_steer_load_bank("data/steering_vectors.bin");
                    const TCSteerVector *sv = tc_steer_bank_find(cs.steer_bank, cname);
                    if (!sv) {
                        printf(ANSI_RED "Concept '%s' not found in bank. Type /steer-list to see concepts.\n\n" ANSI_RESET, cname);
                    } else {
                        cs.current_steer.layer = sv->layer;
                        cs.current_steer.alpha = alpha;
                        cs.current_steer.vec = sv->vec;
                        strncpy(cs.current_concept, sv->name, sizeof(cs.current_concept) - 1);
                        printf(ANSI_MAGENTA "Enabled steering:" ANSI_RESET " [%s] alpha=%+.1f at Layer %d\n",
                               sv->name, alpha, sv->layer);
                        printf(ANSI_DIM "Concept: %s\n" ANSI_RESET, sv->description);

                        int top_pos[3], top_neg[3];
                        float dpos[3], dneg[3];
                        tc_steer_inspect_boost(cs.model, sv->vec, 3, top_pos, dpos, top_neg, dneg);
                        printf("Promoting tokens: ");
                        for (int k = 0; k < 3; k++) {
                            char tok[64];
                            if (cs.bpe) bpe_decode(cs.bpe, &top_pos[k], 1, tok, sizeof(tok));
                            else { tok[0] = tc_decode_id(top_pos[k]); tok[1] = 0; }
                            printf("'%s' ", tok);
                        }
                        printf("\n\n");
                    }
                }
            } else if (strncmp(line, "/doc ", 5) == 0) {
                const char *doc_text = line + 5;
                while (*doc_text == ' ') doc_text++;
                if (strlen(doc_text) > 0) {
                    float dvec[TC_MAX_D_MODEL];
                    embed_text(&cs, doc_text, dvec);
                    tc_vs_add(cs.vs, doc_text, dvec);
                    printf(ANSI_GREEN "Ingested passage [id=%d, %zu chars]:" ANSI_RESET " \"%s\"\n\n",
                           cs.vs->count - 1, strlen(doc_text), doc_text);
                }
            } else if (strcmp(line, "/docs") == 0) {
                printf(ANSI_BOLD "\n--- Glass-Box Knowledge Vector Store (%d passages) ---\n" ANSI_RESET, cs.vs->count);
                if (cs.vs->count == 0) {
                    printf("  (Store is empty. Add passages using: /doc <text>)\n\n");
                } else {
                    for (int i = 0; i < cs.vs->count; i++) {
                        printf("  [%d] \"%s\"\n", i, cs.vs->entries[i].text);
                    }
                    printf("\n");
                }
            } else if (strncmp(line, "/rag ", 5) == 0) {
                const char *query = line + 5;
                while (*query == ' ') query++;
                if (cs.vs->count == 0) {
                    printf(ANSI_YELLOW "Vector store is empty! Add documents first with /doc <text>\n\n" ANSI_RESET);
                } else {
                    float qvec[TC_MAX_D_MODEL];
                    embed_text(&cs, query, qvec);
                    TCVSHit hits[2];
                    int n_hits = tc_vs_search(cs.vs, qvec, 1, hits);
                    if (n_hits > 0) {
                        const char *retrieved = cs.vs->entries[hits[0].idx].text;
                        printf(ANSI_DIM "[Retrieved passage (%d, score=%.3f): \"%s\"]\n" ANSI_RESET,
                               hits[0].idx, hits[0].score, retrieved);
                        char rag_prompt[1024];
                        snprintf(rag_prompt, sizeof(rag_prompt), "Context: %s. Question: %s? Answer:", retrieved, query);
                        cs.last_rag_mode = 1;
                        strncpy(cs.last_retrieved_doc, retrieved, sizeof(cs.last_retrieved_doc) - 1);

                        char ctx_prefix[512];
                        snprintf(ctx_prefix, sizeof(ctx_prefix), "Context: %s. ", retrieved);
                        if (cs.bpe && cs.model->cfg.vocab_size > 256) {
                            int dummy[256];
                            cs.last_ctx_len = bpe_encode(cs.bpe, ctx_prefix, dummy, 256);
                        } else {
                            cs.last_ctx_len = (int)strlen(ctx_prefix);
                        }

                        run_turn(&cs, rag_prompt);
                    }
                }
            } else if (strncmp(line, "/temp ", 6) == 0) {
                float t = (float)atof(line + 6);
                if (t >= 0.0f && t <= 2.0f) {
                    cs.temperature = t;
                    printf("Temperature set to %.2f\n\n", cs.temperature);
                } else {
                    printf(ANSI_YELLOW "Temperature must be between 0.0 and 2.0\n\n" ANSI_RESET);
                }
            } else if (strncmp(line, "/model ", 7) == 0) {
                load_model(&cs, line + 7);
                printf("\n");
            } else if (strcmp(line, "/color on") == 0) {
                cs.color_mode = 1;
                printf("Surprise token coloring: " ANSI_GREEN "ON\n\n" ANSI_RESET);
            } else if (strcmp(line, "/color off") == 0) {
                cs.color_mode = 0;
                printf("Surprise token coloring: OFF\n\n");
            } else if (strcmp(line, "/clear") == 0) {
                cs.last_prompt_len = 0;
                cs.last_total_len = 0;
                cs.last_rag_mode = 0;
                cs.last_ctx_len = 0;
                printf("Conversation context cleared.\n\n");
            } else {
                printf(ANSI_YELLOW "Unknown command '%s'. Type /help for commands.\n\n" ANSI_RESET, line);
            }
            continue;
        }

        cs.last_rag_mode = 0;
        cs.last_ctx_len = 0;
        run_turn(&cs, line);
    }

    if (cs.cache) tc_cache_free(cs.cache);
    if (cs.model) tc_paramset_free(cs.model);
    if (cs.vs) tc_vs_free(cs.vs);
    if (cs.bpe) bpe_free(cs.bpe);
    if (cs.steer_bank) tc_steer_bank_free(cs.steer_bank);
    return 0;
}
