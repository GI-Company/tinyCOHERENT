/* Glass-Box Studio HTTP & JSON Inference Server in pure C
 * Serves ui/ files and provides /api/generate for real-time model telemetry.
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
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define BUFFER_SIZE 65536

static char *read_static_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    size_t n = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[n] = 0;
    *out_len = (long)n;
    return buf;
}

static void send_response(int client_fd, const char *content_type, const char *body, size_t body_len) {
    char header[512];
    snprintf(header, sizeof(header),
             "HTTP/1.1 200 OK\r\n"
             "Content-Type: %s\r\n"
             "Content-Length: %zu\r\n"
             "Connection: close\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "\r\n", content_type, body_len);
    write(client_fd, header, strlen(header));
    write(client_fd, body, body_len);
}

static void send_404(int client_fd) {
    const char *msg = "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nNot Found";
    write(client_fd, msg, strlen(msg));
}

int main(int argc, char **argv) {
    int port = argc > 1 ? atoi(argv[1]) : 8080;
    const char *model_path = argc > 2 ? argv[2] : "build/model_sft.bin";
    const char *bpe_path = argc > 3 ? argv[3] : "data/bpe_merges.txt";

    TCParamSet *model = tc_paramset_load(model_path);
    if (!model) {
        model_path = "build/model_rung4.bin";
        model = tc_paramset_load(model_path);
        if (!model) {
            fprintf(stderr, "Failed to load model from %s\n", model_path);
            return 1;
        }
    }
    BPETokenizer *bpe = NULL;
    if (model->cfg.vocab_size > 256) {
        bpe = bpe_load(bpe_path);
        if (!bpe) { fprintf(stderr, "Failed to load BPE merges from %s\n", bpe_path); return 1; }
    }
    TCCache *cache = tc_cache_create(model->cfg);
    TCSteerBank *steer_bank = tc_steer_load_bank("data/steering_vectors.bin");

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket failed"); return 1; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        return 1;
    }
    if (listen(server_fd, 10) < 0) {
        perror("listen failed");
        return 1;
    }

    printf("\n=======================================================\n");
    printf("   TinyCoherent Glass-Box Web Studio Running!\n");
    printf("=======================================================\n");
    printf("Serving UI at:  http://localhost:%d/\n", port);
    printf("Active Model:   %s (%d params, d_model=%d, n_layers=%d)\n",
           model_path, model->n_floats, model->cfg.d_model, model->cfg.n_layers);
    printf("Press Ctrl+C to stop.\n\n");

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) continue;

        char req_buf[BUFFER_SIZE];
        ssize_t bytes_read = read(client_fd, req_buf, sizeof(req_buf) - 1);
        if (bytes_read <= 0) { close(client_fd); continue; }
        req_buf[bytes_read] = 0;

        if (strncmp(req_buf, "GET / ", 6) == 0 || strncmp(req_buf, "GET /index.html", 15) == 0) {
            long len;
            char *content = read_static_file("ui/index.html", &len);
            if (content) {
                send_response(client_fd, "text/html; charset=utf-8", content, (size_t)len);
                free(content);
            } else send_404(client_fd);
        } else if (strncmp(req_buf, "GET /style.css", 14) == 0) {
            long len;
            char *content = read_static_file("ui/style.css", &len);
            if (content) {
                send_response(client_fd, "text/css; charset=utf-8", content, (size_t)len);
                free(content);
            } else send_404(client_fd);
        } else if (strncmp(req_buf, "GET /app.js", 11) == 0) {
            long len;
            char *content = read_static_file("ui/app.js", &len);
            if (content) {
                send_response(client_fd, "application/javascript; charset=utf-8", content, (size_t)len);
                free(content);
            } else send_404(client_fd);
        } else if (strncmp(req_buf, "GET /api/steer/list", 19) == 0) {
            char json[BUFFER_SIZE];
            int w = snprintf(json, sizeof(json), "{\"concepts\":[");
            if (steer_bank) {
                for (int i = 0; i < steer_bank->count; i++) {
                    const TCSteerVector *sv = &steer_bank->vectors[i];
                    int top_pos[4], top_neg[4];
                    float dpos[4], dneg[4];
                    tc_steer_inspect_boost(model, sv->vec, 4, top_pos, dpos, top_neg, dneg);

                    w += snprintf(json + w, sizeof(json) - (size_t)w,
                                  "%s{\"name\":\"%s\",\"description\":\"%s\",\"layer\":%d,\"promoted\":[",
                                  i > 0 ? "," : "", sv->name, sv->description, sv->layer);
                    for (int k = 0; k < 4; k++) {
                        char tok[64] = {0};
                        if (bpe && model->cfg.vocab_size > 256) bpe_decode(bpe, &top_pos[k], 1, tok, sizeof(tok));
                        else { tok[0] = tc_decode_id(top_pos[k]); tok[1] = 0; }
                        char esc_tok[128] = {0};
                        int ep = 0;
                        for (size_t c = 0; c < strlen(tok); c++) {
                            if (tok[c] == '\"') { esc_tok[ep++] = '\\'; esc_tok[ep++] = '\"'; }
                            else if (tok[c] == '\\') { esc_tok[ep++] = '\\'; esc_tok[ep++] = '\\'; }
                            else esc_tok[ep++] = tok[c];
                        }
                        w += snprintf(json + w, sizeof(json) - (size_t)w,
                                      "%s{\"token\":\"%s\",\"delta\":%.3f}",
                                      k > 0 ? "," : "", esc_tok, dpos[k]);
                    }
                    w += snprintf(json + w, sizeof(json) - (size_t)w, "]}");
                }
            }
            w += snprintf(json + w, sizeof(json) - (size_t)w, "]}");
            send_response(client_fd, "application/json", json, (size_t)w);
        } else if (strncmp(req_buf, "POST /api/generate", 18) == 0) {
            /* Parse simple JSON payload */
            char *body = strstr(req_buf, "\r\n\r\n");
            char prompt[512] = "Once upon a time";
            float temperature = 0.7f;
            int max_tokens = 35;
            char steer_concept[64] = "";
            float steer_alpha = 0.0f;

            if (body) {
                body += 4;
                char *p_pos = strstr(body, "\"prompt\"");
                if (p_pos) {
                    char *p_start = strchr(p_pos + 8, '\"');
                    if (p_start) {
                        p_start++;
                        char *p_end = strchr(p_start, '\"');
                        if (p_end) {
                            size_t l = (size_t)(p_end - p_start);
                            if (l > sizeof(prompt) - 1) l = sizeof(prompt) - 1;
                            memcpy(prompt, p_start, l);
                            prompt[l] = 0;
                        }
                    }
                }
                char *t_pos = strstr(body, "\"temperature\"");
                if (t_pos) {
                    char *col = strchr(t_pos, ':');
                    if (col) temperature = (float)atof(col + 1);
                }
                char *m_pos = strstr(body, "\"max_tokens\"");
                if (m_pos) {
                    char *col = strchr(m_pos, ':');
                    if (col) max_tokens = atoi(col + 1);
                }
                char *sc_pos = strstr(body, "\"steer_concept\"");
                if (sc_pos) {
                    char *sc_start = strchr(sc_pos + 15, '\"');
                    if (sc_start) {
                        sc_start++;
                        char *sc_end = strchr(sc_start, '\"');
                        if (sc_end) {
                            size_t l = (size_t)(sc_end - sc_start);
                            if (l > sizeof(steer_concept) - 1) l = sizeof(steer_concept) - 1;
                            memcpy(steer_concept, sc_start, l);
                            steer_concept[l] = 0;
                        }
                    }
                }
                char *sa_pos = strstr(body, "\"steer_alpha\"");
                if (sa_pos) {
                    char *col = strchr(sa_pos, ':');
                    if (col) steer_alpha = (float)atof(col + 1);
                }
            }

            TCSteerConfig sc = { .layer = -1, .alpha = 0.0f, .vec = NULL };
            if (steer_bank && strlen(steer_concept) > 0 && strcmp(steer_concept, "none") != 0) {
                const TCSteerVector *sv = tc_steer_bank_find(steer_bank, steer_concept);
                if (sv) {
                    sc.layer = sv->layer;
                    sc.alpha = steer_alpha;
                    sc.vec = sv->vec;
                }
            }

            int ids[512];
            int plen = 0;
            char full_prompt[1024];
            if (bpe && model->cfg.vocab_size > 256 && strstr(model_path, "sft") != NULL && strncmp(prompt, "User:", 5) != 0) {
                snprintf(full_prompt, sizeof(full_prompt), "User: %s\nAssistant: ", prompt);
                plen = bpe_encode(bpe, full_prompt, ids, model->cfg.max_seq_len - 45);
            } else if (bpe && model->cfg.vocab_size > 256) {
                plen = bpe_encode(bpe, prompt, ids, model->cfg.max_seq_len - 45);
            } else {
                char clean[512];
                strncpy(clean, prompt, sizeof(clean) - 1);
                clean[sizeof(clean) - 1] = 0;
                tc_sanitize_text(clean);
                int l = (int)strlen(clean);
                if (l > model->cfg.max_seq_len - 45) l = model->cfg.max_seq_len - 45;
                for (int i = 0; i < l; i++) ids[plen++] = tc_encode_char(clean[i]);
            }
            if (plen < 1) { ids[0] = 0; plen = 1; }

            /* Generate response tokens */
            int cur_len = plen;
            int gen_count = 0;
            int V = model->cfg.vocab_size;
            char rolling_gen[1024] = {0};

            char json_out[BUFFER_SIZE];
            int w = snprintf(json_out, sizeof(json_out), "{\"generated_tokens\":[");

            while (cur_len < model->cfg.max_seq_len - 1 && gen_count < max_tokens) {
                float loss;
                if (sc.vec && fabsf(sc.alpha) > 1e-6f) {
                    tc_forward_steered(model, cache, ids, cur_len, NULL, &loss, &sc);
                } else {
                    tc_forward(model, cache, ids, cur_len, NULL, &loss);
                }
                const float *probs = cache->probs + (size_t)(cur_len - 1) * V;

                /* Sample next token */
                int next_id = 0;
                if (temperature <= 0.01f) {
                    float maxp = probs[0];
                    for (int i = 1; i < V; i++) if (probs[i] > maxp) { maxp = probs[i]; next_id = i; }
                } else {
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
                    for (int i = 0; i < V; i++) { scaled[i] = expf(scaled[i] - maxs); sum += scaled[i]; }
                    float r = ((float)rand() / (float)RAND_MAX) * sum;
                    float cum = 0.0f;
                    for (int i = 0; i < V; i++) {
                        cum += scaled[i];
                        if (cum >= r) { next_id = i; break; }
                    }
                }

                float p_next = probs[next_id];
                if (p_next < 1e-9f) p_next = 1e-9f;
                float surprise = -log2f(p_next);

                char piece[64] = {0};
                if (bpe && V > 256) bpe_decode(bpe, &next_id, 1, piece, sizeof(piece));
                else { piece[0] = tc_decode_id(next_id); piece[1] = 0; }

                /* Stop sequence detection */
                if (strstr(piece, "<|") || strstr(piece, "<|endoftext|>") ||
                    strstr(piece, "User:") || strstr(piece, "Assistant:")) {
                    break;
                }
                strncat(rolling_gen, piece, sizeof(rolling_gen) - strlen(rolling_gen) - 1);
                if (strstr(rolling_gen, "<|") || strstr(rolling_gen, "<|endoftext|>") ||
                    strstr(rolling_gen, "\nUser:") || strstr(rolling_gen, "User:")) {
                    break;
                }

                /* Escape quotes for JSON */
                char esc[128] = {0};
                int ep = 0;
                for (size_t i = 0; i < strlen(piece); i++) {
                    if (piece[i] == '\"') { esc[ep++] = '\\'; esc[ep++] = '\"'; }
                    else if (piece[i] == '\n') { esc[ep++] = '\\'; esc[ep++] = 'n'; }
                    else if (piece[i] == '\\') { esc[ep++] = '\\'; esc[ep++] = '\\'; }
                    else esc[ep++] = piece[i];
                }

                w += snprintf(json_out + w, sizeof(json_out) - (size_t)w,
                              "%s{\"text\":\"%s\",\"surprise\":%.3f,\"prob\":%.3f}",
                              gen_count > 0 ? "," : "", esc, surprise, p_next);

                ids[cur_len++] = next_id;
                gen_count++;
                if (gen_count >= 15 && (strchr(piece, '\n') || (strchr(piece, '.') && gen_count >= 25))) break;
            }

            w += snprintf(json_out + w, sizeof(json_out) - (size_t)w, "],\"prompt_tokens\":[");

            /* Prompt tokens */
            for (int i = 0; i < plen; i++) {
                char piece[64] = {0};
                if (bpe && V > 256) bpe_decode(bpe, &ids[i], 1, piece, sizeof(piece));
                else { piece[0] = tc_decode_id(ids[i]); piece[1] = 0; }
                char esc[128] = {0};
                int ep = 0;
                for (size_t j = 0; j < strlen(piece); j++) {
                    if (piece[j] == '\"') { esc[ep++] = '\\'; esc[ep++] = '\"'; }
                    else if (piece[j] == '\n') { esc[ep++] = '\\'; esc[ep++] = 'n'; }
                    else esc[ep++] = piece[j];
                }
                w += snprintf(json_out + w, sizeof(json_out) - (size_t)w,
                              "%s{\"text\":\"%s\"}", i > 0 ? "," : "", esc);
            }

            /* Causal Attributions */
            TCCausalStep c_steps[256];
            int mask_token = (V > 256) ? 32 : tc_encode_char(' ');
            tc_explain_causal(model, cache, ids, cur_len, mask_token, c_steps);

            TCGradStep g_steps[256];
            tc_explain_input_grad(model, cache, ids, cur_len, g_steps);

            w += snprintf(json_out + w, sizeof(json_out) - (size_t)w, "],\"causal_attributions\":[");
            for (int i = 0; i < plen; i++) {
                char piece[64] = {0};
                if (bpe && V > 256) bpe_decode(bpe, &ids[i], 1, piece, sizeof(piece));
                else { piece[0] = tc_decode_id(ids[i]); piece[1] = 0; }
                char esc[128] = {0};
                int ep = 0;
                for (size_t j = 0; j < strlen(piece); j++) {
                    if (piece[j] == '\"') { esc[ep++] = '\\'; esc[ep++] = '\"'; }
                    else if (piece[j] == '\n') { esc[ep++] = '\\'; esc[ep++] = 'n'; }
                    else esc[ep++] = piece[j];
                }
                float c_imp = (i < cur_len - 1) ? c_steps[i].importance : 0.0f;
                float g_imp = (i < cur_len - 1) ? g_steps[i].importance : 0.0f;
                w += snprintf(json_out + w, sizeof(json_out) - (size_t)w,
                              "%s{\"text\":\"%s\",\"causal_score\":%.4f,\"grad_score\":%.6f}",
                              i > 0 ? "," : "", esc, c_imp, g_imp);
            }

            /* Head importances */
            TCHeadImportance heads[128];
            int n_heads = tc_explain_heads(model, cache, ids, cur_len, heads);
            w += snprintf(json_out + w, sizeof(json_out) - (size_t)w, "],\"head_importances\":[");
            for (int i = 0; i < n_heads; i++) {
                w += snprintf(json_out + w, sizeof(json_out) - (size_t)w,
                              "%s{\"layer\":%d,\"head\":%d,\"importance\":%.4f}",
                              i > 0 ? "," : "", heads[i].layer, heads[i].head, heads[i].importance);
            }

            /* Grounding ratio check if Context: prefix exists */
            if (strncmp(prompt, "Context:", 8) == 0) {
                char *q_pos = strstr(prompt, "Question:");
                int ctx_tok_len = plen / 2;
                if (q_pos) {
                    size_t c_bytes = (size_t)(q_pos - prompt);
                    char sub[512] = {0};
                    if (c_bytes < sizeof(sub)) {
                        memcpy(sub, prompt, c_bytes);
                        int d[256];
                        if (bpe && V > 256) ctx_tok_len = bpe_encode(bpe, sub, d, 256);
                    }
                }
                float ctx_tot = 0.0f, qry_tot = 0.0f;
                for (int i = 0; i < plen && i < cur_len - 1; i++) {
                    if (i < ctx_tok_len) ctx_tot += c_steps[i].importance;
                    else qry_tot += c_steps[i].importance;
                }
                float g_ratio = (ctx_tot + qry_tot > 1e-6f) ? (ctx_tot / (ctx_tot + qry_tot) * 100.0f) : 50.0f;
                w += snprintf(json_out + w, sizeof(json_out) - (size_t)w, "],\"grounding_pct\":%.2f", g_ratio);
            } else {
                w += snprintf(json_out + w, sizeof(json_out) - (size_t)w, "]");
            }

            w += snprintf(json_out + w, sizeof(json_out) - (size_t)w,
                          ",\"steer\":{\"active\":%s,\"concept\":\"%s\",\"alpha\":%.2f,\"layer\":%d}}",
                          (sc.vec && fabsf(sc.alpha) > 1e-6f) ? "true" : "false",
                          steer_concept, sc.alpha, sc.layer);

            send_response(client_fd, "application/json", json_out, (size_t)w);
        } else {
            send_404(client_fd);
        }
        close(client_fd);
    }

    close(server_fd);
    tc_cache_free(cache);
    tc_paramset_free(model);
    if (steer_bank) tc_steer_bank_free(steer_bank);
    if (bpe) bpe_free(bpe);
    return 0;
}
