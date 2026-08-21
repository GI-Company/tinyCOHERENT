#include "dag.h"
#include "tokenizer.h"
#include <string.h>
#include <stdio.h>

int tc_graph_run(const TCGraph *g, TCRegistry *reg, const TCVectorStore *vs,
                  const char *initial_prompt, TCNodeResult *out_results) {
    for (int i = 0; i < g->n_nodes; i++) {
        const TCNode *node = &g->nodes[i];
        TCInstance *inst = tc_registry_get(reg, node->instance_name);
        if (!inst) { fprintf(stderr, "dag: unknown instance '%s'\n", node->instance_name); return -1; }
        if (node->op == TC_NODE_RETRIEVE && !vs) { fprintf(stderr, "dag: node '%s' needs a vector store\n", node->id); return -3; }

        const char *input_text = initial_prompt;
        if (node->input_from[0] != 0) {
            int found = -1;
            for (int j = 0; j < i; j++)
                if (strcmp(g->nodes[j].id, node->input_from) == 0) { found = j; break; }
            if (found < 0) { fprintf(stderr, "dag: input_from '%s' not found before node '%s'\n", node->input_from, node->id); return -2; }
            input_text = out_results[found].text;
        }

        char sanitized[TC_NODE_TEXT_CAP];
        strncpy(sanitized, input_text, sizeof(sanitized) - 1);
        sanitized[sizeof(sanitized) - 1] = 0;
        tc_sanitize_text(sanitized);
        int len = (int)strlen(sanitized);
        if (len > inst->params->cfg.max_seq_len) len = inst->params->cfg.max_seq_len;
        int ids[TC_NODE_TEXT_CAP];
        for (int k = 0; k < len; k++) ids[k] = tc_encode_char(sanitized[k]);

        TCNodeResult *res = &out_results[i];
        memset(res, 0, sizeof(*res));

        if (node->op == TC_NODE_GENERATE) {
            unsigned int rng = 1000u + (unsigned int)i;
            int newlen = tc_generate(inst->params, inst->cache, ids, len,
                                      node->max_new_tokens, node->temperature, &rng);
            if (newlen > TC_NODE_TEXT_CAP - 1) newlen = TC_NODE_TEXT_CAP - 1;
            for (int k = 0; k < newlen; k++) res->text[k] = tc_decode_id(ids[k]);
            res->text[newlen] = 0;
        } else if (node->op == TC_NODE_EXPLAIN) {
            strncpy(res->text, sanitized, sizeof(res->text) - 1);
            int n = tc_explain_generate(inst->params, inst->cache, ids, len, res->steps);
            res->n_steps = n;
            float sum = 0.0f;
            for (int k = 0; k < n; k++) sum += res->steps[k].surprise;
            res->avg_surprise = n > 0 ? sum / n : 0.0f;
        } else { /* TC_NODE_RETRIEVE */
            float query_vec[256];
            tc_embed(inst->params, inst->cache, ids, len, query_vec);
            int k = node->top_k > 0 ? node->top_k : 1;
            if (k > TC_MAX_HITS) k = TC_MAX_HITS;
            res->n_hits = tc_vs_search(vs, query_vec, k, res->hits);

            size_t pos = 0;
            for (int h = 0; h < res->n_hits && pos < sizeof(res->text) - 1; h++) {
                const char *hit_text = vs->entries[res->hits[h].idx].text;
                size_t hl = strlen(hit_text);
                if (pos + hl >= sizeof(res->text) - 1) hl = sizeof(res->text) - 1 - pos;
                memcpy(res->text + pos, hit_text, hl);
                pos += hl;
                if (pos < sizeof(res->text) - 1) res->text[pos++] = ' ';
            }
            res->text[pos] = 0;
        }
    }
    return 0;
}
