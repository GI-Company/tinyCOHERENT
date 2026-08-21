/* RAG milestone: retrieval becomes a normal DAG node. Builds a small vector
 * store from short passages (embedded with the "embedder" instance from the
 * previous milestone), then runs a 3-node graph: retrieve -> generate
 * (writer, conditioned on the retrieved passage) -> explain (critic).
 *
 * The embedder is known to be recency-biased (see embed_demo.c): its
 * representation is dominated by the last couple of characters. Rather than
 * hide that, this demo also runs a second query that keeps a stored
 * passage's ending but swaps its subject, to show plainly what retrieval is
 * actually keying on before trusting it for anything real. */
#include "tcmodel.h"
#include "tokenizer.h"
#include "registry.h"
#include "vectorstore.h"
#include "dag.h"
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

static void print_full_ranking(TCParamSet *p, TCCache *cache, TCVectorStore *vs, const char *query) {
    int ids[256];
    int len = encode(query, ids, p->cfg.max_seq_len);
    float qvec[256];
    tc_embed(p, cache, ids, len, qvec);
    TCVSHit hits[8];
    int n = tc_vs_search(vs, qvec, vs->count, hits);
    printf("query: \"%s\"\n", query);
    for (int i = 0; i < n; i++)
        printf("  %.3f  \"%s\"\n", hits[i].score, vs->entries[hits[i].idx].text);
}

int main(int argc, char **argv) {
    /* Generative and embedder weights diverge once the embedder is
     * contrastively fine-tuned (that training updates the whole body, not
     * just pool_w), so writer/critic and the embedder now load different
     * files -- each registry entry gets weights suited to its own job. */
    const char *gen_path = argc > 1 ? argv[1] : "build/model.bin";
    const char *emb_path = argc > 2 ? argv[2] : "build/embedder.bin";

    TCParamSet *p_embedder = tc_paramset_load(emb_path);
    TCParamSet *p_writer = tc_paramset_load(gen_path);
    TCParamSet *p_critic = tc_paramset_load(gen_path);
    if (!p_embedder || !p_writer || !p_critic) {
        fprintf(stderr, "could not load %s / %s -- run `make train` and `make embed_train` first\n", gen_path, emb_path);
        return 1;
    }
    TCRegistry *reg = tc_registry_create();
    tc_registry_add(reg, "embedder", p_embedder);
    tc_registry_add(reg, "writer", p_writer);
    tc_registry_add(reg, "critic", p_critic);
    TCInstance *emb_inst = tc_registry_get(reg, "embedder");

    const char *docs[] = {
        "the cat sat on the mat",
        "the dog ran across the log",
        "the fox jumped over the hill",
        "a bird looked at the box",
        "a mouse walked past the road",
    };
    int n_docs = 5;
    TCVectorStore *vs = tc_vs_create(emb_inst->params->cfg.d_model);
    for (int i = 0; i < n_docs; i++) {
        int ids[256];
        int len = encode(docs[i], ids, emb_inst->params->cfg.max_seq_len);
        float vec[256];
        tc_embed(emb_inst->params, emb_inst->cache, ids, len, vec);
        tc_vs_add(vs, docs[i], vec);
    }

    printf("--- vector store: %d passages ---\n", n_docs);
    for (int i = 0; i < n_docs; i++) printf("  [%d] \"%s\"\n", i, docs[i]);

    printf("\n--- retrieval ranking: exact-match query ---\n");
    print_full_ranking(emb_inst->params, emb_inst->cache, vs, "the cat sat on the mat");

    printf("\n--- retrieval ranking: same ending, different subject ---\n");
    printf("(tests whether retrieval keys on the passage's ending or its subject)\n");
    print_full_ranking(emb_inst->params, emb_inst->cache, vs, "a mouse walked past the mat");

    printf("\n--- 3-node graph: retrieve -> generate -> explain ---\n");
    TCNode nodes[3] = { {0}, {0}, {0} };
    snprintf(nodes[0].id, sizeof(nodes[0].id), "retrieve");
    nodes[0].op = TC_NODE_RETRIEVE;
    snprintf(nodes[0].instance_name, sizeof(nodes[0].instance_name), "embedder");
    nodes[0].top_k = 1;

    snprintf(nodes[1].id, sizeof(nodes[1].id), "gen");
    nodes[1].op = TC_NODE_GENERATE;
    snprintf(nodes[1].instance_name, sizeof(nodes[1].instance_name), "writer");
    snprintf(nodes[1].input_from, sizeof(nodes[1].input_from), "retrieve");
    nodes[1].max_new_tokens = 25;
    nodes[1].temperature = 0.6f;

    snprintf(nodes[2].id, sizeof(nodes[2].id), "score");
    nodes[2].op = TC_NODE_EXPLAIN;
    snprintf(nodes[2].instance_name, sizeof(nodes[2].instance_name), "critic");
    snprintf(nodes[2].input_from, sizeof(nodes[2].input_from), "gen");

    TCGraph g = { nodes, 3 };
    TCNodeResult results[3];
    const char *query = "the cat sat on the mat";
    printf("query prompt fed to the graph: \"%s\"\n\n", query);

    if (tc_graph_run(&g, reg, vs, query, results) != 0) {
        fprintf(stderr, "graph run failed\n");
        return 1;
    }

    printf("node 'retrieve' (embedder) -> top hit (score %.3f): \"%s\"\n",
           results[0].hits[0].score, results[0].text);
    printf("node 'gen' (writer, conditioned on the retrieved passage) -> \"%s\"\n", results[1].text);
    printf("node 'score' (critic) -> mean surprise = %.3f bits/char over that generation\n",
           results[2].avg_surprise);

    tc_vs_free(vs);
    tc_registry_free(reg);
    return 0;
}
