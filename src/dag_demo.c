/* Registry + trivial DAG milestone: load the trained specialist twice under
 * two independent names to prove the registry holds separate instances,
 * then run a 2-node graph where node A generates text and node B (a
 * different registry entry) scores that text with the glass-box surprise
 * trace -- output of one specialist feeding the input of another. */
#include "registry.h"
#include "dag.h"
#include <stdio.h>
#include <stdlib.h>

static char printable(char c) { return c == '\n' ? ' ' : c; }

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "build/model.bin";

    TCRegistry *reg = tc_registry_create();

    TCParamSet *writer = tc_paramset_load(model_path);
    TCParamSet *critic = tc_paramset_load(model_path);
    if (!writer || !critic) {
        fprintf(stderr, "could not load %s -- run `make train` first\n", model_path);
        return 1;
    }
    tc_registry_add(reg, "writer", writer);
    tc_registry_add(reg, "critic", critic);
    tc_registry_list(reg);
    printf("\n");

    TCNode nodes[2] = { {0}, {0} };

    snprintf(nodes[0].id, sizeof(nodes[0].id), "gen");
    nodes[0].op = TC_NODE_GENERATE;
    snprintf(nodes[0].instance_name, sizeof(nodes[0].instance_name), "writer");
    nodes[0].max_new_tokens = 40;
    nodes[0].temperature = 0.7f;

    snprintf(nodes[1].id, sizeof(nodes[1].id), "score");
    nodes[1].op = TC_NODE_EXPLAIN;
    snprintf(nodes[1].instance_name, sizeof(nodes[1].instance_name), "critic");
    snprintf(nodes[1].input_from, sizeof(nodes[1].input_from), "gen");

    TCGraph g = { nodes, 2 };
    TCNodeResult results[2];

    const char *prompt = "the cat sat on";
    printf("graph: [gen: writer, generate] -> [score: critic, explain]\n");
    printf("initial prompt: \"%s\"\n\n", prompt);

    if (tc_graph_run(&g, reg, NULL, prompt, results) != 0) {
        fprintf(stderr, "graph run failed\n");
        return 1;
    }

    printf("node 'gen' (writer) output:\n  \"%s\"\n\n", results[0].text);

    printf("node 'score' (critic) glass-box trace on that output:\n");
    printf("  mean surprise = %.3f bits/char\n\n", results[1].avg_surprise);
    for (int t = 0; t < results[1].n_steps; t++) {
        TCGlassBoxStep *s = &results[1].steps[t];
        printf("  '%c'  p=%.3f  surprise=%.2f bits  top-3: '%c':%.2f '%c':%.2f '%c':%.2f\n",
               printable(s->ch), s->prob, s->surprise,
               printable(s->topk_chars[0]), s->topk_probs[0],
               printable(s->topk_chars[1]), s->topk_probs[1],
               printable(s->topk_chars[2]), s->topk_probs[2]);
    }

    printf("\nsimple DAG routing check: if mean surprise > 2.0 bits, would call a fact-checker specialist -> %s\n",
           results[1].avg_surprise > 2.0f ? "YES (would route)" : "no (confident enough)");

    tc_registry_free(reg);
    return 0;
}
