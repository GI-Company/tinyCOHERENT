#ifndef TC_DAG_H
#define TC_DAG_H

#include "registry.h"
#include "glassbox.h"
#include "vectorstore.h"

/* Trivial DAG: nodes run in the given array order (the caller is
 * responsible for listing them in a valid dependency order -- a real
 * topological sort arrives once graphs stop being hand-written and get
 * complex enough to need one). Each node reads its input text either from
 * the graph's initial prompt or from an upstream node's output text,
 * looked up by node id. */

typedef enum { TC_NODE_GENERATE, TC_NODE_EXPLAIN, TC_NODE_RETRIEVE } TCNodeOp;

typedef struct {
    char id[64];
    TCNodeOp op;
    char instance_name[64]; /* which registry entry runs this node */
    char input_from[64];    /* upstream node id, or "" for the initial prompt */
    int max_new_tokens;     /* TC_NODE_GENERATE only */
    float temperature;      /* TC_NODE_GENERATE only; <=0 means greedy */
    int top_k;               /* TC_NODE_RETRIEVE only: how many hits to fuse into the output text */
} TCNode;

typedef struct {
    TCNode *nodes;
    int n_nodes;
} TCGraph;

#define TC_NODE_TEXT_CAP 512

#define TC_MAX_HITS 8

typedef struct {
    char text[TC_NODE_TEXT_CAP];  /* node's text output: generated text, sanitized input echoed back
                                    * for explain nodes, or the fused top hit(s) for retrieve nodes */
    TCGlassBoxStep steps[256];    /* filled for TC_NODE_EXPLAIN */
    int n_steps;
    float avg_surprise;           /* mean surprise in bits, TC_NODE_EXPLAIN only */
    TCVSHit hits[TC_MAX_HITS];    /* filled for TC_NODE_RETRIEVE, best first */
    int n_hits;
} TCNodeResult;

/* vs may be NULL if the graph has no TC_NODE_RETRIEVE nodes. out_results
 * must have g->n_nodes entries. Returns 0 on success, -1 if a node's
 * instance_name isn't registered, -2 if input_from doesn't match any
 * earlier node's id, -3 if a retrieve node runs with vs == NULL. */
int tc_graph_run(const TCGraph *g, TCRegistry *reg, const TCVectorStore *vs,
                  const char *initial_prompt, TCNodeResult *out_results);

#endif
