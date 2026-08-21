#include "registry.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

TCRegistry *tc_registry_create(void) {
    TCRegistry *r = calloc(1, sizeof(TCRegistry));
    r->capacity = 4;
    r->items = calloc((size_t)r->capacity, sizeof(TCInstance));
    r->count = 0;
    return r;
}

void tc_registry_free(TCRegistry *r) {
    if (!r) return;
    for (int i = 0; i < r->count; i++) {
        tc_cache_free(r->items[i].cache);
        tc_paramset_free(r->items[i].params);
    }
    free(r->items);
    free(r);
}

TCInstance *tc_registry_get(TCRegistry *r, const char *name) {
    for (int i = 0; i < r->count; i++)
        if (strncmp(r->items[i].name, name, sizeof(r->items[i].name)) == 0)
            return &r->items[i];
    return NULL;
}

int tc_registry_add(TCRegistry *r, const char *name, TCParamSet *params) {
    if (tc_registry_get(r, name)) return -1;
    if (r->count == r->capacity) {
        r->capacity *= 2;
        r->items = realloc(r->items, (size_t)r->capacity * sizeof(TCInstance));
    }
    TCInstance *inst = &r->items[r->count++];
    memset(inst, 0, sizeof(*inst));
    strncpy(inst->name, name, sizeof(inst->name) - 1);
    inst->params = params;
    inst->cache = tc_cache_create(params->cfg);
    return 0;
}

int tc_registry_remove(TCRegistry *r, const char *name) {
    for (int i = 0; i < r->count; i++) {
        if (strncmp(r->items[i].name, name, sizeof(r->items[i].name)) == 0) {
            tc_cache_free(r->items[i].cache);
            tc_paramset_free(r->items[i].params);
            r->items[i] = r->items[r->count - 1];
            r->count--;
            return 0;
        }
    }
    return -1;
}

void tc_registry_list(const TCRegistry *r) {
    printf("registry: %d instance(s)\n", r->count);
    for (int i = 0; i < r->count; i++) {
        TCInstance *inst = &r->items[i];
        printf("  - %-12s params=%p (%d floats) cache=%p\n",
               inst->name, (void *)inst->params->buf, inst->params->n_floats, (void *)inst->cache);
    }
}
