#ifndef TC_REGISTRY_H
#define TC_REGISTRY_H

#include "tcmodel.h"

/* A named, independently-loaded model + its own scratch forward/backward
 * cache. Multiple instances can share nothing but the code path -- each
 * owns its weights and cache outright. */
typedef struct {
    char name[64];
    TCParamSet *params;
    TCCache *cache;
} TCInstance;

typedef struct {
    TCInstance *items;
    int count;
    int capacity;
} TCRegistry;

TCRegistry *tc_registry_create(void);
void tc_registry_free(TCRegistry *r); /* frees every instance's params + cache too */

/* Takes ownership of params (registry will free it). Allocates a cache
 * sized to params->cfg. Returns 0 on success, -1 if name already used. */
int tc_registry_add(TCRegistry *r, const char *name, TCParamSet *params);

/* NULL if not found. */
TCInstance *tc_registry_get(TCRegistry *r, const char *name);

/* Frees and removes the named instance. Returns 0 on success, -1 if not found. */
int tc_registry_remove(TCRegistry *r, const char *name);

void tc_registry_list(const TCRegistry *r);

#endif
