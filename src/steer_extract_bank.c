#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tcmodel.h"
#include "steer.h"
#include "bpe.h"

typedef struct {
    const char *name;
    const char *desc;
    const char *pos;
    const char *neg;
} ConceptPair;

static const ConceptPair CONCEPTS[] = {
    {
        "joy",
        "Joy and cheerfulness vs grief and sorrow",
        "The happy celebration brought immense joy, bright smiles, cheerful laughter, delight, and wonderful excitement to everyone.",
        "The grim tragedy brought immense sorrow, cold tears, weeping grief, mourning, and terrible despair to everyone."
    },
    {
        "danger",
        "Extreme peril and threat vs peace and safety",
        "Beware of lethal danger, deadly poison, vicious predators, catastrophic traps, hazard, and violent peril in the dark.",
        "Rest safely in tranquil peace, gentle calm, warm protection, harmless shelter, secure comfort, and serene stillness."
    },
    {
        "magic",
        "Arcane sorcery and enchantment vs mundane ordinary objects",
        "The ancient wizard cast a mystical arcane spell with enchanted crystal runes, glowing magical sorcery, and wonder.",
        "The plain clerk did standard routine work with common metal tools, normal ordinary papers, and mundane items."
    },
    {
        "nature",
        "Wild blooming forest and rivers vs dense stone and industrial metal",
        "The lush wild forest bloomed with tall green trees, flowing fresh rivers, sweet fragrant flowers, and vibrant wildlife.",
        "The dense gray industrial city was built of hard cold stone, steel machinery, noisy factories, and dirty concrete."
    }
};

int main(int argc, char **argv) {
    const char *model_path = (argc > 1) ? argv[1] : "build/model_sft.bin";
    const char *bpe_path = (argc > 2) ? argv[2] : "data/bpe_merges.txt";
    const char *out_path = (argc > 3) ? argv[3] : "data/steering_vectors.bin";

    printf("=== EXTRACTING CURATED STEERING BANK ===\n");
    printf("Model:  %s\n", model_path);
    printf("BPE:    %s\n", bpe_path);
    printf("Output: %s\n", out_path);

    TCParamSet *p = tc_paramset_load(model_path);
    if (!p) {
        model_path = "build/model_rung4.bin";
        p = tc_paramset_load(model_path);
        if (!p) {
            fprintf(stderr, "Error: Could not open model file\n");
            return 1;
        }
    }

    BPETokenizer *bpe = bpe_load(bpe_path);
    if (!bpe) {
        fprintf(stderr, "Error: Could not load BPE merges\n");
        return 1;
    }

    TCCache *c = tc_cache_create(p->cfg);
    TCSteerBank *bank = tc_steer_bank_create();

    int target_layer = p->cfg.n_layers > 2 ? 2 : 1;
    int n_concepts = sizeof(CONCEPTS) / sizeof(CONCEPTS[0]);

    float *vec = malloc(p->cfg.d_model * sizeof(float));

    for (int i = 0; i < n_concepts; i++) {
        const ConceptPair *cp = &CONCEPTS[i];
        int pos_ids[128], neg_ids[128];
        int pos_len = bpe_encode(bpe, cp->pos, pos_ids, 128);
        int neg_len = bpe_encode(bpe, cp->neg, neg_ids, 128);

        int err = tc_steer_extract_direction(p, c, pos_ids, pos_len, neg_ids, neg_len, target_layer, vec);
        if (err != 0) {
            fprintf(stderr, "Error extracting concept: %s\n", cp->name);
            continue;
        }

        tc_steer_bank_add(bank, cp->name, cp->desc, target_layer, p->cfg.d_model, vec);

        printf("\nExtracted concept [%s] at layer %d:\n", cp->name, target_layer);
        printf("  Description: %s\n", cp->desc);

        int top_pos[4], top_neg[4];
        float delta_pos[4], delta_neg[4];
        tc_steer_inspect_boost(p, vec, 4, top_pos, delta_pos, top_neg, delta_neg);

        printf("  Top promoted:   ");
        for (int k = 0; k < 4; k++) {
            char tok[64];
            bpe_decode(bpe, &top_pos[k], 1, tok, sizeof(tok));
            printf("'%s' (%+.2f)  ", tok, delta_pos[k]);
        }
        printf("\n  Top suppressed: ");
        for (int k = 0; k < 4; k++) {
            char tok[64];
            bpe_decode(bpe, &top_neg[k], 1, tok, sizeof(tok));
            printf("'%s' (%+.2f)  ", tok, delta_neg[k]);
        }
        printf("\n");
    }

    if (tc_steer_save_bank(bank, out_path) != 0) {
        fprintf(stderr, "Error saving bank to %s\n", out_path);
        return 1;
    }

    printf("\nSuccessfully saved %d concept steering vectors to %s!\n", bank->count, out_path);

    free(vec);
    tc_steer_bank_free(bank);
    tc_cache_free(c);
    tc_paramset_free(p);
    bpe_free(bpe);
    return 0;
}
