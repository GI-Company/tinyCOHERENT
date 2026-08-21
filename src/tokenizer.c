#include "tokenizer.h"

int tc_encode_char(char c) {
    unsigned char u = (unsigned char)c;
    if (u == '\n') return 0;
    if (u >= 32 && u <= 126) return 1 + (u - 32);
    return 1; /* space, defensive fallback */
}

char tc_decode_id(int id) {
    if (id == 0) return '\n';
    return (char)(32 + (id - 1));
}

void tc_sanitize_text(char *text) {
    for (char *p = text; *p; p++) {
        unsigned char u = (unsigned char)*p;
        if (u == '\n') continue;
        if (u >= 32 && u <= 126) continue;
        *p = ' ';
    }
}
