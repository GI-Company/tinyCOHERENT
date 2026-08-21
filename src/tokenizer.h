#ifndef TC_TOKENIZER_H
#define TC_TOKENIZER_H

/* Fixed 96-symbol vocab: id 0 = '\n', ids 1..95 = ASCII 32..126 (printable).
 * Any other byte should be sanitized to one of these before encoding
 * (see tc_sanitize_text) so encode/decode are always lossless round trips
 * over the reduced alphabet. */

#define TC_VOCAB_SIZE 96

int tc_encode_char(char c);   /* byte -> token id (assumes already sanitized) */
char tc_decode_id(int id);    /* token id -> byte */

/* In place: replaces any byte outside the vocab (except '\n') with a space. */
void tc_sanitize_text(char *text);

#endif
