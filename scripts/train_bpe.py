#!/usr/bin/env python3
"""Fast byte-level BPE tokenizer training based on word frequencies.

Exports merges to a text file with format:
<vocab_size> <num_merges>
<token_id1> <token_id2> <merged_token_id>
"""
import sys
import re
from collections import defaultdict

def train_bpe(input_path, output_path, target_vocab_size=2048, max_bytes=2_000_000):
    print(f"Reading {input_path} (up to {max_bytes} bytes)...")
    with open(input_path, "rb") as f:
        raw = f.read(max_bytes)

    # Split into words/chunks preserving delimiters
    # Using regex to split words keeping spaces and punctuation
    words = re.findall(rb"\s+|\w+|[^\w\s]", raw)
    word_freqs = defaultdict(int)
    for w in words:
        word_freqs[tuple(w)] += 1

    print(f"Found {len(words)} tokens, {len(word_freqs)} unique word pieces.")

    num_merges = target_vocab_size - 256
    merges = []

    # Map from word tuple -> current segmented tuple
    vocab_words = {w: list(w) for w in word_freqs}

    print(f"Training {num_merges} BPE merges to reach vocab={target_vocab_size}...")

    for step in range(num_merges):
        # Count all pairs across unique words
        pair_counts = defaultdict(int)
        for w, freq in word_freqs.items():
            symbols = vocab_words[w]
            for i in range(len(symbols) - 1):
                pair_counts[(symbols[i], symbols[i + 1])] += freq

        if not pair_counts:
            break

        best_pair = max(pair_counts, key=pair_counts.get)
        new_id = 256 + step
        merges.append((best_pair, new_id))

        # Update all words containing this pair
        p0, p1 = best_pair
        for w in word_freqs:
            symbols = vocab_words[w]
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == p0 and symbols[i + 1] == p1:
                    new_symbols.append(new_id)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            vocab_words[w] = new_symbols

        if (step + 1) % 250 == 0 or (step + 1) == num_merges:
            total_tokens = sum(len(vocab_words[w]) * freq for w, freq in word_freqs.items())
            ratio = len(raw) / total_tokens if total_tokens > 0 else 1.0
            print(f"  Step {step + 1}/{num_merges}: merged {best_pair} -> {new_id} (count={pair_counts[best_pair]}), compression={ratio:.2f}x")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{256 + len(merges)} {len(merges)}\n")
        for (p0, p1), new_id in merges:
            f.write(f"{p0} {p1} {new_id}\n")

    print(f"Done! Exported {len(merges)} merges to {output_path} (total vocab {256 + len(merges)})")

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/tinystories_subset.txt"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/bpe_merges.txt"
    vsize = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    train_bpe(inp, out, vsize)
