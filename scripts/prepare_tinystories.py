#!/usr/bin/env python3
"""Extract a size-bounded, sanitize-friendly subset of TinyStories for
data/tinystories_subset.txt. Not committed to git (see SCALE_200k.md) --
TinyStories redistribution terms weren't verified, so this script is the
reproducible source of truth instead of the extracted text.

Usage: python3 scripts/prepare_tinystories.py <path-to-TinyStoriesV2-GPT4-valid.txt>
"""
import sys

TARGET_BYTES = 4_000_000

REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
}

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]

    with open(src, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    stories = [s.strip() for s in raw.split("<|endoftext|>") if s.strip()]

    out_stories = []
    total = 0
    for s in stories:
        if total + len(s) + 2 > TARGET_BYTES:
            break
        out_stories.append(s)
        total += len(s) + 2

    text = "\n\n".join(out_stories) + "\n"
    for k, v in REPLACEMENTS.items():
        text = text.replace(k, v)

    out_path = "data/tinystories_subset.txt"
    with open(out_path, "w", encoding="ascii", errors="replace") as f:
        f.write(text)

    print(f"{len(out_stories)} stories, {len(text)} chars written to {out_path}")

if __name__ == "__main__":
    main()
