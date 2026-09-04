#!/usr/bin/env python3
"""
Generates 3,000+ grounded instruction-following dialogue pairs from data/tinystories_subset.txt.
Formats each example with strict delimiters:
User: <question or instruction>
Assistant: <grounded response>
<|endoftext|>
"""

import re
import random

random.seed(42)

def clean_sentence(s):
    s = s.strip().strip('"').strip("'")
    if not s:
        return ""
    if not s[0].isupper():
        s = s[0].upper() + s[1:]
    if not s.endswith(('.', '!', '?')):
        s += '.'
    return s

def format_name(name):
    if name.lower() in ("he", "she", "they", "it"):
        return name.lower()
    return name

def extract_qa_pairs(story_text):
    pairs = []
    sentences = re.split(r'(?<=[.!?])\s+', story_text.replace('\n', ' '))
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    for s in sentences:
        s_clean = clean_sentence(s)
        # Pattern 1: "X was Y because Z"
        m = re.search(r'\b([A-Z][a-z]+) was ([a-z]+) because (.+)', s_clean)
        if m:
            name, feeling, reason = m.groups()
            name_q = format_name(name)
            reason = reason.rstrip('.!?')
            q = f"Why was {name_q} {feeling}?"
            ans = f"{name} was {feeling} because {reason}."
            pairs.append((q, ans))

        # Pattern 2: "X felt Y because Z"
        m = re.search(r'\b([A-Z][a-z]+) felt ([a-z]+) because (.+)', s_clean)
        if m:
            name, feeling, reason = m.groups()
            name_q = format_name(name)
            reason = reason.rstrip('.!?')
            q = f"How did {name_q} feel, and why?"
            ans = f"{name} felt {feeling} because {reason}."
            pairs.append((q, ans))

        # Pattern 3: "X liked to Y"
        m = re.search(r'\b([A-Z][a-z]+) liked to ([a-z\s]+?)(?: in | on | with | near |\.)', s_clean)
        if m:
            name, activity = m.groups()
            name_q = format_name(name)
            q = f"What did {name_q} like to do?"
            ans = f"{name} liked to {activity.strip()}."
            pairs.append((q, ans))

        # Pattern 4: "X found a Y"
        m = re.search(r'\b([A-Z][a-z]+) found (a [a-z\s]+?)(?: in | under | on |\.)', s_clean)
        if m:
            name, obj = m.groups()
            name_q = format_name(name)
            q = f"What did {name_q} find?"
            ans = f"{name} found {obj.strip()}."
            pairs.append((q, ans))

        # Pattern 5: "X went to Y to Z"
        m = re.search(r'\b([A-Z][a-z]+) went to (the [a-z]+) to ([a-z\s]+)', s_clean)
        if m:
            name, place, purpose = m.groups()
            name_q = format_name(name)
            q = f"Where did {name_q} go and why?"
            ans = f"{name} went to {place} to {purpose}."
            pairs.append((q, ans))

        # Pattern 6: "X decided to Y"
        m = re.search(r'\b([A-Z][a-z]+) decided to ([a-z\s]+)', s_clean)
        if m:
            name, action = m.groups()
            name_q = format_name(name)
            q = f"What did {name_q} decide to do?"
            ans = f"{name} decided to {action}."
            pairs.append((q, ans))

        # Pattern 7: "X learned that Y"
        m = re.search(r'\b([A-Z][a-z]+) learned that ([a-z\s]+)', s_clean)
        if m:
            name, lesson = m.groups()
            name_q = format_name(name)
            q = f"What lesson did {name_q} learn?"
            ans = f"{name} learned that {lesson}."
            pairs.append((q, ans))

        # Pattern 8: Direct question / answer on character actions
        m = re.search(r'\b([A-Z][a-z]+) wanted to ([a-z\s]+)', s_clean)
        if m:
            name, desire = m.groups()
            name_q = format_name(name)
            q = f"What did {name_q} want to do?"
            ans = f"{name} wanted to {desire}."
            pairs.append((q, ans))

        # Pattern 9: Summarization / Short Story
        if 40 < len(s_clean) < 140 and any(s_clean.startswith(prefix) for prefix in ["Once upon a time", "One day", "Suddenly"]):
            q = "Tell me what happened next."
            ans = s_clean
            pairs.append((q, ans))

    return pairs

def main():
    with open("data/tinystories_subset.txt", "r", encoding="utf-8") as f:
        corpus = f.read()

    stories = corpus.split("\n\n")
    all_pairs = []
    for story in stories:
        if len(story.strip()) < 50:
            continue
        p = extract_qa_pairs(story)
        all_pairs.extend(p)

    # Add core grounding QA pairs
    core_qa = [
        ("What is your name?", "I am TinyCoherent, a glass-box language model."),
        ("Who created you?", "I was built from scratch as a glass-box specialist."),
        ("What is a glass-box model?", "A glass-box model provides verified, falsifiable causal attributions for every output."),
        ("How do you know which words were important?", "I use causal occlusion and gradient sensitivity to measure how each input affects my predictions."),
        ("Can you help me answer questions?", "Yes, I can answer questions grounded in facts and stories."),
        ("What should Lily do when she feels sad?", "Lily can talk to her friends or hug her puppy to feel better."),
        ("Why is sharing toys important?", "Sharing toys makes everyone happy and builds strong friendships."),
        ("What happens when children work together?", "When children work together, they can build tall towers and solve big problems."),
    ] * 25
    all_pairs.extend(core_qa)

    # Deduplicate while preserving variety
    seen = set()
    unique_pairs = []
    for q, a in all_pairs:
        key = (q.strip(), a.strip())
        if key not in seen:
            seen.add(key)
            unique_pairs.append((q.strip(), a.strip()))

    random.shuffle(unique_pairs)
    # Take up to 3500 pairs
    selected = unique_pairs[:3500]

    output_path = "data/sft_dialogue.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for q, a in selected:
            f.write(f"User: {q}\n")
            f.write(f"Assistant: {a}\n")
            f.write("<|endoftext|>\n\n")

    print(f"Generated {len(selected)} high-signal SFT dialogue pairs in {output_path}")

if __name__ == "__main__":
    main()
