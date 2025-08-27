import os
os.environ.setdefault("HF_HOME", "/data/mjh24/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/mjh24/hf/transformers")
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-large"  # or "google/flan-t5-base" for more accuracy

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

LABELS = {
    1: "added_context",
    2: "added_information",
    3: "purpose_of_text",
    4: "sibling_content",
    5: "unrelated",
}

def clean_instructional_text(pairs, batch_size=32, max_new_tokens=2, device=None):
    """
    pairs: list of [left, right]
    returns: list of ints (0/1), one per pair; 1 if either text node is classified as 1
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Flatten texts: [L0, R0, L1, R1, ...]
    texts = []
    for left, right in pairs:
        texts.append(str(left) if left is not None else "")
        texts.append(str(right) if right is not None else "")

    texts = list(set(texts))

    # Classify each text node individually
    node_preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        prompts = [f"""Decide if the text looks like a website button or navigation label.
                Return 1 for button/navigation, or 0 otherwise. Output only a single digit.

                Examples of 1: "Read more", "Learn more", "Explore", "Get started", "Try free", "Watch video",
                "Sign in", "Sign up", "My account", "Download", "Contact", "Pricing", "Products", "Docs",
                "Blog", "Home", "About", "Privacy Policy", "Terms", "Fr/En", "Menu", "Next", "Previous", "Back".

                Examples of 0: sentences or descriptive copy, names of people or products in context, long summaries.

                TEXT: {text}

                Answer with 1 or 0 only:
                """ for text in batch_texts
                ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for y in decoded:
            y = y.strip()
            node_preds.append(1 if (len(y) > 0 and y[0] == "1") else 0)

    isTxtInstruct = {}
    for idx, txt in enumerate(texts):
        isTxtInstruct[txt] = node_preds[idx]

    # Reduce back to pair-level: OR(left, right)
    results = []
    for a, b in pairs:
        if isTxtInstruct[a] == 1 or isTxtInstruct[b] == 1:
            results.append(1)
        else:
            results.append(0)

    return results, isTxtInstruct

# def classify_a_relation_batched(pairs, batch_size=16, max_new_tokens=4, device=None, bidirectional=True):
#     """
#     pairs: list of [left, right]
#     returns: list of ints (0-1), one per pair
#     """
#     device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     results = []
#     for i in range(0, len(pairs), batch_size):
#         batch = pairs[i:i+batch_size]

#         prompts = [
#             f"""Decide if the pair [A, B] can be 
# Pair:
# ["{left}","{right}"]

# Answer with 0 or 1 only (no words, no punctuation):"""
#             for left, right in batch
#         ]

#         inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, output_scores=True)

#         decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

#         for text in decoded:
#             text = text.strip()
#             try:
#                 idx = int(text[0])
#                 if idx not in LABELS:
#                     idx = 0
#             except:
#                 idx = 0
#             results.append(idx)

#     return results

# def classify_link_pairs_flan_batched(pairs, batch_size=16, max_new_tokens=4, device=None):
#     """
#     pairs: list of [left, right]
#     returns: list of ints (1-5), one per pair
#     """
#     device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     results = []
#     for i in range(0, len(pairs), batch_size):
#         batch = pairs[i:i+batch_size]

#         prompts = [
#             f"""Classify the relation between these two texts into one category:
# 1 = added_context (one side is the attribute, metadata, or category of the other)
# 2 = added_information (one side adds missing information to the other)
# 3 = purpose_of_text (one side is the title of the other)
# 4 = sibling_content (two peer items/categories)
# 5 = unhelpful

# ["{left}", "{right}"]

# Be absolutely sure, otherwise return "5". Answer with the number only:"""
#             for left, right in batch
#         ]

#         inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, output_scores=True)

#         decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

#         for text in decoded:
#             text = text.strip()
#             try:
#                 idx = int(text[0])
#                 if idx not in LABELS:
#                     idx = 5
#             except:
#                 idx = 5
#             results.append(idx)

#     return results


if __name__ == "__main__":
    sample_pairs = [["british columbia canada", "set in"], ["set in", "british columbia canada"], ["for sexuality and some language", "mpaa reasons"], ["mpaa reasons", "for sexuality and some language"], ["addict", "accident"], ["accident", "addict"], ["other related works", "is related to"], ["is related to", "other related works"], ["drugs", "accident"], ["accident", "drugs"], ["in a minor key", "moods"], ["moods", "in a minor key"], ["drugs", "addict"], ["addict", "drugs"], ["canada", "r"], ["r", "canada"], ["director", "atom egoyan"], ["atom egoyan", "director"], ["panavision", "corrections to this entry"], ["lawyer", "accident"], ["accident", "lawyer"], ["category", "feature"], ["feature", "category"], ["lawyer", "addict"], ["addict", "lawyer"], ["year", "1997"], ["1997", "year"], ["drama", "genres"], ["genres", "drama"], ["panavision", "cinematic process"], ["cinematic process", "panavision"], ["british columbia canada", "corrections to this entry"], ["lawyer", "drugs"], ["drugs", "lawyer"]]
    labels = clean_instructional_text(sample_pairs, batch_size=64)
    for pair, label in zip(sample_pairs, labels):
        print(label, pair)