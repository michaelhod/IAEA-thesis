from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-large"  # or "google/flan-t5-base" for more accuracy

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

LABELS = {
    1: "description_of_text",
    2: "added_context",
    3: "sibling_content",
    4: "contains_instruction",
    5: "unrelated",
}

def classify_link_pairs_flan_batched(pairs, batch_size=16, max_new_tokens=4, device=None):
    """
    pairs: list of [left, right]
    returns: list of ints (1-5), one per pair
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]

        prompts = [
            f"""Classify the relation between these two texts into one category:
1 = description_of_text (one side describes the other, e.g. ["plot synopsis", "This movie is about Madagascar"])
2 = added_context (one side is metadata/attribute of the other, e.g. ["Director", "Christopher Nolan"])
3 = sibling_content (two peer items/categories, e.g. ["afcteams", "nfcteams"])
4 = contains_instruction (a call-to-action link, e.g. ["Read more", "Explore"])
5 = unrelated (no meaningful relation)

["{left}", "{right}"]

Answer with the number only:"""
            for left, right in batch
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in decoded:
            text = text.strip()
            try:
                idx = int(text[0])
                if idx not in LABELS:
                    idx = 5
            except:
                idx = 5
            results.append(idx)

    return results


if __name__ == "__main__":
    sample_pairs = [
        ['learn more', 'learn more about why databricks selected westinghouse ai'],
        ['Director', 'Christopher Nolan'],
        ['afcteams', 'nfcteams'],
        ['Read more', 'Imperial College London'],
        ['Location', 'Bob Mortimer'],
    ]
    labels = classify_link_pairs_flan_batched(sample_pairs, batch_size=4)
    print(labels)  # e.g. [4, 2, 3, 4, 5]
