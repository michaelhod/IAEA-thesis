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

    # Classify each text node individually
    node_preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        prompts = [f"""Decide if the text looks like a website instruction, button, or navigation label.
                Return 1 for instruction/button/navigation, or 0 otherwise. Output only a single digit.

                Examples of 1: "Read more", "Learn more", "Explore", "Get started", "Try free", "Watch video",
                "Sign in", "Sign up", "My account", "Download", "Contact", "Pricing", "Products", "Docs",
                "Blog", "Home", "About", "Privacy Policy", "Terms", "English", "Menu", "Next", "Previous", "Back".

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

    # Reduce back to pair-level: OR(left, right)
    results = []
    for j in range(0, len(node_preds), 2):
        left_is_btn = node_preds[j]
        right_is_btn = node_preds[j+1] if j+1 < len(node_preds) else 0
        results.append(1 if (left_is_btn or right_is_btn) else 0)

    return results

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
1 = added_context (one side is metadata/attribute of the other, e.g. ["Director", "Christopher Nolan"])
2 = added_information (one side adds meaningful information to the other, e.g. ["evinci microreactor", "the next generation small modular reactor for remote applications"])
3 = purpose_of_text (one side classifies the text's purpose, e.g. ["plot synopsis", "This movie is about Madagascar"])
4 = sibling_content (two peer items/categories, e.g. ["afc teams", "nfc teams"])
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
        ['learn more about why databricks selected westinghouse ai for its 2025 data intelligence in energy utility award', 'learn more'],
        ['learn more', 'learn more about why databricks selected westinghouse ai for its 2025 data intelligence in energy utility award'],
        ['the established design of the ap1000 reactor offers unequaled safety economic competitiveness and improved more efficient operations', 'explore'],
        ['explore', 'the established design of the ap1000 reactor offers unequaled safety economic competitiveness and improved more efficient operations'],
        ['for the only smr based on advanced reactor technology thats already licensed and operating ap300 is the proven choice', 'explore'],
        ['explore', 'for the only smr based on advanced reactor technology thats already licensed and operating ap300 is the proven choice'],
        ['access our custom technology library designed for nuclear engineering students and professionals', 'westinghousenavigator'],
        ['explore now', 'westinghousenavigator'],
        ['explore now', 'westinghousenuclearning'],
        ['the evinci microreactor is a nextgeneration micromodular reactor combining innovative technologies with over 60 years of commercial nuclear expertise', 'explore'],
        ['explore', 'the evinci microreactor is a nextgeneration micromodular reactor combining innovative technologies with over 60 years of commercial nuclear expertise'],
        ['take a look', 'westinghousenuclearning'],
        ['explore our interactive 3d visualization tool for innovative answers to operating plant challenges', 'westinghousenuclearning'],
        ['access our custom technology library designed for nuclear engineering students and professionals', 'westinghousenuclearning'],
        ['westinghousenuclearning', 'access our custom technology library designed for nuclear engineering students and professionals'],
        ['explore our interactive 3d visualization tool for innovative answers to operating plant challenges', 'take a look'],
        ['take a look', 'explore our interactive 3d visualization tool for innovative answers to operating plant challenges'],
        ['the next generation small modular reactor for remote applications', 'learn more watch video'],
        ['learn more watch video', 'the next generation small modular reactor for remote applications'],
        ['access our custom technology library designed for nuclear engineering students and professionals', 'explore now'],
        ['explore now', 'access our custom technology library designed for nuclear engineering students and professionals'],
        ['the ap300 smr is the next evolution of the licensed ap1000 technology', 'learn more watch video'],
        ['learn more watch video', 'the ap300 smr is the next evolution of the licensed ap1000 technology'],
        ['enhance your training staffing and outsourcing needs with our training and resource solutions', 'read more'],
        ['westinghouseiq', 'explore our interactive 3d visualization tool for innovative answers to operating plant challenges'],
        ['explore our interactive 3d visualization tool for innovative answers to operating plant challenges', 'westinghouseiq'],
        ['ap300 smr', 'the ap300 smr is the next evolution of the licensed ap1000 technology'],
        ['evinci microreactor', 'the next generation small modular reactor for remote applications'],
        ['the next generation small modular reactor for remote applications', 'evinci microreactor'],
        ['enhance your training staffing and outsourcing needs with our training and resource solutions', 'westinghousenavigator'],
        ['westinghousenavigator', 'enhance your training staffing and outsourcing needs with our training and resource solutions'],
        ['enhance your training staffing and outsourcing needs with our training and resource solutions', 'westinghouseiq'],
        ['westinghouseiq', 'enhance your training staffing and outsourcing needs with our training and resource solutions'],
        ['when it comes to creating a more sustainable planet the need for renewable energy cant replace the need for safe energy with nuclear power you get the best of both worlds', 'safety getting the facts right'],
        ['safety getting the facts right', 'when it comes to creating a more sustainable planet the need for renewable energy cant replace the need for safe energy with nuclear power you get the best of both worlds'],
        ['evinci microreactor', 'learn more watch video'],
        ['learn more watch video', 'evinci microreactor'],
        ['ap1000 pwr', "the world's first proven generation iii pressurized water reactor and passive safety plant available"],
        ["the world's first proven generation iii pressurized water reactor and passive safety plant available", 'ap1000 pwr'],
        ['ap300 smr', 'learn more watch video'],
        ['learn more watch video', 'ap300 smr'],
        ['the ap300 smr is the next evolution of the licensed ap1000 technology', 'evinci microreactor'],
        ['quality environment health safety', 'explore'],
        ['explore', 'quality environment health safety'],
        ['engineering', 'explore'],
        ['explore', 'engineering'],
        ['project management support', 'explore'],
        ['explore', 'project management support'],
    ]
    labels = clean_instructional_text(sample_pairs, batch_size=len(sample_pairs))
    for pair, label in zip(sample_pairs, labels):
        print(label, pair)