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

Answer with 1 or 0 only:""" for text in batch_texts
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

def _classify_pair_relation(pairs, prompt, batch_size=16, max_new_tokens=4, device=None):
    """
    pairs: list of [left, right]
    returns: list of ints (0-1), one per pair
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]

        prompts = [prompt.format(left=l, right=r) for l, r in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in decoded:
            text = text.strip()
            try:
                idx = int(text[0])
                if idx not in LABELS:
                    idx = 0
            except:
                idx = 0
            results.append(idx)

    return results

def classify_link_pairs_flan_batched(pairs, batch_size=16, max_new_tokens=4, device=None):
    #Since it is done 1 by 1, we can pass them indivdually
    prescence_prompt = """Classify the contextual relation between the L and R text:

Choose only one classification:
    1: L is of a similar domain to R;
    0: L is definitely irrelevant to R;

L: {left}

R: {right}

Answer with the number only:"""
    typeofpair_prompt = """For the purpose of fact extraction, classify the relation between the L and R text:

Choose only one classification:
    0: L only contains contextual information that R already has;
    1: L contains any key contextual information that R is missing;

L: {left}

R: {right}

Answer with the number only:"""
    prescence = _classify_pair_relation(pairs, prescence_prompt, batch_size, max_new_tokens, device)
    typeofpair = _classify_pair_relation(pairs, typeofpair_prompt, batch_size, max_new_tokens, device)
    results = []
    for p, t in zip(prescence, typeofpair):
        ans = 3 if p==0 else 2 if t==0 else 1
        results.append(ans)
    return results

@torch.no_grad()
def score_label_next_token(prompt_ids, outputOptions):
    """
    Returns the logits for the FIRST generated token ('0' vs '1').
    For T5, the decoder starts from <pad> and attends to encoder.
    """
    # Encode once
    enc = model.get_encoder()(input_ids=prompt_ids, attention_mask=(prompt_ids != tokenizer.pad_token_id))
    # Start decoder with a single start token (pad) for T5
    start = torch.full((prompt_ids.size(0), 1), tokenizer.pad_token_id, dtype=torch.long, device=prompt_ids.device)
    out = model(encoder_outputs=enc, decoder_input_ids=start)
    logits = out.logits[:, -1, :]                  # [batch, vocab]
    return logits[:, outputOptions]                        # [batch, 2] -> columns: [logit('0'), logit('1')]

def _classify_node(texts, prompt, outputOptions, device=None, calibration_bias=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompts = [prompt.format(txt=str(t).replace("{","{{").replace("}","}}")) for t in texts]
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    logits_01 = score_label_next_token(batch["input_ids"], outputOptions)  # [B,2]

    if calibration_bias is not None:
        logits_01 = logits_01 - calibration_bias  # subtract bias per label

    preds = torch.argmax(logits_01, dim=-1).tolist()  # 0->label "0", 1->label "1"
    return preds

def _estimate_calibration_bias(prompt, outputOptions, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    null_queries = ["", "N/A", "---", "title", "keywords", "summary", "country language"]  # content-free / fragment-y
    prompts = [prompt.format(txt=q) for q in null_queries]
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    logits_01 = score_label_next_token(batch["input_ids"], outputOptions)   # [K,2]
    # Average bias toward each label on null inputs
    bias = logits_01.mean(dim=0, keepdim=True)               # [1,2]
    return bias

def classify_node_needsContext(nodes):
    classify_sentence="""TASK:
    Classify the QUERY's text into only one classification. Focus on sentence structure and ignore the meaning of the QUERY. Output one number only

CLASSES:
    0: The QUERY does not form a complete sentence;
    1: The QUERY looks like a sentence (ignore punctuation);
                   
QUERY: {txt}

ANSWER:"""
    classify_type="""TASK:
    Classify the QUERY text into only one classification. Output one number only

CLASSES:
    0: The text does not reference something unknown;
    1: There are unknowns in what is being talked about;
                   
QUERY: {txt}

ANSWER:"""
    zero_id = tokenizer("0", add_special_tokens=False).input_ids[0] #Token id for 0
    one_id  = tokenizer("1", add_special_tokens=False).input_ids[0] #Token id for 1
    output_tokens = torch.tensor([zero_id, one_id])

    bias = _estimate_calibration_bias(classify_sentence, output_tokens)
    presence = _classify_node(nodes, classify_sentence, output_tokens, calibration_bias=bias)
    
    bias = _estimate_calibration_bias(classify_type, output_tokens)
    typeofsentece = _classify_node(nodes, classify_type, output_tokens, calibration_bias=bias)

    results = []
    for p, t in zip(presence, typeofsentece):
        ans = 1 if t==1 else 0# 1 if p==1 else 2 if t==1 else 3
        results.append(ans)
    return results
    
def classify_node_isCategory(nodes):
    """This classifies if a node is a category name, header or entry (and if it is likely to have a category name, header or entry to add context to it)"""
    classify_category="""TASK:
    Classify the QUERY's text into only one classification. Focus on sentence structure and ignore the meaning of the QUERY. Output one number only

CLASSES:
    0: The QUERY text is not CLASS 1;
    1: The QUERY text is a proper noun, category value, a value or a category entry;
                   
QUERY: {txt}

ANSWER:"""
    zero_id = tokenizer("0", add_special_tokens=False).input_ids[0] #Token id for 0
    one_id  = tokenizer("1", add_special_tokens=False).input_ids[0] #Token id for 1
    output_tokens = torch.tensor([zero_id, one_id])

    bias = _estimate_calibration_bias(classify_category, output_tokens)
    presence = _classify_node(nodes, classify_category, output_tokens, calibration_bias=bias)
    
    results = []
    for p in presence:
        ans = 1 if p==1 else 0
        results.append(ans)
    return results

if __name__ == "__main__":
    #sample_pairs = [["british columbia canada", "set in"], ["set in", "british columbia canada"], ["for sexuality and some language", "mpaa reasons"], ["mpaa reasons", "for sexuality and some language"], ["addict", "accident"], ["accident", "addict"], ["other related works", "is related to"], ["is related to", "other related works"], ["drugs", "accident"], ["accident", "drugs"], ["in a minor key", "moods"], ["moods", "in a minor key"], ["drugs", "addict"], ["addict", "drugs"], ["canada", "r"], ["r", "canada"], ["director", "atom egoyan"], ["atom egoyan", "director"], ["panavision", "corrections to this entry"], ["lawyer", "accident"], ["accident", "lawyer"], ["category", "feature"], ["feature", "category"], ["lawyer", "addict"], ["addict", "lawyer"], ["year", "1997"], ["1997", "year"], ["drama", "genres"], ["genres", "drama"], ["panavision", "cinematic process"], ["cinematic process", "panavision"], ["british columbia canada", "corrections to this entry"], ["lawyer", "drugs"], ["drugs", "lawyer"]]
    sample_pairs = [
        ['ap300 smr', 'the ap300 smr is the next evolution of the licensed ap1000 technology'],
['the ap300 smr is the next evolution of the licensed ap1000 technology', 'ap300 smr'],
['evinci microreactor', 'the next generation small modular reactor for remote applications'],
['the next generation small modular reactor for remote applications', 'evinci microreactor'],
['enhance your training staffing and outsourcing needs with our training and resource solutions', 'westinghousenavigator'],
['westinghousenavigator', 'enhance your training staffing and outsourcing needs with our training and resource solutions'],
['enhance your training staffing and outsourcing needs with our training and resource solutions', 'westinghouseiq'],
['westinghouseiq', 'enhance your training staffing and outsourcing needs with our training and resource solutions'],
['when it comes to creating a more sustainable planet the need for renewable energy cant replace the need for safe energy with nuclear power you get the best of both worlds', 'safety getting the facts right'],
['safety getting the facts right', 'when it comes to creating a more sustainable planet the need for renewable energy cant replace the need for safe energy with nuclear power you get the best of both worlds'],
['ap1000 pwr', "the world's first proven generation iii pressurized water reactor and passive safety plant available"],
["the world's first proven generation iii pressurized water reactor and passive safety plant available", 'ap1000 pwr'],
['the ap300 smr is the next evolution of the licensed ap1000 technology', 'evinci microreactor'],
['evinci microreactor', 'the ap300 smr is the next evolution of the licensed ap1000 technology'],
['ap1000 pwr', 'the ap300 smr is the next evolution of the licensed ap1000 technology'],
['the ap300 smr is the next evolution of the licensed ap1000 technology', 'ap1000 pwr'],
['balancing wind solar and nuclear power will help achieve a carbonfree future and positively impact our changing climate over the past 50 years globally nuclear power has avoided nearly two years of the worlds energyrelated co2 emissions imagine how much more carbon pollution we can prevent', 'carbonfree energy'],
['carbonfree energy', 'balancing wind solar and nuclear power will help achieve a carbonfree future and positively impact our changing climate over the past 50 years globally nuclear power has avoided nearly two years of the worlds energyrelated co2 emissions imagine how much more carbon pollution we can prevent'],
['ap1000 pwr', 'the next generation small modular reactor for remote applications'],
['the next generation small modular reactor for remote applications', 'ap1000 pwr'],
['westinghousenuclearning', 'enhance your training staffing and outsourcing needs with our training and resource solutions'],
['enhance your training staffing and outsourcing needs with our training and resource solutions', 'westinghousenuclearning'],
['ap300 smr', 'the next generation small modular reactor for remote applications'],
['the next generation small modular reactor for remote applications', 'ap300 smr'],
["the fact is it's safetruth is a person working fulltime in a nuclear power plant receives less additional radiation in a year than a flight crew or a business traveler with 250 hours of flying time", 'when it comes to creating a more sustainable planet the need for renewable energy cant replace the need for safe energy with nuclear power you get the best of both worlds'],
['when it comes to creating a more sustainable planet the need for renewable energy cant replace the need for safe energy with nuclear power you get the best of both worlds', "the fact is it's safetruth is a person working fulltime in a nuclear power plant receives less additional radiation in a year than a flight crew or a business traveler with 250 hours of flying time"],
["the world's first proven generation iii pressurized water reactor and passive safety plant available", 'evinci microreactor'],
['evinci microreactor', "the world's first proven generation iii pressurized water reactor and passive safety plant available"],
['solar wind and nuclear energy are essential to a carbonfree future but the sun doesnt always shine and the wind doesnt always blow nuclear power plants are almost always on delivering the highest availability energy source and operating at maximum capacity more than 90% of the time', 'shaping the future with reliable energy'],
['shaping the future with reliable energy', 'solar wind and nuclear energy are essential to a carbonfree future but the sun doesnt always shine and the wind doesnt always blow nuclear power plants are almost always on delivering the highest availability energy source and operating at maximum capacity more than 90% of the time'],
["the world's first proven generation iii pressurized water reactor and passive safety plant available", 'ap300 smr'],
['ap300 smr', "the world's first proven generation iii pressurized water reactor and passive safety plant available"],
['balancing wind solar and nuclear power will help achieve a carbonfree future and positively impact our changing climate over the past 50 years globally nuclear power has avoided nearly two years of the worlds energyrelated co2 emissions imagine how much more carbon pollution we can prevent', 'nuclear energyprovides 55% of the uss and 14% of the worlds carbonfree energy'],
['nuclear energyprovides 55% of the uss and 14% of the worlds carbonfree energy', 'balancing wind solar and nuclear power will help achieve a carbonfree future and positively impact our changing climate over the past 50 years globally nuclear power has avoided nearly two years of the worlds energyrelated co2 emissions imagine how much more carbon pollution we can prevent'],
['project management support', 'quality environment health safety'],
['quality environment health safety', 'project management support'],
['engineering', 'corporate'],
['corporate', 'engineering'],
['safety getting the facts right', "the fact is it's safetruth is a person working fulltime in a nuclear power plant receives less additional radiation in a year than a flight crew or a business traveler with 250 hours of flying time"],
["the fact is it's safetruth is a person working fulltime in a nuclear power plant receives less additional radiation in a year than a flight crew or a business traveler with 250 hours of flying time", 'safety getting the facts right'],
['westinghouse partners with richland county ems to host training video series', 'westinghouse partnered with the richland county ems to host a series of training videos at their facility the training videos filmed onsite at westinghouse in hopkins signals a collaboration that focuses on producing highquality instructional content aimed at improving skills and knowledge among emergency medical personnel'],
['westinghouse partnered with the richland county ems to host a series of training videos at their facility the training videos filmed onsite at westinghouse in hopkins signals a collaboration that focuses on producing highquality instructional content aimed at improving skills and knowledge among emergency medical personnel', 'westinghouse partners with richland county ems to host training video series'],
['carbonfree energy', 'nuclear energyprovides 55% of the uss and 14% of the worlds carbonfree energy'],
['nuclear energyprovides 55% of the uss and 14% of the worlds carbonfree energy', 'carbonfree energy'],
['presidents kaizen week unlocks innovation across americas outage maintenance services', 'at westinghouse we believe that continuous improvement isnt just a goal its a mindset that approach came to life during our recent presidents kaizen week a dynamic crossfunctional initiative aimed at streamlining key business processes using lean methodologies originated by toyota production system for manufacturing improvements lean helps deliver maximum value to customers by identifying and eliminating waste'],
['at westinghouse we believe that continuous improvement isnt just a goal its a mindset that approach came to life during our recent presidents kaizen week a dynamic crossfunctional initiative aimed at streamlining key business processes using lean methodologies originated by toyota production system for manufacturing improvements lean helps deliver maximum value to customers by identifying and eliminating waste', 'presidents kaizen week unlocks innovation across americas outage maintenance services'],
['project management support', 'corporate'],
['quality environment health safety', 'corporate'],
['project management support', 'engineering'],
['quality environment health safety', 'engineering'],
['manufacturing operations maintenance', 'engineering'],
['manufacturing operations maintenance', 'project management support'],
['global directory x', 'westinghousenuclearning'],
['westinghouse joins texas nuclear alliance as a founding member', 'westinghouse ap1000 design receives us licensing extension to 2046'],
['westinghouse ap1000 design receives us licensing extension to 2046', 'westinghouse joins texas nuclear alliance as a founding member'],
['global directory x', 'westinghouseiq'],
['westinghousenuclearning', 'bulgaria bulgarian'],
['bulgaria bulgarian', 'westinghousenuclearning'],
["shaping tomorrow's energythrough advanced nuclear technology", 'westinghousenuclearning'],
['westinghousenuclearning', "shaping tomorrow's energythrough advanced nuclear technology"],
['poland polish', 'westinghousenuclearning'],
['bulgaria bulgarian', 'global directory x'],
['westinghouse joins texas nuclear alliance as a founding member', 'fermi america partners with westinghouse to support licensing for four ap1000 units'],
['fermi america partners with westinghouse to support licensing for four ap1000 units', 'westinghouse joins texas nuclear alliance as a founding member'],
['westinghouseiq', 'bulgaria bulgarian'],
['westinghouse ap1000 design receives us licensing extension to 2046', 'fermi america partners with westinghouse to support licensing for four ap1000 units'],
['fermi america partners with westinghouse to support licensing for four ap1000 units', 'westinghouse ap1000 design receives us licensing extension to 2046'],
['westinghouse partnered with the richland county ems to host a series of training videos at their facility the training videos filmed onsite at westinghouse in hopkins signals a collaboration that focuses on producing highquality instructional content aimed at improving skills and knowledge among emergency medical personnel', 'at westinghouse we believe that continuous improvement isnt just a goal its a mindset that approach came to life during our recent presidents kaizen week a dynamic crossfunctional initiative aimed at streamlining key business processes using lean methodologies originated by toyota production system for manufacturing improvements lean helps deliver maximum value to customers by identifying and eliminating waste'],
['westinghousenuclearning', 'canada english'],
['canada english', 'westinghousenuclearning'],
['poland polish', 'global directory x'],
["shaping tomorrow's energythrough advanced nuclear technology", 'westinghouseiq'],
['presidents kaizen week unlocks innovation across americas outage maintenance services', 'westinghouse partnered with the richland county ems to host a series of training videos at their facility the training videos filmed onsite at westinghouse in hopkins signals a collaboration that focuses on producing highquality instructional content aimed at improving skills and knowledge among emergency medical personnel'],
['westinghouse partnered with the richland county ems to host a series of training videos at their facility the training videos filmed onsite at westinghouse in hopkins signals a collaboration that focuses on producing highquality instructional content aimed at improving skills and knowledge among emergency medical personnel', 'presidents kaizen week unlocks innovation across americas outage maintenance services'],
['westinghouseiq', 'canada english'],
['the established design of the ap1000 reactor offers unequaled safety economic competitiveness and improved more efficient operations', 'westinghousenuclearning'],
['westinghousenuclearning', 'slovakia slovak'],
["shaping tomorrow's energythrough advanced nuclear technology", 'westinghousenavigator'],
['canada english', 'global directory x'],
['bulgaria bulgarian', 'slovakia slovak'],
['westinghousenuclearning', 'the evinci microreactor is a nextgeneration micromodular reactor combining innovative technologies with over 60 years of commercial nuclear expertise'],
['the evinci microreactor is a nextgeneration micromodular reactor combining innovative technologies with over 60 years of commercial nuclear expertise', 'westinghousenuclearning'],
['bulgaria bulgarian', 'slovenia slovenian'],
['bulgaria bulgarian', 'czech republic czech'],
['westinghousenuclearning', 'slovenia slovenian'],
['slovakia slovak', 'poland polish'],
['slovakia slovak', 'global directory x'],
['bulgaria bulgarian', 'sweden swedish'],
['westinghouseiq', 'the evinci microreactor is a nextgeneration micromodular reactor combining innovative technologies with over 60 years of commercial nuclear expertise'],
['slovenia slovenian', 'poland polish'],
['slovenia slovenian', 'global directory x'],
['bulgaria bulgarian', 'ukraine ukrainian'],
['westinghousenuclearning', 'czech republic czech'],
['bulgaria bulgarian', 'japan japanese'],
['czech republic czech', 'poland polish'],
['the established design of the ap1000 reactor offers unequaled safety economic competitiveness and improved more efficient operations', 'for the only smr based on advanced reactor technology thats already licensed and operating ap300 is the proven choice'],
['for the only smr based on advanced reactor technology thats already licensed and operating ap300 is the proven choice', 'the established design of the ap1000 reactor offers unequaled safety economic competitiveness and improved more efficient operations'],
['czech republic czech', 'global directory x'],
['bulgaria bulgarian', 'united kingdom english'],
['westinghousenuclearning', 'sweden swedish'],
['sweden swedish', 'poland polish'],
['westinghouse partners with richland county ems to host training video series', 'presidents kaizen week unlocks innovation across americas outage maintenance services'],
['for the only smr based on advanced reactor technology thats already licensed and operating ap300 is the proven choice', 'westinghousenuclearning'],
['sweden swedish', 'global directory x'],
['the evinci microreactor is a nextgeneration micromodular reactor combining innovative technologies with over 60 years of commercial nuclear expertise', 'the established design of the ap1000 reactor offers unequaled safety economic competitiveness and improved more efficient operations'],
['poland polish', 'ukraine ukrainian'],
['ukraine ukrainian', 'poland polish'],
['ukraine ukrainian', 'global directory x'],
['poland polish', 'japan japanese'],
['japan japanese', 'poland polish'],
['japan japanese', 'global directory x'],
['poland polish', 'united kingdom english'],
['united kingdom english', 'poland polish'],
['news', 'westinghousenuclearning'],
['united kingdom english', 'global directory x'],
['shape your future', 'manufacturing operations maintenance'],
['manufacturing operations maintenance', 'shape your future'],
['shape your future', 'evinci microreactor'],
['westinghousenuclearning', 'shape your future'],
['westinghouse ap1000 design receives us licensing extension to 2046', 'news'],
['westinghouse joins texas nuclear alliance as a founding member', 'news'],
['fermi america partners with westinghouse to support licensing for four ap1000 units', 'news'],
['product spotlights', 'westinghousenuclearning'],
['evinci microreactor', 'product spotlights'],
['product spotlights', 'westinghouseiq'],
['ap1000 pwr', 'product spotlights'],
    ]
    import numpy as np
    sample_pairs = np.unique(sample_pairs)
    labels = classify_node_isCategory(sample_pairs)
    ifnore = [1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 3, 1, 3, 3, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 3, 3, 1, 1, 3, 1, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 2, 3, 1, 1, 1, 3]
    for pair, label, flan in zip(sample_pairs, labels, ifnore):
        if flan != 1:
            pass
        print(label, pair)

    last = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]

    best_attempt = [1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 0, 1, 1, 2]
    # import sys
    # sys.path.insert(1, r"/vol/bitbucket/mjh24/IAEA-thesis")
    # from Stage2.classifyingEdges.metrics import metrics
    # y_true_str = "2 2 1 1 ? ? ? ? 3 3 1 1 3 3 2 2 2 2 1 1 ? ? 1 1 3 3 1 1 3 3 1 1 1 1 2 2 2 2 2 2"
    # y_true = [3 if tok == "?" else int(tok) for tok in y_true_str.split()]
    # metrics(labels[:40],y_true)