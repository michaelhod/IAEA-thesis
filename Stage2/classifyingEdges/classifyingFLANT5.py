import os
os.environ.setdefault("HF_HOME", "/data/mjh24/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/mjh24/hf/transformers")
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-large"  # "google/flan-t5-xl" or "google/flan-t5-base" for more accuracy

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
        prompts = [f"""Decide if the text looks like an website button or navigation label.
Return 1 for button/navigation, or 0 otherwise. Output only a single digit.

Examples of 1: "Read more", "Learn more", "Explore", "Get started", "Try free", "Watch video",
"Sign in", "Sign up", "My account", "Download", "Contact", "Home", "About", "Privacy Policy",
"Terms", "Fr/En", "Menu", "Next", "Previous", "Back", "Dashboard", "Add to cart".

Examples of 0: descriptive copy, names of people or products in context, long summaries, and info labels such as "Best seller", "Made in USA" or "Free shipping on orders over $50"

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
    2: The meaning of L is unclear OR the meaning of R is unclear;
    1: The meaning of L could be related to the meaning of R;
    0: The meaning of L is definitely completely irrelevant to the meaning of R;

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


    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]

        # For each summary node, split by ".", prompts are sentence by sentence. They take the context and the prev sentence
        prompts = []
        for pair in pairs:
            sentences = pair[0].split(".")
            context = pair[1]
            for sentence in sentences:
                prompts.append(prompt.format(txt=sentence, ctx=context))

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0, top_p=0.9, no_repeat_ngram_size=3)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in decoded:
            text = text.strip()
            results.append(text)

    return results


import re
import torch

# --- overlap util (your criterion) ---
def overlap_ratio(src: str, out: str) -> float:
    """Fraction of source words that appear in output (case-insensitive, punctuation stripped)."""
    to_words = lambda s: re.findall(r"\w+", s.lower())
    src_words = to_words(src)
    out_words = set(to_words(out))
    return 0.0 if not src_words else sum(1 for w in src_words if w in out_words) / len(src_words)

def summarise_node(pairs, batch_size=64, sentencethreshold=0.75, contextthreshold=0.25, max_rounds=6, device=None):
    """Takes a list of edges. The first is the node to summarise, the second the context. Splits the summary into sentences"""
    prompt="""TASK:
    Combine the context into the sentence, where it best belongs.
    If unsure, use the context as a name.
    Do not remove important facts. Be specific. Be concise.

CONTEXT: {ctx}
                   
SENTENCE: {txt}

COMBINED:"""
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def split_sents(text):
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sents if s.strip()]

    # index items so we can return results in original order
    work = [{"idx": i, "txt": s, "ctx": c} for i, (s, c) in enumerate(pairs)]
    next_attempt = work[:]                 # initial queue
    first_try  = [None] * len(pairs)        # last outputs

    rounds = 0
    temp = 1e-9
    while next_attempt and rounds < max_rounds:
        runagain = []
        # process in batches
        for b in range(0, len(next_attempt), batch_size):
            
            #build prompts by splitting up the sentences
            batch = next_attempt[b:b+batch_size]
            prompts = [prompt.format(ctx=item["ctx"], txt=item["txt"]) for item in batch]
            # for item in batch:
            #     for sents in split_sents(item["txt"]):
            #         prompts.append(prompt.format(ctx=item["ctx"], txt=sents))

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                out_ids = model.generate(**inputs, num_beams=2, max_new_tokens=64, do_sample=True, temperature=temp, no_repeat_ngram_size=3)
            decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

            # check pass/fail for each item in the batch
            for item, gen_text in zip(batch, decoded):
                gen_text = gen_text.strip()
                if rounds == 0:
                    first_try[item["idx"]] = gen_text
                
                sentence_ov = overlap_ratio(item["txt"], gen_text)
                context_ov = overlap_ratio(item["ctx"], gen_text)
                if sentence_ov >= sentencethreshold and context_ov >= context_ov:
                    first_try[item["idx"]] = gen_text #Overwrite better attempts
                else:
                    runagain.append(item)

        next_attempt = runagain
        rounds += 1

        # make retries a bit more exploratory each round:
        if next_attempt:
            temp = min(0.7, temp + 0.1)

    return first_try

def summarise_cluster(clusters, batch_size=64, device=None):
    """Takes a list of edges. The first is the node to summarise, the second the context. Splits the summary into sentences"""
    prompt="""Instruction:
The different INPUTS are related to each other.
Group the similar INPUTS together to create a list of facts.
Output the relations and the list of facts.
Add no external information.

Example 1:

INPUTS:
- Doctor
- Address
- 43 Palace Gardens, Newcastle
- The date is 1968
- Alexander Evans
- March

OUTPUT (4 facts):
1. The Doctor is Alexander Evans.
2. The Address is 43 Palace Gardens, Newcastle.
3. The date is March, 1968.

Example 2:

INPUTS:
- Challenging
- Thoughts:
- Rewarding
- Acceptable

OUTPUT (3 facts):
1. The Thought is Challenging.
2. Thoughts: Rewarding.
3. Thoughts: Acceptable.

Now do the same for the following INPUTS:

INPUTS:
- {INPUTS}
OUTPUT (~{N} facts):
"""# Within each fact, NEVER use pronouns (e.g., him, these, it).
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def split_sents(text):
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sents if s.strip()]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    for i in range(0, len(clusters), batch_size):
        batch = clusters[i:i+batch_size]

        prompts = [prompt.format(N=len(cluster)/2, INPUTS="\n- ".join(cluster)) for cluster in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, num_beams=4, repetition_penalty=1.3, no_repeat_ngram_size=4)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in decoded:
            text = text.strip()
            results.append(text)

    return results

if __name__ == "__main__":
    sample_pairs = [["british columbia canada", "set in"], ["set in", "british columbia canada"], ["for sexuality and some language", "mpaa reasons"], ["mpaa reasons", "for sexuality and some language"], ["addict", "accident"], ["accident", "addict"], ["other related works", "is related to"], ["is related to", "other related works"], ["drugs", "accident"], ["accident", "drugs"], ["in a minor key", "moods"], ["moods", "in a minor key"], ["drugs", "addict"], ["addict", "drugs"], ["canada", "r"], ["r", "canada"], ["director", "atom egoyan"], ["atom egoyan", "director"], ["panavision", "corrections to this entry"], ["lawyer", "accident"], ["accident", "lawyer"], ["category", "feature"], ["feature", "category"], ["lawyer", "addict"], ["addict", "lawyer"], ["year", "1997"], ["1997", "year"], ["drama", "genres"], ["genres", "drama"], ["panavision", "cinematic process"], ["cinematic process", "panavision"], ["british columbia canada", "corrections to this entry"], ["lawyer", "drugs"], ["drugs", "lawyer"]]
    sample_pairs = [['Football', 'Casino Sites UK'],
 ['Casino Sites UK', 'Football'],
 ['Casino', "Today's Football Scores"],
 ["Today's Football Scores", 'Casino'],
 ['Casino', 'Premier League Scores'],
 ['Premier League Scores', 'Casino'],
 ['Football', 'Free Spins UK'],
 ['Free Spins UK', 'Football'],
 ['Other Sports', 'Free Spins UK'],
 ['Casino', 'Basketball Scores'],
 ['Basketball Scores', 'Casino'],
 ['Football', 'Bingo Sites UK'],
 ['Bingo Sites UK', 'Football'],
 ['Casino', 'Football on TV'],
 ['Football on TV', 'Casino'],
 ['Casino', 'Premier League Standings'],
 ['Premier League Standings', 'Casino'],
 ['Football', 'Free Spins ZA'],
 ['Free Spins ZA', 'Football'],
 ['Other Sports', 'Bingo Sites UK'],
 ['Casino', 'Tennis Scores'],
 ['Tennis Scores', 'Casino'],
 ['Football', 'Free Spins US'],
 ['Free Spins US', 'Football'],
 ['Casino', 'Cricket Scores'],
 ['Football', 'Casino Sites CA'],
 ['Betting', 'Bundesliga Scores'],
 ['Casino', 'NFL Betting Sites'],
 ['Other Sports', 'Casino Sites CA'],
 ['Casino', 'Champions League Scores'],
 ['Champions League Scores', 'Casino'],
 ['Casino', 'FA Cup Scores'],
 ['FA Cup Scores', 'Casino'],
 ['Modern Slavery Statement', 'Cookie Policy'],
 ['Cookie Policy', 'Modern Slavery Statement'],
 ['Casino', 'La Liga Scores'],
 ['La Liga Scores', 'Casino'],
 ['Casino', 'Bundesliga Scores'],
 ['Bundesliga Scores', 'Casino'],
 ['Casino', 'IPL Scores'],
 ['IPL Scores', 'Casino'],
 ['Casino', 'NBA Scores'],
 ['NBA Scores', 'Casino'],
 ['Casino', 'Ice Hockey Scores'],
 ['Casino', 'Championship Scores'],
 ['Championship Scores', 'Casino'],
 ['Casino', 'Serie A Scores'],
 ['Serie A Scores', 'Casino'],
 ['Corporate', 'Modern Slavery Statement'],
 ['Careers', 'Modern Slavery Statement'],
 ['Betting',
  'AFC Bournemouth vs Leicester City Live Scores and Match Information'],
 ['Casino',
  'AFC Bournemouth vs Leicester City Live Scores and Match Information'],
 ['AFC Bournemouth vs Leicester City Live Scores and Match Information',
  'Casino'],
 ['2J. Justin', '33L. Thomas'],
 ['35K. McAteer', '16V. Kristiansen'],
 ['16V. Kristiansen', '35K. McAteer'],
 ['18J. Ayew', '35K. McAteer'],
 ['7D. Brooks', '224A. Semenyo'],
 ['224A. Semenyo', '19J. Kluivert'],
 ['19J. Kluivert', '224A. Semenyo'],
 ['93', '47'],
 ['47', '93'],
 ['3W. Faes', '33L. Thomas'],
 ['Modern Slavery Statement',
  'The latest football scores, line-ups and more for AFC Bournemouth vs Leicester City.'],
 ['The latest football scores, line-ups and more for AFC Bournemouth vs Leicester City.',
  'Modern Slavery Statement'],
 ['93', '23'],
 ['23', '93'],
 ['93', '22'],
 ['93', '31'],
 ['L. SinisterraHamstring strain', 'A. Scott'],
 ['L. CookStraight red card', 'J. Evans'],
 ['Ligue 1France', 'RegionEnglandChampions LeagueSpainItalyGermany'],
 ['L. CookStraight red card', 'A. Scott'],
 ['L. CookStraight red card', 'B. Winterburn'],
 ['L. CookStraight red card', 'J. Monga'],
 ['L. CookStraight red card', 'D. Huijsen'],
 ['2J. Justin', '20P. Daka'],
 ['20P. Daka', '2J. Justin'],
 ['41J. Stolarczyk', '20P. Daka'],
 ['20P. Daka', '41J. Stolarczyk'],
 ['41J. Stolarczyk', '2J. Justin'],
 ['2J. Justin', '41J. Stolarczyk'],
 ['41J. Stolarczyk', '35K. McAteer'],
 ['35K. McAteer', '41J. Stolarczyk'],
 ['13K. Arrizabalaga', '224A. Semenyo'],
 ['9Evanilson', '13K. Arrizabalaga'],
 ['13K. Arrizabalaga', '9Evanilson'],
 ['9Evanilson', '3M. Kerkez'],
 ['13K. Arrizabalaga', '5M. Senesi'],
 ['41J. Stolarczyk', '22O. Skipp'],
 ['22O. Skipp', '41J. Stolarczyk'],
 ['18J. Ayew', '20P. Daka'],
 ['4C. Coady', '41J. Stolarczyk'],
 ['41J. Stolarczyk', '24B. Soumar'],
 ['24B. Soumar', '41J. Stolarczyk'],
 ['9Evanilson', '224A. Semenyo'],
 ['224A. Semenyo', '9Evanilson'],
 ['41J. Stolarczyk', '33L. Thomas'],
 ['41J. Stolarczyk', '16V. Kristiansen'],
 ['18J. Ayew', '41J. Stolarczyk'],
 ['7D. Brooks', '13K. Arrizabalaga'],
 ['9Evanilson', '16M. Tavernier'],
 ['16M. Tavernier', '9Evanilson'],
 ['9Evanilson', '5M. Senesi'],
 ['RegionEnglandChampions LeagueSpainItalyGermany', 'BrazilBrazil'],
 ['BrazilBrazil', 'RegionEnglandChampions LeagueSpainItalyGermany'],
 ['9Evanilson', '19J. Kluivert'],
 ['Serie AItaly', 'BrazilBrazil'],
 ['BrazilBrazil', 'Serie AItaly'],
 ['LaLigaSpain', 'BrazilBrazil'],
 ['BrazilBrazil', 'LaLigaSpain'],
 ['13K. Arrizabalaga', 'AFC Bournemouth4-2-3-1'],
 ['Premier LeagueEngland', 'BrazilBrazil'],
 ['BrazilBrazil', 'Premier LeagueEngland'],
 ['Andoni IraolaRuud van Nistelrooy', 'L. SinisterraHamstring strain'],
 ['Andoni IraolaRuud van Nistelrooy', 'R. ChristieGroin injury'],
 ['Andoni IraolaRuud van Nistelrooy', 'M. HermansenGroin injury'],
 ['Andoni IraolaRuud van Nistelrooy', 'D. OuattaraGroin injury'],
 ['Andoni IraolaRuud van Nistelrooy', 'E. nalCruciate ligament injury'],
 ['Andoni IraolaRuud van Nistelrooy', 'L. CookStraight red card'],
 ['Andoni IraolaRuud van Nistelrooy', 'S. MavididiMuscle injury'],
 ['Andoni IraolaRuud van Nistelrooy', 'H. SouttarAchilles tendon injury'],
 ['Andoni IraolaRuud van Nistelrooy', 'A. FatawuACL knee injury']]
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
    sample_pairs = [["Explore", "Lets not do that again"], ["Explore the training videos", "Lets do that again"],["Explore the mountains in the summer to avoid the cold", "Lets do that again"]]
    #sample_pairs = [['Family Viewing 1987, Atom Egoyan', 'In the Bedroom 2001, Todd Field', 
                     #"Atom Egoyan's haunting adaptation of the Russell Banks novel The Sweet Hereafter was the Canadian filmmaker's most successful film to date, taking home a Special Grand Jury Prize at the 1997 Cannes Film Festival and scoring a pair of Academy Award nominations, including Best Director. Restructured to fit Egoyan's signature mosaic narrative style, the story concerns the cultural aftershocks which tear apart a small British Columbia town in the wake of a school-bus accident which leaves a number of local children dead. Ian Holm stars as Mitchell Stephens, a big-city lawyer who arrives in the interest of uniting the survivors to initiate a lawsuit his maneuvering only drives the community further apart, reopening old wounds and jeopardizing any hopes of emotional recovery. Like so many of Egoyan's features, The Sweet Hereafter is a serious and painfully honest exploration of family grief no character is immune from the sense of utter devastation which grips the film, not even the attorney, whose interests are in part motivated by his own remorse over the fate of his daughter, an HIV-positive drug addict.", 
                     #'The Five Senses 1999, Jeremy Podeswa', 'The Ice Storm 1997, Ang Lee', 'Blue 1993, Krzysztof Kieslowski', "L'Humanit 1999, Bruno Dumont", 'Eureka 2000, Shinji Aoyama', 'Corrections to this Entry', 'Similar Works', 'Is related to:', 'Work Rating', "The Son's Room 2001, Nanni Moretti", 'The Bed You Sleep In 1993, Jon Jost', 'The Pledge 2001, Sean Penn', 
    #                 'Director', 'Atom Egoyan', 'Other Related Works', 'The War Zone 1999, Tim Roth', 'by Jason Ankeny', 'Plot Synopsis']]
    import numpy as np
    # sample_pairs = [["The knight layed down his sword", "for a prince"],
    #                 ["The strongest man in the kingdom", "a beggar's boy"],
    #                 ["It was huge", "apple"],
    #                 ["The king is not a fool at all", "The Queen"]]
    txt = clean_instructional_text(sample_pairs)
    #ifnore = [1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 3, 1, 3, 3, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 3, 3, 1, 1, 3, 1, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 2, 3, 1, 1, 1, 3]
    print(txt)
    last = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]

    best_attempt = [1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 0, 1, 1, 2]
    # import sys
    # sys.path.insert(1, r"/vol/bitbucket/mjh24/IAEA-thesis")
    # from Stage2.classifyingEdges.metrics import metrics
    # y_true_str = "2 2 1 1 ? ? ? ? 3 3 1 1 3 3 2 2 2 2 1 1 ? ? 1 1 3 3 1 1 3 3 1 1 1 1 2 2 2 2 2 2"
    # y_true = [3 if tok == "?" else int(tok) for tok in y_true_str.split()]
    # metrics(labels[:40],y_true)