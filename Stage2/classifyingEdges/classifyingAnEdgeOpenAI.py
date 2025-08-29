#Turns out GPT is very bad at this task, almost as bad as FLAN-T5. It tried to be too smart I think?

from openai import OpenAI
import tiktoken
import os, re, sys

OPENAI_MODEL = "gpt-5-nano"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Context window controls (tokens). Set real numbers via env for your model.
CONTEXT_WINDOW_TOKENS = 400000
SAFETY_MARGIN_TOKENS  = 1000    # buffer to avoid overflows

# USD per 1K tokens (configure to your model’s pricing)
PRICE_IN_PER_1M  = 0.05
PRICE_OUT_PER_1M = 0.4

# Per-text truncation for long inputs
MAX_TOKENS_PER_TEXT = 128
MAX_NEW_TOKENS_GUESS_PER_LABEL = 2

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are a classifier. Given pairs of linked texts from an HTML document, decide their relationship for the purpose of fact extraction.
Output ONLY the category number for each pair.

Categories (choose exactly one per pair):
1 = Context/information: Two entries can be combined into a single fact-like statement (e.g., “X is Y,” “X has Y,” “X happened in Y”), even if each entry alone already looks like a complete statement. As long as the combination is linguistically plausible, count it as Category 1. Do not require correctness or exclusivity — overlap or redundancy is fine if the pair enables a new inference. 
    (e.g., ["Director", "Andy Murray"] → "Andy Murray is a director"; 
    ["the ap1000 pwr is the best in the market", "the next generation small modular reactor for remote applications"] -> “The ap1000 pwr is the next generation small modular reactor for remote applications.”).
2 = Sibling items: both entries stand alone and do not add factual content to each other. This includes generic labels, buttons, or sibling list items. 
    (e.g., ["ap300 smr", "the ap300 smr is the next evolution of the licensed ap1000 technology"] → the second already contains the fact; the first adds nothing new. 
    ["afc teams", "nfc teams"] -> sibling content, no fact to extract). 
3 = No helpful relation: the entries are not meaningfully connected for extracting facts. 
    (e.g., ["Read more", "Imperial College London"]). 

Important: 
- Always think in terms of whether combining the two entries enables a fact that one alone does not provide. 
- Always evaluate the relationship only within the specific pair of entries to compare. Do not use information from other pairs, prior knowledge, or assumptions about what the entries might mean elsewhere. The decision must be based solely on whether combining these two entries alone produces a new factual statement that neither provides by itself. 
- If neither entry is useful to the other, the pair is category 3.
- Output format: Output exactly {N} integers separated by single spaces, e.g. "1 2 3".
"""

USER_HEADER = "Classify each pair below one by one. Output a plain, space-separated list of exactly {N} integers only.\n\n"
USER_FOOTER = "\n\nOutput {N} integers only:"

# ---------- token helpers ----------
ENC = tiktoken.get_encoding("o200k_base")#"cl100k_base")

def _count_tokens(s: str) -> int:
    return len(ENC.encode(s))

def _truncate_to_tokens(s: str, max_tokens: int) -> str:
    toks = ENC.encode(s)
    if len(toks) <= max_tokens:
        return s
    return ENC.decode(toks[:max_tokens]) + " …"

def _estimate_cost(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in / 1000000.0) * PRICE_IN_PER_1M + (tokens_out / 1000000.0) * PRICE_OUT_PER_1M

# ---------- batching that maximizes the context window ----------
def _create_entry_line(idx: int, left: str, right: str) -> str:
    l_t = _truncate_to_tokens(left, MAX_TOKENS_PER_TEXT)
    r_t = _truncate_to_tokens(right, MAX_TOKENS_PER_TEXT)
    return f"{idx}.\nLEFT: {l_t}\nRIGHT: {r_t}\n\n"

def _create_batches(pairs, max_batch_size=None):
    """Yields batches to maximally fills the context window."""
    sys_tokens, header_tokens, footer_tokens = _count_tokens(SYSTEM_PROMPT), _count_tokens(USER_HEADER), _count_tokens(USER_FOOTER)
    max_batch_size = max_batch_size if max_batch_size else len(pairs)

    i = 0
    while i < len(pairs):
        # start a new batch
        batch_start = i
        used = sys_tokens + header_tokens + footer_tokens
        lines = []
        total_out_tokens = 0

        # greedily add pairs until token limit would be exceeded, or batch limit is reached
        while i < len(pairs) and i < max_batch_size+batch_start:
            line = _create_entry_line((i - batch_start) + 1, pairs[i][0], pairs[i][1])
            line_tokens = _count_tokens(line)
            next_used = used + line_tokens
            next_out_guess = total_out_tokens + MAX_NEW_TOKENS_GUESS_PER_LABEL
            if next_used + next_out_guess + SAFETY_MARGIN_TOKENS > CONTEXT_WINDOW_TOKENS:
                break
            lines.append(line)
            used = next_used
            total_out_tokens = next_out_guess
            i += 1

        # fallback: if even one pair doesn't fit (extreme long text), force single-pair batch
        if batch_start == i:
            raise Exception(f"Caught in an infinite loop at entry number {i}\n{pairs[i]}\nNo more batches are being created")

        yield (batch_start, i, "".join(lines), used, total_out_tokens)

def classify_link_pairs_openAI(pairs, dry_run_confirm=True, max_batch_size: int | None = 30, return_raw_response_and_cost=False):
    """
    pairs: list[[left, right]]
    return: list[int] with values in {1,2,3}; printed as "1 2 3 ..."
    """
    results = []

    batches = list(_create_batches(pairs, max_batch_size))
    
    total_in_tokens  = sum(b[3] for b in batches)
    total_out_tokens = sum(b[4] for b in batches)
    est_cost = _estimate_cost(total_in_tokens, total_out_tokens+15000*len(batches))
    if dry_run_confirm:
        print(f"Expected total price: ${est_cost}")
        ans = input("Type Yes to continue this batch (anything else aborts): ").strip()
        if ans != "Yes":
            print("Aborted by user.")
            sys.exit(1)

    runningCostTotal = 0
    rawText = ""

    for batch_idx, (lo, hi, lines_joined, used_in_tokens, out_tokens_guess) in enumerate(batches, start=1):

        user_prompt = USER_HEADER.format(N=(hi - lo)) + lines_joined + USER_FOOTER.format(N=(hi - lo))

        # token+cost estimate
        est_cost = _estimate_cost(used_in_tokens, out_tokens_guess+15000)

        print(f"\nBatch {batch_idx} (pairs {lo}..{hi-1}): ~input tokens={used_in_tokens:,}, ~output tokens={out_tokens_guess:,}, "
              f"est. cost=${est_cost:.4f} (IN=${PRICE_IN_PER_1M}/1M, OUT=${PRICE_OUT_PER_1M}/1M)")

        # Call API (no JSON; tiny output)
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT.format(N=(hi - lo))},
                {"role": "user", "content": user_prompt},
            ],
            reasoning={"effort": "medium"},   # "minimal"/"low"/"medium"/"high" depending on model
            text={"verbosity": "low"},         # "low" | "medium" | "high" (GPT-5)
            #max_output_tokens=(hi - lo) * 10,  # upper bound for safety
            #temperature=0.0
        )

        text = resp.output_text.strip()
        usage = resp.usage
        reasoning_tokens = usage.output_tokens_details.reasoning_tokens
        actual_out_tokens = usage.output_tokens - reasoning_tokens
        actual_in_tokens = usage.input_tokens
        actual_total_tokens = usage.total_tokens
        batchCost =  _estimate_cost(actual_in_tokens, usage.output_tokens)
        runningCostTotal += batchCost
        print("Total_tokens=", actual_total_tokens, " {input_tokens=", actual_in_tokens, " reasoning_tokens=", reasoning_tokens, " output_tokens=", actual_out_tokens, "}")
        print("Running Total cost: $", runningCostTotal, " This batch cost: $", batchCost)

        # Parse exactly (hi - lo) labels in {1,2,3}
        want = hi - lo
        nums = re.findall(r"\b[1-3]\b", text)
        if len(nums) != want:
            # soft fallback: split by whitespace, take first char if in 1-3
            tokens = text.split()
            soft = [t[0] for t in tokens if t and t[0] in "123"]
            if len(soft) != want:
                raise ValueError(
                    f"Expected {want} labels for batch {batch_idx}, got {len(nums)} (soft {len(soft)}). "
                    f"Raw output:\n{text}"
                )
            nums = soft

        rawText += "[NEWBATCH]"+ text

        results.extend(int(x) for x in nums)

    return results, runningCostTotal, rawText if return_raw_response_and_cost else results


if __name__ == "__main__":
    # Example usage
    # sample_pairs = [
    #     ["learn more", "learn more about why databricks selected westinghouse ai for its 2025 data intelligence in energy utility award, including judging criteria and methodology."],
    #     ["Director", "Christopher Nolan"],
    #     ["afcteams", "nfcteams"],
    #     ["Read more", "Imperial College London"],
    #     ["This is a very long paragraph describing the AP1000 reactor in detail, including safety systems, passive cooling, and economic competitiveness across global markets.", 
    #      "Explore the AP1000 overview page"],
    #     ["Another extremely long body of text ... " * 50, "Products"],
    # ]

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

    labels, _, _ = classify_link_pairs_openAI(sample_pairs[:40], dry_run_confirm=True)
    print("\nFinal labels:")
    print(" ".join(str(x) for x in labels))

    import sys
    sys.path.insert(1, r"/vol/bitbucket/mjh24/IAEA-thesis")
    from Stage2.classifyingEdges.metrics import metrics
    y_true_str = "2 2 1 1 ? ? ? ? 3 3 1 1 3 3 2 2 2 2 1 1 ? ? 1 1 3 3 1 1 3 3 1 1 1 1 2 2 2 2 2 2"
    y_true = [3 if tok == "?" else int(tok) for tok in y_true_str.split()]
    metrics(labels[:len(y_true)],y_true)

    # y_old_prompt = "1 1 1 1 2 2 2 2 3 1 1 1 2 2 1 1 1 1 2 2 2 2 1 1 1 1 2 2 1 1 1 1 1 1 2 2 2 2 1" #only 39 values
    # y_old_prompt = [int(i) for i in y_old_prompt.split()]
    # y_new_prompt = [2, 2, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
