#Turns out GPT is very bad at this task, almost as bad as FLAN-T5. It tried to be too smart I think?

from openai import OpenAI
import tiktoken
import os, re, sys

OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Context window controls (tokens). Set real numbers via env for your model.
CONTEXT_WINDOW_TOKENS = 400000
SAFETY_MARGIN_TOKENS  = 1000    # buffer to avoid overflows

# USD per 1K tokens (configure to your model’s pricing)
PRICE_IN_PER_1M  = 0.4
PRICE_OUT_PER_1M = 1.6

# Per-text truncation for long inputs
MAX_TOKENS_PER_TEXT = 128
MAX_NEW_TOKENS_GUESS_PER_LABEL = 2

client = OpenAI(api_key="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")#OPENAI_API_KEY)

SYSTEM_PROMPT_PAIRWISE = """You are a classifier.
Decide the relation between INPUT 1 and INPUT 2.

CLASSES:
    0 = Both belong in the same category (e.g., both are a product, both are a feeling, both are dates).
    1 = Both belong in seperate categories"""

SYSTEM_PROMPT_SENTENCE = """TASK:
    You are a fact-finding classifier. Read the QUERY text and assign it one classification. Output the classification number for each pair

CLASSES:
    0: All subjects are known. The text does not reference or allude to something unknown;
    1: There are key unknown subjects referenced;
"""

SYSTEM_PROMPT = """You are a classifier. Given pairs of linked texts from an HTML document, decide their relationship for fact extraction.
Output the classification number for each pair.

Categories (choose exactly one per pair):
1 = Context/information: LEFT contains any key contextual information that RIGHT is missing when extracting facts from RIGHT. Do not require correctness or exclusivity, as long as the fact is linguistically plausible.
    (e.g., ["Director", "Andy Murray"] → "Andy Murray is a director"; 
    ["the IPhone XE is the best in the market", "we included it in this year's small phone competition"] -> “The IPhone XE will be included in this year's small phone competition”).
2 = Sibling items: LEFT only contains contextual information that RIGHT already has when extracting facts from RIGHT. 
    (e.g., ["England football team", "the England football team will play tomorrow"] → the second already contains the fact; the first adds nothing new. 
    ["afc teams", "nfc teams"] -> sibling content, no fact to extract). 
3 = No helpful relation: the entries are not meaningfully connected for extracting facts.  

Important: 
- Always think in terms of whether combining the two entries enables a fact that one alone does not provide. 
- Judge only within each pair, no outside knowledge.
"""

USER_HEADER = "\nClassify each, one by one:\n\n"
USER_FOOTER = "\n\nOutput {N} space separated integers ONLY:"

# ---------- token helpers ----------
ENC = tiktoken.get_encoding("cl100k_base")#"o200k_base")

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

def classify_link_pairs_openAI(pairs, dry_run_confirm=True, max_batch_size: int | None = 1, return_raw_response_and_cost=False):
    """
    pairs: list[[left, right]]
    return: list[int] with values in {1,2,3}; printed as "1 2 3 ..."
    """
    results = []

    batches = list(_create_batches(pairs, max_batch_size))
    
    total_in_tokens  = sum(b[3] for b in batches)
    total_out_tokens = sum(b[4] for b in batches)
    est_cost = _estimate_cost(total_in_tokens, total_out_tokens)
    if dry_run_confirm:
        print(f"Expected total price: ${est_cost}")
        ans = input("Type Yes to continue this batch (anything else aborts): ").strip()
        if ans != "Yes":
            print("Aborted by user.")
            sys.exit(1)

    runningCostTotal = 0
    rawText = ""

    for batch_idx, (lo, hi, lines_joined, used_in_tokens, out_tokens_guess) in enumerate(batches, start=1):

        user_prompt = USER_HEADER + lines_joined + USER_FOOTER.format(N=(hi - lo))

        # token+cost estimate
        est_cost = _estimate_cost(used_in_tokens, out_tokens_guess)

        print(f"\nBatch {batch_idx} (pairs {lo}..{hi-1}): ~input tokens={used_in_tokens:,}, ~output tokens={out_tokens_guess:,}, "
              f"est. cost=${est_cost:.4f} (IN=${PRICE_IN_PER_1M}/1M, OUT=${PRICE_OUT_PER_1M}/1M)")

        # Call API (no JSON; tiny output)
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT.format(N=(hi - lo))},
                {"role": "user", "content": user_prompt},
            ],
            #reasoning={"effort": "medium"},   # "minimal"/"low"/"medium"/"high" depending on model
            #text={"verbosity": "low"},         # "low" | "medium" | "high" (GPT-5)
            #max_output_tokens=(hi - lo) * 10,  # upper bound for safety
            temperature=0.0,
            #presence_penalty=0.0 # Make it more negative to make the answer more on topic
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
        print("output so far: ", rawText)
        print("results so far: ", results)

    return (results, runningCostTotal, rawText) if return_raw_response_and_cost else results

def classify_needsContext_openAI(texts, dry_run_confirm=True, batch_size: int | None = 1, return_raw_response_and_cost=False):
    """
    pairs: list[[left, right]]
    return: list[int] with values in {1,2,3}; printed as "1 2 3 ..."
    """
    results = [] 
    
    in_SYST  = _count_tokens(SYSTEM_PROMPT_SENTENCE+USER_HEADER+USER_FOOTER)
    in_txts = _count_tokens("\n\n".join(texts))
    total_out_tokens = _count_tokens("1 "*len(texts))
    est_cost = _estimate_cost(in_txts+in_SYST*(len(texts)/batch_size), total_out_tokens)
    if dry_run_confirm:
        print(f"Expected total price: ${est_cost}")
        ans = input("Type Yes to continue this batch (anything else aborts): ").strip()
        if ans != "Yes":
            print("Aborted by user.")
            sys.exit(1)

    runningCostTotal = 0
    rawText = ""

    for batch_idx in range(0, len(texts), batch_size):
        batch = texts[batch_idx:batch_idx+batch_size]

        user_prompt = USER_HEADER + "QUERY: " +"\n\nQUERY: ".join(batch) + USER_FOOTER.format(N=(batch_size))
        used_in_tokens = in_SYST + _count_tokens(user_prompt)
        out_tokens_guess = _count_tokens("1 "*batch_size)

        print(f"\nBatch {batch_idx} ~input tokens={used_in_tokens:,}, ~output tokens={out_tokens_guess:,}, "
              f"est. cost=${est_cost:.4f} (IN=${PRICE_IN_PER_1M}/1M, OUT=${PRICE_OUT_PER_1M}/1M)")

        # Call API (no JSON; tiny output)
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT_SENTENCE},
                {"role": "user", "content": user_prompt},
            ],
            #reasoning={"effort": "minimal"},   # "minimal"/"low"/"medium"/"high" depending on model
            #text={"verbosity": "low"},         # "low" | "medium" | "high" (GPT-5)
            #max_output_tokens=(hi - lo) * 10,  # upper bound for safety
            temperature=0.0,
            #presence_penalty=0.0 # Make it more negative to make the answer more on topic
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
        nums = re.findall(r"\b[0-1]\b", text)
        if len(nums) != len(batch):
            # soft fallback: split by whitespace, take first char if in 1-3
            tokens = text.split()
            soft = [t[0] for t in tokens if t and t[0] in "01"]
            if len(soft) != batch_size:
                raise ValueError(
                    f"Expected {batch_size} labels for batch {batch_idx}, got {len(nums)} (soft {len(soft)}). "
                    f"Raw output:\n{text}"
                )
            nums = soft

        rawText += "[NEWBATCH]"+ text

        results.extend(int(x) for x in nums)
        print("output so far: ", rawText)
        print("results so far: ", results)

    return (results, runningCostTotal, rawText) if return_raw_response_and_cost else results

def classify_pairwiseEdges_openAI(triplets, dry_run_confirm=True, batch_size: int | None = 1, return_raw_response_and_cost=False):
    """
    pairs: list[[left, right]]
    return: list[int] with values in {1,2,3}; printed as "1 2 3 ..."
    """
    results = [] 
    
    in_SYST  = _count_tokens(SYSTEM_PROMPT_PAIRWISE+USER_HEADER+USER_FOOTER)
    in_txts = _count_tokens("\n\n".join(["INPUT: ".join(triplet) for triplet in triplets]))
    total_out_tokens = _count_tokens("1 "*len(triplets))
    est_cost = _estimate_cost(in_txts+in_SYST*(len(triplets)/batch_size), total_out_tokens)
    if dry_run_confirm:
        print(f"Expected total price: ${est_cost}")
        ans = input("Type Yes to continue this batch (anything else aborts): ").strip()
        if ans != "Yes":
            print("Aborted by user.")
            sys.exit(1)

    runningCostTotal = 0
    rawText = ""

    for batch_idx in range(0, len(triplets), batch_size):
        batch = triplets[batch_idx:batch_idx+batch_size]

        user_prompt = "".join([f"\n\n{idx})\nINPUT 1: " + triplet[1] + "\nINPUT 2: " + triplet[2] for idx, triplet in enumerate(batch)])
        user_prompt = "\n\nCLASSIFY one by one:" + user_prompt + f"\n\nCLASSIFICATION (output {len(batch)} space separated numbers only):"
        used_in_tokens = in_SYST + _count_tokens(user_prompt)
        out_tokens_guess = _count_tokens("1 "*batch_size)

        print(f"\nBatch {batch_idx} ~input tokens={used_in_tokens:,}, ~output tokens={out_tokens_guess:,}, "
              f"est. cost=${est_cost:.4f} (IN=${PRICE_IN_PER_1M}/1M, OUT=${PRICE_OUT_PER_1M}/1M)")

        # Call API (no JSON; tiny output)
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT_PAIRWISE},
                {"role": "user", "content": user_prompt},
            ],
            #reasoning={"effort": "high"},   # "minimal"/"low"/"medium"/"high" depending on model
            #text={"verbosity": "low"},         # "low" | "medium" | "high" (GPT-5)
            #max_output_tokens=(hi - lo) * 10,  # upper bound for safety
            temperature=0.0,
            #presence_penalty=0.0 # Make it more negative to make the answer more on topic
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
        nums = re.findall(r"\b[0-2]\b", text)
        if len(nums) != len(batch):
            # soft fallback: split by whitespace, take first char if in 1-3
            tokens = text.split()
            soft = [t[0] for t in tokens if t and t[0] in "012"]
            if len(soft) != batch_size:
                raise ValueError(
                    f"Expected {batch_size} labels for batch {batch_idx}, got {len(nums)} (soft {len(soft)}). "
                    f"Raw output:\n{text}"
                )
            nums = soft

        rawText += "[NEWBATCH]"+ text

        results.extend(int(x) for x in nums)
        print("output so far: ", rawText)
        print("results so far: ", results)

    return (results, runningCostTotal, rawText) if return_raw_response_and_cost else results

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
["the fact is it's safetruth. a person working fulltime in a nuclear power plant receives less additional radiation in a year than a flight crew or a business traveler with 250 hours of flying time", 'when it comes to creating a more sustainable planet the need for renewable energy cant replace the need for safe energy with nuclear power you get the best of both worlds'],
['when it comes to creating a more sustainable planet the need for renewable energy cant replace the need for safe energy with nuclear power you get the best of both worlds', "the fact is it's safetruth. a person working fulltime in a nuclear power plant receives less additional radiation in a year than a flight crew or a business traveler with 250 hours of flying time"],
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
['safety getting the facts right', "the fact is it's safetruth. a person working fulltime in a nuclear power plant receives less additional radiation in a year than a flight crew or a business traveler with 250 hours of flying time"],
["the fact is it's safetruth. a person working fulltime in a nuclear power plant receives less additional radiation in a year than a flight crew or a business traveler with 250 hours of flying time", 'safety getting the facts right'],
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
    sample_pairs = [["The weather temperature", "cold", "hot"],
                    ["The strongest man in the universe", "the king", "a dragon"],
                    ["It is blue", "My car", "My house"],
                    ["I love it", "Ice cream", "Sugar"],
                    ['','news', 'westinghousenuclearning'],
                    ['','united kingdom english', 'global directory x'],
                    ['','shape your future', 'manufacturing operations maintenance'],
                    ['','evinci microreactor', 'ap1000 pwr'],
                    ['','product spotlights', 'westinghouseiq'],
                    ['','ap1000 pwr', 'product spotlights'],]
    # ifnore = np.array([1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 3, 1, 3, 3, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 3, 3, 1, 1, 3, 1, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 2, 3, 1, 1, 1, 3])
    # sample_pairs = sample_pairs[ifnore!=1]
    # sample_pairs = sample_pairs.tolist()

    labels = classify_pairwiseEdges_openAI(sample_pairs, dry_run_confirm=True, batch_size=4)
    print("\nFinal labels:")
    print(" ".join(str(x) for x in labels))
    for pair, label in zip(sample_pairs, labels):
        print(label, pair)

    nano_4_old = [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1] # cost $ 0.000258
    mini_4_old = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0] # cost $ 0.0010287999999999999
    nano_5_med = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0] # cost $ 0.0039208
    nano_5_low = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0] # cost $ 0.0011048
    nano_5_min = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] # cost $ 0.00018320000000000006
    nano_4 = [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] # cost $ 0.00026770000000000006
    mini_4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0] # cost $ 0.0010708

    import sys
    sys.path.insert(1, r"/vol/bitbucket/mjh24/IAEA-thesis")
    from Stage2.classifyingEdges.metrics import metrics
    y_true_str = "2 2 1 1 ? ? ? ? 3 3 1 1 3 3 2 2 2 2 1 1 ? ? 1 1 3 3 1 1 3 3 1 1 1 1 2 2 2 2 2 2"
    y_true = [3 if tok == "?" else int(tok) for tok in y_true_str.split()]
    #metrics(labels[:len(y_true)],y_true)

    # y_old_prompt = "1 1 1 1 2 2 2 2 3 1 1 1 2 2 1 1 1 1 2 2 2 2 1 1 1 1 2 2 1 1 1 1 1 1 2 2 2 2 1" #only 39 values
    # y_old_prompt = [int(i) for i in y_old_prompt.split()]
    low_y_5_per_but_low_reasoning = [1, 2, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 1, 1, 3, 1, 2, 3, 3, 3, 3, 3] # cost 0.001826  ~500 reasoning tokens per batch for 5 on low
    min_y_1_per_row_no_reasoning = [2, 2, 1, 3, 3, 3, 3, 3, 1, 1, 1, 1, 3, 2, 1, 2, 1, 1, 1, 2, 3, 3, 2, 2, 1, 1, 3, 3, 1, 1, 2, 3, 1, 1, 3, 3, 3, 3, 3, 3] # cost 0.001169    0 reasoning tokens, 1 per batch
    min_y_1_per_row_no_reasoning_reduced_prompt = [1, 1, 2, 1, 3, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 3, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2, 3, 1, 3, 2, 2, 2, 2, 2] # cost 0.0005   1 per batch
    min_y_1_per_row_stripping_examples = [1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 3, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3] # cost ~0.000789   1 per batch
    med_y_40_per_row_stripping_examples = "2 1 3 1 3 3 3 3 3 3 2 1 3 3 3 1 2 2 3 1 3 3 2 1 3 3 3 2 3 2 1 2 3 3 3 3 3 3 3" #only 39 values Cost ~0.00434 10560 reasoning tokens 40 per batch
    med_y_40_per_row_stripping_examples = [int(i) for i in med_y_40_per_row_stripping_examples.split()]
    low_y_4_per_row_stripping_examples = [2, 1, 1, 1, 3, 1, 1, 1, 3, 3, 2, 2, 3, 3, 3, 3, 1, 2, 3, 3, 3, 2, 2, 1, 3, 3, 3, 3, 2, 3, 1, 2, 1, 1, 3, 3, 3, 3, 1, 1]# cost 0.003 
    # y_new_prompt = [2, 2, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3] # This cost 0.00233  ~5500 reasoning tokens
    #metrics(low_y_4_per_row_stripping_examples, y_true[:40])