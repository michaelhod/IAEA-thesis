#Turns out GPT is very bad at this task, almost as bad as FLAN-T5. It tried to be too smart I think?

from openai import OpenAI
import tiktoken
import os, re, sys

OPENAI_MODEL = "gpt-4.1-nano"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Context window controls (tokens). Set real numbers via env for your model.
CONTEXT_WINDOW_TOKENS = 400000
SAFETY_MARGIN_TOKENS  = 1000    # buffer to avoid overflows

# USD per 1K tokens (configure to your model’s pricing)
PRICE_IN_PER_1M  = 0.1
PRICE_OUT_PER_1M = 0.4

# Per-text truncation for long inputs
MAX_TOKENS_PER_TEXT = 128
MAX_NEW_TOKENS_GUESS_PER_LABEL = 64

client = OpenAI(api_key="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")#OPENAI_API_KEY)

ADD_CONTEXT_PROMPT = """Summarise the INPUT provided into facts. Do not remove important facts. Do not add facts.
If there is missing information within the INPUT (e.g. pronouns, alluding to something) use the CONTEXT to enrich it.
Do not use the CONTEXT if the fact is complete.
Do not ouput a fact directly from the CONTEXT. Output only the facts from the INPUT. Be concise.
If there are no fact, output "NO FACTS"
"""

SUMMARISE_PROMPT = """A fact is a declarative, verifiable claim with a concrete subject and predicate that can be true or false.
Summarise the INPUT provided into a minimal list of self-contained facts.
If the INPUT contains no facts, output "NO FACTS". Ignore sentences that contain no facts.
Within each fact, NEVER use pronouns (e.g., him, these, it).
The previous fact must not imply something in the next fact.
Explicitly state everything, even if it means repeating words.
Be concise. Seperate the facts with "\\n".
"""

USER_HEADER = "\n\n"
USER_FOOTER = "\n\nSummary:"

# ---------- token helpers ----------
ENC = tiktoken.get_encoding("cl100k_base")#"o200k_base")

def _count_tokens(s: str) -> int:
    return len(ENC.encode(s))

def _truncate_to_tokens(s: str, max_tokens: int) -> str:
    toks = ENC.encode(s)
    if len(toks) <= max_tokens:
        return s
    return ENC.decode(toks[:max_tokens]) + " …"

def _estimate_cost(tokens_in: int, tokens_out: int, price_in=PRICE_IN_PER_1M, price_out=PRICE_OUT_PER_1M) -> float:
    return (tokens_in / 1000000.0) * price_in + (tokens_out / 1000000.0) * price_out

# ---------- batching that maximizes the context window ----------
def _create_entry_line(idx: int, left: str, right: str) -> str:
    l_t = _truncate_to_tokens(left, MAX_TOKENS_PER_TEXT)
    r_t = _truncate_to_tokens(right, MAX_TOKENS_PER_TEXT)
    return f"CONTEXT: {r_t}\nINPUT: {l_t}\n\n"

def _create_batches(pairs, max_batch_size=None):
    """Yields batches to maximally fills the context window."""
    sys_tokens, header_tokens, footer_tokens = _count_tokens(ADD_CONTEXT_PROMPT), _count_tokens(USER_HEADER), _count_tokens(USER_FOOTER)
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

def add_context(pairs, dry_run_confirm=True, max_batch_size: int | None = 1, return_raw_response_and_cost=False):
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

    for batch_idx, (lo, hi, lines_joined, used_in_tokens, out_tokens_guess) in enumerate(batches, start=1):

        user_prompt = USER_HEADER + lines_joined + USER_FOOTER

        # token+cost estimate
        est_cost = _estimate_cost(used_in_tokens, out_tokens_guess)

        print(f"\nBatch {batch_idx} (pairs {lo}..{hi-1}): ~input tokens={used_in_tokens:,}, ~output tokens={out_tokens_guess:,}, "
              f"est. cost=${est_cost:.4f} total cost=${runningCostTotal:.4f} (IN=${PRICE_IN_PER_1M}/1M, OUT=${PRICE_OUT_PER_1M}/1M)")

        # Call API (no JSON; tiny output)
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": ADD_CONTEXT_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            #reasoning={"effort": "medium"},   # "minimal"/"low"/"medium"/"high" depending on model
            #text={"verbosity": "low"},         # "low" | "medium" | "high" (GPT-5)
            max_output_tokens=int(len(user_prompt)/2),  # upper bound for safety
            temperature=0.0,
            truncation='auto',
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

        results.append(text)
        print("results so far: ", results)

    return (results, runningCostTotal) if return_raw_response_and_cost else results

def summairse(texts, dry_run_confirm=True, batch_size: int | None = 1, return_raw_response_and_cost=False):
    """
    pairs: list[[left, right]]
    return: list[int] with values in {1,2,3}; printed as "1 2 3 ..."
    """
    results = [] 
    
    in_SYST  = _count_tokens(SUMMARISE_PROMPT+USER_HEADER+USER_FOOTER)
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

    for batch_idx in range(0, len(texts), batch_size):
        batch = texts[batch_idx:batch_idx+batch_size]

        user_prompt = USER_HEADER + "INPUT: " +"\n\nINPUT: ".join(batch) + USER_FOOTER.format(N=(batch_size))
        used_in_tokens = in_SYST + _count_tokens(user_prompt)
        out_tokens_guess = _count_tokens("1 "*batch_size)

        print(f"\nBatch {batch_idx}: ~input tokens={used_in_tokens:,}, ~output tokens={out_tokens_guess:,}, "
              f"est. cost=${est_cost:.4f} total cost=${runningCostTotal:.4f} (IN=${PRICE_IN_PER_1M}/1M, OUT=${PRICE_OUT_PER_1M}/1M)")

        def count_sents(text):
            sents = re.split(r'(?<=[.!?])\s+', text.strip())
            return len([s.strip() for s in sents if s.strip()])

        modelType = "gpt-4.1-nano"
        cost_per_M = (0.1, 0.4)
        if count_sents(user_prompt) > 3:
            modelType = "gpt-4.1-mini"
            cost_per_M = (0.4, 1.6)
        print(modelType)
        # Call API (no JSON; tiny output)
        resp = client.responses.create(
            model=modelType,
            input=[
                {"role": "system", "content": SUMMARISE_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            #reasoning={"effort": "medium"},   # "minimal"/"low"/"medium"/"high" depending on model
            #text={"verbosity": "low"},         # "low" | "medium" | "high" (GPT-5)
            max_output_tokens=int(len(user_prompt)/2),  # upper bound for safety
            temperature=0.0,
            truncation='auto',
            #presence_penalty=0.0 # Make it more negative to make the answer more on topic
        )

        text = resp.output_text.strip()
        usage = resp.usage
        reasoning_tokens = usage.output_tokens_details.reasoning_tokens
        actual_out_tokens = usage.output_tokens - reasoning_tokens
        actual_in_tokens = usage.input_tokens
        actual_total_tokens = usage.total_tokens
        batchCost =  _estimate_cost(actual_in_tokens, usage.output_tokens, cost_per_M[0], cost_per_M[1])
        runningCostTotal += batchCost
        print("Total_tokens=", actual_total_tokens, " {input_tokens=", actual_in_tokens, " reasoning_tokens=", reasoning_tokens, " output_tokens=", actual_out_tokens, "}")
        print("Running Total cost: $", runningCostTotal, " This batch cost: $", batchCost)

        results.append(text)
        print("results so far: ", results)

    return (results, runningCostTotal) if return_raw_response_and_cost else results

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
#         ['ap300 smr', 'the ap300 smr is the next evolution of the licensed ap1000 technology'],
# ['the ap300 smr is the next evolution of the licensed ap1000 technology', 'ap300 smr'],
# ['evinci microreactor', 'the next generation small modular reactor for remote applications'],
# ['the next generation small modular reactor for remote applications', 'evinci microreactor'],
# ['enhance your training staffing and outsourcing needs with our training and resource solutions', 'westinghousenavigator'],
# ['westinghousenavigator', 'enhance your training staffing and outsourcing needs with our training and resource solutions'],
# ['enhance your training staffing and outsourcing needs with our training and resource solutions', 'westinghouseiq'],
# ['westinghouseiq', 'enhance your training staffing and outsourcing needs with our training and resource solutions'],
# ['when it comes to creating a more sustainable planet the need for renewable energy cant replace the need for safe energy with nuclear power you get the best of both worlds', 'safety getting the facts right'],
# ['safety getting the facts right', 'when it comes to creating a more sustainable planet the need for renewable energy cant replace the need for safe energy with nuclear power you get the best of both worlds'],
# ["BC Canada", "set in"],
# ["set in", "BC Canada"],
# ["Other related works", "is related to"],
# ["director", "atom eygoran"],
# ['ap1000 pwr', "the world's first proven generation iii pressurized water reactor and passive safety plant available"],
# ["the world's first proven generation iii pressurized water reactor and passive safety plant available", 'ap1000 pwr'],
# ['the ap300 smr is the next evolution of the licensed ap1000 technology', 'evinci microreactor'],
# ['evinci microreactor', 'the ap300 smr is the next evolution of the licensed ap1000 technology'],
# ['ap1000 pwr', 'the ap300 smr is the next evolution of the licensed ap1000 technology'],
# ['the ap300 smr is the next evolution of the licensed ap1000 technology', 'ap1000 pwr'],
['balancing wind solar and nuclear power will help achieve a carbonfree future and positively impact our changing climate over the past 50 years globally nuclear power has avoided nearly two years of the worlds energyrelated co2 emissions imagine how much more carbon pollution we can prevent', 'carbonfree energy'],
# ['carbonfree energy', 'balancing wind solar and nuclear power will help achieve a carbonfree future and positively impact our changing climate over the past 50 years globally nuclear power has avoided nearly two years of the worlds energyrelated co2 emissions imagine how much more carbon pollution we can prevent'],
# ['ap1000 pwr', 'the next generation small modular reactor for remote applications'],
# ['the next generation small modular reactor for remote applications', 'ap1000 pwr'],
# ['westinghousenuclearning', 'enhance your training staffing and outsourcing needs with our training and resource solutions'],
# ['enhance your training staffing and outsourcing needs with our training and resource solutions', 'westinghousenuclearning'],
# ['ap300 smr', 'the next generation small modular reactor for remote applications'],
# ['the next generation small modular reactor for remote applications', 'ap300 smr'],
# ["the fact is it's safetruth. a person working fulltime in a nuclear power plant receives less additional radiation in a year than a flight crew or a business traveler with 250 hours of flying time", 'when it comes to creating a more sustainable planet the need for renewable energy cant replace the need for safe energy with nuclear power you get the best of both worlds'],
# ['when it comes to creating a more sustainable planet the need for renewable energy cant replace the need for safe energy with nuclear power you get the best of both worlds', "the fact is it's safetruth. a person working fulltime in a nuclear power plant receives less additional radiation in a year than a flight crew or a business traveler with 250 hours of flying time"],
# ["the world's first proven generation iii pressurized water reactor and passive safety plant available", 'evinci microreactor'],
# ['evinci microreactor', "the world's first proven generation iii pressurized water reactor and passive safety plant available"],
# ['solar wind and nuclear energy are essential to a carbonfree future but the sun doesnt always shine and the wind doesnt always blow nuclear power plants are almost always on delivering the highest availability energy source and operating at maximum capacity more than 90% of the time', 'shaping the future with reliable energy'],
# ['shaping the future with reliable energy', 'solar wind and nuclear energy are essential to a carbonfree future but the sun doesnt always shine and the wind doesnt always blow nuclear power plants are almost always on delivering the highest availability energy source and operating at maximum capacity more than 90% of the time'],
# ["the world's first proven generation iii pressurized water reactor and passive safety plant available", 'ap300 smr'],
# ['ap300 smr', "the world's first proven generation iii pressurized water reactor and passive safety plant available"],
# ['balancing wind solar and nuclear power will help achieve a carbonfree future and positively impact our changing climate over the past 50 years globally nuclear power has avoided nearly two years of the worlds energyrelated co2 emissions imagine how much more carbon pollution we can prevent', 'nuclear energyprovides 55% of the uss and 14% of the worlds carbonfree energy'],
# ['nuclear energyprovides 55% of the uss and 14% of the worlds carbonfree energy', 'balancing wind solar and nuclear power will help achieve a carbonfree future and positively impact our changing climate over the past 50 years globally nuclear power has avoided nearly two years of the worlds energyrelated co2 emissions imagine how much more carbon pollution we can prevent'],
# ['project management support', 'quality environment health safety'],
# ['quality environment health safety', 'project management support'],
# ['engineering', 'corporate'],
# ['corporate', 'engineering'],
# ['safety getting the facts right', "the fact is it's safetruth. a person working fulltime in a nuclear power plant receives less additional radiation in a year than a flight crew or a business traveler with 250 hours of flying time"],
# ["the fact is it's safetruth. a person working fulltime in a nuclear power plant receives less additional radiation in a year than a flight crew or a business traveler with 250 hours of flying time", 'safety getting the facts right'],
# ['westinghouse partners with richland county ems to host training video series', 'westinghouse partnered with the richland county ems to host a series of training videos at their facility the training videos filmed onsite at westinghouse in hopkins signals a collaboration that focuses on producing highquality instructional content aimed at improving skills and knowledge among emergency medical personnel'],
# ['westinghouse partnered with the richland county ems to host a series of training videos at their facility the training videos filmed onsite at westinghouse in hopkins signals a collaboration that focuses on producing highquality instructional content aimed at improving skills and knowledge among emergency medical personnel', 'westinghouse partners with richland county ems to host training video series'],
# ['carbonfree energy', 'nuclear energyprovides 55% of the uss and 14% of the worlds carbonfree energy'],
# ['nuclear energyprovides 55% of the uss and 14% of the worlds carbonfree energy', 'carbonfree energy'],
# ['presidents kaizen week unlocks innovation across americas outage maintenance services', 'at westinghouse we believe that continuous improvement isnt just a goal its a mindset that approach came to life during our recent presidents kaizen week a dynamic crossfunctional initiative aimed at streamlining key business processes using lean methodologies originated by toyota production system for manufacturing improvements lean helps deliver maximum value to customers by identifying and eliminating waste'],
# ['at westinghouse we believe that continuous improvement isnt just a goal its a mindset that approach came to life during our recent presidents kaizen week a dynamic crossfunctional initiative aimed at streamlining key business processes using lean methodologies originated by toyota production system for manufacturing improvements lean helps deliver maximum value to customers by identifying and eliminating waste', 'presidents kaizen week unlocks innovation across americas outage maintenance services'],
# ['project management support', 'corporate'],
# ['quality environment health safety', 'corporate'],
# ['project management support', 'engineering'],
# ['quality environment health safety', 'engineering'],
# ['manufacturing operations maintenance', 'engineering'],
# ['manufacturing operations maintenance', 'project management support'],
# ['global directory x', 'westinghousenuclearning'],
# ['westinghouse joins texas nuclear alliance as a founding member', 'westinghouse ap1000 design receives us licensing extension to 2046'],
# ['westinghouse ap1000 design receives us licensing extension to 2046', 'westinghouse joins texas nuclear alliance as a founding member'],
# ['global directory x', 'westinghouseiq'],
# ['westinghousenuclearning', 'bulgaria bulgarian'],
# ['bulgaria bulgarian', 'westinghousenuclearning'],
# ["shaping tomorrow's energythrough advanced nuclear technology", 'westinghousenuclearning'],
# ['westinghousenuclearning', "shaping tomorrow's energythrough advanced nuclear technology"],
# ['poland polish', 'westinghousenuclearning'],
# ['bulgaria bulgarian', 'global directory x'],
# ['westinghouse joins texas nuclear alliance as a founding member', 'fermi america partners with westinghouse to support licensing for four ap1000 units'],
# ['fermi america partners with westinghouse to support licensing for four ap1000 units', 'westinghouse joins texas nuclear alliance as a founding member'],
# ['westinghouseiq', 'bulgaria bulgarian'],
# ['westinghouse ap1000 design receives us licensing extension to 2046', 'fermi america partners with westinghouse to support licensing for four ap1000 units'],
# ['fermi america partners with westinghouse to support licensing for four ap1000 units', 'westinghouse ap1000 design receives us licensing extension to 2046'],
# ['westinghouse partnered with the richland county ems to host a series of training videos at their facility the training videos filmed onsite at westinghouse in hopkins signals a collaboration that focuses on producing highquality instructional content aimed at improving skills and knowledge among emergency medical personnel', 'at westinghouse we believe that continuous improvement isnt just a goal its a mindset that approach came to life during our recent presidents kaizen week a dynamic crossfunctional initiative aimed at streamlining key business processes using lean methodologies originated by toyota production system for manufacturing improvements lean helps deliver maximum value to customers by identifying and eliminating waste'],
# ['westinghousenuclearning', 'canada english'],
# ['canada english', 'westinghousenuclearning'],
# ['poland polish', 'global directory x'],
# ["shaping tomorrow's energythrough advanced nuclear technology", 'westinghouseiq'],
# ['presidents kaizen week unlocks innovation across americas outage maintenance services', 'westinghouse partnered with the richland county ems to host a series of training videos at their facility the training videos filmed onsite at westinghouse in hopkins signals a collaboration that focuses on producing highquality instructional content aimed at improving skills and knowledge among emergency medical personnel'],
# ['westinghouse partnered with the richland county ems to host a series of training videos at their facility the training videos filmed onsite at westinghouse in hopkins signals a collaboration that focuses on producing highquality instructional content aimed at improving skills and knowledge among emergency medical personnel', 'presidents kaizen week unlocks innovation across americas outage maintenance services'],
# ['westinghouseiq', 'canada english'],
# ['the established design of the ap1000 reactor offers unequaled safety economic competitiveness and improved more efficient operations', 'westinghousenuclearning'],
# ['westinghousenuclearning', 'slovakia slovak'],
# ["shaping tomorrow's energythrough advanced nuclear technology", 'westinghousenavigator'],
# ['canada english', 'global directory x'],
# ['bulgaria bulgarian', 'slovakia slovak'],
# ['westinghousenuclearning', 'the evinci microreactor is a nextgeneration micromodular reactor combining innovative technologies with over 60 years of commercial nuclear expertise'],
# ['the evinci microreactor is a nextgeneration micromodular reactor combining innovative technologies with over 60 years of commercial nuclear expertise', 'westinghousenuclearning'],
# ['bulgaria bulgarian', 'slovenia slovenian'],
# ['bulgaria bulgarian', 'czech republic czech'],
# ['westinghousenuclearning', 'slovenia slovenian'],
# ['slovakia slovak', 'poland polish'],
# ['slovakia slovak', 'global directory x'],
# ['bulgaria bulgarian', 'sweden swedish'],
# ['westinghouseiq', 'the evinci microreactor is a nextgeneration micromodular reactor combining innovative technologies with over 60 years of commercial nuclear expertise'],
# ['slovenia slovenian', 'poland polish'],
# ['slovenia slovenian', 'global directory x'],
# ['bulgaria bulgarian', 'ukraine ukrainian'],
# ['westinghousenuclearning', 'czech republic czech'],
# ['bulgaria bulgarian', 'japan japanese'],
# ['czech republic czech', 'poland polish'],
# ['the established design of the ap1000 reactor offers unequaled safety economic competitiveness and improved more efficient operations', 'for the only smr based on advanced reactor technology thats already licensed and operating ap300 is the proven choice'],
# ['for the only smr based on advanced reactor technology thats already licensed and operating ap300 is the proven choice', 'the established design of the ap1000 reactor offers unequaled safety economic competitiveness and improved more efficient operations'],
# ['czech republic czech', 'global directory x'],
# ['bulgaria bulgarian', 'united kingdom english'],
# ['westinghousenuclearning', 'sweden swedish'],
# ['sweden swedish', 'poland polish'],
# ['westinghouse partners with richland county ems to host training video series', 'presidents kaizen week unlocks innovation across americas outage maintenance services'],
# ['for the only smr based on advanced reactor technology thats already licensed and operating ap300 is the proven choice', 'westinghousenuclearning'],
# ['sweden swedish', 'global directory x'],
# ['the evinci microreactor is a nextgeneration micromodular reactor combining innovative technologies with over 60 years of commercial nuclear expertise', 'the established design of the ap1000 reactor offers unequaled safety economic competitiveness and improved more efficient operations'],
# ['poland polish', 'ukraine ukrainian'],
# ['ukraine ukrainian', 'poland polish'],
# ['ukraine ukrainian', 'global directory x'],
# ['poland polish', 'japan japanese'],
# ['japan japanese', 'poland polish'],
# ['japan japanese', 'global directory x'],
# ['poland polish', 'united kingdom english'],
# ['united kingdom english', 'poland polish'],
# ['news', 'westinghousenuclearning'],
# ['united kingdom english', 'global directory x'],
# ['shape your future', 'manufacturing operations maintenance'],
# ['manufacturing operations maintenance', 'shape your future'],
# ['shape your future', 'evinci microreactor'],
# ['westinghousenuclearning', 'shape your future'],
# ['westinghouse ap1000 design receives us licensing extension to 2046', 'news'],
# ['westinghouse joins texas nuclear alliance as a founding member', 'news'],
# ['fermi america partners with westinghouse to support licensing for four ap1000 units', 'news'],
# ['product spotlights', 'westinghousenuclearning'],
# ['evinci microreactor', 'product spotlights'],
# ['product spotlights', 'westinghouseiq'],
# ['ap1000 pwr', 'product spotlights'],
    ]
    sample_pairs = ["balancing wind solar and nuclear power will help achieve a carbonfree future and positively impact our changing climate, over the past 50 years globally nuclear power has avoided nearly two years of the worlds energyrelated co2 emissions imagine how much more carbon pollution we can prevent", 
                    "Shaping Tomorrow's EnergyThrough Advanced Nuclear Technology", 
                    "Westinghouse Expands Supply Chain with Six UK Companies",
                    "Fermi America Partners with Westinghouse to Support Licensing for Four AP1000 Units",
                    "Atom Egoyan's haunting adaptation of the Russell Banks novel The Sweet Hereafter was the Canadian filmmaker's most successful film to date taking home a Special Grand Jury Prize at the 1997 Cannes Film Festival and scoring a pair of Academy Award nominations including Best Director. Restructured to fit Egoyan's signature mosaic narrative style the story concerns the cultural aftershocks which tear apart a small British Columbia town in the wake of a schoolbus accident which leaves a number of local children dead. Ian Holm stars as Mitchell Stephens a bigcity lawyer who arrives in the interest of uniting the survivors to initiate a lawsuit his maneuvering only drives the community further apart reopening old wounds and jeopardizing any hopes of emotional recovery. Like so many of Egoyan's features The Sweet Hereafter is a serious and painfully honest exploration of family grief no character is immune from the sense of utter devastation which grips the film not even the attorney whose interests are in part motivated by his own remorse over the fate of his daughter an HIVpositive drug addict."]
    labels = summairse(sample_pairs, dry_run_confirm=False, batch_size=1)
    print()
    for pair, label in zip(sample_pairs, labels):
        print("\t",pair)
        print(label)