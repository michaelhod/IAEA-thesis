# pip install openai>=1.40.0 tiktoken
from openai import OpenAI
import tiktoken
import os, re, sys, math

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Context window controls (tokens). Set real numbers via env for your model.
CONTEXT_WINDOW_TOKENS = 400,000
SAFETY_MARGIN_TOKENS  = 1000    # buffer to avoid overflows

# USD per 1K tokens (configure to your model’s pricing)
PRICE_IN_PER_1M  = 0.05
PRICE_OUT_PER_1M = 0.4

# Per-text truncation for long inputs
MAX_TOKENS_PER_TEXT = 128
MAX_NEW_TOKENS_GUESS_PER_LABEL = 2

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are a classifier. Given pairs of linked texts from an HTML document, classify the relationship of each pair into one of four categories and output ONLY the numbers.

Categories (choose exactly one per pair):
1 = Category/title: one is the other's title
2 = Context/metadata: one contains key context the other lacks
3 = Sibling items: similar/peer items without added context
4 = No helpful relation

Output format:
- For N pairs, output exactly N integers separated by single spaces, e.g. 1 3 2 4
- No brackets, commas, or extra text.
"""

USER_HEADER = "Classify each pair below. Output a plain, space-separated list of integers only.\n\n"
USER_FOOTER = "\n\nOutput:"

# ---------- token helpers ----------
ENC = tiktoken.get_encoding("cl100k_base")

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
    return f"{idx}. 1: {l_t}\n   2: {r_t}\n\n"

def _create_batches(pairs):
    """Yields batches to maximally fills the context window."""
    sys_tokens, header_tokens, footer_tokens = _count_tokens(SYSTEM_PROMPT), _count_tokens(USER_HEADER), _count_tokens(USER_FOOTER)

    i = 0
    while i < len(pairs):
        # start a new batch
        batch_start = i
        used = sys_tokens + header_tokens + footer_tokens
        lines = []
        total_out_tokens = 0

        # greedily add pairs until token limit would be exceeded
        while i < len(pairs):
            line = _create_entry_line((i - batch_start) + 1, pairs[i][0], pairs[i][1])
            line_tokens = _count_tokens(line)
            next_used = used + line_tokens
            next_out_guess = (total_out_tokens + MAX_NEW_TOKENS_GUESS_PER_LABEL)
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

def classify_link_pairs_minimal(pairs, dry_run_confirm=True):
    """
    pairs: list[[left, right]]
    return: list[int] with values in {1,2,3,4}; printed as "1 2 3 ..."
    """
    results = []

    batches = _create_batches(pairs)
    
    in_tokens = _count_tokens(SYSTEM_PROMPT)*len(batches) + sum(batches[:,3])
    out_tokens_guess = sum(batches[:,4])
    est_cost = _estimate_cost(in_tokens, out_tokens_guess)
    if dry_run_confirm:
        print(f"Expected total price: ${est_cost}")
        ans = input("Type Yes to continue this batch (anything else aborts): ").strip()
        if ans != "Yes":
            print("Aborted by user.")
            sys.exit(1)

    for batch_idx, (lo, hi, lines_joined, _, out_guess) in enumerate(batches, start=1):

        user_prompt = USER_HEADER + lines_joined + USER_FOOTER

        # token+cost estimate
        in_tokens = _count_tokens(SYSTEM_PROMPT) + _count_tokens(user_prompt)
        out_tokens_guess = max(4, out_guess)
        est_cost = _estimate_cost(in_tokens, out_tokens_guess)

        print(f"\nBatch {batch_idx} (pairs {lo}..{hi-1}): ~input tokens={in_tokens:,}, ~output tokens={out_tokens_guess:,}, "
              f"est. cost=${est_cost:.4f} (IN=${PRICE_IN_PER_1M}/1M, OUT=${PRICE_OUT_PER_1M}/1M)")

        # Call API (no JSON; tiny output)
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        text = resp.output[0].content[0].text.strip()

        # Parse exactly (hi - lo) labels in {1,2,3,4}
        want = hi - lo
        nums = re.findall(r"\b[1-4]\b", text)
        # if len(nums) != want:
        #     # soft fallback: split by whitespace, take first char if in 1-4
        #     tokens = text.split()
        #     soft = [t[0] for t in tokens if t and t[0] in "1234"]
        #     if len(soft) != want:
        #         raise ValueError(
        #             f"Expected {want} labels for batch {batch_idx}, got {len(nums)} (soft {len(soft)}). "
        #             f"Raw output:\n{text}"
        #         )
        #     nums = soft

        results.extend(int(x) for x in nums)

    return results


if __name__ == "__main__":
    # Example usage
    sample_pairs = [
        ["learn more", "learn more about why databricks selected westinghouse ai for its 2025 data intelligence in energy utility award, including judging criteria and methodology."],
        ["Director", "Christopher Nolan"],
        ["afcteams", "nfcteams"],
        ["Read more", "Imperial College London"],
        ["This is a very long paragraph describing the AP1000 reactor in detail, including safety systems, passive cooling, and economic competitiveness across global markets.", 
         "Explore the AP1000 overview page"],
        ["Another extremely long body of text ... " * 50, "Products"],
    ]

    labels = classify_link_pairs_minimal(sample_pairs, dry_run_confirm=True)
    print("\nFinal labels:")
    print(" ".join(str(x) for x in labels))