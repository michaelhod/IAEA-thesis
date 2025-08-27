from openai import OpenAI
import os, json, time

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")  # or "gpt-4.1-mini" to save cost
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# Fixed label mapping
LABELS = {
    1: "description_of_text",
    2: "added_context",
    3: "sibling_content",
    4: "contains_instruction",
    5: "unrelated",
}

SYSTEM_PROMPT = """\
You are a classifier. Given two linked text snippets from an HTML document,
classify their relationship into one of four categories, and return ONLY an
array of integers, no explanations.

Categories:
1 = Category: one is the other's title or category heading,
2 = Contextual information: one contains key contextual information that the other is missing",
3 = they are similar items that don't add missing contextual information to eachother,
4 = there is no helpful relation between them,
"""

def classify_link_pairs_minimal(pairs, batch_size=32):
    results = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]

        # Format numbered prompt
        items = []
        for idx, (l, r) in enumerate(batch, 1):
            items.append(f"{idx}. 1: {l}\n   2: {r}")
        user_prompt = "Classify each pair below. Output an array of integers.\n\n" + "\n\n".join(items)

        # Call API
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={ "type": "json" },
            temperature=0,
        )

        # Parse result
        text = resp.output[0].content[0].text
        arr = json.loads(text)
        if not isinstance(arr, list):
            raise ValueError(f"Expected JSON list, got: {text}")
        results.extend(arr)
    return results


if __name__ == "__main__":
    sample_pairs = [
        ['learn more', 'learn more about why databricks selected westinghouse ai'],
        ['Director', 'Christopher Nolan'],
        ['afcteams', 'nfcteams'],
        ['Read more', 'Imperial College London'],
        ['Location', 'Bob Mortimer'],
    ]
    labels = classify_link_pairs_minimal(sample_pairs)
    print(labels)