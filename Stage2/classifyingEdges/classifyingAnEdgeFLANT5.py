import os
os.environ.setdefault("HF_HOME", "/data/mjh24/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/mjh24/hf/transformers")
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
from typing import List, Tuple, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# LABELS_ID2NAME: Dict[int, str] = {
#     1: "added_context",
#     2: "added_information",
#     3: "purpose_of_text",
#     4: "sibling_content",
#     5: "unhelpful",
# }
# LABELS_NAME2ID = {v: k for k, v in LABELS_ID2NAME.items()}

LABEL_DEFS: Dict[int, str] = {
    1: "L is R's title or category heading",
    2: "L contains key contextual information that R is missing",
    # 3: "one side is an example of the other",
    3: "L and R contain the same information",
    4: "there is no helpful relation between them",
}
CANDIDATE_DEFS: List[str] = [LABEL_DEFS[i] for i in sorted(LABEL_DEFS.keys())]
UNHELP_DEF = LABEL_DEFS[4]

# ======================================
# Zero-shot DeBERTa-v3 (MoritzLaurer) backend
# ======================================

class DebertaZeroShot:
    """
    Zero-shot NLI pipeline with DeBERTa-v3 trained on MNLI+ANLI+FEVER+WANLI.
    """
    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        device: Optional[int] = None,
    ):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline(
            task="zero-shot-classification",
            model=mdl,
            tokenizer=tok,
            device=self.device,
        )

    @torch.no_grad()
    def label_scores(
        self,
        premises: List[str],
        candidate_defs: List[str],
        hypothesis_template: str = "For the purpose of fact extraction, the relation between L and R is that {}.",
        batch_size: int = 16,
    ) -> List[Dict[str, float]]:
        """
        Returns list of dicts mapping candidate_def -> probability (sigmoid if multi_label=True).
        """
        out = self.pipe(
            sequences=premises,
            candidate_labels=candidate_defs,
            hypothesis_template=hypothesis_template,
            multi_label=True,   # independent sigmoid per label
            batch_size=batch_size,
        )
        if isinstance(out, dict):
            out = [out]
        results = []
        for item in out:
            results.append(dict(zip(item["labels"], item["scores"])))
        return results


def classify_link_pairs_zero_shot(
    pairs: List[Tuple[str, str]],
    batch_size: int = 16,
    bidirectional: bool = True,
    confidence_Factor = 0.7,
    return_scores: bool = False,
) -> Tuple[List[int], Optional[List[Dict[int, float]]]]:
    """
    Classify pairs using zero-shot NLI over label definitions.
    Gate: if best non-5 score < none_threshold => predict 5.
    confidence_Factor: How much more does the winning category need to be over "unrelated". if P(added_info)=0.99 but P(unhelpful)=0.7, output unhelpful 
    """
    zsl = DebertaZeroShot()

    seq1 = [f"L: {L}\nR: {R}" for (L, R) in pairs]
    s1 = zsl.label_scores(seq1, CANDIDATE_DEFS, batch_size=batch_size)

    if bidirectional:
        seq2 = [f"L: {R}\nR: {L}" for (L, R) in pairs]
        s2 = zsl.label_scores(seq2, CANDIDATE_DEFS, batch_size=batch_size)

    final_labels: List[int] = []
    debug_scores: List[Dict[int, float]] = []

    for i in range(len(pairs)):
        avg: Dict[int, float] = {}
        # average per-definition scores across directions
        for lab_id, defn in LABEL_DEFS.items():
            if bidirectional:
                avg_prob = 0.5 * (s1[i].get(defn, 0.0) + s2[i].get(defn, 0.0))
            else:
                avg_prob = s1[i].get(defn, 0.0)
            avg[lab_id] = float(avg_prob)

        best_non4 = max(avg[k] for k in (1, 2, 3, 4))
        if avg[4] > best_non4*confidence_Factor:
            pred = 4
        else:
            pred = int(max(avg.items(), key=lambda kv: kv[1])[0])


        final_labels.append(pred)
        if return_scores:
            debug_scores.append(avg)

    return (final_labels, debug_scores) if return_scores else final_labels


# ======================
# Demo / smoke test
# ======================

if __name__ == "__main__":
    sample_pairs: List[Tuple[str, str]] = [
        ("british columbia canada", "set in"),
        ("set in", "british columbia canada"),
        ("for sexuality and some language", "mpaa reasons"),
        ("mpaa reasons", "for sexuality and some language"),
        ("addict", "accident"),
        ("accident", "addict"),
        ("other related works", "is related to"),
        ("is related to", "other related works"),
        ("drugs", "accident"),
        ("accident", "drugs"),
        ("in a minor key", "moods"),
        ("moods", "in a minor key"),
        ("drugs", "addict"),
        ("addict", "drugs"),
        ("director", "Atom Egoyan"),
        ("Car brand", "Toyota"),
        ("genres", "drama"),
    ]
    # -------- Zero-shot backend --------
    print("\n=== Zero-shot (MoritzLaurer DeBERTa-v3) ===")
    labels_zs, scores_zs = classify_link_pairs_zero_shot(
        sample_pairs,
        batch_size=8,
        bidirectional=True,
        confidence_Factor=0.7,
        return_scores=True,
    )
    for (pair, lab, sc) in zip(sample_pairs, labels_zs, scores_zs or []):
        print(lab, pair, "| scores:", {k: round(v, 3) for k, v in sc.items()})

    # Quick summary
    def hist(xs):
        h = {i: 0 for i in range(1, 6)}
        for x in xs:
            h[x] += 1
        return h