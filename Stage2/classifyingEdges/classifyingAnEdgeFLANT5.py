import os
os.environ.setdefault("HF_HOME", "/data/mjh24/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/mjh24/hf/transformers")
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
from typing import List, Tuple, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

LABEL_DEFS: Dict[int, str] = {
    1: "L contains key contextual information directly relevant to R that R is missing",
    # 3: "one side is an example of the other",
    2: "L only contains key contextual information directly relevant to R that R has",
    3: "there is no helpful relation between them",
}

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
        hypothesis_template: str = "For the purpose of fact extraction by combining L and R, the relation between L and R is that {}.",
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
    confidence_Factor = 0.85,
    return_scores: bool = False,
    labels = LABEL_DEFS
) -> Tuple[List[int], Optional[List[Dict[int, float]]]]:
    """
    Classify pairs using zero-shot NLI over label definitions.
    Gate: if best non-5 score < none_threshold => predict 5.
    confidence_Factor: How much more does the winning category need to be over "unrelated". if P(added_info)=0.99 but P(unhelpful)=0.7, output unhelpful 
    """
    candidate_defs: List[str] = [labels[i] for i in sorted(labels.keys())]
    zsl = DebertaZeroShot()

    seq1 = [f"L: {L}\nR: {R}" for (L, R) in pairs]
    s1 = zsl.label_scores(seq1, candidate_defs, batch_size=batch_size)

    if bidirectional:
        seq2 = [f"L: {R}\nR: {L}" for (L, R) in pairs]
        s2 = zsl.label_scores(seq2, candidate_defs, batch_size=batch_size)

    final_labels: List[int] = []
    debug_scores: List[Dict[int, float]] = []
    keys = sorted(labels.keys())

    for i in range(len(pairs)):
        avg: Dict[int, float] = {}
        # average per-definition scores across directions
        for lab_id, defn in labels.items():
            if bidirectional:
                avg_prob = 0.5 * (s1[i].get(defn, 0.0) + s2[i].get(defn, 0.0))
            else:
                avg_prob = s1[i].get(defn, 0.0)
            avg[lab_id] = float(avg_prob)

        last_key = keys[-1]
        best_non3 = max(avg[k] for k in keys[:-1])
        if avg[last_key] > best_non3*confidence_Factor:
            pred = last_key
        else:
            pred = int(max(avg.items(), key=lambda kv: kv[1])[0])
        # This logic picks 3 if it is higher than the best score * factor
        # Then it picks 2 if it is higher than the best score * factor
        # if avg[3] > best_non3*confidence_Factor_over_norelation:
        #     pred = 3
        # elif avg[2] > best_non3*confidence_Factor_over_irrelevantrelation:
        #     pred = 2 
        # else:
        #     pred = int(max(avg.items(), key=lambda kv: kv[1])[0])


        final_labels.append(pred)
        if return_scores:
            debug_scores.append(avg)

    return (final_labels, debug_scores) if return_scores else final_labels

def classify_link_pairs_zero_shot_two_step(
    pairs: List[Tuple[str, str]],
    batch_size: int = 16,
    bidirectional: bool = False,
    return_scores=False
    ):
    label_presence = {0: "L is of a different domain to R",
                    1: "L is of a similar domain to R"}
    label_type = {0: "L and R say the same thing", #"L and R contain the exact same key information"
                1: "L contains key contextual information that R is missing"}

    prescence, p_score = classify_link_pairs_zero_shot(pairs, batch_size, bidirectional, confidence_Factor=1, return_scores=True, labels=label_presence)
    typeofpair, type_score = classify_link_pairs_zero_shot(pairs, batch_size, bidirectional, confidence_Factor=1, return_scores=True, labels=label_type)
    results = []
    for p, t in zip(prescence, typeofpair):
        ans = 3 if p==0 else 2 if t==0 else 1
        results.append(ans)
    return (results, (p_score, type_score)) if return_scores else results


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

    # -------- Zero-shot backend --------
    print("\n=== Zero-shot (MoritzLaurer DeBERTa-v3) ===")
    labels_zs, scores_zs = classify_link_pairs_zero_shot_two_step(
        sample_pairs,
        batch_size=8,
        bidirectional=False,
        return_scores=True
    )
    for (pair, lab, sc_p, sc_t) in zip(sample_pairs, labels_zs, scores_zs[0], scores_zs[1] or []):
        print(lab, pair, "| scores: exists: ", {k: round(v, 3) for k, v in sc_p.items()}, " type: ", {k: round(v, 3) for k, v in sc_t.items()})

    # Quick summary
    def hist(xs):
        h = {i: 0 for i in range(1, 6)}
        for x in xs:
            h[x] += 1
        return h