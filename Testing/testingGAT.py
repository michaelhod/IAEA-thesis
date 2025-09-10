# %%
import sys
sys.path.insert(1, r"/vol/bitbucket/mjh24/IAEA-thesis")
from Stage1.GAT.GATModel import GraphAttentionNetwork
import torch
from Stage2.txtGraphExtraction.extract_mini_txt_graphs_helper import main as txtExtractor
from Stage1.tree_helpers import normalise_text
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
import re

# %%
SRC_GRAPHS = Path("/vol/bitbucket/mjh24/IAEA-thesis/data/allmovie")
with open("/vol/bitbucket/mjh24/IAEA-thesis/data/swde_expanded_dataset/dataset/movie/movie-allmovie(2000).json") as f:
    labels = json.load(f)

buttonfilter = "Post_button_filter"
clusteringfilter = "Post_clustering_filter"
lowprobfilter = "Post_low_prop_filter"
semanticsfilter = "Post_semantics_filter"

TESTINGFILTER = clusteringfilter
BUILD_LABELS = True

graph_folders = sorted([f/TESTINGFILTER for f in SRC_GRAPHS.iterdir() if f.is_dir()], key=lambda p: str(p)[-len(TESTINGFILTER)-5:-len(TESTINGFILTER)-1])
folder_map = {str(f)[-len(TESTINGFILTER)-5:-len(TESTINGFILTER)-1] + ".htm": f for f in graph_folders}
if BUILD_LABELS:
    results = {}
    list_of_k = []
    SimilarWorks = {"Is featured in:",
    "Is preceded by:",
    "Influenced:",
    "Is followed by:" ,
    "Is spoofed in:" ,
    "Has been remade as:",
    "Is a version of:",
    "Is influenced by:" ,
    "Is re-edited from:" ,
    "Has been re-edited into:",
    "Is a spoof of:",
    "Is related to:"}

    # assume `data` is the JSON you loaded with json.load(file)
    for parent_key, attributes in labels.items():
        edges = np.load(folder_map[parent_key]/"txts.npy")
        print("Loaded: ", parent_key)
        
        results[parent_key] = {}

        for attr_key, values in attributes.items():
            attr_key = attr_key.split("|")[-1].strip()
            for value in values:
                count = 0
                found = False
                for src, tgt in edges:
                    tgt = tgt.replace("\xa0", " ").strip()
                    tgt = re.sub(r"\s+", " ", tgt).strip()
                    src = src.replace("\xa0", " ").strip()
                    src = re.sub(r"\s+", " ", src).strip()
                    if "Similar Works" == src or src in SimilarWorks:
                        if len(value.split()) == 1:
                            tgt = tgt.split(" ")[0].strip()
                    if normalise_text(value.strip()) == normalise_text(tgt.strip()) or (len(value.split()) > 1 and normalise_text(value) in normalise_text(tgt)):
                        count += 1
                        if normalise_text(attr_key) in normalise_text(src):
                            found = True
                            break
                if found:
                    results[parent_key][(attr_key, value)] = count
                    list_of_k.append(count)
                else:
                    print((attr_key, value), "not found")
                    results[parent_key][(attr_key, value)] = -1
                    list_of_k.append(-1)

    # %%
    json_ready = {
        parent: {f"{k[0]} || {k[1]}": v for k, v in kvs.items()}
        for parent, kvs in results.items()
    }

    with open(f"./{TESTINGFILTER}_results.json", "w") as f: #commenting out so cannot accidentally be overwritten
        json.dump(json_ready, f, indent=4)

else:
# %%
    with open(f"./{TESTINGFILTER}_results.json", "r") as f:
        results = json.load(f)

# %%
# Count the % correct in a document for the 2000 documents (Recall per document)
distributions = []
ks = [1,2,3,4,5,10]
for k in ks:
    percentages = []
    for parent_key, kv_pairs in results.items():
        counts = kv_pairs.values()  # all the count numbers for this page
        total = len(counts)-1
        if total == 0:
            percentages.append(0.0)  # avoid division by zero
            continue

        # how many counts are in [0, 5]
        correct = sum(1 for c in counts if 0 < c <= k)

        # percentage for this page
        percentages.append(correct / total)
    
    distributions.append(percentages)


plt.figure(figsize=(9, 5))
plt.boxplot(distributions, tick_labels=[str(k) for k in ks], showmeans=True)
plt.xlabel("k (top-k per node/text endpoint)", fontdict={"fontsize":20})
plt.ylabel(r"Precision per document (% found)", fontdict={"fontsize":15})
plt.title("Proportion of positive hits per HTML file", fontdict={"fontsize":20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig("./{TESTINGFILTER}_kdist.png")

# %%
# Number of hits in k for all 2000*200 edges

all_counts = []
bars = []

for kv_pairs in results.values():
    for c in kv_pairs.values():
        all_counts.append(c)
for k in ks:
    hits = sum(1 for c in all_counts if 0 < c <= k)
    proportion = hits / len(all_counts)
    bars.append(proportion)

# plt.figure(figsize=(9, 5))
# plt.bar([str(k) for k in ks], bars)
# plt.xlabel("k (top-k per node/text endpoint)")
# plt.ylabel(r"Recall (% found)")
# plt.title("Proportion of positive hits overall")
# plt.tight_layout()
# plt.show()

# %%
#Get the top k edges per node. Then test the prospective edges 1=edge can be found. 2=edge cannot be found. Therefore this gives n F1 score that can be used. 
TP = sum(1 for c in all_counts if 0 < c <= 1)
FN = sum(1 for c in all_counts if c != 1) - 2000 # -2000 to ignore not found as will be skewed by "topic_entity"

FP = 0
TN = 0

for parent_key, attributes in labels.items():
    graph = folder_map[parent_key]
    edges = np.load(graph / "txts.npy")
    print("Loaded: ", parent_key)

    # Build a reverse lookup: value -> correct keys
    value_to_keys = {}
    for k, vals in attributes.items():
        for v in vals:
            value_to_keys.setdefault(v, set()).add(normalise_text(k))

    seen_right = set()

    for src, tgt in edges:
        # correct keys for this tgt (empty if tgt not in labels)
        correct_keys = value_to_keys.get(tgt, set())

        if tgt in value_to_keys and tgt not in seen_right:
            seen_right.add(tgt)
            # FP if left side is not the exact correct key
            if normalise_text(src) not in correct_keys:
                FP += 1
        else:
            # subsequent times → TN 
            if tgt in value_to_keys and normalise_text(src) not in correct_keys:
                TN += 1
                
print(TP, TN, FP, FN)

# %%
# Precision:  0.807973551948587
# Recall:  0.7364009274121633
# Accuracy:  0.9950741995666535
# F1:  0.7705287557273277
importantValues1 = [("Precision: ", TP/(TP+FP)), ("Recall: ", TP/(TP+FN)), ("Accuracy: ", (TP+TN)/(TP+FN+TN+FP)), ("F1: ", 2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP))+(TP/(TP+FN))))]
print("Precision: ", TP/(TP+FP))
print("Recall: ", TP/(TP+FN))
print("Accuracy: ", (TP+TN)/(TP+FN+TN+FP)) #TN is large. Massive class imbalance makes this look much better than it is
print("F1: ", 2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP))+(TP/(TP+FN))))

# %%
# This is the same as above, but the FP are not restricted to those values we expect to see. They are ALL the positive edges the model predicts BUT note that this is before pruning
TP = sum(1 for c in all_counts if 0 < c <= 1)
FN = sum(1 for c in all_counts if c != 1) - 2000

FP = 0
TN = 0

for parent_key, attributes in labels.items():
    graph = folder_map[parent_key]
    edges = np.load(graph / "txts.npy")
    print("Loaded: ", parent_key)

    # Build a reverse lookup: value -> correct keys
    value_to_keys = {}
    for k, vals in attributes.items():
        for v in vals:
            value_to_keys.setdefault(v, set()).add(normalise_text(k))

    seen_right = set()

    for src, tgt in edges:
        # correct keys for this tgt (empty if tgt not in labels)
        correct_keys = value_to_keys.get(tgt, set())

        if tgt not in seen_right:
            seen_right.add(tgt)
            # FP if left side is not the exact correct key
            if normalise_text(src) not in correct_keys:
                FP += 1
        else:
            # subsequent times → TN 
            if tgt in value_to_keys and normalise_text(src) not in correct_keys:
                TN += 1
                
# %%
# Precision:  0.3462412668429113
# Recall:  0.7364009274121633
# Accuracy:  0.9816746791196899
# F1:  0.47101875645428565
importantValues2=[("Precision: ", TP/(TP+FP)), ("Recall: ", TP/(TP+FN)), ("Accuracy: ", (TP+TN)/(TP+FN+TN+FP)), ("F1: ", 2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP))+(TP/(TP+FN))))]
print("Precision: ", TP/(TP+FP))
print("Recall: ", TP/(TP+FN))
print("Accuracy: ", (TP+TN)/(TP+FN+TN+FP)) #TN is large. Massive class imbalance makes this look much better than it is
print("F1: ", 2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP))+(TP/(TP+FN))))

# %%
# If the top predicted edge is assumed to be the positive prediction, and all others are negative, then we can get an edge-by-edge recall=precision=F1 score. This is because there is only one true edge per item, therefore: incorrect = FN=FP=1 → F1 reduces to accuracy.
per_category_scores = {}
for parent_key, kv_pairs in results.items():
    for kv_pair, k_hit in kv_pairs.items():
        key, value = kv_pair.split(" || ")
        
        if key not in per_category_scores:
            per_category_scores[key] = [0,0,0] # TP, FP, FN

        if k_hit == 1:
            per_category_scores[key][0] += 1
        else:
            per_category_scores[key][2] += 1

for parent_key, attributes in labels.items():
    graph = folder_map[parent_key]
    edges = np.load(graph / "txts.npy")
    print("Loaded: ", parent_key)

    # Build a reverse lookup: value -> correct keys
    value_to_keys = {}
    for k, vals in attributes.items():
        for v in vals:
            value_to_keys.setdefault(v, set()).add(normalise_text(k))

    seen_right = set()

    for src, tgt in edges:
        # correct keys for this tgt (empty if tgt not in labels)
        correct_keys = value_to_keys.get(tgt, set())
        if tgt in value_to_keys and tgt not in seen_right:
            seen_right.add(tgt)
            # FP if left side is not the exact correct key
            if normalise_text(src) not in correct_keys:
                if src in per_category_scores:
                    per_category_scores[src][1] += 1

# %%
output = []

counts = {}
for parent_key, kv_pairs in results.items():
    for kv_pair, k_hit in kv_pairs.items():
        key, value = kv_pair.split(" || ")
        
        if key not in counts:
            counts[key] = 0
        counts[key] += 1

for k, v in per_category_scores.items():
    TP, FP, FN = v
    if TP == 0:
        print("No matches for: ", k, v)
        pr,rec,f1=0,0,0
    else:
        pr, rec, f1 = TP/(TP+FP), TP/(TP+FN), 2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP))+(TP/(TP+FN)))
    output.append([k, pr, rec, f1])

output = sorted(output, key=lambda p: (p[-1], sum(p[-3:])))[::-1]

f1Scores = []
f1freq = []
for k, pr, rec, f1 in output:
    print(k, end=", ")
    print("Precision: ", pr, end=", ")
    print("Recall: ", rec, end=", ")
    print("F1: ", f1)
    f1freq.extend([f1]*counts[k])
    f1Scores.extend([f1])

plt.figure(figsize=(9, 5))
plt.hist(f1Scores, bins=30, color="skyblue", edgecolor="black", linewidth=1.2, alpha=0.85)
plt.xlabel("F1 score of a label", fontsize=18)
plt.ylabel("Number of F1 scores", fontsize=18)
plt.title("Distribution of Unique F1 Scores for Ground Truth Labels", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f"./{TESTINGFILTER}_f1dist.png")

plt.figure(figsize=(9, 5))
plt.hist(f1freq, bins=30, color="skyblue", edgecolor="black", linewidth=1.2, alpha=0.85)
plt.xlabel("F1 score of a label", fontsize=18)
plt.ylabel("Frequency of F1 score", fontsize=18)
plt.title("Frequency Distribution of F1 Scores for Ground Truth Labels", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig(f"./{TESTINGFILTER}_f1freq.png")

# %%

title_results = {}

# assume `data` is the JSON you loaded with json.load(file)
for parent_key in labels:
    titlePrediction = np.load(folder_map[parent_key]/"title_txt.npy")
    print("Loaded: ", parent_key)
    
    pageTitle = labels[parent_key]["topic_entity_name"][0]
    
    for idx, txt in enumerate(titlePrediction):
        if normalise_text(txt) == normalise_text(pageTitle):
            title_results[parent_key] = idx+1
            break
    
    if parent_key not in title_results:
        print(pageTitle, "not found")
        title_results[parent_key] = -1

# %%
# Count the % correct in a document for the 2000 documents (Recall per document)
percentages = []
ks = [1,2,3,4,5,10,25]
counts = title_results.values()
print(counts)
for k in ks:
    correct = sum(1 for c in counts if 0 < c <= k)
    percentages.append(correct / len(counts))

print(TESTINGFILTER)
print("Percentages", percentages)
print(importantValues1)
print(importantValues2)
# plt.figure(figsize=(9, 5))
# plt.bar([str(k) for k in ks], percentages)
# plt.xlabel("k (top-k hits)", fontdict={"fontsize":20})
# plt.ylabel(r"Proportion of positive hits", fontdict={"fontsize":15})
# plt.title("Proportion of positive hits per HTML file", fontdict={"fontsize":20})
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=15)
# plt.tight_layout()
# plt.show()


