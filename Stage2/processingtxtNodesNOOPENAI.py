# %%
import sys
sys.path.insert(1, r"/vol/bitbucket/mjh24/IAEA-thesis")
import os
os.environ.setdefault("HF_HOME", "/data/mjh24/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/mjh24/hf/transformers")
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
import torch
from Stage1.ExtractingGraphs.seleniumDriver import *
from Stage2.txtGraphExtraction.extract_mini_txt_graphs_helper import *
from Stage1.tree_helpers import *
from Stage2.txtGraphPruning.prune_txt_graph import main as prune_txt_graph
from Stage2.txtGraphPruning.graph_clustering import leiden_clustering, mini_graphs_from_clusters, louvain_clustering
from Stage2.classifyingEdges.classifyingFLANT5 import clean_instructional_text, classify_link_pairs_flan_batched
from collections import defaultdict
from pathlib import Path
import numpy as np
import pickle

CLIP_TO_N_EDGES_PER_NODE = 10 # This is to clip all the outgoing edges per node to a certain number
device = device or ("cuda" if torch.cuda.is_available() else "cpu")
REMOVE_DUPES = False


SRC_FOLDER = Path("/vol/bitbucket/mjh24/IAEA-thesis/data/swde/sourceCode/sourceCode/movie/movie/movie-allmovie(2000)")
SRC_GRAPHS = Path("/vol/bitbucket/mjh24/IAEA-thesis/data/allmovie")

def saveArrays(path,files,filenames):
    for file, name in zip(files,filenames):
        np.save(path/name, file)

LiedenfitnessResults = {}

graph_folders = sorted([f for f in SRC_GRAPHS.iterdir() if f.is_dir()], key=lambda p: str(p)[-4:])
graph_folders = graph_folders[226:114:-1]
for graph in graph_folders:
    sorted_label_index_extracted, xpaths_extracted, txts_extracted, probs_extracted = np.load(graph/"sorted_label_index.npy"), np.load(graph/"xpaths.npy"), np.load(graph/"txts.npy"), np.load(graph/"probs.npy")
    print("loaded: ", graph)

    # %%
    # Start the process of gathering facts
    LISTOFFACTS = []
    LISTOFXPATHS = []
    SRC_POST_BUTTON = graph / "Post_button_filter"
    SRC_POST_LOWPROB = graph / "Post_low_prop_filter"
    SRC_POST_SEMANTIC = graph / "Post_semantics_filter"
    SRC_POST_CLUSTERING = graph / "Post_clustering_filter"
    SRC_POST_BUTTON.mkdir(parents=True, exist_ok=True)
    SRC_POST_LOWPROB.mkdir(parents=True, exist_ok=True)
    SRC_POST_SEMANTIC.mkdir(parents=True, exist_ok=True)
    SRC_POST_CLUSTERING.mkdir(parents=True, exist_ok=True)

    # %%
    # Normalise the text
    normtxt = []
    for a, b in txts_extracted:
        normtxt.append([normalise_text(a, ",:\\-.%'\\s", lower=False), normalise_text(b, ",:\\-.%'\\s", lower=False)])
    txts, probs, sorted_label_index, xpaths = np.array(normtxt), np.array(probs_extracted), np.array(sorted_label_index_extracted), np.array(xpaths_extracted)

    # %% [markdown]
    # Extract unique nodes from the edges and create a mapping from node to edge

    # %%
    node_unique_label_index, node_to_edge_pos = np.unique(sorted_label_index, return_index=True)
    node_to_edge_pos = np.array([[int(idx/2),0] if idx%2==0 else [int(idx/2),1] for idx in node_to_edge_pos])
    node_to_edge_x_pos, node_to_edge_y_pos = node_to_edge_pos[:,0], node_to_edge_pos[:,1]

    node_unique_txts, node_unique_xpaths = txts[node_to_edge_x_pos,node_to_edge_y_pos], xpaths[node_to_edge_x_pos, node_to_edge_y_pos]

    # %% [markdown]
    # ### Step 3. ###
    # This filters out all the nodes where the text is a button/navigational link.
    # 
    # This uses a FLAN-T5 model

    # %%
    # Remove all html instructional labels
    def get_sents(text):
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sents if s.strip()] if sents!=[''] else ['-']
    
    txtsShortened = [[get_sents(a)[0], get_sents(b)[0]] for a, b in txts]
    isButton, buttonTxt = clean_instructional_text(txtsShortened, batch_size=512)
    opmask = np.array(isButton, dtype=bool)
    mask = np.logical_not(opmask)

#    print("All the nodes filtered out:\n\n")
    isButtonNode = np.zeros_like(node_unique_txts, dtype=bool)
    for key, value in buttonTxt.items():
        if value == 1:
            idx = np.where(node_unique_txts==key)
#            print(len(idx[0]),"-",key)
            isButtonNode[idx] = True

    # %%
    txts, probs, sorted_label_index, xpaths = txts[mask], probs[mask], sorted_label_index[mask], xpaths[mask]
    node_unique_label_index, node_to_edge_pos, node_unique_txts, node_unique_xpaths = node_unique_label_index[~isButtonNode], node_to_edge_pos[~isButtonNode], node_unique_txts[~isButtonNode], node_unique_xpaths[~isButtonNode]

    saveArrays(SRC_POST_BUTTON, [txts, probs, sorted_label_index, xpaths, node_unique_label_index, node_to_edge_pos, node_unique_txts, node_unique_xpaths], ["txts.npy", "probs.npy", "sorted_label_index.npy", "xpaths.npy", "node_unique_label_index.npy", "node_to_edge_pos.npy", "node_unique_txts.npy", "node_unique_xpaths.npy"])

    # %% [markdown]
    # ### Step 4. ###
    # This filters out all the low probability edges using the finding from the probability paper. At least two edges per node are saved

    # %%
    # -- RUN THE MAIN PRUNING MASK --
    mask = prune_txt_graph(probs, sorted_label_index, toloerance=0.01, remove_dupes=REMOVE_DUPES)

    # Concatanate and apply masks if we want specific text
    mask = np.array(mask, dtype=bool) 
    txts, probs, sorted_label_index, xpaths = txts[mask], probs[mask], sorted_label_index[mask], xpaths[mask]
    # -- PRUNING FINISHED --
    saveArrays(SRC_POST_LOWPROB, [txts, probs, sorted_label_index, xpaths], ["txts.npy", "probs.npy", "sorted_label_index.npy", "xpaths.npy"])

    # %% [markdown]
    # ### Step 5. ###
    # This filters out all the edges where the two text nodes make no sense together (i.e. they talk about different things)

    # %%
    # Classify the edges
    classificationFlan = classify_link_pairs_flan_batched(txts, batch_size=512)

    #metrics(classificationFlan[:len(y_true)], y_true)

    # %%
    mask = np.array([(False if i==3 else True) for i in classificationFlan])
    txts, probs, sorted_label_index, xpaths = txts[mask], probs[mask], sorted_label_index[mask], xpaths[mask]
    saveArrays(SRC_POST_SEMANTIC, [txts, probs, sorted_label_index, xpaths], ["txts.npy", "probs.npy", "sorted_label_index.npy", "xpaths.npy"])

    # %% [markdown]
    # ### 7b. Cluster the non sentence nodes together ###
    # This uses Leiden clustering. Louvain also available
    # 
    # This makes a new, simple classification of sentence as "more than x words" (instead of that above using FLAN-T5)

    # %%
    #To still do: Link all the small text nodes
    #Output XPaths so we know where the nodes came from
    NUMWORDSPERSENTENCE = np.inf
    manyWordsMask = np.zeros(len(sorted_label_index), dtype=bool)
    for idx, (first, second) in enumerate(txts):
        if len(first.split()) > NUMWORDSPERSENTENCE or len(second.split()) > NUMWORDSPERSENTENCE:
            manyWordsMask[idx] = True

    no_sent_label_index, no_sent_txts, no_sent_probs, no_sent_xpaths = sorted_label_index[~manyWordsMask], txts[~manyWordsMask], probs[~manyWordsMask], xpaths[~manyWordsMask]

    node_meta = defaultdict(lambda: {"texts": set(), "xpaths": set()})
    for (u, v), (t1, t2), (x1, x2) in zip(no_sent_label_index, no_sent_txts, no_sent_xpaths):
        node_meta[u]["texts"].add(t1); node_meta[u]["xpaths"].add(x1)
        node_meta[v]["texts"].add(t2); node_meta[v]["xpaths"].add(x2)

    clusters, fitness = leiden_clustering(no_sent_label_index, no_sent_probs, return_fitness=True)
    clusters = [cluster[::-1] for cluster in clusters] #Reorder IF LEIDEN as the output above seems to put titles at the bottom
    LiedenfitnessResults[str(graph)[-4:]] = fitness.score
    # %% [markdown]
    # Tried extra clustering within clusters but it didn't add much. Would of had to be fine tuned per website to give any useful information. semantic grouping would have been much better

    # %%
    mini_graphs, mini_probs = mini_graphs_from_clusters(no_sent_label_index, no_sent_probs, clusters)

    # %% [markdown]
    # ### 8. Final fact output ###
    # Outputing all the facts and groupings found, with xpaths to lead back to the source

    # %%
    new_edges = []
    new_txts = []
    new_xpaths = []
    new_probs = []
    for g, p in zip(mini_graphs, mini_probs):
        for edge, prob in zip(g, p):
            u, v = edge
            if [int(u), int(v)] not in sorted_label_index.tolist() and [int(v), int(u)] not in sorted_label_index.tolist():
                print(edge, "not found", end=", ")
                continue
            u_txt, u_xpath = "|".join(node_meta[u]["texts"]), "|".join(node_meta[u]["xpaths"])
            v_txt, v_xpath = "|".join(node_meta[v]["texts"]), "|".join(node_meta[v]["xpaths"])
            new_edges.append(edge)
            new_txts.append([u_txt, v_txt])
            new_xpaths.append([u_xpath, v_xpath])
            new_probs.append(prob)

    new_edges, new_txts, new_xpaths, new_probs = np.array(new_edges), np.array(new_txts), np.array(new_xpaths), np.array(new_probs)

    saveArrays(SRC_POST_CLUSTERING, [new_txts, new_probs, new_edges, new_xpaths], ["txts.npy", "probs.npy", "sorted_label_index.npy", "xpaths.npy"])

    # %%
    for i, cluster in enumerate(clusters):
        textInCluster, xpathInCluster = [], []
        for n in cluster:
            t, xp = ", ".join(node_meta[n]["texts"]), ", ".join(node_meta[n]["xpaths"])
            textInCluster.append(t)
            xpathInCluster.append(xp)
        LISTOFFACTS.append(textInCluster)
        LISTOFXPATHS.append(xpathInCluster)

    with open(SRC_POST_CLUSTERING/"TEXT_CLUSTERS.pkl", "wb") as f:
        pickle.dump(LISTOFFACTS, f)
    with open(SRC_POST_CLUSTERING/"XPATH_CLUSTERS.pkl", "wb") as f:
        pickle.dump(LISTOFXPATHS, f)

with open(SRC_GRAPHS/"Leiden_Fitness_Results225_1000.json", "w") as f:
    json.dump(LiedenfitnessResults, f)