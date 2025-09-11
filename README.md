# IAEA-Thesis: Fact Extraction from HTML with Graph Transformers

This repository contains the code and experiments for my MSc thesis at Imperial College London:  
**“Using Graph Neural Networks and LLMs to Extract Knowledge from HTML Files for the Purpose of Creating a Fact Database.”**

The project develops a **cheaper and more scalable alternative to LLM-only pipelines** (e.g. ChatGPT) for extracting verifiable facts from open-source webpages, particularly for nuclear non-proliferation analysis at the IAEA.  

---

## 🚀 Contributions

This work extends prior research (notably GraphScholarBERT) with two key innovations:

1. **Graph Transformer for HTML**  
   - Captures long-range dependencies in webpage layouts more effectively than Graph Attention Networks (GAT).  
   - Improves generalisation across different domains.

2. **Text-Independent Edge Classification**  
   - Removes reliance on parallel BERT edge embeddings and keyword queries.  
   - Uses graph features + lightweight filtering to predict relationships between HTML nodes.  
   - Produces **explainable outputs** by attaching XPaths to every extracted fact.

Additional contributions:
- **Explainability** – each extracted fact is linked back to its original HTML location(s) via XPath.  
- **Fact Clustering** – semi-structured text nodes are grouped into “mini-graphs” before being processed by smaller LLMs (FLAN-T5) or OpenAI APIs.  
- **Pipeline Efficiency** – runs in ~1m15s per webpage at ~$0.002 USD/article, enabling large-scale fact databases (40k articles ≈ $100 and ~4.5 days with 8 parallel pipelines).  

---

## 📊 Results

- Achieved strong performance on the **SWDE expanded dataset** (movies, NBA players, universities).  
- Graph Transformer + filtering/clustering: **Precision 0.83, Recall 0.87, F1 0.85**.  
- Outperforms DOM-LM, MarkupLM, and GraphScholarBERT in zero-shot generalisation.  
- Demonstrated scalability and cost-effectiveness compared to GPT-based extraction.  

---

## 📂 Repository Structure

```text
IAEA-thesis/
│── data/                       # Preprocessing scripts & dataset references (SWDE, Web Data Commons)
│── Stage1/                     # Graph Transformer and converting website to Graphs
│── Stage2/                     # Full extraction pipeline (graph building, edge prediction, filtering, clustering, fact extraction)
│── Testing/                    # Experimental outputs (metrics, loss curves)
│── Single_Webstie_Download/    # Downloading websites
```

---

## 🔧 Usage

Run the pipeline on a webpage:
Change the url in Step 1. in
```
Stage2/processingtxtNodes/ipynb
```

This outputs a set of facts + XPaths at the bottom of the notebook

---

## 📖 Thesis

Full details, methods, and evaluation can be found in the [Final Report](./Final%20Report.pdf).

---

## 🙏 Acknowledgments

Supervised by **Dr. Ovidiu Serban (Imperial College London)**  
Co-supervised by **Beth James (Ridgeway Information)** and **Stephen Francis (IAEA)**
